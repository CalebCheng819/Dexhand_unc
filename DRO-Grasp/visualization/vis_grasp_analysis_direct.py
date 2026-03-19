"""
Direct visualization + failure analysis for DexGrasp evaluation results.

Usage (analysis only, no viser):
    python vis_grasp_analysis_direct.py --run_dir /data/caleb/dexhand_output/dexgrasp_static_joint_value_seed2025_ep200 --stats_only

Usage (prepare Isaac per-grasp cache for 2 objects):
    python vis_grasp_analysis_direct.py --run_dir ... --prepare_only --top_k 1

Usage (full viser viewer):
    python vis_grasp_analysis_direct.py --run_dir ... --top_k 2 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import trimesh

# DRO-Grasp root (for hand model, validation, data paths)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Dexhand_unc project root (for dataset, configs)
PROJ_DIR = os.path.dirname(ROOT_DIR)
# ROOT_DIR must come first so DRO-Grasp's utils/ overrides Dexhand_unc's utils/
if PROJ_DIR not in sys.path:
    sys.path.append(PROJ_DIR)  # low priority (dataset, configs only)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)  # high priority (DRO-Grasp utils)

from utils.hand_model import create_hand_model
from validation.validate_utils import validate_isaac


# ───────────────────────── helpers ──────────────────────────

@dataclass
class ObjectEvalData:
    object_name: str
    q_pred: torch.Tensor    # (N, dof)  – exported model predictions
    q_isaac: torch.Tensor   # (N, dof)  – final hand pose after Isaac physics
    success: torch.Tensor   # (N,) bool
    success_rate_percent: float
    success_indices: List[int]
    failure_indices: List[int]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True,
                   help="Root dir of one training run, e.g. .../dexgrasp_static_joint_value_seed2025_ep200")
    p.add_argument("--object_name", type=str, default="",
                   help="Single object to visualize (e.g. contactdb+apple). Empty = auto-select.")
    p.add_argument("--top_k", type=int, default=2,
                   help="Number of best + worst objects to visualize (if object_name is empty).")
    p.add_argument("--max_samples", type=int, default=128, help="Max grasps per object.")
    p.add_argument("--chunk_size", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--robot_name", type=str, default="shadowhand")
    p.add_argument("--q_dof_mismatch", type=str, default="tail",
                   choices=["tail", "head", "error"])
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument("--stats_only", action="store_true",
                   help="Print distribution analysis and exit (no Isaac, no viser).")
    p.add_argument("--prepare_only", action="store_true",
                   help="Run Isaac eval and cache results, but don't launch viser.")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    return p.parse_args()


def _load_all_pred_q(export_dir: str) -> torch.Tensor:
    index_path = os.path.join(export_dir, "export_index.json")
    with open(index_path) as f:
        idx = json.load(f)
    parts = []
    for g in idx["groups"]:
        p = g["q_path"]
        if not os.path.isabs(p):
            p = os.path.join(export_dir, p)
        parts.append(torch.load(p, weights_only=True))
    return torch.cat(parts, 0)


def _load_per_object_csv(run_dir: str):
    path = os.path.join(run_dir, "isaac_eval", "isaac_per_object.csv")
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def _select_objects(run_dir: str, object_name: str, top_k: int) -> List[str]:
    if object_name:
        return [object_name]
    rows = _load_per_object_csv(run_dir)
    rows = sorted(rows, key=lambda r: float(r["success_rate_percent"]), reverse=True)
    top = rows[:max(1, top_k)]
    bottom = list(reversed(rows[-max(1, top_k):]))
    seen, out = set(), []
    for r in top + bottom:
        o = r["object_name"]
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out


def _adapt_dof(q: torch.Tensor, dof: int, mode: str) -> torch.Tensor:
    if q.shape[-1] == dof:
        return q
    if mode == "error":
        raise RuntimeError(f"DOF mismatch: {q.shape[-1]} vs {dof}")
    if q.shape[-1] > dof:
        return q[:, -dof:] if mode == "tail" else q[:, :dof]
    pad = torch.zeros(q.shape[0], dof - q.shape[-1])
    return torch.cat([pad, q], -1) if mode == "tail" else torch.cat([q, pad], -1)


def _get_pred_q_for_object(export_dir: str, robot_name: str, object_name: str,
                            max_samples: int, dof: int, dof_mode: str) -> torch.Tensor:
    index_path = os.path.join(export_dir, "export_index.json")
    with open(index_path) as f:
        idx = json.load(f)
    for g in idx["groups"]:
        if g["robot_name"] == robot_name and g["object_name"] == object_name:
            p = g["q_path"]
            if not os.path.isabs(p):
                p = os.path.join(export_dir, p)
            q = torch.load(p, weights_only=True)
            if max_samples > 0:
                q = q[:max_samples]
            return _adapt_dof(q, dof, dof_mode)
    raise RuntimeError(f"Object {object_name} not found in export_index")


def _compute_or_load(run_dir: str, export_dir: str, robot_name: str, object_name: str,
                     dof: int, dof_mode: str, max_samples: int, chunk_size: int,
                     gpu: int, force: bool) -> ObjectEvalData:
    cache_dir = os.path.join(run_dir, "isaac_eval", "vis_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{robot_name}_{object_name.replace('+', '__')}.pt")

    if os.path.exists(cache_path) and not force:
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        q_pred = data["q_pred"].float()
        q_isaac = data["q_isaac"].float()
        success = data["success"].bool()
    else:
        q_pred = _get_pred_q_for_object(export_dir, robot_name, object_name,
                                         max_samples, dof, dof_mode)
        slist, qlist = [], []
        for s in range(0, q_pred.shape[0], chunk_size):
            e = min(s + chunk_size, q_pred.shape[0])
            sc, qc = validate_isaac(robot_name, object_name, q_pred[s:e], gpu=gpu)
            slist.append(sc.bool().cpu())
            qlist.append(qc.float().cpu())
        success = torch.cat(slist, 0)
        q_isaac = torch.cat(qlist, 0)
        torch.save({"q_pred": q_pred, "q_isaac": q_isaac, "success": success}, cache_path)

    si = torch.where(success)[0].tolist()
    fi = torch.where(~success)[0].tolist()
    sr = len(si) / max(1, len(success)) * 100
    return ObjectEvalData(object_name, q_pred, q_isaac, success, sr, si, fi)


def _object_mesh(object_name: str) -> trimesh.Trimesh:
    ds, obj = object_name.split("+")
    mesh_path = os.path.join(ROOT_DIR, "data", "data_urdf", "object", ds, obj, f"{obj}.stl")
    m = trimesh.load_mesh(mesh_path)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    return m


# ───────────────── statistical analysis ─────────────────────

def _print_distribution_analysis(run_dir: str, export_dir: str):
    """Compare predicted q distribution to training GT distribution."""
    from termcolor import cprint

    cprint("\n" + "="*70, "cyan")
    cprint("  DISTRIBUTION ANALYSIS: Predicted q vs Ground-Truth q", "cyan")
    cprint("="*70, "cyan")

    # Load all predicted q
    pred_q = _load_all_pred_q(export_dir)
    cprint(f"\nPredicted: {pred_q.shape[0]} grasps, {pred_q.shape[1]} DOF", "white")

    # Per-object CSV summary
    rows = _load_per_object_csv(run_dir)
    cprint("\n── Per-object Isaac success rates ──", "yellow")
    rates = []
    for r in sorted(rows, key=lambda x: float(x["success_rate_percent"]), reverse=True):
        sr = float(r["success_rate_percent"])
        rates.append(sr)
        bar = "█" * int(sr / 5)
        cprint(f"  {r['object_name']:35s} {sr:5.1f}%  {bar}", "white" if sr > 10 else "red")

    overall = float(sum(int(r["success_num"]) for r in rows)) / max(1, sum(int(r["num_grasps"]) for r in rows)) * 100
    cprint(f"\n  Overall: {overall:.2f}%  ({sum(int(r['success_num']) for r in rows)}/{sum(int(r['num_grasps']) for r in rows)})", "yellow")

    # Diversity analysis
    cprint("\n── Prediction diversity (std per DOF) ──", "yellow")
    pred_std = pred_q.std(0)
    low_diversity_dims = (pred_std < 0.1).nonzero(as_tuple=True)[0].tolist()
    cprint(f"  Dims with std < 0.1 (collapsed): {low_diversity_dims}", "red")
    cprint(f"  Mean std across dims: {pred_std.mean():.4f}", "white")
    cprint(f"  Min std: dim{pred_std.argmin()} = {pred_std.min():.4f}", "white")

    # Saturation check
    cprint("\n── Saturation at ±1.0 limits (> 5%) ──", "yellow")
    for i in range(pred_q.shape[1]):
        sp = (pred_q[:, i] >= 0.999).float().mean().item()
        sn = (pred_q[:, i] <= -0.999).float().mean().item()
        if sp > 0.05 or sn > 0.05:
            cprint(f"  dim{i:2d}: +1.0={sp*100:.1f}%  -1.0={sn*100:.1f}%", "white")

    # Load GT from CMapDataset directly
    try:
        cmap_path = os.path.join(ROOT_DIR, "data", "CMapDataset_filtered", "cmap_dataset.pt")
        split_path = os.path.join(ROOT_DIR, "data", "CMapDataset_filtered", "split_train_validate_objects.json")
        cmap = torch.load(cmap_path, map_location="cpu", weights_only=False)
        with open(split_path) as f:
            split_data = json.load(f)
        val_objects = set(split_data["validate"])

        # Each metadata item: (q_tensor, object_name, robot_name)
        gt_qs = []
        for item in cmap["metadata"]:
            q_raw, obj_name, rname = item[0], item[1], item[2]
            if rname == "shadowhand" and obj_name in val_objects:
                gt_qs.append(q_raw)
        if not gt_qs:
            raise RuntimeError("No shadowhand validate items found")
        gt_q_raw = torch.stack(gt_qs, 0).float()  # (N, 22) raw DRO dof

        # DRO q is 22-dim; exported pred_q is 30-dim (env_act_dim=6 + rot_act_dim=24)
        # The last 24 dims of pred_q correspond to joint values, first 6 are wrist pose
        # DRO q_dof_mismatch=tail means pred_q[:, -22:] ≈ gt_q_raw
        pred_joint = pred_q[:, -gt_q_raw.shape[1]:]  # last 22 dims
        cprint(f"\n  GT (DRO raw 22-DOF): {gt_q_raw.shape}, Pred joint tail: {pred_joint.shape}", "white")

        cprint(f"\n── Pred vs GT: bias and std ratio (flagged) ──", "yellow")
        cprint(f"  {'dim':>4} | {'GT_mean':>8} | {'PR_mean':>8} | {'GT_std':>7} | {'PR_std':>7} | {'BIAS':>6} | {'ratio':>6} | issues", "white")
        cprint("  " + "-"*72, "white")
        big_issues = []
        n = min(gt_q_raw.shape[0], pred_joint.shape[0])
        for i in range(pred_joint.shape[1]):
            gm = gt_q_raw[:n, i].mean().item()
            pm = pred_joint[:n, i].mean().item()
            gs = gt_q_raw[:n, i].std().item()
            ps = pred_joint[:n, i].std().item()
            bias = pm - gm
            ratio = ps / (gs + 1e-6)
            flags = []
            if abs(bias) > 0.15: flags.append("BIAS")
            if ratio < 0.5: flags.append("LOW_STD")
            if flags:
                color = "red" if "BIAS" in flags else "yellow"
                cprint(f"  d{i:2d}  | {gm:+8.3f} | {pm:+8.3f} | {gs:7.3f} | {ps:7.3f} | {bias:+6.3f} | {ratio:5.2f}x | {' '.join(flags)}", color)
                big_issues.append((i, bias, ratio, flags))

        l1 = (pred_joint[:n] - gt_q_raw[:n]).abs().mean(0)
        top5 = l1.topk(min(5, l1.numel()))
        cprint(f"\n── Top-5 L1 error dims (pred vs GT) ──", "yellow")
        for v, didx in zip(top5.values, top5.indices):
            cprint(f"  dim{didx.item():2d}: L1={v.item():.4f}", "red" if v > 0.5 else "white")
        cprint(f"  mean L1 across all dims: {l1.mean().item():.4f}", "white")

        cprint(f"\n── Root cause summary ──", "yellow")
        n_low = sum(1 for _,_,r,f in big_issues if 'LOW_STD' in f)
        n_bias = sum(1 for _,b,_,f in big_issues if 'BIAS' in f)
        cprint(f"  [1] MEAN COLLAPSE: {n_low}/{pred_joint.shape[1]} dims have std ratio < 0.5", "red")
        cprint(f"      → Model blurs over bimodal pose distributions (e.g. wrist left/right)", "red")
        cprint(f"  [2] BIAS: {n_bias} dims have |mean error| > 0.15 rad", "red")
        cprint(f"  [3] OBS COLLAPSE: stats-encoder (mean/std/min/max of canonical hand cloud)", "red")
        cprint(f"      is IDENTICAL for all samples → model cannot distinguish grasp types", "red")
        cprint(f"  [4] SOLUTION: richer observation (pointnet on object + target hand config)", "red")

    except Exception as e:
        cprint(f"\n  [GT comparison skipped: {e}]", "yellow")

    cprint("\n" + "="*70 + "\n", "cyan")


# ───────────────────── main ──────────────────────────────────

def main():
    args = _parse_args()
    run_dir = os.path.abspath(args.run_dir)
    export_dir = os.path.join(run_dir, "final_q_export")

    if not os.path.exists(export_dir):
        raise FileNotFoundError(f"export_dir not found: {export_dir}\n"
                                f"Run export_dexgrasp_final_q.py first.")

    # Always print statistics
    _print_distribution_analysis(run_dir, export_dir)

    if args.stats_only:
        return

    # Select objects and run/load Isaac eval
    object_names = _select_objects(run_dir, args.object_name, args.top_k)

    hand = create_hand_model(args.robot_name, device="cpu")
    dof = int(hand.dof)

    from termcolor import cprint
    entries: List[ObjectEvalData] = []
    for obj in object_names:
        cprint(f"[Isaac] Computing/loading: {obj} ...", "cyan")
        entry = _compute_or_load(
            run_dir=run_dir, export_dir=export_dir,
            robot_name=args.robot_name, object_name=obj,
            dof=dof, dof_mode=args.q_dof_mismatch,
            max_samples=args.max_samples, chunk_size=args.chunk_size,
            gpu=args.gpu, force=args.force_recompute,
        )
        entries.append(entry)
        cprint(f"  {obj}: {entry.success_rate_percent:.1f}%  "
               f"({len(entry.success_indices)} ok / {entry.success.numel()} total)", "green")

    if args.prepare_only:
        cprint("[Done] prepare_only – cache ready.", "yellow")
        return

    # ── viser viewer ──────────────────────────────────────────
    import viser

    server = viser.ViserServer(host=args.host, port=args.port)
    cprint(f"\nViser running at http://{args.host}:{args.port}", "blue")

    obj_slider  = server.gui.add_slider("object_idx",           min=0, max=max(0, len(entries)-1), step=1, initial_value=0)
    mode_slider = server.gui.add_slider("mode(0=succ,1=fail)",  min=0, max=1, step=1, initial_value=0)
    grasp_slider= server.gui.add_slider("grasp_idx",            min=0, max=1000, step=1, initial_value=0)
    show_pred   = server.gui.add_checkbox("show_pred(blue)",    initial_value=True)
    show_isaac  = server.gui.add_checkbox("show_isaac(result)", initial_value=True)

    def _update(_=None):
        entry = entries[int(obj_slider.value)]
        show_success = int(mode_slider.value) == 0
        pool = entry.success_indices if show_success else entry.failure_indices
        mode_label = "SUCCESS" if show_success else "FAILURE"
        if not pool:
            cprint(f"[{entry.object_name}] no {mode_label} samples.", "yellow")
            return
        choose = pool[int(grasp_slider.value) % len(pool)]
        ok = bool(entry.success[choose].item())

        # object mesh
        obj_mesh = _object_mesh(entry.object_name)
        server.scene.add_mesh_simple("object", obj_mesh.vertices, obj_mesh.faces,
                                     color=(239, 132, 167), opacity=0.85)

        # predicted hand (blue, semi-transparent)
        if show_pred.value:
            pm = hand.get_trimesh_q(entry.q_pred[choose])["visual"]
            server.scene.add_mesh_simple("pred", pm.vertices, pm.faces,
                                         color=(80, 160, 255), opacity=0.35)
        else:
            try: server.scene.remove("pred")
            except Exception: pass

        # Isaac final hand (green=success, red=failure)
        if show_isaac.value:
            im = hand.get_trimesh_q(entry.q_isaac[choose])["visual"]
            color = (50, 200, 80) if ok else (220, 60, 60)
            server.scene.add_mesh_simple("isaac", im.vertices, im.faces,
                                         color=color, opacity=0.80)
        else:
            try: server.scene.remove("isaac")
            except Exception: pass

        cprint(
            f"[{mode_label}] {entry.object_name}  idx={choose}  success={ok}  "
            f"SR={entry.success_rate_percent:.1f}%",
            "green" if ok else "red",
        )

    for ctrl in [obj_slider, mode_slider, grasp_slider, show_pred, show_isaac]:
        ctrl.on_update(_update)
    _update()

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
