"""
Visualize success/failure grasps from exported DexGrasp q-groups with DRO visualization stack.

This script:
1) reads one action run under a tip-isaac run_root,
2) (re)computes per-grasp Isaac success labels for selected objects,
3) visualizes object mesh + predicted hand mesh + Isaac-final hand mesh in viser.

Run with --prepare_only first to build cache quickly without launching viewer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import trimesh
import viser
from termcolor import cprint

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model
from validation.validate_utils import validate_isaac


@dataclass
class ObjectEvalData:
    object_name: str
    q_eval: torch.Tensor      # (N, dof)
    q_isaac: torch.Tensor     # (N, dof)
    success: torch.Tensor     # (N,) bool
    success_rate_percent: float
    success_indices: List[int]
    failure_indices: List[int]


def _parse_args():
    parser = argparse.ArgumentParser(description="Visualize success/failure grasps in DRO viser.")
    parser.add_argument("--run_root", type=str, required=True, help="Tip run root, e.g. .../cmp_mse_repr_ablation_tip_ep200_...")
    parser.add_argument("--action", type=str, default="rot_euler", help="Action representation to visualize.")
    parser.add_argument("--robot_name", type=str, default="shadowhand")
    parser.add_argument("--object_name", type=str, default="", help="Optional single object to visualize.")
    parser.add_argument("--top_k_objects", type=int, default=2, help="If object_name empty, select top-K + bottom-K objects.")
    parser.add_argument("--max_samples_per_object", type=int, default=128, help="Limit grasps per object for speed.")
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--q_dof_mismatch", type=str, default="tail", choices=["tail", "head", "error"])
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--prepare_only", action="store_true", help="Only compute cache and print summary.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def _read_tsv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _find_action_run_dir(run_root: str, action: str) -> str:
    summary_tsv = os.path.join(run_root, "reports", "summary.tsv")
    rows = _read_tsv(summary_tsv)
    matches = [r for r in rows if r.get("action", "") == action and r.get("status", "") == "success"]
    if len(matches) == 0:
        raise RuntimeError(f"No successful action row found for action={action} in {summary_tsv}")
    run_name = matches[0]["run_name"]
    run_dir = os.path.join(run_root, run_name)
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    return run_dir


def _select_objects(run_dir: str, object_name: str, top_k: int) -> List[str]:
    if object_name:
        return [object_name]
    per_object_csv = os.path.join(run_dir, "isaac_eval", "isaac_per_object.csv")
    rows = _read_csv(per_object_csv)
    rows = sorted(rows, key=lambda r: _to_float(r.get("success_rate_percent", "0")), reverse=True)
    top = rows[: max(1, top_k)]
    bottom = list(reversed(rows[-max(1, top_k):]))
    objs = [r["object_name"] for r in top + bottom]
    # keep order, remove duplicates
    out = []
    seen = set()
    for o in objs:
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out


def _adapt_q_dof(q_batch: torch.Tensor, expected_dof: int, mode: str) -> torch.Tensor:
    q_dim = int(q_batch.shape[-1])
    if q_dim == expected_dof:
        return q_batch
    if mode == "error":
        raise RuntimeError(f"q dof mismatch: got {q_dim}, expected {expected_dof}")
    if q_dim > expected_dof:
        return q_batch[:, -expected_dof:] if mode == "tail" else q_batch[:, :expected_dof]
    pad = expected_dof - q_dim
    zeros = torch.zeros((q_batch.shape[0], pad), dtype=q_batch.dtype)
    return torch.cat([zeros, q_batch], dim=-1) if mode == "tail" else torch.cat([q_batch, zeros], dim=-1)


def _load_group_q(export_index_path: str, robot_name: str, object_name: str) -> torch.Tensor:
    with open(export_index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    export_root = os.path.dirname(export_index_path)
    groups = index.get("groups", [])
    for g in groups:
        if g.get("robot_name") == robot_name and g.get("object_name") == object_name:
            q_path = g["q_path"]
            if not os.path.isabs(q_path):
                q_path = os.path.join(export_root, q_path)
            q = torch.load(q_path, map_location="cpu").to(torch.float32)
            if q.ndim == 1:
                q = q.unsqueeze(0)
            return q
    raise RuntimeError(f"Group not found for [{robot_name}/{object_name}] in {export_index_path}")


def _object_mesh(object_name: str) -> trimesh.Trimesh:
    ds, obj = object_name.split("+")
    mesh_path = os.path.join(ROOT_DIR, "data", "data_urdf", "object", ds, obj, f"{obj}.stl")
    m = trimesh.load_mesh(mesh_path)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    return m


def _compute_or_load_object_eval(
    run_dir: str,
    robot_name: str,
    object_name: str,
    expected_dof: int,
    q_dof_mismatch: str,
    max_samples: int,
    chunk_size: int,
    gpu: int,
    force_recompute: bool,
) -> ObjectEvalData:
    cache_dir = os.path.join(run_dir, "isaac_eval", "vis_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{robot_name}_{object_name.replace('+', '__')}.pt")

    if os.path.exists(cache_path) and not force_recompute:
        data = torch.load(cache_path, map_location="cpu")
        success = data["success"].to(torch.bool).cpu()
        q_eval = data["q_eval"].to(torch.float32).cpu()
        q_isaac = data["q_isaac"].to(torch.float32).cpu()
    else:
        export_index_path = os.path.join(run_dir, "export_index.json")
        q_batch = _load_group_q(export_index_path, robot_name=robot_name, object_name=object_name)
        if max_samples > 0:
            q_batch = q_batch[:max_samples]
        q_eval = _adapt_q_dof(q_batch, expected_dof=expected_dof, mode=q_dof_mismatch).cpu()

        success_list = []
        q_isaac_list = []
        n = int(q_eval.shape[0])
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            q_chunk = q_eval[start:end]
            success_chunk, q_isaac_chunk = validate_isaac(robot_name, object_name, q_chunk, gpu=gpu)
            success_list.append(success_chunk.to(torch.bool).cpu())
            q_isaac_list.append(q_isaac_chunk.to(torch.float32).cpu())

        success = torch.cat(success_list, dim=0)
        q_isaac = torch.cat(q_isaac_list, dim=0)
        torch.save({"success": success, "q_eval": q_eval, "q_isaac": q_isaac}, cache_path)

    success_indices = torch.where(success)[0].tolist()
    failure_indices = torch.where(~success)[0].tolist()
    success_rate = (len(success_indices) / int(success.numel()) * 100.0) if success.numel() > 0 else 0.0
    return ObjectEvalData(
        object_name=object_name,
        q_eval=q_eval,
        q_isaac=q_isaac,
        success=success,
        success_rate_percent=success_rate,
        success_indices=success_indices,
        failure_indices=failure_indices,
    )


def main():
    args = _parse_args()
    run_root = os.path.abspath(args.run_root)
    run_dir = _find_action_run_dir(run_root, args.action)

    hand = create_hand_model(args.robot_name, device="cpu")
    expected_dof = int(hand.dof)

    object_names = _select_objects(run_dir, args.object_name, args.top_k_objects)
    cprint(f"[Vis] action={args.action}, objects={object_names}", "cyan")

    entries: List[ObjectEvalData] = []
    for obj in object_names:
        entry = _compute_or_load_object_eval(
            run_dir=run_dir,
            robot_name=args.robot_name,
            object_name=obj,
            expected_dof=expected_dof,
            q_dof_mismatch=args.q_dof_mismatch,
            max_samples=int(args.max_samples_per_object),
            chunk_size=int(args.chunk_size),
            gpu=int(args.gpu),
            force_recompute=bool(args.force_recompute),
        )
        entries.append(entry)
        cprint(
            f"[Prepared] {obj}: total={entry.success.numel()} success={len(entry.success_indices)} "
            f"fail={len(entry.failure_indices)} sr={entry.success_rate_percent:.2f}%",
            "green",
        )

    if args.prepare_only:
        cprint("[Done] prepare_only=true, cache generated.", "yellow")
        return

    server = viser.ViserServer(host=args.host, port=int(args.port))
    object_slider = server.gui.add_slider("object_idx", min=0, max=max(0, len(entries) - 1), step=1, initial_value=0)
    mode_slider = server.gui.add_slider("mode(0=success,1=failure)", min=0, max=1, step=1, initial_value=0)
    grasp_slider = server.gui.add_slider("grasp_idx", min=0, max=1000, step=1, initial_value=0)

    def _update():
        entry = entries[int(object_slider.value)]
        show_success = int(mode_slider.value) == 0
        idx_pool = entry.success_indices if show_success else entry.failure_indices
        mode_name = "success" if show_success else "failure"
        if len(idx_pool) == 0:
            cprint(f"[{entry.object_name}] no {mode_name} samples.", "yellow")
            return
        choose = idx_pool[int(grasp_slider.value) % len(idx_pool)]
        ok = bool(entry.success[choose].item())

        object_mesh = _object_mesh(entry.object_name)
        server.scene.add_mesh_simple(
            "object",
            object_mesh.vertices,
            object_mesh.faces,
            color=(239, 132, 167),
            opacity=0.85,
        )

        pred_mesh = hand.get_trimesh_q(entry.q_eval[choose])["visual"]
        server.scene.add_mesh_simple(
            "robot_pred",
            pred_mesh.vertices,
            pred_mesh.faces,
            color=(102, 192, 255),
            opacity=0.35,
        )

        isaac_mesh = hand.get_trimesh_q(entry.q_isaac[choose])["visual"]
        isaac_color = (50, 180, 80) if ok else (220, 70, 70)
        server.scene.add_mesh_simple(
            "robot_isaac",
            isaac_mesh.vertices,
            isaac_mesh.faces,
            color=isaac_color,
            opacity=0.75,
        )

        cprint(
            f"[{args.action}] object={entry.object_name} mode={mode_name} local={int(grasp_slider.value)%len(idx_pool)} "
            f"global={choose} success={ok}",
            "cyan",
        )

    object_slider.on_update(lambda _: _update())
    mode_slider.on_update(lambda _: _update())
    grasp_slider.on_update(lambda _: _update())
    _update()

    cprint(f"Viser running at http://{args.host}:{args.port}", "blue")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
