"""
Save static success/failure pose visualizations (with object point cloud) from exported DexGrasp q-groups.

This script is intended for report-ready snapshots:
1) load one export_index.json (from isaac_small_eval/epoch_xxxx/export),
2) compute per-grasp Isaac success labels and final q (with cache),
3) save PNGs for success/failure examples per object.

Example:
  /data2/caleb/conda_envs/dro_ig_py38/bin/python \
    DRO-Grasp/visualization/save_exported_isaac_pose_cases.py \
    --export_index /data/caleb/dexhand_output/.../isaac_small_eval/epoch_0349/export/export_index.json \
    --out_dir /data/caleb/dexhand_output/.../isaac_small_eval/epoch_0349/pose_cases \
    --robot_name shadowhand --gpu 0 --cases_per_mode 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from termcolor import cprint  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.hand_model import create_hand_model  # noqa: E402
from validation.validate_utils import validate_isaac  # noqa: E402


@dataclass
class ObjectEvalData:
    object_name: str
    q_pred: torch.Tensor
    q_isaac: torch.Tensor
    success: torch.Tensor
    success_rate_percent: float
    success_indices: List[int]
    failure_indices: List[int]


def _parse_args():
    parser = argparse.ArgumentParser(description="Save success/failure pose visualizations with point cloud.")
    parser.add_argument("--export_index", type=str, required=True, help="Path to export_index.json")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save output PNGs")
    parser.add_argument("--robot_name", type=str, default="shadowhand")
    parser.add_argument("--object_names", type=str, default="", help="Comma separated object names; empty means all in export_index")
    parser.add_argument("--max_samples_per_object", type=int, default=128)
    parser.add_argument("--cases_per_mode", type=int, default=1, help="How many success / failure images per object")
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--q_dof_mismatch", type=str, default="tail", choices=["tail", "head", "error"])
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=42.0)
    parser.add_argument("--point_size", type=float, default=5.0)
    return parser.parse_args()


def _load_export_index(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_q_path(export_root: str, rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(export_root, rel_or_abs)


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


def _load_object_pointcloud(object_name: str) -> np.ndarray:
    ds, obj = object_name.split("+")
    pc_path = os.path.join(ROOT_DIR, "data", "PointCloud", "object", ds, f"{obj}.pt")
    pc = torch.load(pc_path, map_location="cpu")
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise RuntimeError(f"Unexpected point cloud shape for {object_name}: {tuple(pc.shape)}")
    return pc[:, :3].cpu().numpy()


def _load_object_mesh(object_name: str) -> trimesh.Trimesh:
    ds, obj = object_name.split("+")
    mesh_path = os.path.join(ROOT_DIR, "data", "data_urdf", "object", ds, obj, f"{obj}.stl")
    mesh = trimesh.load_mesh(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    return mesh


def _pick_indices(indices: List[int], n: int) -> List[int]:
    if len(indices) == 0 or n <= 0:
        return []
    if len(indices) <= n:
        return list(indices)
    # Evenly-spaced picks for diversity.
    xs = np.linspace(0, len(indices) - 1, num=n, dtype=int)
    return [indices[int(i)] for i in xs.tolist()]


def _compute_or_load_object_eval(
    export_index: Dict,
    export_root: str,
    object_name: str,
    robot_name: str,
    hand_dof: int,
    q_dof_mismatch: str,
    max_samples: int,
    chunk_size: int,
    gpu: int,
    cache_dir: str,
    force_recompute: bool,
) -> ObjectEvalData:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{robot_name}_{object_name.replace('+', '__')}.pt")
    if os.path.exists(cache_path) and not force_recompute:
        data = torch.load(cache_path, map_location="cpu")
        q_pred = data["q_pred"].to(torch.float32).cpu()
        q_isaac = data["q_isaac"].to(torch.float32).cpu()
        success = data["success"].to(torch.bool).cpu()
    else:
        group = None
        for g in export_index.get("groups", []):
            if str(g.get("robot_name", "")) == robot_name and str(g.get("object_name", "")) == object_name:
                group = g
                break
        if group is None:
            raise RuntimeError(f"Group not found for [{robot_name}/{object_name}]")
        q_path = _resolve_q_path(export_root, str(group["q_path"]))
        q_pred_raw = torch.load(q_path, map_location="cpu").to(torch.float32)
        if q_pred_raw.ndim == 1:
            q_pred_raw = q_pred_raw.unsqueeze(0)
        if max_samples > 0:
            q_pred_raw = q_pred_raw[:max_samples]
        q_pred = _adapt_q_dof(q_pred_raw, expected_dof=hand_dof, mode=q_dof_mismatch).cpu()

        success_list = []
        q_isaac_list = []
        n = int(q_pred.shape[0])
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            q_chunk = q_pred[start:end]
            success_chunk, q_isaac_chunk = validate_isaac(robot_name, object_name, q_chunk, gpu=gpu)
            success_list.append(success_chunk.to(torch.bool).cpu())
            q_isaac_list.append(q_isaac_chunk.to(torch.float32).cpu())
        success = torch.cat(success_list, dim=0)
        q_isaac = torch.cat(q_isaac_list, dim=0)
        torch.save({"q_pred": q_pred, "q_isaac": q_isaac, "success": success}, cache_path)

    success_indices = torch.where(success)[0].tolist()
    failure_indices = torch.where(~success)[0].tolist()
    success_rate = (len(success_indices) / int(success.numel()) * 100.0) if success.numel() > 0 else 0.0
    return ObjectEvalData(
        object_name=object_name,
        q_pred=q_pred,
        q_isaac=q_isaac,
        success=success,
        success_rate_percent=success_rate,
        success_indices=success_indices,
        failure_indices=failure_indices,
    )


def _set_equal_axes(ax, xyz: np.ndarray):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0)
    radius = max(radius, 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _draw_pose_case(
    object_name: str,
    case_label: str,
    idx: int,
    q_pred_i: torch.Tensor,
    q_isaac_i: torch.Tensor,
    is_success: bool,
    hand,
    out_path: str,
    elev: float,
    azim: float,
    point_size: float,
    success_rate: float,
):
    obj_pc = _load_object_pointcloud(object_name)
    obj_mesh = _load_object_mesh(object_name)
    pred_mesh = hand.get_trimesh_q(q_pred_i)["visual"]
    isaac_mesh = hand.get_trimesh_q(q_isaac_i)["visual"]

    fig = plt.figure(figsize=(8.4, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    # object point cloud
    ax.scatter(
        obj_pc[:, 0],
        obj_pc[:, 1],
        obj_pc[:, 2],
        s=point_size,
        c="#f08ab6",
        alpha=0.70,
        linewidths=0.0,
        label="object point cloud",
    )

    # object mesh (light transparent)
    obj_faces = obj_mesh.vertices[obj_mesh.faces]
    obj_poly = Poly3DCollection(obj_faces, alpha=0.10, facecolor="#f08ab6", edgecolor="none")
    ax.add_collection3d(obj_poly)

    # predicted hand (blue)
    pred_faces = pred_mesh.vertices[pred_mesh.faces]
    pred_poly = Poly3DCollection(pred_faces, alpha=0.25, facecolor="#5ca9ff", edgecolor="none")
    ax.add_collection3d(pred_poly)

    # isaac final hand (green/red)
    final_color = "#32c850" if is_success else "#dc3c3c"
    isaac_faces = isaac_mesh.vertices[isaac_mesh.faces]
    isaac_poly = Poly3DCollection(isaac_faces, alpha=0.78, facecolor=final_color, edgecolor="none")
    ax.add_collection3d(isaac_poly)

    # equal limits from combined points
    xyz = np.concatenate(
        [
            obj_pc[:, :3],
            np.asarray(pred_mesh.vertices, dtype=np.float32),
            np.asarray(isaac_mesh.vertices, dtype=np.float32),
        ],
        axis=0,
    )
    _set_equal_axes(ax, xyz)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(False)
    ax.set_title(
        f"{object_name} | {case_label.upper()} | idx={idx} | object_sr={success_rate:.2f}%",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    args = _parse_args()
    export_index_path = os.path.abspath(args.export_index)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    export_index = _load_export_index(export_index_path)
    export_root = os.path.dirname(export_index_path)

    hand = create_hand_model(args.robot_name, device=torch.device("cpu"))
    hand_dof = int(hand.dof)

    if args.object_names.strip():
        object_names = [x.strip() for x in args.object_names.split(",") if x.strip()]
    else:
        object_names = []
        for g in export_index.get("groups", []):
            if str(g.get("robot_name", "")) == args.robot_name:
                object_names.append(str(g.get("object_name", "")))
        # keep stable order + unique
        seen = set()
        object_names = [o for o in object_names if not (o in seen or seen.add(o))]

    if len(object_names) == 0:
        raise RuntimeError("No objects selected from export_index.")

    cache_dir = os.path.join(out_dir, "cache")
    summary_rows = []
    for obj in object_names:
        cprint(f"[Prepare] {obj}", "cyan")
        entry = _compute_or_load_object_eval(
            export_index=export_index,
            export_root=export_root,
            object_name=obj,
            robot_name=args.robot_name,
            hand_dof=hand_dof,
            q_dof_mismatch=args.q_dof_mismatch,
            max_samples=int(args.max_samples_per_object),
            chunk_size=int(args.chunk_size),
            gpu=int(args.gpu),
            cache_dir=cache_dir,
            force_recompute=bool(args.force_recompute),
        )
        cprint(
            f"  total={entry.success.numel()} success={len(entry.success_indices)} "
            f"failure={len(entry.failure_indices)} sr={entry.success_rate_percent:.2f}%",
            "green",
        )

        succ_pick = _pick_indices(entry.success_indices, int(args.cases_per_mode))
        fail_pick = _pick_indices(entry.failure_indices, int(args.cases_per_mode))

        for idx in succ_pick:
            out_png = os.path.join(
                out_dir,
                f"{obj.replace('+', '__')}_success_idx{idx:04d}.png",
            )
            _draw_pose_case(
                object_name=obj,
                case_label="success",
                idx=idx,
                q_pred_i=entry.q_pred[idx],
                q_isaac_i=entry.q_isaac[idx],
                is_success=True,
                hand=hand,
                out_path=out_png,
                elev=float(args.elev),
                azim=float(args.azim),
                point_size=float(args.point_size),
                success_rate=float(entry.success_rate_percent),
            )
            summary_rows.append((obj, "success", idx, out_png))

        for idx in fail_pick:
            out_png = os.path.join(
                out_dir,
                f"{obj.replace('+', '__')}_failure_idx{idx:04d}.png",
            )
            _draw_pose_case(
                object_name=obj,
                case_label="failure",
                idx=idx,
                q_pred_i=entry.q_pred[idx],
                q_isaac_i=entry.q_isaac[idx],
                is_success=False,
                hand=hand,
                out_path=out_png,
                elev=float(args.elev),
                azim=float(args.azim),
                point_size=float(args.point_size),
                success_rate=float(entry.success_rate_percent),
            )
            summary_rows.append((obj, "failure", idx, out_png))

    summary_txt = os.path.join(out_dir, "case_list.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        for obj, mode, idx, path in summary_rows:
            f.write(f"{obj}\t{mode}\t{idx}\t{path}\n")

    cprint(f"[Done] Saved {len(summary_rows)} images to: {out_dir}", "yellow")
    cprint(f"[Done] Case list: {summary_txt}", "yellow")


if __name__ == "__main__":
    main()

