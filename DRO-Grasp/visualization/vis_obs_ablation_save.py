"""
Save observation ablation visualizations for:
1) fingertip points + object point cloud
2) full-hand surface points + object point cloud

This script is non-interactive and writes PNG + CSV outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DRO_ROOT = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(DRO_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.hand_model import create_hand_model  # noqa: E402


def _stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def _load_split_objects(dro_root: str, split: str) -> set:
    split_path = os.path.join(dro_root, "data", "CMapDataset_filtered", "split_train_validate_objects.json")
    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    if split not in split_data:
        raise KeyError(f"split={split} not in split file keys={list(split_data.keys())}")
    return set(split_data[split])


def _load_metadata(dro_root: str) -> list:
    dataset_path = os.path.join(dro_root, "data", "CMapDataset_filtered", "cmap_dataset.pt")
    raw = torch.load(dataset_path, map_location="cpu")
    metadata = raw.get("metadata", None)
    if metadata is None:
        raise RuntimeError(f"Missing metadata in {dataset_path}")
    return metadata


def _extract_item(meta_item) -> Tuple[str, str]:
    if not isinstance(meta_item, (list, tuple)) or len(meta_item) < 3:
        raise RuntimeError(f"Unexpected metadata item format: {type(meta_item)}")
    object_name = str(meta_item[1])
    robot_name = str(meta_item[2])
    return object_name, robot_name


def _load_object_pc(dro_root: str, object_name: str) -> torch.Tensor:
    dataset_type, obj = object_name.split("+")
    pc_path = os.path.join(dro_root, "data", "PointCloud", "object", dataset_type, f"{obj}.pt")
    pc = torch.load(pc_path, map_location="cpu")
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise RuntimeError(f"Invalid object point cloud shape: {tuple(pc.shape)} at {pc_path}")
    return pc[:, :3].to(torch.float32)


def _load_object_mesh(dro_root: str, object_name: str) -> trimesh.Trimesh:
    dataset_type, obj = object_name.split("+")
    mesh_path = os.path.join(dro_root, "data", "data_urdf", "object", dataset_type, obj, f"{obj}.stl")
    mesh = trimesh.load_mesh(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Unsupported mesh type for {mesh_path}: {type(mesh)}")
    return mesh


def _sample_object_pc(object_pc: torch.Tensor, num_points: int, seed: int, object_name: str) -> torch.Tensor:
    if object_pc.shape[0] <= num_points:
        return object_pc
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + _stable_hash_int(object_name))
    idx = torch.randperm(object_pc.shape[0], generator=gen)[:num_points]
    return object_pc[idx]


def _compute_metrics(hand_pts: torch.Tensor, object_pts: torch.Tensor) -> Dict[str, float]:
    d = torch.cdist(hand_pts.unsqueeze(0), object_pts.unsqueeze(0)).squeeze(0)  # (H, O)
    hand_to_obj = d.min(dim=1).values
    obj_to_hand = d.min(dim=0).values
    return {
        "mean_hand_to_obj": float(hand_to_obj.mean().item()),
        "mean_obj_to_hand": float(obj_to_hand.mean().item()),
        "coverage_obj_1cm": float((obj_to_hand < 0.01).float().mean().item()),
        "coverage_obj_2cm": float((obj_to_hand < 0.02).float().mean().item()),
    }


def _set_equal_axis(ax, pts: np.ndarray) -> None:
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    cx = float((x.max() + x.min()) * 0.5)
    cy = float((y.max() + y.min()) * 0.5)
    cz = float((z.max() + z.min()) * 0.5)
    r = float(max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()) * 0.55 + 1e-6)
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)


def _plot_pair(
    out_png: str,
    object_pts: torch.Tensor,
    tip_pts: torch.Tensor,
    full_pts: torch.Tensor,
    hand_mesh: trimesh.Trimesh,
    object_mesh: trimesh.Trimesh,
    title_prefix: str,
) -> None:
    obj = object_pts.cpu().numpy()
    tip = tip_pts.cpu().numpy()
    full = full_pts.cpu().numpy()
    hand_v = np.asarray(hand_mesh.vertices)
    obj_v = np.asarray(object_mesh.vertices)
    all_pts = np.concatenate([obj, tip, full, hand_v, obj_v], axis=0)

    fig = plt.figure(figsize=(18, 5), dpi=160)
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    ax1.scatter(obj[:, 0], obj[:, 1], obj[:, 2], s=1.5, c="#cc3344", alpha=0.35, label="object")
    ax1.scatter(tip[:, 0], tip[:, 1], tip[:, 2], s=35.0, c="#2277ff", alpha=0.95, label="tip")
    _set_equal_axis(ax1, all_pts)
    ax1.set_title(f"{title_prefix}\nTip ({tip.shape[0]} pts)")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.scatter(obj[:, 0], obj[:, 1], obj[:, 2], s=1.5, c="#cc3344", alpha=0.35, label="object")
    ax2.scatter(full[:, 0], full[:, 1], full[:, 2], s=2.0, c="#2277ff", alpha=0.65, label="full hand")
    _set_equal_axis(ax2, all_pts)
    ax2.set_title(f"{title_prefix}\nFull Hand ({full.shape[0]} pts)")
    ax2.legend(loc="upper right", fontsize=8)

    hf = np.asarray(hand_mesh.faces)
    of = np.asarray(object_mesh.faces)
    ax3.plot_trisurf(obj_v[:, 0], obj_v[:, 1], obj_v[:, 2], triangles=of, color="#cc3344", alpha=0.40, linewidth=0.05)
    ax3.plot_trisurf(hand_v[:, 0], hand_v[:, 1], hand_v[:, 2], triangles=hf, color="#2277ff", alpha=0.40, linewidth=0.05)
    _set_equal_axis(ax3, all_pts)
    ax3.set_title(f"{title_prefix}\nMesh Overlay")

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=22, azim=-52)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _parse_args():
    parser = argparse.ArgumentParser(description="Save tip-vs-full-hand observation ablation visualizations.")
    parser.add_argument("--dro_root", type=str, default=DRO_ROOT)
    parser.add_argument("--robot_name", type=str, default="shadowhand")
    parser.add_argument("--split", type=str, default="validate", choices=["train", "validate"])
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--object_points", type=int, default=512)
    parser.add_argument("--full_hand_points", type=int, default=512)
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = os.path.abspath(args.out_dir)
    png_dir = os.path.join(out_dir, "png")
    os.makedirs(png_dir, exist_ok=True)

    split_objects = _load_split_objects(args.dro_root, args.split)
    metadata = _load_metadata(args.dro_root)
    filtered: List[Tuple[int, str]] = []
    for i, item in enumerate(metadata):
        object_name, robot_name = _extract_item(item)
        if robot_name != args.robot_name:
            continue
        if object_name not in split_objects:
            continue
        filtered.append((i, object_name))
    if len(filtered) == 0:
        raise RuntimeError(f"No metadata for robot={args.robot_name}, split={args.split}")

    n = min(int(args.num_samples), len(filtered))
    picked = random.sample(filtered, n)

    hand_model = create_hand_model(args.robot_name, device="cpu")
    q_init = hand_model.get_canonical_q().to(torch.float32)
    tip_pts = hand_model.compute_tip_positions(q_init.unsqueeze(0))[0].to(torch.float32)
    full_mesh = hand_model.get_trimesh_q(q_init)["visual"]
    full_pts = torch.from_numpy(full_mesh.sample(int(args.full_hand_points))).to(torch.float32)

    rows: List[Dict[str, object]] = []
    for local_idx, (meta_idx, object_name) in enumerate(picked):
        object_pc_full = _load_object_pc(args.dro_root, object_name)
        object_pts = _sample_object_pc(object_pc_full, int(args.object_points), int(args.seed), object_name)
        object_mesh = _load_object_mesh(args.dro_root, object_name)

        tip_metrics = _compute_metrics(tip_pts, object_pts)
        full_metrics = _compute_metrics(full_pts, object_pts)

        tag = f"{local_idx:03d}_meta{meta_idx}_{object_name.replace('+', '_')}"
        png_path = os.path.join(png_dir, f"{tag}.png")
        _plot_pair(
            out_png=png_path,
            object_pts=object_pts,
            tip_pts=tip_pts,
            full_pts=full_pts,
            hand_mesh=full_mesh,
            object_mesh=object_mesh,
            title_prefix=f"{args.robot_name} | {object_name}",
        )

        rows.append(
            {
                "sample_tag": tag,
                "meta_index": meta_idx,
                "robot_name": args.robot_name,
                "object_name": object_name,
                "mode": "tip",
                **tip_metrics,
                "png_path": os.path.relpath(png_path, out_dir),
            }
        )
        rows.append(
            {
                "sample_tag": tag,
                "meta_index": meta_idx,
                "robot_name": args.robot_name,
                "object_name": object_name,
                "mode": "full_hand",
                **full_metrics,
                "png_path": os.path.relpath(png_path, out_dir),
            }
        )

    detail_csv = os.path.join(out_dir, "ablation_detail.csv")
    detail_fields = [
        "sample_tag",
        "meta_index",
        "robot_name",
        "object_name",
        "mode",
        "mean_hand_to_obj",
        "mean_obj_to_hand",
        "coverage_obj_1cm",
        "coverage_obj_2cm",
        "png_path",
    ]
    _write_csv(detail_csv, rows, detail_fields)

    summary_rows: List[Dict[str, object]] = []
    for mode in ("tip", "full_hand"):
        mode_rows = [r for r in rows if r["mode"] == mode]
        summary_rows.append(
            {
                "mode": mode,
                "num_samples": len(mode_rows),
                "mean_hand_to_obj": float(np.mean([r["mean_hand_to_obj"] for r in mode_rows])),
                "mean_obj_to_hand": float(np.mean([r["mean_obj_to_hand"] for r in mode_rows])),
                "coverage_obj_1cm": float(np.mean([r["coverage_obj_1cm"] for r in mode_rows])),
                "coverage_obj_2cm": float(np.mean([r["coverage_obj_2cm"] for r in mode_rows])),
            }
        )
    summary_csv = os.path.join(out_dir, "ablation_summary.csv")
    summary_fields = [
        "mode",
        "num_samples",
        "mean_hand_to_obj",
        "mean_obj_to_hand",
        "coverage_obj_1cm",
        "coverage_obj_2cm",
    ]
    _write_csv(summary_csv, summary_rows, summary_fields)

    meta = {
        "created_utc": datetime.utcnow().isoformat(),
        "dro_root": os.path.abspath(args.dro_root),
        "robot_name": args.robot_name,
        "split": args.split,
        "num_samples": n,
        "seed": int(args.seed),
        "object_points": int(args.object_points),
        "full_hand_points": int(args.full_hand_points),
        "detail_csv": os.path.relpath(detail_csv, out_dir),
        "summary_csv": os.path.relpath(summary_csv, out_dir),
        "png_dir": os.path.relpath(png_dir, out_dir),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("******************************** [Obs Ablation Save] ********************************")
    print(f"out_dir: {out_dir}")
    print(f"samples: {n}")
    print(f"detail_csv: {detail_csv}")
    print(f"summary_csv: {summary_csv}")
    print(f"png_dir: {png_dir}")


if __name__ == "__main__":
    main()
