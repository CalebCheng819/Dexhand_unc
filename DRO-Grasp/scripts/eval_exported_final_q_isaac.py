from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import torch
from termcolor import cprint

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model
from validation.validate_utils import validate_isaac


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate exported final-q groups with DRO IsaacGym validator.")
    parser.add_argument("--export_index", type=str, required=True, help="Path to export_index.json from export_dexgrasp_final_q.py")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for IsaacGym subprocess.")
    parser.add_argument("--chunk_size", type=int, default=64, help="Chunk size for each Isaac validation call.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for Isaac eval metrics.")
    parser.add_argument("--max_groups", type=int, default=-1, help="Optional group limit for smoke test.")
    parser.add_argument(
        "--q_dof_mismatch",
        type=str,
        default="tail",
        choices=["tail", "head", "error"],
        help="How to adapt q dimension mismatch against DRO robot dof.",
    )
    return parser.parse_args()


def _safe_diversity(q_success: torch.Tensor) -> float:
    if q_success is None or q_success.numel() == 0 or q_success.shape[0] <= 1:
        return 0.0
    return float(torch.std(q_success, dim=0).mean().item())


def _adapt_q_dof(q_batch: torch.Tensor, expected_dof: int, mode: str, robot_name: str, object_name: str) -> torch.Tensor:
    q_dim = int(q_batch.shape[-1])
    if q_dim == expected_dof:
        return q_batch

    if mode == "error":
        raise RuntimeError(
            f"q dof mismatch for [{robot_name}/{object_name}]: got {q_dim}, expected {expected_dof} (mode=error)."
        )

    if q_dim > expected_dof:
        if mode == "tail":
            out = q_batch[:, -expected_dof:]
        else:
            out = q_batch[:, :expected_dof]
        cprint(
            f"[DOF Adapt] [{robot_name}/{object_name}] got={q_dim} expected={expected_dof}, mode={mode}, slicing to {out.shape[-1]}",
            "yellow",
        )
        return out

    # q_dim < expected_dof
    pad = expected_dof - q_dim
    zeros = torch.zeros((q_batch.shape[0], pad), dtype=q_batch.dtype, device=q_batch.device)
    if mode == "tail":
        out = torch.cat([zeros, q_batch], dim=-1)
    else:
        out = torch.cat([q_batch, zeros], dim=-1)
    cprint(
        f"[DOF Adapt] [{robot_name}/{object_name}] got={q_dim} expected={expected_dof}, mode={mode}, padding zeros -> {out.shape[-1]}",
        "yellow",
    )
    return out


def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.export_index, "r", encoding="utf-8") as f:
        index = json.load(f)
    export_root = os.path.dirname(os.path.abspath(args.export_index))

    groups = list(index.get("groups", []))
    if args.max_groups >= 0:
        groups = groups[: args.max_groups]

    per_object_rows = []
    all_success_q = []
    total_num = 0
    success_num = 0
    hand_dof_cache = {}

    for group_idx, group in enumerate(groups):
        robot_name = str(group["robot_name"])
        object_name = str(group["object_name"])
        q_path = str(group["q_path"])
        if not os.path.isabs(q_path):
            q_path = os.path.join(export_root, q_path)
        if not os.path.exists(q_path):
            raise FileNotFoundError(f"q_path not found for group {group_idx}: {q_path}")

        q_batch = torch.load(q_path, map_location="cpu").to(torch.float32)
        if q_batch.ndim == 1:
            q_batch = q_batch.unsqueeze(0)
        if robot_name not in hand_dof_cache:
            hand = create_hand_model(robot_name, torch.device("cpu"))
            hand_dof_cache[robot_name] = int(hand.dof)
        q_batch = _adapt_q_dof(
            q_batch=q_batch,
            expected_dof=hand_dof_cache[robot_name],
            mode=str(args.q_dof_mismatch),
            robot_name=robot_name,
            object_name=object_name,
        )
        n = int(q_batch.shape[0])

        obj_success = 0
        obj_success_q = []
        for start in range(0, n, args.chunk_size):
            end = min(start + args.chunk_size, n)
            q_chunk = q_batch[start:end]
            success, _ = validate_isaac(robot_name, object_name, q_chunk, gpu=int(args.gpu))
            success = success.to(torch.bool).cpu()
            succ_num = int(success.sum().item())
            obj_success += succ_num
            if succ_num > 0:
                obj_success_q.append(q_chunk[success])

        obj_success_rate = (obj_success / n * 100.0) if n > 0 else 0.0
        obj_success_q_cat = torch.cat(obj_success_q, dim=0) if obj_success_q else torch.empty((0, q_batch.shape[-1]), dtype=torch.float32)
        obj_diversity = _safe_diversity(obj_success_q_cat)
        if obj_success_q:
            all_success_q.append(obj_success_q_cat)

        row = {
            "robot_name": robot_name,
            "object_name": object_name,
            "num_grasps": n,
            "success_num": obj_success,
            "success_rate_percent": obj_success_rate,
            "diversity_rad": obj_diversity,
        }
        per_object_rows.append(row)

        cprint(
            f"[{group_idx + 1}/{len(groups)}] [{robot_name}/{object_name}] "
            f"success={obj_success}/{n} ({obj_success_rate:.2f}%) diversity={obj_diversity:.6f}",
            "green",
        )

        total_num += n
        success_num += obj_success

    all_success_q_cat = torch.cat(all_success_q, dim=0) if all_success_q else torch.empty((0, 24), dtype=torch.float32)
    summary = {
        "created_utc": datetime.utcnow().isoformat(),
        "checkpoint_path": str(index.get("checkpoint_path", "")),
        "train_cfg_path": str(index.get("train_cfg_path", "")),
        "action_type": str(index.get("action_type", "")),
        "split": str(index.get("split", "")),
        "num_groups_eval": int(len(groups)),
        "num_grasps_total": int(total_num),
        "success_num_total": int(success_num),
        "success_rate_percent": float((success_num / total_num * 100.0) if total_num > 0 else 0.0),
        "diversity_rad": float(_safe_diversity(all_success_q_cat)),
        "gpu": int(args.gpu),
        "chunk_size": int(args.chunk_size),
        "export_index": os.path.abspath(args.export_index),
    }

    per_object_csv = os.path.join(args.output_dir, "isaac_per_object.csv")
    with open(per_object_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["robot_name", "object_name", "num_grasps", "success_num", "success_rate_percent", "diversity_rad"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_object_rows)

    summary_json = os.path.join(args.output_dir, "isaac_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary_csv = os.path.join(args.output_dir, "isaac_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("******************************** [DRO Isaac Eval] ********************************")
    print(summary)
    print(f"Saved: {per_object_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
