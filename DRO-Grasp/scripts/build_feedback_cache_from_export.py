from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime

import torch
from termcolor import cprint

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model
from validation.validate_utils import validate_isaac


def _parse_args():
    parser = argparse.ArgumentParser(description="Build Isaac feedback pseudo labels from exported candidate q groups.")
    parser.add_argument("--export_index", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument(
        "--q_dof_mismatch",
        type=str,
        default="tail",
        choices=["tail", "head", "error"],
    )
    parser.add_argument(
        "--score_weights",
        type=float,
        nargs=2,
        default=[0.7, 0.3],
        metavar=("W_SUCCESS", "W_STABILITY"),
    )
    return parser.parse_args()


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


def _stability_score(q_input: torch.Tensor, q_isaac: torch.Tensor) -> torch.Tensor:
    # Convert q deviation to [0, 1] stability-like score.
    diff = torch.mean(torch.abs(q_input - q_isaac), dim=1)
    return torch.exp(-10.0 * diff).clamp(0.0, 1.0)


def _make_key(robot_name: str, object_name: str, sample_id: int) -> str:
    return f"{robot_name}|{object_name}|{int(sample_id)}"


def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    with open(args.export_index, "r", encoding="utf-8") as f:
        index = json.load(f)
    export_root = os.path.dirname(os.path.abspath(args.export_index))
    groups = list(index.get("groups", []))

    w_success = float(args.score_weights[0])
    w_stability = float(args.score_weights[1])

    hand_dof_cache = {}
    per_candidate_rows = []
    grouped_candidates = defaultdict(list)

    for group_idx, group in enumerate(groups):
        robot_name = str(group["robot_name"])
        object_name = str(group["object_name"])
        q_path = str(group["q_path"])
        meta_path = str(group.get("meta_path", ""))

        if not os.path.isabs(q_path):
            q_path = os.path.join(export_root, q_path)
        if not os.path.exists(q_path):
            raise FileNotFoundError(f"q_path not found for group {group_idx}: {q_path}")

        if not meta_path:
            cprint(f"Skip [{robot_name}/{object_name}] because meta_path is missing.", "yellow")
            continue
        if not os.path.isabs(meta_path):
            meta_path = os.path.join(export_root, meta_path)
        if not os.path.exists(meta_path):
            cprint(f"Skip [{robot_name}/{object_name}] because meta_path not found: {meta_path}", "yellow")
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_rows = json.load(f)

        q_batch = torch.load(q_path, map_location="cpu").to(torch.float32)
        if q_batch.ndim == 1:
            q_batch = q_batch.unsqueeze(0)
        if len(meta_rows) != int(q_batch.shape[0]):
            cprint(
                f"Skip [{robot_name}/{object_name}] due size mismatch: q={q_batch.shape[0]} meta={len(meta_rows)}",
                "yellow",
            )
            continue

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
        for start in range(0, n, args.chunk_size):
            end = min(start + args.chunk_size, n)
            q_chunk = q_batch[start:end]
            success, q_isaac = validate_isaac(robot_name, object_name, q_chunk, gpu=int(args.gpu))
            success = success.to(torch.bool).cpu()
            q_isaac = q_isaac.to(torch.float32).cpu()
            stability = _stability_score(q_chunk.cpu(), q_isaac)

            for local_idx in range(end - start):
                meta = meta_rows[start + local_idx]
                sample_id = int(meta.get("sample_id", -1))
                candidate_id = int(meta.get("candidate_id", -1))
                succ = int(success[local_idx].item())
                stab = float(stability[local_idx].item())
                combined = float(w_success * succ + w_stability * stab)

                row = {
                    "robot_name": robot_name,
                    "object_name": object_name,
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "success": succ,
                    "stability_score": stab,
                    "combined_score": combined,
                }
                per_candidate_rows.append(row)
                key = _make_key(robot_name, object_name, sample_id)
                grouped_candidates[key].append(row)

        cprint(f"[{group_idx + 1}/{len(groups)}] processed [{robot_name}/{object_name}] with {n} candidates", "green")

    labels = {}
    per_object_stats = defaultdict(lambda: {"samples": 0, "success_samples": 0, "mean_target_score": []})

    for key, rows in grouped_candidates.items():
        best_row = max(rows, key=lambda x: x["combined_score"]) if rows else None
        has_success = any(int(x["success"]) == 1 for x in rows)
        if has_success:
            stability_label = max(float(x["stability_score"]) for x in rows if int(x["success"]) == 1)
        else:
            stability_label = max(float(x["stability_score"]) for x in rows) if rows else 0.0
        target_score = float(best_row["combined_score"]) if best_row is not None else 0.0

        robot_name, object_name, sample_id_str = key.split("|", 2)
        sample_id = int(sample_id_str)
        labels[key] = {
            "robot_name": robot_name,
            "object_name": object_name,
            "sample_id": sample_id,
            "success": int(has_success),
            "stability_score": float(stability_label),
            "target_score": float(target_score),
            "best_candidate_id": int(best_row["candidate_id"]) if best_row is not None else -1,
            "num_candidates": int(len(rows)),
        }

        obj_key = f"{robot_name}|{object_name}"
        per_object_stats[obj_key]["samples"] += 1
        per_object_stats[obj_key]["success_samples"] += int(has_success)
        per_object_stats[obj_key]["mean_target_score"].append(float(target_score))

    summary = {
        "created_utc": datetime.utcnow().isoformat(),
        "export_index": os.path.abspath(args.export_index),
        "num_candidate_rows": int(len(per_candidate_rows)),
        "num_labeled_samples": int(len(labels)),
        "score_weights": [w_success, w_stability],
        "q_dof_mismatch": str(args.q_dof_mismatch),
        "gpu": int(args.gpu),
    }

    per_object_rows = []
    for obj_key, stat in sorted(per_object_stats.items(), key=lambda x: x[0]):
        robot_name, object_name = obj_key.split("|", 1)
        samples = int(stat["samples"])
        succ_samples = int(stat["success_samples"])
        row = {
            "robot_name": robot_name,
            "object_name": object_name,
            "num_samples": samples,
            "success_samples": succ_samples,
            "success_rate_percent": float((succ_samples / samples) * 100.0 if samples > 0 else 0.0),
            "mean_target_score": float(sum(stat["mean_target_score"]) / max(1, len(stat["mean_target_score"]))),
        }
        per_object_rows.append(row)

    candidate_csv = os.path.join(args.output_dir, "feedback_candidates.csv")
    with open(candidate_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "robot_name",
            "object_name",
            "sample_id",
            "candidate_id",
            "success",
            "stability_score",
            "combined_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_candidate_rows)

    per_object_csv = os.path.join(args.output_dir, "feedback_per_object.csv")
    with open(per_object_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "robot_name",
            "object_name",
            "num_samples",
            "success_samples",
            "success_rate_percent",
            "mean_target_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_object_rows)

    labels_json = os.path.join(args.output_dir, "feedback_labels.json")
    payload = {
        "summary": summary,
        "labels": labels,
    }
    with open(labels_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_json = os.path.join(args.output_dir, "feedback_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_object": per_object_rows}, f, indent=2)

    cache_epoch_path = os.path.join(args.cache_dir, f"labels_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    shutil.copyfile(labels_json, cache_epoch_path)
    latest_path = os.path.join(args.cache_dir, "labels_latest.json")
    tmp_latest = latest_path + ".tmp"
    shutil.copyfile(labels_json, tmp_latest)
    os.replace(tmp_latest, latest_path)

    print("******************************** [Feedback Cache Build] ********************************")
    print(summary)
    print(f"Saved candidate rows: {candidate_csv}")
    print(f"Saved per-object stats: {per_object_csv}")
    print(f"Saved labels: {labels_json}")
    print(f"Saved summary: {summary_json}")
    print(f"Saved cache snapshot: {cache_epoch_path}")
    print(f"Updated latest cache: {latest_path}")


if __name__ == "__main__":
    main()
