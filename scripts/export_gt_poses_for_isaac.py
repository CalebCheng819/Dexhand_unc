"""Export GT target_q poses from CMapDataset for Isaac Gym evaluation.

Usage:
    python scripts/export_gt_poses_for_isaac.py \
        --dataset_path DRO-Grasp/data/CMapDataset_filtered/cmap_dataset.pt \
        --split_json DRO-Grasp/data/CMapDataset_filtered/split_train_validate_objects.json \
        --output_dir /tmp/gt_export \
        --split validate \
        --robot_name shadowhand
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str,
                        default=os.path.join(ROOT_DIR, "DRO-Grasp/data/CMapDataset_filtered/cmap_dataset.pt"))
    parser.add_argument("--split_json", type=str,
                        default=os.path.join(ROOT_DIR, "DRO-Grasp/data/CMapDataset_filtered/split_train_validate_objects.json"))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="validate", choices=["train", "validate"])
    parser.add_argument("--robot_name", type=str, default="shadowhand")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset from {args.dataset_path} ...")
    data = torch.load(args.dataset_path, map_location="cpu")

    # Understand structure
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        metadata = data.get("metadata", data.get("data", None))
    elif isinstance(data, (list, tuple)):
        metadata = data
    else:
        raise ValueError(f"Unknown dataset type: {type(data)}")

    print(f"Total entries: {len(metadata)}")
    # Print first entry to understand structure
    entry0 = metadata[0]
    print(f"Entry type: {type(entry0)}")
    if isinstance(entry0, (list, tuple)):
        print(f"Entry length: {len(entry0)}")
        for i, v in enumerate(entry0):
            if hasattr(v, 'shape'):
                print(f"  [{i}]: tensor shape={v.shape}")
            else:
                print(f"  [{i}]: {type(v).__name__} = {v!r}")
    elif isinstance(entry0, dict):
        for k, v in entry0.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: tensor shape={v.shape}")
            else:
                print(f"  {k}: {type(v).__name__} = {v!r}")

    # Load split info
    with open(args.split_json) as f:
        split_data = json.load(f)

    if args.split == "validate":
        split_objects = set(split_data.get("validate", split_data.get("val", [])))
    else:
        split_objects = set(split_data.get("train", []))

    print(f"\nSplit '{args.split}' has {len(split_objects)} objects:")
    print(sorted(split_objects))

    # Parse entries
    grouped_q = defaultdict(list)
    total = 0
    skipped_robot = 0
    skipped_split = 0

    for entry in metadata:
        if isinstance(entry, (list, tuple)):
            # (target_q, object_name, robot_name) — best guess from context
            if len(entry) == 3:
                target_q, object_name, robot_name = entry
            elif len(entry) == 2:
                target_q, object_name = entry
                robot_name = args.robot_name
            else:
                raise ValueError(f"Unexpected entry length: {len(entry)}")
        elif isinstance(entry, dict):
            target_q = entry.get("target_q", entry.get("q"))
            object_name = entry.get("object_name", entry.get("object"))
            robot_name = entry.get("robot_name", entry.get("robot", args.robot_name))
        else:
            raise ValueError(f"Unknown entry type: {type(entry)}")

        robot_name = str(robot_name)
        object_name = str(object_name)

        if robot_name != args.robot_name:
            skipped_robot += 1
            continue

        if object_name not in split_objects:
            skipped_split += 1
            continue

        grouped_q[object_name].append(target_q.float().cpu())
        total += 1

    print(f"\nFiltered: {total} samples kept, {skipped_robot} wrong robot, {skipped_split} wrong split")
    print(f"Groups ({len(grouped_q)}): {sorted(grouped_q.keys())}")

    # Save q groups
    q_group_dir = os.path.join(args.output_dir, "q_groups")
    os.makedirs(q_group_dir, exist_ok=True)

    index = {
        "created_utc": datetime.utcnow().isoformat(),
        "checkpoint_path": "GT_POSES",
        "train_cfg_path": "GT_POSES",
        "split": args.split,
        "action_type": "joint_value",
        "export_step_mode": "full_last",
        "inference_seed": -1,
        "total_samples": total,
        "num_groups": len(grouped_q),
        "groups": [],
    }

    import csv
    group_csv_path = os.path.join(args.output_dir, "group_counts.csv")
    with open(group_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["robot_name", "object_name", "count", "q_path"])
        writer.writeheader()

        for object_name in sorted(grouped_q.keys()):
            q_list = grouped_q[object_name]
            q_tensor = torch.stack(q_list, dim=0)
            print(f"  {object_name}: {q_tensor.shape}")

            robot_dir = os.path.join(q_group_dir, args.robot_name)
            os.makedirs(robot_dir, exist_ok=True)
            q_path = os.path.join(robot_dir, f"{object_name}.pt")
            torch.save(q_tensor, q_path)

            rel_q_path = os.path.relpath(q_path, args.output_dir)
            row = {
                "robot_name": args.robot_name,
                "object_name": object_name,
                "count": int(q_tensor.shape[0]),
                "q_path": rel_q_path,
            }
            writer.writerow(row)
            index["groups"].append(row)

    index_path = os.path.join(args.output_dir, "export_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"\nSaved export_index.json: {index_path}")
    print(f"Saved group_counts.csv:  {group_csv_path}")
    print(f"Total samples: {total}, groups: {len(grouped_q)}")


if __name__ == "__main__":
    main()
