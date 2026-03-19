from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import torch
from omegaconf import OmegaConf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from dataset.DexGraspDataset import create_dataloader
from model.pl_module import TrainingModule
from utils.action_schema import compute_action_schema
from utils.action_utils import action_seq_relative_to_absolute, projection_q
from utils.hand_model import create_hand_model


def _parse_args():
    parser = argparse.ArgumentParser(description="Export stochastic candidate q groups for Isaac feedback cache.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--train_cfg_path", type=str, default="")
    parser.add_argument("--action_type", type=str, default="")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validate"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--max_samples_per_object", type=int, default=256)
    parser.add_argument("--candidates_k", type=int, default=8)
    parser.add_argument("--objects", type=str, nargs="*", default=[])
    parser.add_argument("--inference_seed", type=int, default=-1)
    return parser.parse_args()


def _infer_train_cfg_path(checkpoint_path: str) -> str:
    run_root = os.path.abspath(os.path.join(os.path.dirname(checkpoint_path), ".."))
    candidates = [
        os.path.join(run_root, "log", "hydra", ".hydra", "config.yaml"),
        os.path.join(run_root, "log", "hydra", "config.yaml"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not infer Hydra config for checkpoint: {checkpoint_path}")


def _load_train_cfg(args) -> tuple:
    train_cfg_path = args.train_cfg_path if args.train_cfg_path else _infer_train_cfg_path(args.checkpoint_path)
    cfg = OmegaConf.load(train_cfg_path)

    if args.action_type:
        cfg.dataset.action_type = str(args.action_type)

    schema = compute_action_schema(
        env_act_dim=int(cfg.dataset.env_act_dim),
        num_joints=int(cfg.dataset.num_joints),
        action_type=str(cfg.dataset.action_type),
    )
    cfg.dataset.rot_dim = schema.rot_dim
    cfg.dataset.rot_act_dim = schema.rot_act_dim
    cfg.dataset.act_dim = schema.act_dim

    return cfg, train_cfg_path


def _setup_inference_determinism(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _assemble_full_q(pred_q_1d: torch.Tensor, raw_target_q_1d: torch.Tensor, q_dof_mismatch: str) -> torch.Tensor:
    pred_dim = int(pred_q_1d.numel())
    tgt_dim = int(raw_target_q_1d.numel())
    if tgt_dim == pred_dim:
        return pred_q_1d

    if tgt_dim > pred_dim and q_dof_mismatch in {"tail", "head"}:
        if q_dof_mismatch == "tail":
            prefix = raw_target_q_1d[: tgt_dim - pred_dim]
            return torch.cat([prefix, pred_q_1d], dim=0)
        suffix = raw_target_q_1d[pred_dim:]
        return torch.cat([pred_q_1d, suffix], dim=0)

    return pred_q_1d


@torch.no_grad()
def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg, train_cfg_path = _load_train_cfg(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    inference_seed = int(args.inference_seed) if int(args.inference_seed) >= 0 else int(getattr(cfg, "seed", 2025))
    _setup_inference_determinism(inference_seed)

    action_type = str(cfg.dataset.action_type)
    action_mode = str(getattr(cfg.dataset, "action_mode", "absolute")).lower()
    env_act_dim = int(cfg.dataset.env_act_dim)
    num_joints = int(getattr(cfg.dataset, "num_joints", 24))
    q_dof_mismatch = str(getattr(cfg.dataset, "q_dof_mismatch", "tail"))

    objects_filter = {str(x) for x in args.objects}
    use_filter = len(objects_filter) > 0

    train_loader, val_loader = create_dataloader(cfg.dataset, is_train=True)
    loader = train_loader if args.split == "train" else val_loader
    ds_ref = loader.dataset

    module = TrainingModule.load_from_checkpoint(args.checkpoint_path, map_location=device, cfg=cfg, strict=False)
    module = module.to(device).eval()
    model = module.ema_model if (module.use_ema_for_eval and module.use_ema and module.ema_model is not None) else module.model

    hand_models = {}
    grouped_q = defaultdict(list)
    grouped_meta = defaultdict(list)
    object_counts = defaultdict(int)

    total_selected_samples = 0
    total_entries = 0
    candidates_k = max(1, int(args.candidates_k))

    for batch_idx, batch in enumerate(loader):
        if args.max_batches >= 0 and batch_idx >= args.max_batches:
            break

        obs = module._build_obs_seq(batch)
        cand_list = [model.get_action(obs, return_full=True, stochastic=True) for _ in range(candidates_k)]

        for i, meta in enumerate(batch["meta"]):
            robot_name = str(meta["robot_name"])
            object_name = str(meta["object_name"])
            sample_id = int(meta["sample_id"])

            if use_filter and object_name not in objects_filter:
                continue
            if args.max_samples_per_object >= 0 and object_counts[object_name] >= args.max_samples_per_object:
                continue

            if robot_name not in hand_models:
                hand_models[robot_name] = create_hand_model(robot_name, device=device)
            hand_model = hand_models[robot_name]

            for k in range(candidates_k):
                pred_seq = cand_list[k][i : i + 1]
                if action_mode == "relative" and action_type != "joint_value":
                    pred_seq = action_seq_relative_to_absolute(
                        pred_seq,
                        action_type=action_type,
                        action_mode=action_mode,
                        hand_model=hand_model,
                        env_act_dim=env_act_dim,
                        J=num_joints,
                    )
                final_action = pred_seq[:, -1]
                action_joint = final_action[:, env_act_dim:]
                if action_type == "joint_value":
                    final_q_joint = action_joint
                else:
                    final_q_joint = projection_q(hand_model, action_joint, input_q_type=action_type)
                if env_act_dim > 0:
                    final_q = torch.cat([final_action[:, :env_act_dim], final_q_joint], dim=1)
                else:
                    final_q = final_q_joint

                pred_q_1d = final_q.squeeze(0).detach().cpu().to(torch.float32)
                raw_target_q = ds_ref.samples[sample_id].target_q.detach().cpu().to(torch.float32)
                export_q = _assemble_full_q(pred_q_1d, raw_target_q, q_dof_mismatch=q_dof_mismatch)

                key = (robot_name, object_name)
                grouped_q[key].append(export_q)
                grouped_meta[key].append(
                    {
                        "sample_id": int(sample_id),
                        "candidate_id": int(k),
                    }
                )
                total_entries += 1

            object_counts[object_name] += 1
            total_selected_samples += 1

        if use_filter and args.max_samples_per_object >= 0:
            if all(object_counts[obj] >= args.max_samples_per_object for obj in objects_filter):
                break

    grouped_q_tensors = {key: torch.stack(v, dim=0).to(torch.float32) for key, v in grouped_q.items()}

    q_group_dir = os.path.join(args.output_dir, "q_groups")
    meta_group_dir = os.path.join(args.output_dir, "candidate_meta")
    os.makedirs(q_group_dir, exist_ok=True)
    os.makedirs(meta_group_dir, exist_ok=True)

    index = {
        "created_utc": datetime.utcnow().isoformat(),
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
        "train_cfg_path": os.path.abspath(train_cfg_path),
        "split": args.split,
        "action_type": action_type,
        "inference_seed": int(inference_seed),
        "candidates_k": int(candidates_k),
        "objects_filter": sorted(list(objects_filter)),
        "total_selected_samples": int(total_selected_samples),
        "total_entries": int(total_entries),
        "num_groups": int(len(grouped_q_tensors)),
        "groups": [],
    }

    group_csv_path = os.path.join(args.output_dir, "group_counts.csv")
    with open(group_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["robot_name", "object_name", "count", "q_path", "meta_path", "unique_samples"],
        )
        writer.writeheader()

        for (robot_name, object_name), q_tensor in sorted(grouped_q_tensors.items(), key=lambda x: (x[0][0], x[0][1])):
            robot_dir = os.path.join(q_group_dir, robot_name)
            robot_meta_dir = os.path.join(meta_group_dir, robot_name)
            os.makedirs(robot_dir, exist_ok=True)
            os.makedirs(robot_meta_dir, exist_ok=True)

            q_path = os.path.join(robot_dir, f"{object_name}.pt")
            meta_path = os.path.join(robot_meta_dir, f"{object_name}.json")
            torch.save(q_tensor, q_path)
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(grouped_meta[(robot_name, object_name)], mf)

            sample_ids = [int(x["sample_id"]) for x in grouped_meta[(robot_name, object_name)]]
            row = {
                "robot_name": robot_name,
                "object_name": object_name,
                "count": int(q_tensor.shape[0]),
                "q_path": os.path.relpath(q_path, args.output_dir),
                "meta_path": os.path.relpath(meta_path, args.output_dir),
                "unique_samples": int(len(set(sample_ids))),
            }
            writer.writerow(row)
            index["groups"].append(row)

    index_path = os.path.join(args.output_dir, "export_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print("******************************** [DexGrasp Feedback Candidate Export] ********************************")
    print(f"checkpoint: {args.checkpoint_path}")
    print(f"train_cfg:  {train_cfg_path}")
    print(f"split:      {args.split}")
    print(f"seed:       {inference_seed}")
    print(f"device:     {device}")
    print(f"candidates: {candidates_k}")
    print(f"samples:    {total_selected_samples}")
    print(f"entries:    {total_entries}")
    print(f"groups:     {len(grouped_q_tensors)}")
    print(f"saved index: {index_path}")
    print(f"saved group csv: {group_csv_path}")


if __name__ == "__main__":
    main()
