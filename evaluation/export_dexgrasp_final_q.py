from __future__ import annotations

import argparse
import csv
import hashlib
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
    parser = argparse.ArgumentParser(description="Export DexGrasp checkpoint predictions to final q for DRO IsaacGym validation.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint (*.ckpt).")
    parser.add_argument("--train_cfg_path", type=str, default="", help="Optional Hydra config.yaml path for this checkpoint.")
    parser.add_argument("--action_type", type=str, default="", help="Optional override for dataset.action_type.")
    parser.add_argument("--split", type=str, default="validate", choices=["train", "validate"], help="Dataset split to export.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for grouped q files.")
    parser.add_argument("--max_batches", type=int, default=-1, help="Limit number of dataloader batches; -1 means all.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Limit number of samples; -1 means all.")
    parser.add_argument(
        "--export_step_mode",
        type=str,
        default="full_last",
        choices=["window_last", "full_last"],
        help="Which predicted step to export as final grasp q. In static single-step mode both choices are equivalent.",
    )
    parser.add_argument(
        "--inference_seed",
        type=int,
        default=-1,
        help="Optional deterministic seed for diffusion sampling. -1 means stochastic.",
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
        help="Disable static_single_step export cache so each sample runs fresh inference (needed for DDPM stochastic diversity).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use DDPM stochastic sampling (100 steps) instead of DDIM (10 steps). Produces diverse predictions per sample.",
    )
    parser.add_argument(
        "--rerank_enable",
        action="store_true",
        help="Enable candidate rerank before final q export.",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=1,
        help="Number of stochastic candidates per sample when rerank is enabled.",
    )
    parser.add_argument(
        "--score_weights",
        type=float,
        nargs=2,
        default=[0.7, 0.3],
        metavar=("W_SUCCESS", "W_CONTACT"),
        help="Score weights for rerank: score = w_success*pred_success + w_contact*contact_proxy.",
    )
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
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _assemble_full_q(pred_q_1d: torch.Tensor, raw_target_q_1d: torch.Tensor, q_dof_mismatch: str) -> torch.Tensor:
    pred_dim = int(pred_q_1d.numel())
    tgt_dim = int(raw_target_q_1d.numel())
    if tgt_dim == pred_dim:
        return pred_q_1d

    # Preserve the non-modeled prefix/suffix from raw target q (e.g., root pose),
    # then replace modeled segment with predicted q.
    if tgt_dim > pred_dim and q_dof_mismatch in {"tail", "head"}:
        if q_dof_mismatch == "tail":
            prefix = raw_target_q_1d[: tgt_dim - pred_dim]
            return torch.cat([prefix, pred_q_1d], dim=0)
        suffix = raw_target_q_1d[pred_dim:]
        return torch.cat([pred_q_1d, suffix], dim=0)

    return pred_q_1d


def _static_export_cache_path(
    checkpoint_path: str,
    train_cfg_path: str,
    action_type: str,
    split: str,
    max_batches: int,
    max_samples: int,
    inference_seed: int,
) -> str:
    run_root = os.path.abspath(os.path.join(os.path.dirname(checkpoint_path), ".."))
    cache_root = os.path.join(run_root, "export_cache", "static_single_step")
    os.makedirs(cache_root, exist_ok=True)
    cache_key = {
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "train_cfg_path": os.path.abspath(train_cfg_path),
        "action_type": str(action_type),
        "split": str(split),
        "max_batches": int(max_batches),
        "max_samples": int(max_samples),
        "inference_seed": int(inference_seed),
    }
    digest = hashlib.md5(json.dumps(cache_key, sort_keys=True).encode("utf-8")).hexdigest()
    return os.path.join(cache_root, f"{digest}.pt")


@torch.no_grad()
def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg, train_cfg_path = _load_train_cfg(args)
    static_single_step = int(cfg.model.pred_horizon) == 1 and int(cfg.model.act_horizon) == 1
    if static_single_step:
        device = torch.device("cpu")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    inference_seed = int(args.inference_seed) if int(args.inference_seed) >= 0 else int(getattr(cfg, "seed", 2025))
    _setup_inference_determinism(inference_seed)
    action_type = str(cfg.dataset.action_type)
    action_mode = str(getattr(cfg.dataset, "action_mode", "absolute")).lower()
    env_act_dim = int(cfg.dataset.env_act_dim)
    num_joints = int(getattr(cfg.dataset, "num_joints", 24))
    q_dof_mismatch = str(getattr(cfg.dataset, "q_dof_mismatch", "tail"))
    rerank_enable = bool(args.rerank_enable)
    num_candidates = max(1, int(args.num_candidates))
    score_weights = (float(args.score_weights[0]), float(args.score_weights[1]))
    total_samples = 0
    grouped_q_tensors: dict[tuple[str, str], torch.Tensor] = {}
    grouped_sample_ids: dict[tuple[str, str], list[int]] = {}
    cache_path = ""
    cache_used = False

    cache_allowed = static_single_step and (not args.disable_cache) and (not args.stochastic) and (not rerank_enable) and num_candidates <= 1
    if cache_allowed:
        cache_path = _static_export_cache_path(
            checkpoint_path=args.checkpoint_path,
            train_cfg_path=train_cfg_path,
            action_type=action_type,
            split=args.split,
            max_batches=args.max_batches,
            max_samples=args.max_samples,
            inference_seed=inference_seed,
        )
        if os.path.exists(cache_path):
            cache_data = torch.load(cache_path, map_location="cpu")
            total_samples = int(cache_data["total_samples"])
            grouped_q_tensors = {
                (str(item["robot_name"]), str(item["object_name"])): item["q_tensor"].to(torch.float32)
                for item in cache_data["groups"]
            }
            grouped_sample_ids = {
                (str(item["robot_name"]), str(item["object_name"])): [int(x) for x in item.get("sample_ids", [])]
                for item in cache_data["groups"]
            }
            cache_used = True

    if not cache_used:
        train_loader, val_loader = create_dataloader(cfg.dataset, is_train=True)
        loader = train_loader if args.split == "train" else val_loader

        module = TrainingModule.load_from_checkpoint(args.checkpoint_path, map_location=device, cfg=cfg, strict=False)
        module = module.to(device).eval()
        model = module.ema_model if (module.use_ema_for_eval and module.use_ema and module.ema_model is not None) else module.model
        ds_ref = loader.dataset
        window_start = int(cfg.model.obs_horizon) - 1
        window_end = window_start + int(cfg.model.act_horizon)

        hand_models = {}
        grouped_q = defaultdict(list)
        grouped_sample_ids = defaultdict(list)

        for batch_idx, batch in enumerate(loader):
            if args.max_batches >= 0 and batch_idx >= args.max_batches:
                break

            obs = module._build_obs_seq(batch)
            if rerank_enable and num_candidates > 1:
                cand_list = [model.get_action(obs, return_full=True, stochastic=True) for _ in range(num_candidates)]
                cand_tensor = torch.stack(cand_list, dim=1)  # (B,K,T,D)
                pc_heatmap = batch["pc_heatmap"].to(device) if "pc_heatmap" in batch else None
                _, _, _, best_idx = module.score_action_candidates(
                    obs_seq=obs,
                    action_candidates=cand_tensor,
                    pc_heatmap=pc_heatmap,
                    score_weights=score_weights,
                )
                b_idx = torch.arange(cand_tensor.shape[0], device=device)
                pred_seq_full = cand_tensor[b_idx, best_idx]
            else:
                pred_seq_full = model.get_action(obs, return_full=True, stochastic=args.stochastic)
            pred_seq_window = pred_seq_full[:, window_start:window_end]
            if args.export_step_mode == "window_last":
                final_action_non_relative = pred_seq_window[:, -1]
            else:
                final_action_non_relative = pred_seq_full[:, -1]
            meta_list = batch["meta"]

            for i, meta in enumerate(meta_list):
                robot_name = str(meta["robot_name"])
                object_name = str(meta["object_name"])
                sample_id = int(meta["sample_id"])
                key = (robot_name, object_name)

                if robot_name not in hand_models:
                    hand_models[robot_name] = create_hand_model(robot_name, device=device)
                hand_model = hand_models[robot_name]

                if action_mode == "relative" and action_type != "joint_value":
                    seq_i = pred_seq_full[i : i + 1]  # (1, T, D)
                    seq_i_abs = action_seq_relative_to_absolute(
                        seq_i,
                        action_type=action_type,
                        action_mode=action_mode,
                        hand_model=hand_model,
                        env_act_dim=env_act_dim,
                        J=num_joints,
                    )
                    seq_i_window_abs = seq_i_abs[:, window_start:window_end]
                    if args.export_step_mode == "window_last":
                        action_i_full = seq_i_window_abs[:, -1]
                    else:
                        action_i_full = seq_i_abs[:, -1]
                else:
                    action_i_full = final_action_non_relative[i : i + 1]
                action_i_joint = action_i_full[:, env_act_dim:]
                if action_type == "joint_value":
                    final_q_joint = action_i_joint
                else:
                    final_q_joint = projection_q(hand_model, action_i_joint, input_q_type=action_type)

                if env_act_dim > 0:
                    final_q = torch.cat([action_i_full[:, :env_act_dim], final_q_joint], dim=1)
                else:
                    final_q = final_q_joint

                pred_q_1d = final_q.squeeze(0).detach().cpu().to(torch.float32)
                raw_target_q = ds_ref.samples[sample_id].target_q.detach().cpu().to(torch.float32)
                export_q = _assemble_full_q(pred_q_1d, raw_target_q, q_dof_mismatch=q_dof_mismatch)

                grouped_q[key].append(export_q)
                grouped_sample_ids[key].append(sample_id)
                total_samples += 1

                if args.max_samples >= 0 and total_samples >= args.max_samples:
                    break

            if args.max_samples >= 0 and total_samples >= args.max_samples:
                break

        grouped_q_tensors = {
            key: torch.stack(q_list, dim=0).to(torch.float32)
            for key, q_list in grouped_q.items()
        }
        if cache_allowed and cache_path:
            torch.save(
                {
                    "total_samples": int(total_samples),
                    "groups": [
                        {
                            "robot_name": robot_name,
                            "object_name": object_name,
                            "q_tensor": q_tensor.cpu().to(torch.float32),
                            "sample_ids": [int(x) for x in grouped_sample_ids.get((robot_name, object_name), [])],
                        }
                        for (robot_name, object_name), q_tensor in sorted(
                            grouped_q_tensors.items(),
                            key=lambda x: (x[0][0], x[0][1]),
                        )
                    ],
                },
                cache_path,
            )

    q_group_dir = os.path.join(args.output_dir, "q_groups")
    os.makedirs(q_group_dir, exist_ok=True)

    index = {
        "created_utc": datetime.utcnow().isoformat(),
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
        "train_cfg_path": os.path.abspath(train_cfg_path),
        "split": args.split,
        "action_type": action_type,
        "export_step_mode": str(args.export_step_mode),
        "inference_seed": int(inference_seed),
        "total_samples": int(total_samples),
        "num_groups": int(len(grouped_q_tensors)),
        "rerank_enable": bool(rerank_enable),
        "num_candidates": int(num_candidates),
        "score_weights": [float(score_weights[0]), float(score_weights[1])],
        "groups": [],
    }
    if static_single_step:
        index["static_single_step_cache_path"] = os.path.abspath(cache_path) if cache_path else ""
        index["static_single_step_cache_used"] = bool(cache_used)

    group_csv_path = os.path.join(args.output_dir, "group_counts.csv")
    sample_id_group_dir = os.path.join(args.output_dir, "sample_ids")
    os.makedirs(sample_id_group_dir, exist_ok=True)
    with open(group_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["robot_name", "object_name", "count", "q_path", "sample_ids_path"])
        writer.writeheader()

        for (robot_name, object_name), q_tensor in sorted(grouped_q_tensors.items(), key=lambda x: (x[0][0], x[0][1])):
            robot_dir = os.path.join(q_group_dir, robot_name)
            robot_sid_dir = os.path.join(sample_id_group_dir, robot_name)
            os.makedirs(robot_dir, exist_ok=True)
            os.makedirs(robot_sid_dir, exist_ok=True)
            q_path = os.path.join(robot_dir, f"{object_name}.pt")
            sid_path = os.path.join(robot_sid_dir, f"{object_name}.pt")
            torch.save(q_tensor, q_path)
            sample_ids = grouped_sample_ids.get((robot_name, object_name), [])
            if len(sample_ids) != int(q_tensor.shape[0]):
                sample_ids = [-1] * int(q_tensor.shape[0])
            torch.save(torch.tensor(sample_ids, dtype=torch.long), sid_path)

            rel_q_path = os.path.relpath(q_path, args.output_dir)
            rel_sid_path = os.path.relpath(sid_path, args.output_dir)
            row = {
                "robot_name": robot_name,
                "object_name": object_name,
                "count": int(q_tensor.shape[0]),
                "q_path": rel_q_path,
                "sample_ids_path": rel_sid_path,
            }
            writer.writerow(row)
            index["groups"].append(row)

    index_path = os.path.join(args.output_dir, "export_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print("******************************** [DexGrasp Final-Q Export] ********************************")
    print(f"checkpoint: {args.checkpoint_path}")
    print(f"train_cfg:  {train_cfg_path}")
    print(f"action:     {action_type}")
    print(f"step_mode:  {args.export_step_mode}")
    print(f"rerank:     {rerank_enable}")
    print(f"candidates: {num_candidates}")
    print(f"seed:       {inference_seed}")
    print(f"device:     {device}")
    if static_single_step:
        print(f"cache:      {cache_path}")
        print(f"cache_used: {cache_used}")
    print(f"split:      {args.split}")
    print(f"samples:    {total_samples}")
    print(f"groups:     {len(grouped_q_tensors)}")
    print(f"saved index: {index_path}")
    print(f"saved group csv: {group_csv_path}")


if __name__ == "__main__":
    main()
