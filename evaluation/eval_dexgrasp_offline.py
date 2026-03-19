from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from dataset.DexGraspDataset import create_dataloader
from model.pl_module import TrainingModule
from utils.action_schema import compute_action_schema
from utils.action_utils import action_seq_relative_to_absolute, decode_rotations_to_R, projection_q
from utils.hand_model import create_hand_model


def _ensure_eval_runtime_cfg(cfg):
    """Inject minimal training/runtime fields so weights-only checkpoints can be loaded."""
    default_output = os.environ.get("DEXHAND_EVAL_RUNTIME_DIR", "/tmp/dexgrasp_eval_runtime")
    with open_dict(cfg):
        if "output_dir" not in cfg or not cfg.output_dir:
            cfg.output_dir = default_output
        if "log_dir" not in cfg or not cfg.log_dir:
            cfg.log_dir = os.path.join(cfg.output_dir, "log")
        if "file_dir" not in cfg or not cfg.file_dir:
            cfg.file_dir = os.path.join(cfg.output_dir, "files")

        if "training" not in cfg or cfg.training is None:
            cfg.training = OmegaConf.create({})

        t = cfg.training
        if "save_dir" not in t or not t.save_dir:
            t.save_dir = os.path.join(cfg.output_dir, "state_dict")
        if "use_ema" not in t:
            t.use_ema = False
        if "use_ema_for_eval" not in t:
            t.use_ema_for_eval = False
        if "ema_decay" not in t:
            t.ema_decay = 0.995
        if "ema_warmup_steps" not in t:
            t.ema_warmup_steps = 0
        if "debug_shapes" not in t:
            t.debug_shapes = False
        if "action_stats_path" not in t:
            t.action_stats_path = None

        if "loss_mode" not in t:
            t.loss_mode = "hybrid"
        if "lambda_geo" not in t:
            t.lambda_geo = 0.1
        if "lambda_task" not in t:
            t.lambda_task = 0.0
        if "lambda_task_warmup_steps" not in t:
            t.lambda_task_warmup_steps = 0

        if "lr" not in t:
            t.lr = 1e-4
        if "betas" not in t:
            t.betas = [0.95, 0.999]
        if "weight_decay" not in t:
            t.weight_decay = 1e-6
        if "total_iters" not in t:
            t.total_iters = 1

        if "task_loss" not in t or t.task_loss is None:
            t.task_loss = OmegaConf.create({})
        if "type" not in t.task_loss:
            t.task_loss.type = "smooth"
        if "only_rot_segment" not in t.task_loss:
            t.task_loss.only_rot_segment = True
        if "t_start" not in t.task_loss:
            t.task_loss.t_start = 0
        if "t_end" not in t.task_loss:
            t.task_loss.t_end = int(cfg.model.act_horizon)


@torch.no_grad()
def _geodesic_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    rel = R_gt.transpose(-1, -2) @ R_pred
    tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = torch.clamp((tr - 1.0) * 0.5, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos)
    return theta * (180.0 / np.pi)


@torch.no_grad()
def _final_q_l1(pred: torch.Tensor, gt: torch.Tensor, action_type: str, hand_model, env_act_dim: int) -> torch.Tensor:
    if action_type == "joint_value":
        pred_q = pred[:, -1, env_act_dim:]
        gt_q = gt[:, -1, env_act_dim:]
    else:
        pred_q = projection_q(hand_model, pred[:, -1, env_act_dim:], input_q_type=action_type)
        gt_q = projection_q(hand_model, gt[:, -1, env_act_dim:], input_q_type=action_type)
    return torch.mean(torch.abs(pred_q - gt_q), dim=1)


@torch.no_grad()
def run_eval(cfg):
    _ensure_eval_runtime_cfg(cfg)

    schema = compute_action_schema(
        env_act_dim=int(cfg.dataset.env_act_dim),
        num_joints=int(cfg.dataset.num_joints),
        action_type=str(cfg.dataset.action_type),
    )
    cfg.dataset.rot_dim = schema.rot_dim
    cfg.dataset.rot_act_dim = schema.rot_act_dim
    cfg.dataset.act_dim = schema.act_dim

    if not cfg.checkpoint_path:
        raise ValueError("Please provide checkpoint_path for offline eval")

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    _, val_loader = create_dataloader(cfg.dataset, is_train=False)
    module = TrainingModule.load_from_checkpoint(cfg.checkpoint_path, map_location=device, cfg=cfg, strict=False)
    module = module.to(device)
    module.eval()
    model = module.ema_model if (module.use_ema_for_eval and module.use_ema and module.ema_model is not None) else module.model

    hand_model = create_hand_model("shadowhand", device=device)

    env_act_dim = int(cfg.dataset.env_act_dim)
    action_type = str(cfg.dataset.action_type)
    action_mode = str(getattr(cfg.dataset, "action_mode", "absolute")).lower()
    num_joints = int(getattr(cfg.dataset, "num_joints", 24))

    agg = defaultdict(list)
    max_batches = cfg.eval.max_batches

    for bidx, batch in enumerate(val_loader):
        if max_batches is not None and bidx >= int(max_batches):
            break

        # Keep eval path consistent with training/validation when using pointnet contact features.
        obs = module._build_obs_seq(batch)
        gt = batch["actions"].to(device)

        if action_mode == "relative" and action_type != "joint_value":
            pred = model.get_action(obs, return_full=True)
            gt_eval = gt
            if gt_eval.shape[1] != pred.shape[1]:
                t = min(int(gt_eval.shape[1]), int(pred.shape[1]))
                gt_eval = gt_eval[:, :t]
                pred = pred[:, :t]
            pred = action_seq_relative_to_absolute(
                pred,
                action_type=action_type,
                action_mode=action_mode,
                hand_model=hand_model,
                env_act_dim=env_act_dim,
                J=num_joints,
            )
            gt_eval = action_seq_relative_to_absolute(
                gt_eval,
                action_type=action_type,
                action_mode=action_mode,
                hand_model=hand_model,
                env_act_dim=env_act_dim,
                J=num_joints,
            )
        else:
            pred = model.get_action(obs)
            start = int(cfg.model.obs_horizon) - 1
            end = start + int(cfg.model.act_horizon)
            gt_eval = gt[:, start:end]

            if gt_eval.shape[1] != pred.shape[1]:
                t = min(int(gt_eval.shape[1]), int(pred.shape[1]))
                gt_eval = gt_eval[:, :t]
                pred = pred[:, :t]

        mse = torch.mean((pred - gt_eval) ** 2, dim=(1, 2))
        agg["action_mse"].extend(mse.cpu().tolist())

        smooth = torch.mean((pred[:, 1:] - pred[:, :-1]) ** 2, dim=(1, 2)) if pred.shape[1] > 1 else torch.zeros(pred.shape[0], device=pred.device)
        agg["smoothness_l2"].extend(smooth.cpu().tolist())

        if action_type in {"rot_quat", "rot_6d", "rot_vec", "rot_euler", "rot_mat"}:
            pred_rot = pred[:, :, env_act_dim:]
            gt_rot = gt_eval[:, :, env_act_dim:]
            R_pred = decode_rotations_to_R(pred_rot, action_type, J=int(cfg.dataset.num_joints))
            R_gt = decode_rotations_to_R(gt_rot, action_type, J=int(cfg.dataset.num_joints))
            theta_deg = _geodesic_deg(R_pred, R_gt)
            agg["rot_geodesic_deg_mean"].extend(theta_deg.mean(dim=(1, 2)).cpu().tolist())
            agg["final_step_rot_geodesic_deg"].extend(theta_deg[:, -1].mean(dim=1).cpu().tolist())
        else:
            agg["rot_geodesic_deg_mean"].extend([0.0] * pred.shape[0])
            agg["final_step_rot_geodesic_deg"].extend([0.0] * pred.shape[0])

        q_l1 = _final_q_l1(pred, gt_eval, action_type, hand_model, env_act_dim)
        agg["final_step_q_l1"].extend(q_l1.cpu().tolist())

    summary = {k: float(np.mean(v)) if len(v) else 0.0 for k, v in agg.items()}
    summary["action_type"] = action_type
    summary["checkpoint_path"] = cfg.checkpoint_path
    summary["obs_encoder"] = str(cfg.dataset.observation.encoder)
    summary["trajectory_interp"] = str(cfg.dataset.trajectory.interp)

    output_dir = cfg.eval.output_dir if cfg.eval.output_dir else os.path.join(ROOT_DIR, "output", "dexgrasp_eval")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"offline_metrics_{action_type}.csv")
    json_path = os.path.join(output_dir, f"offline_metrics_{action_type}.json")

    fieldnames = [
        "action_type",
        "obs_encoder",
        "trajectory_interp",
        "checkpoint_path",
        "action_mse",
        "rot_geodesic_deg_mean",
        "final_step_rot_geodesic_deg",
        "final_step_q_l1",
        "smoothness_l2",
    ]

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: summary.get(k, "") for k in fieldnames})

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("******************************** [DexGrasp Offline Eval] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print(summary)
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval_dexgrasp")
def main(cfg):
    run_eval(cfg)


if __name__ == "__main__":
    main()
