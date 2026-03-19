from __future__ import annotations

import csv
import json
import os
import sys

import hydra
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from dataset.DexBenchHDF5Dataset import create_dataloader
from model.dexbench.pl_module import DexBenchTrainingModule
from utils.dexbench_rotations import get_rot_repr_dim, rot_geodesic_deg_torch


def _masked_mse(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)
    diff = torch.square(pred - gt) * weight
    denom = weight.sum() * pred.shape[-1]
    if float(denom.item()) <= 0.0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return diff.sum() / denom


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(device=values.device, dtype=values.dtype)
    denom = weight.sum()
    if float(denom.item()) <= 0.0:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return (values * weight).sum() / denom


def _last_valid(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    counts = mask.sum(dim=1).long()
    valid = counts > 0
    last_idx = torch.clamp(counts - 1, min=0)
    batch_idx = torch.arange(values.shape[0], device=values.device)
    gathered = values[batch_idx, last_idx]
    return gathered, valid.to(dtype=values.dtype)


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval_dexbench")
def main(cfg):
    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path is required")

    rot_dim = get_rot_repr_dim(str(cfg.dataset.action_type))
    cfg.dataset.rot_dim = rot_dim
    cfg.dataset.env_act_dim = int(cfg.dataset.raw_act_dim) - len(cfg.dataset.rotation_indices)
    cfg.dataset.act_dim = int(cfg.dataset.env_act_dim + rot_dim)

    _, val_loader, train_ds, _ = create_dataloader(cfg.dataset)
    cfg.dataset.obs_dim = int(train_ds.raw_obs_dim)

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    module = DexBenchTrainingModule.load_from_checkpoint(cfg.checkpoint_path, map_location=device, cfg=cfg)
    module = module.to(device).eval()

    action_mse_list, smooth_list, geo_list, geo_last_list = [], [], [], []

    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if cfg.eval.max_batches is not None and bi >= int(cfg.eval.max_batches):
                break
            obs = batch["observations"].to(device)
            gt = batch["actions"].to(device)
            valid_mask = batch["valid_mask"].to(device=device, dtype=obs.dtype)

            pred = module.model.get_action(obs)
            start = int(cfg.model.obs_horizon) - 1
            end = start + int(cfg.model.act_horizon)
            gt_eval = gt[:, start:end]
            valid_eval = valid_mask[:, start:end].to(device=pred.device, dtype=pred.dtype)

            action_mse_list.append(_masked_mse(pred, gt_eval, valid_eval).item())
            if pred.shape[1] > 1:
                pair_mask = valid_eval[:, 1:] * valid_eval[:, :-1]
                smooth_list.append(_masked_mse(pred[:, 1:], pred[:, :-1], pair_mask).item())
            else:
                smooth_list.append(0.0)

            env_dim = int(cfg.dataset.env_act_dim)
            if str(cfg.dataset.action_type) == "joint_value":
                geo_list.append(0.0)
                geo_last_list.append(0.0)
            else:
                pred_rot = pred[:, :, env_dim: env_dim + rot_dim]
                gt_rot = gt_eval[:, :, env_dim: env_dim + rot_dim]
                geo = rot_geodesic_deg_torch(pred_rot, gt_rot, str(cfg.dataset.action_type))
                geo_list.append(_masked_mean(geo, valid_eval).item())
                geo_last_vals, last_valid = _last_valid(geo, valid_eval)
                geo_last_list.append(_masked_mean(geo_last_vals, last_valid).item())

    summary = {
        "action_type": str(cfg.dataset.action_type),
        "checkpoint_path": str(cfg.checkpoint_path),
        "obs_component_signature": str(getattr(train_ds, "obs_component_signature", "")),
        "obs_stats_path": str(getattr(train_ds, "obs_stats_path", "")),
        "action_mse": float(np.mean(action_mse_list)) if action_mse_list else 0.0,
        "smoothness_l2": float(np.mean(smooth_list)) if smooth_list else 0.0,
        "rot_geodesic_deg_mean": float(np.mean(geo_list)) if geo_list else 0.0,
        "final_step_rot_geodesic_deg": float(np.mean(geo_last_list)) if geo_last_list else 0.0,
    }

    out_dir = cfg.eval.output_dir if cfg.eval.output_dir else os.path.join(ROOT_DIR, "output", "dexbench_eval")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"offline_metrics_{cfg.dataset.action_type}.csv")
    json_path = os.path.join(out_dir, f"offline_metrics_{cfg.dataset.action_type}.json")

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(summary)
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
