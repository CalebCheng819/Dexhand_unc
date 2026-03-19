from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
from diffusers.optimization import get_scheduler

from model.dexbench.diffusion_policy import DexBenchDiffusionPolicy
from utils.dexbench_rotations import get_rot_repr_dim, rot_geodesic_deg_torch, rot_geodesic_torch


class DexBenchTrainingModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = DexBenchDiffusionPolicy(cfg)
        self.action_type = str(cfg.dataset.action_type)
        training_cfg = cfg.training if hasattr(cfg, "training") else None
        self.rotation_loss_weight = float(getattr(training_cfg, "rotation_loss_weight", 1.0)) if training_cfg is not None else 1.0
        self._warned_non_sample_prediction = False

    @staticmethod
    def _masked_mse(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)
        diff = torch.square(pred - gt) * weight
        denom = weight.sum() * pred.shape[-1]
        if float(denom.item()) <= 0.0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return diff.sum() / denom

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(device=values.device, dtype=values.dtype)
        denom = weight.sum()
        if float(denom.item()) <= 0.0:
            return torch.zeros((), device=values.device, dtype=values.dtype)
        return (values * weight).sum() / denom

    @staticmethod
    def _last_valid(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        counts = mask.sum(dim=1).long()
        valid = counts > 0
        last_idx = torch.clamp(counts - 1, min=0)
        batch_idx = torch.arange(values.shape[0], device=values.device)
        gathered = values[batch_idx, last_idx]
        return gathered, valid.to(dtype=values.dtype)

    def training_step(self, batch, batch_idx):
        out = self.model(batch["observations"], batch["actions"])
        pred = out["pred"]
        gt = out["gt"]
        valid_mask = batch["valid_mask"].to(device=pred.device, dtype=pred.dtype)

        if str(self.model.prediction_type) != "sample":
            if not self._warned_non_sample_prediction:
                rank_zero_warn = getattr(pl.utilities.rank_zero, "rank_zero_warn", None)
                if rank_zero_warn is not None:
                    rank_zero_warn(
                        f"DexBench SO(3) training loss requires prediction_type='sample'; "
                        f"falling back to masked MSE for prediction_type={self.model.prediction_type}."
                    )
                self._warned_non_sample_prediction = True
            loss = self._masked_mse(pred, gt, valid_mask)
            self.log("loss", loss, prog_bar=True)
            return loss

        if self.action_type == "joint_value":
            loss_non_rot = self._masked_mse(pred, gt, valid_mask)
            loss_rot = torch.zeros((), device=pred.device, dtype=pred.dtype)
            loss = loss_non_rot
        else:
            env_dim = int(self.cfg.dataset.env_act_dim)
            rot_dim = get_rot_repr_dim(self.action_type)
            pred_env = pred[..., :env_dim]
            gt_env = gt[..., :env_dim]
            pred_rot = pred[..., env_dim: env_dim + rot_dim]
            gt_rot = gt[..., env_dim: env_dim + rot_dim]

            loss_non_rot = self._masked_mse(pred_env, gt_env, valid_mask)
            loss_rot = self._masked_mean(rot_geodesic_torch(pred_rot, gt_rot, self.action_type), valid_mask)
            loss = loss_non_rot + self.rotation_loss_weight * loss_rot

        self.log("loss", loss, prog_bar=True)
        self.log("loss_non_rot_mse", loss_non_rot, prog_bar=False)
        self.log("loss_rot_geo_rad", loss_rot, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        obs = batch["observations"]
        gt = batch["actions"]
        valid_mask = batch["valid_mask"].to(device=obs.device, dtype=obs.dtype)
        pred = self.model.get_action(obs)

        start = int(self.cfg.model.obs_horizon) - 1
        end = start + int(self.cfg.model.act_horizon)
        gt_eval = gt[:, start:end]
        valid_eval = valid_mask[:, start:end].to(device=pred.device, dtype=pred.dtype)

        action_mse = self._masked_mse(pred, gt_eval, valid_eval)
        if pred.shape[1] > 1:
            pair_mask = valid_eval[:, 1:] * valid_eval[:, :-1]
            smooth = self._masked_mse(pred[:, 1:], pred[:, :-1], pair_mask)
        else:
            smooth = torch.zeros((), device=pred.device, dtype=pred.dtype)

        env_dim = int(self.cfg.dataset.env_act_dim)
        rot_dim = get_rot_repr_dim(self.action_type)
        if self.action_type == "joint_value":
            geo = torch.zeros((), device=pred.device, dtype=pred.dtype)
            geo_last = torch.zeros((), device=pred.device, dtype=pred.dtype)
        else:
            pred_rot = pred[:, :, env_dim: env_dim + rot_dim]
            gt_rot = gt_eval[:, :, env_dim: env_dim + rot_dim]
            geo_all = rot_geodesic_deg_torch(pred_rot, gt_rot, self.action_type)
            geo = self._masked_mean(geo_all, valid_eval)
            geo_last_vals, last_valid = self._last_valid(geo_all, valid_eval)
            geo_last = self._masked_mean(geo_last_vals, last_valid)

        self.log("val_action_mse", action_mse, prog_bar=True, sync_dist=True)
        self.log("val_smoothness_l2", smooth, prog_bar=False, sync_dist=True)
        self.log("val_rot_geodesic_deg", geo, prog_bar=True, sync_dist=True)
        self.log("val_final_rot_geodesic_deg", geo_last, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg.training.lr),
            betas=tuple(self.cfg.training.betas),
            weight_decay=float(self.cfg.training.weight_decay),
        )
        sch = get_scheduler(
            name="cosine",
            optimizer=opt,
            num_warmup_steps=100,
            num_training_steps=int(self.cfg.training.total_iters),
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "step",
                "frequency": 1,
            },
        }
