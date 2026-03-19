from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import DDIMScheduler, DDPMScheduler

from model.diffusion_policy.conditional_unet1d import ConditionalUnet1D


class DexBenchDiffusionPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.obs_dim = int(cfg.dataset.obs_dim)
        self.act_dim = int(cfg.dataset.act_dim)
        self.obs_horizon = int(cfg.model.obs_horizon)
        self.act_horizon = int(cfg.model.act_horizon)
        self.pred_horizon = int(cfg.model.pred_horizon)
        self.num_diffusion_iters = int(cfg.model.num_diffusion_iters)
        self.prediction_type = str(cfg.model.diffusion_prediction_type)

        self.net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=self.obs_dim * self.obs_horizon,
            diffusion_step_embed_dim=int(cfg.model.diffusion_step_embed_dim),
            down_dims=list(cfg.model.unet_dims),
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=bool(cfg.model.scheduler_clip_sample),
            prediction_type=self.prediction_type,
        )
        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=bool(cfg.model.scheduler_clip_sample),
            prediction_type=self.prediction_type,
        )

    def forward(self, obs_seq: torch.Tensor, action_seq: torch.Tensor):
        B = obs_seq.shape[0]
        device = obs_seq.device

        obs_cond = obs_seq.flatten(start_dim=1)
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device).long()

        noisy = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        pred = self.net(noisy, timesteps, global_cond=obs_cond)

        if self.prediction_type == "epsilon":
            gt = noise
        elif self.prediction_type == "sample":
            gt = action_seq
        elif self.prediction_type == "v_prediction":
            abar = self.noise_scheduler.alphas_cumprod[timesteps].to(action_seq.dtype).view(B, 1, 1)
            gt = torch.sqrt(abar) * noise - torch.sqrt(1.0 - abar) * action_seq
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        return {"pred": pred, "gt": gt, "timesteps": timesteps}

    @torch.no_grad()
    def get_action(self, obs_seq: torch.Tensor, num_inference_steps: int = 10) -> torch.Tensor:
        B = obs_seq.shape[0]
        device = obs_seq.device

        self.ddim_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        obs_cond = obs_seq.flatten(start_dim=1)

        sample = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
        for t in self.ddim_scheduler.timesteps:
            pred = self.net(sample, t, global_cond=obs_cond)
            sample = self.ddim_scheduler.step(model_output=pred, timestep=t, sample=sample).prev_sample

        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return sample[:, start:end]
