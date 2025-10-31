import time
import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler, DDIMScheduler

from model.diffusion_policy.conditional_unet1d import ConditionalUnet1D
from utils.action_utils import ROT_DIMS


class DiffusionPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.action_type = cfg.dataset.action_type
        self.obs_dim = cfg.dataset.obs_dim
        self.act_dim = cfg.dataset.act_dim
        self.obs_horizon = cfg.model.obs_horizon
        self.act_horizon = cfg.model.act_horizon
        self.pred_horizon = cfg.model.pred_horizon
        self.num_diffusion_iters = cfg.model.num_diffusion_iters
        self.diffusion_prediction_type = cfg.model.diffusion_prediction_type

        if self.action_type == 'joint_value':
            self.noise_pred_net = ConditionalUnet1D(
                input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
                global_cond_dim=self.obs_dim * self.obs_horizon,
                diffusion_step_embed_dim=cfg.model.diffusion_step_embed_dim,
                down_dims=cfg.model.unet_dims,
            )
        else:#分层处理不同维度的动作
    
            env_act_dim = cfg.dataset.env_act_dim
            self.sub_ranges = [(0, env_act_dim)] if env_act_dim > 0 else []
            rot_dims = ROT_DIMS[cfg.dataset.action_type]
            assert self.act_dim - env_act_dim == 24 * rot_dims, \
                f"Action dimension mismatch! Got {self.act_dim}, expected {cfg.dataset.env_act_dim + 24 * rot_dims}."
            self.sub_ranges.extend([(env_act_dim + idx * rot_dims, env_act_dim + (idx + 1) * rot_dims) for idx in range(24)])
            self.noise_pred_net = nn.ModuleList([
                ConditionalUnet1D(
                    input_dim=(end - start),
                    global_cond_dim=self.obs_dim * self.obs_horizon,
                    diffusion_step_embed_dim=cfg.model.diffusion_step_embed_dim,
                    down_dims=cfg.model.unet_dims,
                ) for start, end in self.sub_ranges
            ])

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.model.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=cfg.model.scheduler_clip_sample,
            prediction_type=self.diffusion_prediction_type
        )
        self.noise_ddim_scheduler = DDIMScheduler(
            num_train_timesteps=cfg.model.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=cfg.model.scheduler_clip_sample,
            prediction_type=self.diffusion_prediction_type,
        )

    def forward(self, obs_seq, action_seq):
        device = obs_seq.device
        B = obs_seq.shape[0]

        # observation as FiLM conditioning
        obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

        # sample noise to add to actions & diffusion iteration for each data point
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device).long()

        # add noise to the clean actions according to the noise magnitude at each diffusion iteration
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        if self.action_type == 'joint_value':
            pred = self.noise_pred_net(noisy_action_seq, timesteps, global_cond=obs_cond)
        else:
            sub_noisy_action_seqs = [noisy_action_seq[:, :, start: end] for start, end in self.sub_ranges]
            pred = []
            for idx, sub_noise_pred_net in enumerate(self.noise_pred_net):
                sub_pred = sub_noise_pred_net(sub_noisy_action_seqs[idx], timesteps, global_cond=obs_cond)
                pred.append(sub_pred)
            pred = torch.cat(pred, dim=-1)

        return {
            'gt': noise if self.diffusion_prediction_type == 'epsilon' else action_seq,
            'pred': pred,
            'timesteps': timesteps,   # ★ 新增：供几何项反解 x0 使用
        }

    def get_action(self, obs_seq):
        self.noise_ddim_scheduler.set_timesteps(num_inference_steps=10)

        device = obs_seq.device
        B = obs_seq.shape[0]
        with torch.no_grad():
            obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

            # initialize action from Gaussian noise
            noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

            for t in self.noise_ddim_scheduler.timesteps:  # inverse order
                if self.action_type == 'joint_value':
                    model_output = self.noise_pred_net(
                        sample=noisy_action_seq,
                        timestep=t,
                        global_cond=obs_cond,
                    )
                else:
                    sub_noisy_action_seqs = [noisy_action_seq[:, :, start: end] for start, end in self.sub_ranges]
                    model_output = []
                    for idx, sub_noise_pred_net in enumerate(self.noise_pred_net):
                        pred = sub_noise_pred_net(sample=sub_noisy_action_seqs[idx], timestep=t, global_cond=obs_cond)
                        model_output.append(pred)
                    model_output = torch.cat(model_output, dim=-1)

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_ddim_scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)


def create_model(cfg):
    model = DiffusionPolicy(cfg)
    return model
