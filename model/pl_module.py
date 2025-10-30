import os
import sys
import numpy as np
import torch
import pytorch_lightning as pl
from termcolor import colored
from torch.nn import functional as F
from diffusers.optimization import get_scheduler

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from model.diffusion_policy.diffusion_policy import create_model
from env.adroit_env import AdroitEnvWrapper
from utils.action_utils import decode_rotations_to_R


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = create_model(cfg)
        os.makedirs(cfg.training.save_dir, exist_ok=True)

    def ddp_print(self, *args, **kwargs):
        if self.global_rank == 0:
            print(*args, **kwargs)

    # ---------- SO(3) 几何距离：旋转矩阵 → 角度 ----------
    def geodesic_loss_R(self,R_pred, R_gt, reduction='mean', squared=True):#老版计算，现已不用
        # R_pred, R_gt: (...,3,3)
        R = R_gt.transpose(-1,-2) @ R_pred
        trace = R[...,0,0] + R[...,1,1] + R[...,2,2]
        cos = torch.clamp((trace - 1.0)*0.5, -1+1e-7, 1-1e-7)
        theta = torch.acos(cos)
        if squared: theta = theta**2
        if reduction=='mean': return theta.mean()
        if reduction=='sum':  return theta.sum()
        return theta

    # 1) 计算逐样本的几何损失（不做 batch 平均）
    # 举例：geodesic_loss_R 返回逐元素角度，我们自己在非 batch 维上 reduce 到 (B,)
    def geodesic_angle_per_sample(self, R_pred, R_gt, eps=1e-7):
        # R_pred, R_gt: (B, T, J, 3, 3)
        R = R_gt.transpose(-1, -2) @ R_pred                 # (B,T,J,3,3)
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]  # (B,T,J)
        cos = torch.clamp((trace - 1.0) * 0.5, -1 + eps, 1 - eps)
        theta = torch.acos(cos)                              # (B,T,J) 弧度
        # 对时间与关节做平均，得到每个样本一个标量 (B,)
        return theta.mean(dim=(1, 2))

    # def training_step(self, batch, batch_idx):
    #     model_output = self.model(
    #         obs_seq=batch["observations"],
    #         action_seq=batch["actions"],
    #     )
    #     loss = F.mse_loss(model_output["gt"], model_output["pred"])
    #     self.log("loss", loss, prog_bar=True)
    #     return loss
    def training_step(self, batch, batch_idx):
        model_output = self.model(
            obs_seq=batch["observations"],
            action_seq=batch["actions"],
        )
        eps_pred  = model_output["pred"]                  # (B,T,act_dim)  预测噪声
        noise     = model_output["gt"]                    # (B,T,act_dim)  真实噪声
        timesteps = model_output["timesteps"]             # (B,)           本次 forward 采样的 t
        
        # 主损失：ε 的 MSE（保持扩散稳定）
        L_main = F.mse_loss(eps_pred, noise)
        #loss = L_main       
        # 仅当有旋转类 action_type 时，才做几何项
        pt = self.model.diffusion_prediction_type
        if pt == 'epsilon':

            if self.model.action_type in {'rot_quat','rot_6d','rot_vec','rot_euler'}:
                
                # 反解 x0_est
                x0 = batch["actions"]

                alphas_cumprod = self.model.noise_scheduler.alphas_cumprod.to(noise.device)
                a_bar = alphas_cumprod[timesteps].view(-1,1,1)     # (B,1,1)
                # 数值安全（可选但推荐）
                a_bar = torch.clamp(a_bar, 1e-5, 1 - 1e-5)
                                            # (B,T,act_dim)
                x_t = self.model.noise_scheduler.add_noise(x0, noise, timesteps)
                x0_est = (x_t - torch.sqrt(1 - a_bar) * eps_pred) / torch.sqrt(a_bar)

                # 取旋转段并解码为矩阵
                env_dim = self.cfg.dataset.env_act_dim
                

                # geodesic
                #L_geo = self.geodesic_loss_R(R_pred, R_gt, squared=True)

                # 权重：与有效信号幅度同量级
                w_t = (a_bar ** 0.5).detach()                             # (B,1,1)
                w_t = w_t.squeeze(-1).squeeze(-1)                         # (B,)

                # 3) 加权并做 batch 平均，得到标量 L_geo
                # L_geo = (theta_per_B * w_t).mean()     
                # loss=theta_per_B.mean()       
                #loss  = L_main + self.cfg.training.lambda_geo * L_geo
                #loss=L_geo

                # self.log_dict({"L_main": L_main, "L_geo": L_geo, "L_total": loss}, prog_bar=True)
            # else:
            #     # 非旋转类型（如 joint_value），可在此加 S¹ 几何或 FK→SO(3)（见前文）
            #     # self.log("L_main", L_main, prog_bar=True)
            #     return L_main
        
        elif pt == 'sample':
            x0_pred = model_output["pred"]                # (B,T,act_dim) 预测的无噪声动作
            x0_est = model_output["pred"]
            # 权重：可以设为全1；若也想弱化高噪步，可同样取 sqrt(a_bar)
            #w_t = torch.ones(x0_est.shape[0], device=x0_est.device)         # (B,)
            alphas_cumprod = self.model.noise_scheduler.alphas_cumprod.to(x0_pred.device)
            w_t = torch.sqrt(torch.clamp(alphas_cumprod[timesteps], 1e-5, 1-1e-5)).detach()#带权重

        x0 = batch["actions"]
        env_dim = self.cfg.dataset.env_act_dim
        rot_pred = x0_est[:, :, env_dim:]                  # (B,T,24*rot_dims)
        rot_gt   = x0[:,     :, env_dim:]
        R_pred   = decode_rotations_to_R(rot_pred, self.model.action_type)  # (B,T,24,3,3)
        R_gt     = decode_rotations_to_R(rot_gt,   self.model.action_type)
        theta_per_B = self.geodesic_angle_per_sample(R_pred, R_gt)  # (B,)
        theta_per_B = (theta_per_B ** 2)
        if self.cfg.training.loss_mode == 'geo_only':
            loss = theta_per_B.mean()
        elif self.cfg.training.loss_mode == 'hybrid':
            L_geo = (theta_per_B * w_t).mean()
            loss = L_main + self.cfg.training.lambda_geo * L_geo

        elif self.cfg.training.loss_mode == 'mae':
            loss=L_main
        else:
            raise ValueError(f"Unknown loss_mode: {self.cfg.training.loss_mode}")
        self.log_dict({"L_main": L_main, "L_geo": L_geo, "loss": loss,"theta_per_B": theta_per_B.mean()}, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.env = AdroitEnvWrapper(
            env_name=self.cfg.dataset.env_name,
            action_type=self.cfg.dataset.action_type,
            obs_horizon=self.cfg.model.obs_horizon,
            act_horizon=self.cfg.model.act_horizon,
            device=self.device,
            render_mode=None#修改为rgb_array
        )
        self.eval_results = []

    def validation_step(self, batch, batch_idx):
        self.env.reset()
        while not self.env.is_done:
            obs_seq = self.env.get_obs_seq()
            action_seq = self.model.get_action(obs_seq)
            self.env.step(action_seq)
        self.eval_results.append(self.env.is_success)

    def on_validation_epoch_end(self):
        self.env.close()
        success_rate = np.mean(self.eval_results)
        self.ddp_print(colored(f"\nValidation Success Rate: {success_rate * 100:.2f}%", 'blue'))
        self.log("success_rate", success_rate, sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.lr,
            betas=self.cfg.training.betas,
            weight_decay=self.cfg.training.weight_decay
        )
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=self.cfg.training.total_iters,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
