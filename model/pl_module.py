import os
import sys
import numpy as np
import torch
import copy
import pytorch_lightning as pl
from termcolor import colored
from torch.nn import functional as F
from diffusers.optimization import get_scheduler
from typing import Optional  # 放在文件开头的 import 里

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from model.diffusion_policy.diffusion_policy import create_model
from env.adroit_env import AdroitEnvWrapper
from utils.action_utils import decode_rotations_to_R
def Rseq_to_qseq_along_axes(R_seq, hand_model):
    """
    R_seq: (B,T,J,3,3)  每个关节的绝对旋转矩阵（或相对到关节本地轴，二者一致时可直接用）
    return q: (B,T,D)   与 hand_model.pk_chain 的关节顺序一致
    """
    B,T,J = R_seq.shape[:3]
    # rotvec: (B,T,J,3)
    omega = R_to_rotvec_torch(R_seq.reshape(B*T*J,3,3)).reshape(B,T,J,3)

    # 取出每个关节轴 a_j, 组为 (J,3)，并广播到 (B,T,J,3)
    axes = torch.stack([hand_model.joint_axes[name] for name in hand_model.joint_orders], dim=0)  # (J,3)
    axes = axes.to(R_seq.device).unsqueeze(0).unsqueeze(0).expand(B,T,-1,-1)

    # θ_j = <ω_j, a_j>
    q = (omega * axes).sum(dim=-1)  # (B,T,J)

    # clip to joint limits
    q_lower, q_upper = hand_model.get_joint_limits()   # (D,), (D,)
    q = torch.max(q, q_lower.view(1,1,-1))
    q = torch.min(q, q_upper.view(1,1,-1))
    return q


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = create_model(cfg)
        os.makedirs(cfg.training.save_dir, exist_ok=True)
        # ===== EMA 开关与参数（带默认值，防止 cfg 不含 train 键时报错）=====
        self.use_ema          = getattr(cfg.training, "use_ema", True)
        self.ema_decay_target = getattr(cfg.training, "ema_decay", 0.998)
        self.use_ema_for_eval = getattr(cfg.training, "use_ema_for_eval", True)
        self.ema_warmup_steps = getattr(cfg.training, "ema_warmup_steps", 0)

        # ===== 创建 EMA 模型 =====
        self.ema_model = None
        if self.use_ema:
            # 用 deepcopy 确保结构/初值完全一致
            self.ema_model = copy.deepcopy(self.model)
            # 影子模型不参与梯度
            for p in self.ema_model.parameters():
                p.requires_grad = False
        # ===== 动作统计（用于 smooth 标准化）=====
        # 如果 only_rot_segment=True，这里注册的是旋转段的维度；先占位，运行时再对齐拷贝
        D_task = getattr(cfg.model, "act_dim", None)  # 如果你能拿到动作维度，填对更好
        if D_task is None:
            D_task = 1  # 先占位，不影响后续 copy_ 覆盖
        self.register_buffer("act_mean", torch.zeros(D_task))
        self.register_buffer("act_std",  torch.ones(D_task))

        # 可选：从文件载入统计（离线算好的）
        stats_path = getattr(cfg.training, "action_stats_path", None)
        if stats_path and os.path.isfile(stats_path):
            s = torch.load(stats_path, map_location="cpu")
            self.act_mean = s["mean"].to(self.act_mean.dtype)
            self.act_std  = s["std"].clamp_min(1e-6).to(self.act_std.dtype)


    @torch.no_grad()
    def _ema_update(self, cur_decay: float):
        # 1) 同步 BN buffers（running_mean/var 等），保持统计一致
        for b_ema, b in zip(self.ema_model.buffers(), self.model.buffers()):
            b_ema.copy_(b)

        # 2) 指数滑动平均更新参数
        for p_ema, p in zip(self.ema_model.parameters(), self.model.parameters()):
            p_ema.data.mul_(cur_decay).add_(p.data, alpha=1.0 - cur_decay)

    def _ema_cur_decay(self):
        """带预热的当前 decay 值。"""
        if self.ema_warmup_steps and self.global_step < self.ema_warmup_steps:
            ratio = float(self.global_step) / max(1, self.ema_warmup_steps)
            return self.ema_decay_target * ratio
        return self.ema_decay_target

    def on_before_zero_grad(self, optimizer):
        if self.use_ema and self.ema_model is not None:
            cur_decay = self._ema_cur_decay()
            self._ema_update(cur_decay)


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

    def _current_lambda_task(self):
        """对 lambda_task 做 warmup（线性预热）"""
        lambda_task = getattr(self.cfg.training, "lambda_task", 0.0)
        warmup = getattr(self.cfg.training, "lambda_task_warmup_steps", 0)
        if warmup and self.global_step < warmup:
            return lambda_task * float(self.global_step) / max(1, warmup)
        return lambda_task



    def _select_x_for_task(self, x0_est: torch.Tensor):
        """根据 only_rot_segment & action_type 切出用于任务损失的动作段"""
        only_rot = getattr(self.cfg.training.task_loss, "only_rot_segment", False) \
                   if getattr(self.cfg, "training", None) and getattr(self.cfg.training, "task_loss", None) else False
        env_dim = getattr(self.cfg.dataset, "env_act_dim", 0)
        if only_rot and self.model.action_type in {'rot_quat','rot_6d','rot_vec','rot_euler'}:
            return x0_est[:, :, env_dim:]  # (B,T,D_task)
        return x0_est  # (B,T,D)

    def _smooth_loss_standardized(
                self,
                x_for_task: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        标准化 + 时间平滑（支持可变长 mask）
        x_for_task: (B, T, D_task)
        valid_mask: (B, T) 或 None
        """
        x = x_for_task
        # 保证 act_mean/std 与当前维度一致
        D_task = x.size(-1)
        if self.act_mean.numel() != D_task:
            # 维度不一致时，按当前维度重建 buffer（只发生在首次/only_rot 场景）
            self.act_mean = torch.zeros(D_task, device=x.device)
            self.act_std  = torch.ones(D_task,  device=x.device)
        act_mean = self.act_mean.to(x.device)
        act_std  = self.act_std.clamp_min(1e-6).to(x.device)

        x_norm = (x - act_mean) / act_std

        # T<2 无需平滑
        if x_norm.size(1) < 2:
            return x_norm.new_tensor(0.0)

        diff = x_norm[:, 1:] - x_norm[:, :-1]  # (B, T-1, D_task)

        if valid_mask is not None:
            m = valid_mask.float()
            if m.size(1) < 2:
                return x_norm.new_tensor(0.0)
            m_diff = (m[:, 1:] * m[:, :-1]).unsqueeze(-1)  # (B, T-1, 1)
            num = ((diff ** 2) * m_diff).sum()
            den = (m_diff.sum() * diff.size(-1)).clamp_min(1e-8)
            return num / den
        else:
            return (diff ** 2).mean()


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
        # === L_task：任务损失 ===
        # 依赖 x0_est（无噪声预测动作），已经在上面 pt 分支里得到
        device = self.device
        L_task = torch.tensor(0.0, device=device)

        # 从配置里读 task 损失类型与权重
        task_cfg = getattr(self.cfg.training, "task_loss", None)
        lambda_task = getattr(self.cfg.training, "lambda_task", 0.0)

        if task_cfg is not None and lambda_task > 0:
            task_type = getattr(task_cfg, "type", "smooth")  # 'smooth' | 'l2' | 'fingertip_pos'
            # 你是否只对旋转段施加任务损失（可选）
            only_rot = getattr(task_cfg, "only_rot_segment", False)
            env_dim = getattr(self.cfg.dataset, "env_act_dim", 0)

            if only_rot and self.model.action_type in {'rot_quat', 'rot_6d', 'rot_vec', 'rot_euler'}:
                x_for_task = x0_est[:, :, env_dim:]  # 只对旋转部分做任务损失
            else:
                x_for_task = x0_est  # 对完整动作做任务损失

            if task_type == "smooth":
                # 时间平滑：\sum_t ||x_{t} - x_{t-1}||^2
                # 与扩散兼容性好、实现简单且稳定
                # diff = x_for_task[:, 1:] - x_for_task[:, :-1]  # (B, T-1, D_task)
                # L_task = (diff ** 2).mean()
                # 1) 选择用于任务损失的动作段
                x_for_task = self._select_x_for_task(x0_est)  # (B,T,D_task)

                # 2) 拿到可变长序列的 mask（如果你的 batch 有这个键）
                valid_mask = batch.get("valid_mask", None)  # (B,T) or None

                # 3) 标准化 + 平滑（带 mask）
                L_task = self._smooth_loss_standardized(x_for_task, valid_mask)

            elif task_type == "l2":
                # 幅值正则：\sum_t ||x_t||^2
                L_task = (x_for_task ** 2).mean()

            elif task_type == "fingertip_pos":
                # 需要：batch["tip_goal"] 形状 (B, T, K, 3) 以及一个 FK 函数把 x0_est -> 指尖位置
                # 注意：若 action_type = joint_value，FK 直接吃关节角最方便；
                #      若 action_type 为 rot_*，你需要一个 decode + 逆解/近似FK
                assert "tip_goal" in batch, "使用 fingertip_pos 任务损失需要 batch['tip_goal']"
                tip_goal = batch["tip_goal"]  # (B, T, K, 3)

                # 根据配置决定取哪个时间段（比如只对 act_horizon 部分做任务）
                t_start = getattr(task_cfg, "t_start", 0)
                t_end = getattr(task_cfg, "t_end", x_for_task.shape[1])  # 默认全时段
                x_slice = x0_est[:, t_start:t_end]  # (B, T', D)

                # 把预测动作映射到指尖位置（你需要实现该函数，见下方 2)）
                tip_pred = self.forward_kinematics_to_tips(x_slice)  # (B, T', K, 3)

                goal_slice = tip_goal[:, t_start:t_end]
                L_task = F.mse_loss(tip_pred, goal_slice)

            else:
                raise ValueError(f"Unknown task_loss.type: {task_type}")
        x0 = batch["actions"]
        env_dim = self.cfg.dataset.env_act_dim
        rot_pred = x0_est[:, :, env_dim:]                  # (B,T,24*rot_dims)
        rot_gt   = x0[:,     :, env_dim:]
        R_pred   = decode_rotations_to_R(rot_pred, self.model.action_type)  # (B,T,24,3,3)
        R_gt     = decode_rotations_to_R(rot_gt,   self.model.action_type)
        theta_per_B = self.geodesic_angle_per_sample(R_pred, R_gt)  # (B,)
        theta_per_B = (theta_per_B ** 2)
        lambda_task_cfg = getattr(self.cfg.training, "lambda_task", 0.0)  # 旧变量仍可用于 log
        lambda_task = self._current_lambda_task()  # ✅ 用 warmup 后的权重
        if self.cfg.training.loss_mode == 'geo_only':
            loss = theta_per_B.mean()
            L_geo=theta_per_B.mean()
        elif self.cfg.training.loss_mode == 'hybrid':
            L_geo = (theta_per_B * w_t).mean()
            loss = L_main + self.cfg.training.lambda_geo * L_geo+lambda_task * L_task

        elif self.cfg.training.loss_mode == 'mse':
            loss=L_main+ lambda_task * L_task
            L_geo=theta_per_B.mean()#为了log输出，不参与loss计算
        else:
            raise ValueError(f"Unknown loss_mode: {self.cfg.training.loss_mode}")
        self.log_dict({"L_main": L_main, "L_geo": L_geo, "loss": loss,"theta_per_B": theta_per_B.mean(),"L_task": L_task.detach(), "lambda_task_now": torch.tensor(lambda_task, device=self.device)}, prog_bar=True)
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
            m = self.ema_model if (self.use_ema_for_eval and self.use_ema and self.ema_model is not None) else self.model
            action_seq = m.get_action(obs_seq)
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
