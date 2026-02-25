import os
import sys
import time
import imageio
import numpy as np
import minari
import gymnasium as gym
import gymnasium_robotics
import torch
from datetime import datetime
from collections import deque
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model
from utils.action_utils import projection_q, ROT_DIMS,relative_rot_to_absolute,decode_rotations_to_R,ROT_DIMS



class AdroitEnvWrapper:
    def __init__(self, env_name, action_type, obs_horizon, act_horizon, device, render_mode=None,action_mode: str = "absolute", ):#rgb array方式
        self.env = gym.make(env_name, max_episode_steps=200, render_mode=render_mode)
        self.action_type = action_type
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.render_mode = render_mode
        self.device = device
        self.action_mode = action_mode  # 新增
        self.obs_buffer = deque(maxlen=obs_horizon)
        self.hand_model = create_hand_model('shadowhand', device)

        self.rot_dim =ROT_DIMS[action_type]

    def reset(self):
        self.is_done = False
        self.is_success = None
        self.frames = []
        self.obs_buffer.clear()
        self.R_abs_prev = None  # 新增：每个 episode 重置

        obs, info = self.env.reset()
        for _ in range(self.obs_horizon):
            self.obs_buffer.append(obs)
        if self.render_mode is not None:
            frame = self.env.render()
            self.frames.append(frame)

    def close(self):
        self.env.close()

    def get_obs_seq(self):
        return torch.from_numpy(np.array(self.obs_buffer, dtype=np.float32)[None, :]).to(self.device)

    # def step(self, actions):
    #     assert actions.shape[0] == 1, "[Adroit Env] Only support batch size = 1!"
    #     actions = actions.squeeze(0)  # remove batch dim
    #     actions = actions.unsqueeze(0) if len(actions.shape) == 1 else actions  # ensure seq dim
    #     if self.action_type != 'joint_value':#需要转换回joint_value
    #         hand_model = create_hand_model('shadowhand', device=actions.device)
    #         rot_dim = 24 * ROT_DIMS[self.action_type]
    #         q = projection_q(hand_model, actions[:, -rot_dim:], input_q_type=self.action_type)
    #         q_control = hand_model.q_real2control(q)
    #         actions = torch.cat([actions[:, :-rot_dim], q_control], dim=-1)
    #     actions = actions.cpu().numpy()
    #
    #     for action in actions:
    #         obs, reward, terminated, truncated, info = self.env.step(action)
    #         self.obs_buffer.append(obs)
    #         if self.render_mode is not None:
    #             frame = self.env.render()
    #             self.frames.append(frame)
    #         if info["success"] or terminated or truncated:
    #             self.is_success = info["success"]
    #             self.is_done = True
    #             break
    def step(self, actions):
        assert actions.shape[0] == 1, "[Adroit Env] Only support batch size = 1!"
        actions = actions.squeeze(0)  # (T, D) 或 (D,)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)  # (1, D)
        actions = actions.to(self.device).float()

        # 1) 如果是 joint_value，直接走 env，不用管旋转表示
        if self.action_type == 'joint_value':
            actions_np = actions.cpu().numpy()
        else:
            # 非 joint_value：最后 rot_dim 是 hand 的旋转表示
            assert self.rot_dim > 0
            other = actions[:, :-self.rot_dim*24]         # (T, D - rot_dim)
            rot_part = actions[:, -self.rot_dim*24:]      # (T, rot_dim)

            # 2) 根据 action_mode 选择 absolute / relative 逻辑
            if self.action_mode == "relative":
                # 模型输出的是相对增量，需要还原成绝对表示
                rot_abs = relative_rot_to_absolute(
                    rel_seq=rot_part,
                    action_type=self.action_type,
                    J=24,
                    order='xyz',
                    R0=self.R_abs_prev         # 可能是 None（首次）或上一步的绝对 R
                )                             # (T, rot_dim)

                # 更新 R_abs_prev：用这一段序列的最后一帧作为下次的起点
                dim = ROT_DIMS[self.action_type]
                last_abs_repr = rot_abs[-1].reshape(1, 24, dim)  # (1, 24, dim)
                self.R_abs_prev = decode_rotations_to_R(
                    last_abs_repr, self.action_type
                )                                               # (1, 24, 3, 3)

                rot_for_projection = rot_abs                    # (T, rot_dim)
            else:
                # action_mode == 'absolute'：保持你原来的行为
                rot_for_projection = rot_part

                # 顺便更新一下 R_abs_prev（如果你以后想在中途切换模式也不至于乱）
                dim = ROT_DIMS[self.action_type]
                last_abs_repr = rot_for_projection[-1].reshape(1, 24, dim)
                self.R_abs_prev = decode_rotations_to_R(
                    last_abs_repr, self.action_type
                )

            # 3) 用 hand_model 把旋转表示投到真实关节角，再转成控制量
            hand_model = self.hand_model  # 复用，不要每步 create_hand_model
            q = projection_q(
                hand_model,
                rot_for_projection,   # (T, rot_dim)
                input_q_type=self.action_type
            )                         # (T, 24) 或类似，取决于你的 hand_model 实现

            q_control = hand_model.q_real2control(q)  # (T, 24)

            # 把非旋转部分和控制量拼起来，作为最终动作
            actions_full = torch.cat([other, q_control], dim=-1)  # (T, D_env)
            actions_np = actions_full.cpu().numpy()

        # 4) 按你原来的方式喂给 gym env
        for action in actions_np:
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.obs_buffer.append(obs)
            if self.render_mode is not None:
                frame = self.env.render()
                self.frames.append(frame)
            if info.get("success", False) or terminated or truncated:
                self.is_success = info.get("success", False)
                self.is_done = True
                break


    def save_gif(self, save_dir, file_name):
        assert self.render_mode is not None, "[Adroit Env] render_mode is None!"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(f"{save_dir}/{file_name}", exist_ok=True)
        gif_filename = f"{self.is_success}-{timestamp}"
        imageio.mimsave(f"{save_dir}/{file_name}/{gif_filename}.gif", self.frames)
