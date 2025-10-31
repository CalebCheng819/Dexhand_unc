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
from utils.action_utils import projection_q, ROT_DIMS


class AdroitEnvWrapper:
    def __init__(self, env_name, action_type, obs_horizon, act_horizon, device, render_mode=None):#rgb array方式
        self.env = gym.make(env_name, max_episode_steps=200, render_mode=render_mode)
        self.action_type = action_type
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.render_mode = render_mode
        self.device = device

        self.obs_buffer = deque(maxlen=obs_horizon)
        self.hand_model = create_hand_model('shadowhand', device)

    def reset(self):
        self.is_done = False
        self.is_success = None
        self.frames = []
        self.obs_buffer.clear()

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

    def step(self, actions):
        assert actions.shape[0] == 1, "[Adroit Env] Only support batch size = 1!"
        actions = actions.squeeze(0)  # remove batch dim
        actions = actions.unsqueeze(0) if len(actions.shape) == 1 else actions  # ensure seq dim
        if self.action_type != 'joint_value':#需要转换回joint_value
            hand_model = create_hand_model('shadowhand', device=actions.device)
            rot_dim = 24 * ROT_DIMS[self.action_type]
            q = projection_q(hand_model, actions[:, -rot_dim:], input_q_type=self.action_type)
            q_control = hand_model.q_real2control(q)
            actions = torch.cat([actions[:, :-rot_dim], q_control], dim=-1)
        actions = actions.cpu().numpy()

        for action in actions:
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.obs_buffer.append(obs)
            if self.render_mode is not None:
                frame = self.env.render()
                self.frames.append(frame)
            if info["success"] or terminated or truncated:
                self.is_success = info["success"]
                self.is_done = True
                break

    def save_gif(self, save_dir, file_name):
        assert self.render_mode is not None, "[Adroit Env] render_mode is None!"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(f"{save_dir}/{file_name}", exist_ok=True)
        gif_filename = f"{self.is_success}-{timestamp}"
        imageio.mimsave(f"{save_dir}/{file_name}/{gif_filename}.gif", self.frames)
