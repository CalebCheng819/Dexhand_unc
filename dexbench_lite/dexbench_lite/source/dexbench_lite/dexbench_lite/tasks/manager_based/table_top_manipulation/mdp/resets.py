from __future__ import annotations

import torch

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    quat_mul,
)

def reset_root_pose_uniform(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict,
):
    """Reset root pose (position + orientation) without writing velocities.

    This is safe for kinematic / fixed-base articulations.
    pose_range values are treated as offsets from default_root_state.
    """
    asset = env.scene[asset_cfg.name]

    if isinstance(env_ids, slice):
        env_ids_t = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids_t = env_ids.to(device=env.device, dtype=torch.long)

    root_state = asset.data.default_root_state[env_ids_t].clone()  # (N, 13)
    num = env_ids_t.shape[0]

    # Translation offsets
    offsets = torch.zeros((num, 3), device=env.device)
    x_range = pose_range.get("x", (0.0, 0.0))
    y_range = pose_range.get("y", (0.0, 0.0))
    z_range = pose_range.get("z", (0.0, 0.0))
    offsets[:, 0].uniform_(x_range[0], x_range[1])
    offsets[:, 1].uniform_(y_range[0], y_range[1])
    offsets[:, 2].uniform_(z_range[0], z_range[1])
    root_state[:, 0:3] = root_state[:, 0:3] + env.scene.env_origins[env_ids_t] + offsets

    # Orientation offsets (roll/pitch/yaw)
    roll_range = pose_range.get("roll", (0.0, 0.0))
    pitch_range = pose_range.get("pitch", (0.0, 0.0))
    yaw_range = pose_range.get("yaw", (0.0, 0.0))
    euler = torch.zeros((num, 3), device=env.device)
    euler[:, 0].uniform_(roll_range[0], roll_range[1])
    euler[:, 1].uniform_(pitch_range[0], pitch_range[1])
    euler[:, 2].uniform_(yaw_range[0], yaw_range[1])
    delta_quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
    root_state[:, 3:7] = quat_mul(root_state[:, 3:7], delta_quat)

    # Write pose only (no velocity)
    asset.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids_t)

