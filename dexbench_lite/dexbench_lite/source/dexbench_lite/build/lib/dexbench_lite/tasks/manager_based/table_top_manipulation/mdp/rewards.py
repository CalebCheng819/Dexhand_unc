# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.envs.mdp.observations import last_action

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(-1000, 1000)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-1000, 1000)

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reaching the object using a tanh-kernel on end-effector distance.

    The reward is close to 1 when the maximum distance between the object and any end-effector body is small.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    object_pos = object.data.root_pos_w
    # object_ee_distance = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    object_ee_distance_sum = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).sum(dim=-1)

    # return 1 - torch.tanh(object_ee_distance / std)
    return torch.exp(-10 * object_ee_distance_sum)


def lift_when_grasping_reward(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.07,
) -> torch.Tensor:
    """Reward lifting the object when grasping."""
    asset: Articulation = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    object_pos = object.data.root_pos_w
    asset_to_obj_distances = torch.norm(asset_pos - object_pos[:, None, :], dim=-1) 
    thumb_to_obj_distance = asset_to_obj_distances[:, 0]
    index_to_obj_distance = asset_to_obj_distances[:, 1]
    middle_to_obj_distance = asset_to_obj_distances[:, 2]
    ring_to_obj_distance = asset_to_obj_distances[:, 3]

    good_contact_cond1 = (thumb_to_obj_distance <= threshold) & (
        (index_to_obj_distance <= threshold) | (middle_to_obj_distance <= threshold) | (ring_to_obj_distance <= threshold)
    )
    latest_action: torch.Tensor = last_action(env) 
    z_lift_action = latest_action[..., 2] 
    # clip to 0 
    z_lift_action = z_lift_action.clamp(0, 1)
    reward = good_contact_cond1 * z_lift_action
    return reward


def contacts_flexible(
    env: ManagerBasedRLEnv,
    threshold: float,
    fingertip_names: list[str] | None = None,
) -> torch.Tensor:
    """Reward when thumb + at least one other finger are in contact with object.
    
    Flexible version that accepts fingertip names as parameters.
    
    Args:
        env: The environment.
        threshold: Contact force threshold in Newtons.
        fingertip_names: List of fingertip body names. Expected order: [thumb, index, middle, ring].
                        Defaults to ["thumb_fingertip", "fingertip", "fingertip2", "fingertip3"].
    
    Returns:
        Tensor of shape (num_envs,) with 1.0 when good contact condition is met, 0.0 otherwise.
    """
    if fingertip_names is None:
        fingertip_names = ["thumb_fingertip", "fingertip", "fingertip2", "fingertip3"]
    
    def _get_sensor(base_name: str) -> ContactSensor:
        sensors = env.scene.sensors
        key = f"{base_name}_object_s"
        if key in sensors:
            return sensors[key]
        tip_key = f"{base_name}_tip_object_s"
        if tip_key in sensors:
            return sensors[tip_key]
        # Fall back: raise clear error listing available sensors
        available = list(sensors.keys())
        raise KeyError(f"Contact sensor for '{base_name}' not found. Tried '{key}' and '{tip_key}'. Available: {available}")
    
    # Get contact sensors for each fingertip (at least thumb, index, middle required)
    thumb_contact_sensor: ContactSensor = _get_sensor(fingertip_names[0])
    index_contact_sensor: ContactSensor = _get_sensor(fingertip_names[1])
    middle_contact_sensor: ContactSensor = _get_sensor(fingertip_names[2])
    
    # Check if contact force is above threshold
    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_contact = middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    
    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)
    middle_contact_mag = torch.norm(middle_contact, dim=-1)
    
    # Good contact: thumb + at least one other finger
    # Include ring finger if available
    if len(fingertip_names) > 3:
        ring_contact_sensor: ContactSensor = _get_sensor(fingertip_names[3])
        ring_contact = ring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
        ring_contact_mag = torch.norm(ring_contact, dim=-1)
        good_contact_cond = (thumb_contact_mag > threshold) & (
            (index_contact_mag > threshold) | (middle_contact_mag > threshold) | (ring_contact_mag > threshold)
        )
    else:
        good_contact_cond = (thumb_contact_mag > threshold) & (
            (index_contact_mag > threshold) | (middle_contact_mag > threshold)
        )
    
    return good_contact_cond.float()



def position_command_error_exp_from_metrics(
    env: ManagerBasedRLEnv, std: float, command_name: str
) -> torch.Tensor:
    """Reward tracking of commanded position using command metrics (world-frame distance)."""
    command_term = env.command_manager.get_term(command_name)
    distance = command_term.metrics["position_error"]
    return torch.exp(-distance / max(std, 1e-6))




def success_reward_from_metrics(
    env: ManagerBasedRLEnv,
    command_name: str,
    pos_std: float,
    rot_std: float | None = None,
) -> torch.Tensor:
    """Reward success using command metrics (world-frame errors)."""
    command_term = env.command_manager.get_term(command_name)
    pos_dist = command_term.metrics["position_error"]
    if not rot_std:
        return (1 - torch.tanh(pos_dist / pos_std)) ** 2
    rot_dist = command_term.metrics["orientation_error"]
    return (1 - torch.tanh(pos_dist / pos_std)) * (1 - torch.tanh(rot_dist / rot_std))


