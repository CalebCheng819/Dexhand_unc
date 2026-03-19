# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the dexsuite task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def out_of_bound(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    in_bound_range: dict[str, tuple[float, float]] = {},
) -> torch.Tensor:
    """Termination condition for the object falls out of bound.

    Args:
        env: The environment.
        asset_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        in_bound_range: The range in x, y, z such that the object is considered in range
    """
    object: RigidObject = env.scene[asset_cfg.name]
    range_list = [in_bound_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)

    object_pos_local = object.data.root_pos_w - env.scene.env_origins
    outside_bounds = ((object_pos_local < ranges[:, 0]) | (object_pos_local > ranges[:, 1])).any(dim=1)
    return outside_bounds


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminating environment when violation of velocity limits detects, this usually indicates unstable physics caused
    by very bad, or aggressive action"""
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Compute violation mask: joint_vel.abs() > (joint_vel_limits * 2)
    joint_vel_abs = robot.data.joint_vel.abs()
    joint_vel_limit_threshold = robot.data.joint_vel_limits * 10
    violation_mask = joint_vel_abs > joint_vel_limit_threshold  # [num_envs, num_joints]
    
    # # Check which environments have any violations
    abnormal_envs = violation_mask.any(dim=1)  # [num_envs]
    return abnormal_envs

def object_at_goal_position(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for when the object is close enough to the goal position.
    
    This checks if the object's position is within the specified threshold of the goal position.
    Orientation is ignored - only position matters.
    
    Args:
        env: The environment.
        command_name: The name of the command term that contains the goal position.
        threshold: Distance threshold in meters. Defaults to 0.05.
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").
    
    Returns:
        A boolean tensor indicating which environments have succeeded (object at goal position).
    """
    # Get the command term which has position_error metric
    command_term = env.command_manager.get_term(command_name)
    
    # Check if position error is below threshold
    position_error = command_term.metrics["position_error"]
    success = position_error < threshold
    
    return success



def debug_reset_reasons(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    prefix: str = "Env reset",
):
    """Print reset reasons for each env_id based on active termination terms."""
    if isinstance(env_ids, slice):
        env_ids_t = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids_t = env_ids.to(device=env.device, dtype=torch.long)

    term_names = env.termination_manager.active_terms
    if not term_names:
        for env_id in env_ids_t.tolist():
            print(f"{prefix} [{env_id}]: no termination terms active")
        return

    term_buffers = {name: env.termination_manager.get_term(name) for name in term_names}
    for env_id in env_ids_t.tolist():
        reasons = [name for name in term_names if bool(term_buffers[name][env_id])]
        if reasons:
            print(f"{prefix} [{env_id}]: {', '.join(reasons)}")
        else:
            print(f"{prefix} [{env_id}]: unknown")




