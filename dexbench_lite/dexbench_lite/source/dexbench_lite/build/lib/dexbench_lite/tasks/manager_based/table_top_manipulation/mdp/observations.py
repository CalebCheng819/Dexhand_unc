# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import (
    quat_apply_inverse,
    subtract_frame_transforms,
)


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_state_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body state (pos, quat, lin vel, ang vel) in the base asset's root frame.

    The state for each body is stacked horizontally as
    ``[position(3), quaternion(4)(wxyz), linvel(3), angvel(3)]`` and then concatenated over bodies.

    Args:
        env: The environment.
        body_asset_cfg: Scene entity for the articulated body whose links are observed.
        base_asset_cfg: Scene entity providing the reference (root) frame.

    Returns:
        Tensor of shape ``(num_envs, num_bodies * 13)`` with per-body states expressed in the base root frame.
    """
    body_asset: Articulation = env.scene[body_asset_cfg.name]
    base_asset: Articulation = env.scene[base_asset_cfg.name]
    # get world pose of bodies
    body_pos_w = body_asset.data.body_pos_w[:, body_asset_cfg.body_ids].view(-1, 3)
    body_quat_w = body_asset.data.body_quat_w[:, body_asset_cfg.body_ids].view(-1, 4)
    body_lin_vel_w = body_asset.data.body_lin_vel_w[:, body_asset_cfg.body_ids].view(-1, 3)
    body_ang_vel_w = body_asset.data.body_ang_vel_w[:, body_asset_cfg.body_ids].view(-1, 3)
    num_bodies = int(body_pos_w.shape[0] / env.num_envs)
    # get world pose of base frame
    root_pos_w = base_asset.data.root_link_pos_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 3)
    root_quat_w = base_asset.data.root_link_quat_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 4)
    # transform from world body pose to local body pose
    body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    body_lin_vel_b = quat_apply_inverse(root_quat_w, body_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(root_quat_w, body_ang_vel_w)
    # concate and return
    out = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=1)
    return out.view(env.num_envs, -1)


def _get_asset_root_state_w(asset: Articulation | RigidObject) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(asset, Articulation):
        return (
            asset.data.root_link_pos_w,
            asset.data.root_link_quat_w,
            asset.data.root_link_lin_vel_w,
            asset.data.root_link_ang_vel_w,
        )
    if isinstance(asset, RigidObject):
        return (
            asset.data.root_pos_w,
            asset.data.root_quat_w,
            asset.data.root_lin_vel_w,
            asset.data.root_ang_vel_w,
        )
    raise TypeError(f"Unsupported asset type for root_state_b: {type(asset)}")


def root_state_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Root state (pos, quat, lin vel, ang vel) in the base asset's root frame."""
    asset = env.scene[asset_cfg.name]
    base_asset = env.scene[base_asset_cfg.name]

    asset_pos_w, asset_quat_w, asset_lin_vel_w, asset_ang_vel_w = _get_asset_root_state_w(asset)
    base_pos_w, base_quat_w, _, _ = _get_asset_root_state_w(base_asset)

    asset_pos_b, asset_quat_b = subtract_frame_transforms(base_pos_w, base_quat_w, asset_pos_w, asset_quat_w)
    asset_lin_vel_b = quat_apply_inverse(base_quat_w, asset_lin_vel_w)
    asset_ang_vel_b = quat_apply_inverse(base_quat_w, asset_ang_vel_w)
    return torch.cat((asset_pos_b, asset_quat_b, asset_lin_vel_b, asset_ang_vel_b), dim=-1)


class object_world_frame_vis(ManagerTermBase):
    """Visualize object and world frames with axis markers."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        frame_scale = cfg.params.get("frame_scale", (0.08, 0.08, 0.08))

        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG

        frame_cfg = FRAME_MARKER_CFG.copy()
        frame_cfg.markers["frame"].scale = frame_scale
        self.object_frame_vis = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/ObjectFrame"))
        self.world_frame_vis = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/WorldFrame"))

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        object_cfg = object_cfg or self.object_cfg
        obj: RigidObject = env.scene[object_cfg.name]

        marker_indices = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self.object_frame_vis.visualize(
            translations=obj.data.root_pos_w,
            orientations=obj.data.root_quat_w,
            marker_indices=marker_indices,
        )

        world_pos = env.scene.env_origins
        world_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device, dtype=world_pos.dtype).unsqueeze(0).repeat(env.num_envs, 1)
        self.world_frame_vis.visualize(
            translations=world_pos,
            orientations=world_quat,
            marker_indices=marker_indices,
        )

        return torch.zeros(env.num_envs, 0, device=env.device)


def fingers_contact_force_b(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """base-frame contact forces from listed sensors, concatenated per env.

    Args:
        env: The environment.
        contact_sensor_names: Names of contact sensors in ``env.scene.sensors`` to read.

    Returns:
        Tensor of shape ``(num_envs, 3 * num_sensors)`` with forces stacked horizontally as
        ``[fx, fy, fz]`` per sensor.
    """
    force_w = [env.scene.sensors[name].data.force_matrix_w.view(env.num_envs, 3) for name in contact_sensor_names]
    force_w = torch.stack(force_w, dim=1)  # Shape: (num_envs, num_sensors, 3)
    robot: Articulation = env.scene[asset_cfg.name]
    forces_b = quat_apply_inverse(robot.data.root_link_quat_w.unsqueeze(1).repeat(1, force_w.shape[1], 1), force_w)
    # Flatten to (num_envs, 3 * num_sensors) for concatenation with other observations
    return forces_b.view(env.num_envs, -1)

