from __future__ import annotations

from dataclasses import dataclass

from utils.action_utils import ROT_DIMS


@dataclass(frozen=True)
class ActionSchema:
    env_act_dim: int
    num_joints: int
    action_type: str
    rot_dim: int
    rot_act_dim: int
    act_dim: int


def compute_action_schema(env_act_dim: int, num_joints: int, action_type: str) -> ActionSchema:
    if action_type not in ROT_DIMS:
        raise ValueError(f"Unsupported action_type: {action_type}")
    if env_act_dim < 0:
        raise ValueError(f"env_act_dim must be >= 0, got {env_act_dim}")
    if num_joints <= 0:
        raise ValueError(f"num_joints must be > 0, got {num_joints}")

    rot_dim = int(ROT_DIMS[action_type])
    rot_act_dim = int(num_joints * rot_dim)
    act_dim = int(env_act_dim + rot_act_dim)
    return ActionSchema(
        env_act_dim=int(env_act_dim),
        num_joints=int(num_joints),
        action_type=str(action_type),
        rot_dim=rot_dim,
        rot_act_dim=rot_act_dim,
        act_dim=act_dim,
    )
