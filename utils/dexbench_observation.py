from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from typing import Iterable

import h5py
import numpy as np
import torch

OBS_COMPONENT_DIMS = OrderedDict(
    [
        ("obs/actions", 28),
        ("obs/object_pos_b", 13),
        ("obs/object_state_robot_b", 13),
        ("obs/joint_pos", 28),
        ("obs/joint_vel", 28),
        ("obs/hand_tips_state_b", 78),
        ("obs/contact", 15),
        ("goal_pose", 7),
    ]
)

DEFAULT_OBS_COMPONENTS = list(OBS_COMPONENT_DIMS.keys())


def canonicalize_obs_components(components: Iterable[str] | None) -> list[str]:
    if components is None:
        return list(DEFAULT_OBS_COMPONENTS)
    out = [str(x) for x in list(components)]
    unknown = [name for name in out if name not in OBS_COMPONENT_DIMS]
    if unknown:
        raise ValueError(f"Unknown DexBench observation components: {unknown}")
    return out


def get_obs_component_dim(name: str) -> int:
    if name not in OBS_COMPONENT_DIMS:
        raise KeyError(f"Unknown DexBench observation component: {name}")
    return int(OBS_COMPONENT_DIMS[name])


def get_obs_dim(components: Iterable[str] | None) -> int:
    return int(sum(get_obs_component_dim(name) for name in canonicalize_obs_components(components)))


def get_obs_component_signature(components: Iterable[str] | None) -> str:
    parts = [f"{name}:{get_obs_component_dim(name)}" for name in canonicalize_obs_components(components)]
    return "|".join(parts)


def get_obs_component_hash(components: Iterable[str] | None) -> str:
    sig = get_obs_component_signature(components)
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:12]


def derive_obs_stats_path(
    hdf5_path: str,
    components: Iterable[str] | None,
    split_ratio: float,
    explicit_path: str | None = None,
) -> str:
    if explicit_path:
        return str(explicit_path)
    root, ext = os.path.splitext(str(hdf5_path))
    if not ext:
        ext = ".hdf5"
    split_tag = f"{float(split_ratio):.4f}".replace(".", "p")
    comp_hash = get_obs_component_hash(components)
    return f"{root}.obs_stats.{comp_hash}.train_{split_tag}.npz"


def _to_2d_np(arr: np.ndarray, component: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Observation component {component} must be 2D after normalization, got shape={arr.shape}")
    if arr.shape[1] != get_obs_component_dim(component):
        raise ValueError(
            f"Observation component {component} has wrong dim: got {arr.shape[1]}, expected {get_obs_component_dim(component)}"
        )
    return arr


def expand_goal_pose_array(goal_pose: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(goal_pose, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(f"goal_pose must have shape (7,) or (N,7), got shape={arr.shape}")
    if arr.shape[0] == 1:
        return np.repeat(arr, length, axis=0)
    if arr.shape[0] == length:
        return arr
    raise ValueError(f"goal_pose length mismatch: got {arr.shape[0]}, expected 1 or {length}")


def load_hdf5_obs_component(grp: h5py.Group, component: str, episode_length: int | None = None) -> np.ndarray:
    if episode_length is None:
        if "processed_actions" in grp:
            episode_length = int(grp["processed_actions"].shape[0])
        elif "actions" in grp:
            episode_length = int(grp["actions"].shape[0])
        else:
            raise KeyError("Cannot infer episode length: expected processed_actions or actions in episode group")

    if component == "goal_pose":
        if "goal_pose" not in grp:
            raise KeyError("Missing goal_pose dataset in episode group")
        return expand_goal_pose_array(grp["goal_pose"][...], episode_length)

    if not component.startswith("obs/"):
        raise KeyError(f"Unsupported DexBench observation component path: {component}")
    key = component.split("/", 1)[1]
    if "obs" not in grp or key not in grp["obs"]:
        raise KeyError(f"Missing observation component '{component}' in HDF5 episode group")
    return _to_2d_np(grp["obs"][key][...], component)


def concatenate_hdf5_observations(grp: h5py.Group, components: Iterable[str] | None) -> np.ndarray:
    comp_list = canonicalize_obs_components(components)
    episode_length = int(grp["processed_actions"].shape[0]) if "processed_actions" in grp else None
    arrays = [load_hdf5_obs_component(grp, name, episode_length=episode_length) for name in comp_list]
    lengths = [arr.shape[0] for arr in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"Observation component length mismatch: {dict(zip(comp_list, lengths))}")
    return np.concatenate(arrays, axis=-1).astype(np.float32)


def concatenate_component_dict_np(component_map: dict[str, np.ndarray], components: Iterable[str] | None) -> np.ndarray:
    comp_list = canonicalize_obs_components(components)
    arrays = [_to_2d_np(component_map[name], name) for name in comp_list]
    lengths = [arr.shape[0] for arr in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"Observation component length mismatch: {dict(zip(comp_list, lengths))}")
    return np.concatenate(arrays, axis=-1).astype(np.float32)


def concatenate_component_dict_torch(component_map: dict[str, torch.Tensor], components: Iterable[str] | None) -> torch.Tensor:
    comp_list = canonicalize_obs_components(components)
    tensors: list[torch.Tensor] = []
    prefix_shape = None
    for name in comp_list:
        x = component_map[name]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != get_obs_component_dim(name):
            raise ValueError(
                f"Observation component {name} has wrong dim: got {x.shape[-1]}, expected {get_obs_component_dim(name)}"
            )
        if prefix_shape is None:
            prefix_shape = tuple(x.shape[:-1])
        elif tuple(x.shape[:-1]) != prefix_shape:
            raise ValueError(f"Observation component batch mismatch for {name}: {tuple(x.shape[:-1])} vs {prefix_shape}")
        tensors.append(x)
    return torch.cat(tensors, dim=-1)


def compute_obs_stats_from_hdf5(
    hdf5_path: str,
    demo_keys: Iterable[str],
    components: Iterable[str] | None,
    std_floor: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, int]:
    comp_list = canonicalize_obs_components(components)
    obs_dim = get_obs_dim(comp_list)
    total_sum = np.zeros(obs_dim, dtype=np.float64)
    total_sq_sum = np.zeros(obs_dim, dtype=np.float64)
    total_count = 0

    with h5py.File(hdf5_path, "r") as f:
        for key in demo_keys:
            obs = concatenate_hdf5_observations(f["data"][key], comp_list).astype(np.float64)
            total_sum += obs.sum(axis=0)
            total_sq_sum += np.square(obs).sum(axis=0)
            total_count += int(obs.shape[0])

    if total_count <= 0:
        raise RuntimeError("Cannot compute observation stats from empty DexBench split")

    mean = total_sum / float(total_count)
    var = np.maximum(total_sq_sum / float(total_count) - np.square(mean), 0.0)
    std = np.sqrt(var)
    std = np.maximum(std, float(std_floor))
    return mean.astype(np.float32), std.astype(np.float32), int(total_count)


def save_obs_stats(
    path: str,
    mean: np.ndarray,
    std: np.ndarray,
    components: Iterable[str] | None,
    num_samples: int,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    comp_list = canonicalize_obs_components(components)
    np.savez(
        path,
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(std, dtype=np.float32),
        components_json=np.asarray(json.dumps(comp_list)),
        signature=np.asarray(get_obs_component_signature(comp_list)),
        num_samples=np.asarray(int(num_samples), dtype=np.int64),
    )


def load_obs_stats(path: str, components: Iterable[str] | None = None) -> dict[str, np.ndarray | int | list[str] | str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Observation stats file not found: {path}")
    data = np.load(path, allow_pickle=False)
    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)
    components_json = str(data["components_json"].item())
    stored_components = json.loads(components_json)
    signature = str(data["signature"].item())
    expected = canonicalize_obs_components(components) if components is not None else stored_components
    expected_signature = get_obs_component_signature(expected)
    if signature != expected_signature:
        raise ValueError(
            f"Observation stats signature mismatch: stats={signature}, expected={expected_signature}"
        )
    if mean.shape[0] != get_obs_dim(expected) or std.shape[0] != get_obs_dim(expected):
        raise ValueError(
            f"Observation stats dimension mismatch: mean/std dim={mean.shape[0]}/{std.shape[0]}, expected={get_obs_dim(expected)}"
        )
    return {
        "mean": mean,
        "std": std,
        "components": stored_components,
        "signature": signature,
        "num_samples": int(data["num_samples"].item()),
    }


def normalize_obs_np(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return ((obs - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)


def normalize_obs_torch(obs: torch.Tensor, mean: np.ndarray | torch.Tensor, std: np.ndarray | torch.Tensor) -> torch.Tensor:
    if not isinstance(mean, torch.Tensor):
        mean = torch.as_tensor(mean, device=obs.device, dtype=obs.dtype)
    else:
        mean = mean.to(device=obs.device, dtype=obs.dtype)
    if not isinstance(std, torch.Tensor):
        std = torch.as_tensor(std, device=obs.device, dtype=obs.dtype)
    else:
        std = std.to(device=obs.device, dtype=obs.dtype)
    view_shape = (1,) * (obs.ndim - 1) + (obs.shape[-1],)
    return (obs - mean.view(view_shape)) / std.view(view_shape)


def make_runtime_obs_component_map(env, dexbench_mdp, scene_entity_cfg_cls) -> dict[str, torch.Tensor]:
    robot_cfg = getattr(env.cfg, "robot_config", None)
    if robot_cfg is None:
        raise AttributeError("DexBench env config missing robot_config; cannot construct runtime observation components.")
    cache = getattr(env, "_dexbench_obs_runtime_cache", None)
    if cache is None:
        fingertip_body_names = list(getattr(robot_cfg, "fingertip_body_names", []))
        hand_tips_body_names = list(getattr(robot_cfg, "hand_tips_body_names", []))
        if not fingertip_body_names or not hand_tips_body_names:
            raise ValueError("DexBench robot_config must define fingertip_body_names and hand_tips_body_names")
        robot = env.scene["robot"]
        hand_tip_body_ids, _ = robot.find_bodies(hand_tips_body_names, preserve_order=True)
        cache = {
            "fingertip_body_names": fingertip_body_names,
            "hand_tip_body_ids": hand_tip_body_ids,
            "contact_sensor_names": [f"{name}_object_s" for name in fingertip_body_names],
        }
        setattr(env, "_dexbench_obs_runtime_cache", cache)

    component_map = {
        "obs/actions": dexbench_mdp.last_action(env),
        "obs/object_pos_b": dexbench_mdp.root_state_b(
            env,
            asset_cfg=scene_entity_cfg_cls("object"),
            base_asset_cfg=scene_entity_cfg_cls("table"),
        ),
        "obs/object_state_robot_b": dexbench_mdp.root_state_b(
            env,
            asset_cfg=scene_entity_cfg_cls("object"),
            base_asset_cfg=scene_entity_cfg_cls("robot"),
        ),
        "obs/joint_pos": dexbench_mdp.joint_pos(env),
        "obs/joint_vel": dexbench_mdp.joint_vel(env),
        "obs/hand_tips_state_b": dexbench_mdp.body_state_b(
            env,
            body_asset_cfg=scene_entity_cfg_cls("robot", body_ids=cache["hand_tip_body_ids"]),
            base_asset_cfg=scene_entity_cfg_cls("robot"),
        ),
        "obs/contact": dexbench_mdp.fingers_contact_force_b(
            env,
            contact_sensor_names=cache["contact_sensor_names"],
        ),
        "goal_pose": env.command_manager.get_command("object_pose"),
    }
    return component_map
