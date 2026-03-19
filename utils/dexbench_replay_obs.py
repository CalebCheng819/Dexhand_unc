from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from typing import Iterable

import numpy as np
import torch


# Deploy-time observation groups from env.reset/env.step.
DEPLOY_OBS_GROUP_DIMS = OrderedDict(
    [
        ("policy", 54),
        ("proprio", 149),
    ]
)
DEFAULT_DEPLOY_OBS_GROUPS = list(DEPLOY_OBS_GROUP_DIMS.keys())
POLICY_ACTION_DIM = 28


def canonicalize_deploy_obs_groups(groups: Iterable[str] | None) -> list[str]:
    if groups is None:
        return list(DEFAULT_DEPLOY_OBS_GROUPS)
    out = [str(x) for x in list(groups)]
    unknown = [name for name in out if name not in DEPLOY_OBS_GROUP_DIMS]
    if unknown:
        raise ValueError(f"Unknown deploy obs groups: {unknown}")
    return out


def _canonicalize_policy_actions_scale(policy_actions_scale: float | int | None) -> float:
    scale = 1.0 if policy_actions_scale is None else float(policy_actions_scale)
    if scale < 0.0:
        raise ValueError(f"policy_actions_scale must be >= 0, got {scale}")
    return scale


def get_policy_actions_slice(groups: Iterable[str] | None) -> slice | None:
    group_list = canonicalize_deploy_obs_groups(groups)
    offset = 0
    for name in group_list:
        dim = int(DEPLOY_OBS_GROUP_DIMS[name])
        if name == "policy":
            return slice(offset, offset + POLICY_ACTION_DIM)
        offset += dim
    return None


def get_deploy_obs_dim(groups: Iterable[str] | None) -> int:
    group_list = canonicalize_deploy_obs_groups(groups)
    return int(sum(int(DEPLOY_OBS_GROUP_DIMS[name]) for name in group_list))


def get_deploy_obs_signature(
    groups: Iterable[str] | None,
    policy_actions_scale: float | int | None = 1.0,
) -> str:
    group_list = canonicalize_deploy_obs_groups(groups)
    signature = "|".join(f"{name}:{int(DEPLOY_OBS_GROUP_DIMS[name])}" for name in group_list)
    scale = _canonicalize_policy_actions_scale(policy_actions_scale)
    if scale != 1.0:
        signature = f"{signature}|policy_actions_scale:{scale:.6f}"
    return signature


def _to_2d_np(arr, expected_dim: int, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 1:
        out = out[None, :]
    if out.ndim != 2:
        raise ValueError(f"Deploy obs group '{name}' must be 1D/2D, got shape={out.shape}")
    if out.shape[1] != int(expected_dim):
        raise ValueError(f"Deploy obs group '{name}' dim mismatch: got {out.shape[1]}, expected {expected_dim}")
    return out


def concat_deploy_obs_dict_np(
    obs_dict: dict,
    groups: Iterable[str] | None,
    policy_actions_scale: float | int | None = 1.0,
) -> np.ndarray:
    group_list = canonicalize_deploy_obs_groups(groups)
    scale = _canonicalize_policy_actions_scale(policy_actions_scale)
    arrays = []
    for name in group_list:
        arr = _to_2d_np(obs_dict[name], int(DEPLOY_OBS_GROUP_DIMS[name]), name)
        if name == "policy" and scale != 1.0:
            arr = arr.copy()
            arr[:, :POLICY_ACTION_DIM] *= scale
        arrays.append(arr)
    lengths = [arr.shape[0] for arr in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"Deploy obs group length mismatch: {dict(zip(group_list, lengths))}")
    return np.concatenate(arrays, axis=-1).astype(np.float32)


def concat_deploy_obs_dict_torch(
    obs_dict: dict,
    groups: Iterable[str] | None,
    policy_actions_scale: float | int | None = 1.0,
) -> torch.Tensor:
    group_list = canonicalize_deploy_obs_groups(groups)
    scale = _canonicalize_policy_actions_scale(policy_actions_scale)
    tensors = []
    prefix_shape = None
    for name in group_list:
        x = obs_dict[name]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        expected_dim = int(DEPLOY_OBS_GROUP_DIMS[name])
        if x.shape[-1] != expected_dim:
            raise ValueError(f"Deploy obs group '{name}' dim mismatch: got {x.shape[-1]}, expected {expected_dim}")
        if prefix_shape is None:
            prefix_shape = tuple(x.shape[:-1])
        elif tuple(x.shape[:-1]) != prefix_shape:
            raise ValueError(
                f"Deploy obs group batch mismatch for '{name}': {tuple(x.shape[:-1])} vs {prefix_shape}"
            )
        if name == "policy" and scale != 1.0:
            x = x.clone()
            x[..., :POLICY_ACTION_DIM] = x[..., :POLICY_ACTION_DIM] * float(scale)
        tensors.append(x)
    return torch.cat(tensors, dim=-1)


def derive_replay_obs_stats_path(
    replay_obs_dir: str,
    groups: Iterable[str] | None,
    split_ratio: float,
    policy_actions_scale: float | int | None = 1.0,
    explicit_path: str | None = None,
) -> str:
    if explicit_path:
        return str(explicit_path)
    group_list = canonicalize_deploy_obs_groups(groups)
    scale = _canonicalize_policy_actions_scale(policy_actions_scale)
    # Backward compatibility: scale=1.0 keeps the historical path format.
    if scale == 1.0:
        signature = "|".join(group_list)
    else:
        signature = f"{'|'.join(group_list)}|policy_actions_scale:{scale:.6f}"
    group_hash = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    split_tag = f"{float(split_ratio):.4f}".replace(".", "p")
    return os.path.join(str(replay_obs_dir), f"replay_obs_stats.{group_hash}.train_{split_tag}.npz")


def save_replay_obs_stats(
    path: str,
    mean: np.ndarray,
    std: np.ndarray,
    groups: Iterable[str] | None,
    num_samples: int,
    policy_actions_scale: float | int | None = 1.0,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    group_list = canonicalize_deploy_obs_groups(groups)
    scale = _canonicalize_policy_actions_scale(policy_actions_scale)
    np.savez(
        path,
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(std, dtype=np.float32),
        groups_json=np.asarray(json.dumps(group_list)),
        signature=np.asarray(get_deploy_obs_signature(group_list, policy_actions_scale=scale)),
        policy_actions_scale=np.asarray(scale, dtype=np.float32),
        num_samples=np.asarray(int(num_samples), dtype=np.int64),
    )


def load_replay_obs_stats(
    path: str,
    groups: Iterable[str] | None = None,
    policy_actions_scale: float | int | None = 1.0,
) -> dict[str, np.ndarray | int | str | list[str] | float]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Replay obs stats file not found: {path}")
    data = np.load(path, allow_pickle=False)
    mean = np.asarray(data["mean"], dtype=np.float32)
    std = np.asarray(data["std"], dtype=np.float32)
    stored_groups = json.loads(str(data["groups_json"].item()))
    signature = str(data["signature"].item())
    stored_scale = float(data["policy_actions_scale"].item()) if "policy_actions_scale" in data else 1.0
    expected_scale = _canonicalize_policy_actions_scale(policy_actions_scale)
    expected_groups = canonicalize_deploy_obs_groups(groups) if groups is not None else stored_groups
    expected_signature = get_deploy_obs_signature(expected_groups, policy_actions_scale=expected_scale)
    if signature != expected_signature:
        raise ValueError(f"Replay obs stats signature mismatch: stats={signature}, expected={expected_signature}")
    expected_dim = get_deploy_obs_dim(expected_groups)
    if mean.shape[0] != expected_dim or std.shape[0] != expected_dim:
        raise ValueError(
            f"Replay obs stats dimension mismatch: mean/std={mean.shape[0]}/{std.shape[0]}, expected={expected_dim}"
        )
    return {
        "mean": mean,
        "std": std,
        "groups": stored_groups,
        "signature": signature,
        "policy_actions_scale": stored_scale,
        "num_samples": int(data["num_samples"].item()),
    }
