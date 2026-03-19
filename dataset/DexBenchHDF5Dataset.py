from __future__ import annotations

import json
import os
from glob import glob
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.dexbench_observation import (
    canonicalize_obs_components,
    compute_obs_stats_from_hdf5,
    concatenate_hdf5_observations,
    derive_obs_stats_path,
    get_obs_component_signature,
    get_obs_dim,
    load_obs_stats,
    normalize_obs_np,
    save_obs_stats,
)
from utils.dexbench_replay_obs import (
    canonicalize_deploy_obs_groups,
    concat_deploy_obs_dict_np,
    derive_replay_obs_stats_path,
    get_deploy_obs_dim,
    get_deploy_obs_signature,
    get_policy_actions_slice,
    load_replay_obs_stats,
    save_replay_obs_stats,
)
from utils.dexbench_rotations import encode_euler_xyz_np, get_rot_repr_dim


@dataclass
class EpisodeRef:
    demo_key: str
    start: int
    end: int


class DexBenchHDF5Dataset(Dataset):
    def __init__(self, cfg, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.hdf5_path = str(cfg.hdf5_path)
        self.obs_horizon = int(cfg.obs_horizon)
        self.pred_horizon = int(cfg.pred_horizon)
        self.action_key = str(cfg.action_key)
        self.rotation_indices = list(cfg.rotation_indices)
        self.action_type = str(cfg.action_type)
        self.split_ratio = float(cfg.split_ratio)
        self.obs_components = canonicalize_obs_components(getattr(cfg, "obs_components", None))
        self.obs_component_signature = get_obs_component_signature(self.obs_components)
        self.normalize_obs = bool(getattr(cfg, "normalize_obs", True))
        self.obs_stats_std_floor = float(getattr(cfg, "obs_stats_std_floor", 1.0e-6))
        self.obs_stats_path = derive_obs_stats_path(
            self.hdf5_path,
            self.obs_components,
            self.split_ratio,
            explicit_path=str(getattr(cfg, "obs_stats_path", "") or ""),
        )

        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(
                f"HDF5 not found: {self.hdf5_path}. "
                f"Build the augmented DexBench file first, e.g. with scripts/rebuild_dexbench_hdf5_obs.py."
            )

        self.obs_mean: np.ndarray | None = None
        self.obs_std: np.ndarray | None = None

        self._load_index()
        self._init_obs_stats()

    def _load_index(self):
        with h5py.File(self.hdf5_path, "r") as f:
            demo_keys = sorted(list(f["data"].keys()), key=lambda x: int(x.split("_")[-1]))

            n_total = len(demo_keys)
            n_train = max(1, int(n_total * self.split_ratio))
            self.train_demo_keys = demo_keys[:n_train]
            self.val_demo_keys = demo_keys[n_train:] if n_train < n_total else demo_keys[-1:]
            self.demo_keys = self.train_demo_keys if self.split == "train" else self.val_demo_keys

            self.refs: list[EpisodeRef] = []
            pad_before = self.obs_horizon - 1
            pad_after = self.pred_horizon - self.obs_horizon
            for key in self.demo_keys:
                T = int(f["data"][key][self.action_key].shape[0])
                for s in range(-pad_before, T - self.pred_horizon + pad_after + 1):
                    self.refs.append(EpisodeRef(demo_key=key, start=s, end=s + self.pred_horizon))

            sample_key = self.demo_keys[0]
            try:
                obs0 = concatenate_hdf5_observations(f["data"][sample_key], self.obs_components)
            except KeyError as exc:
                missing = str(exc)
                sample_grp = f["data"][sample_key]
                available_obs = sorted(list(sample_grp["obs"].keys())) if "obs" in sample_grp else []
                has_states = "states" in sample_grp
                raise KeyError(
                    f"DexBench HDF5 is missing required observation components for signature "
                    f"'{self.obs_component_signature}': {missing}. "
                    f"Detected obs keys={available_obs}, states_present={has_states}. "
                    f"If this file was recorded via IsaacLab recorder (state/action-first export), "
                    f"rebuild with scripts/rebuild_dexbench_hdf5_obs.py "
                    f"(optionally add --validate_states)."
                ) from exc
            act0 = f["data"][sample_key][self.action_key][...].astype(np.float32)

        self.raw_obs_dim = int(obs0.shape[1])
        self.raw_act_dim = int(act0.shape[1])
        self.env_keep_indices = [i for i in range(self.raw_act_dim) if i not in self.rotation_indices]
        self.rot_repr_dim = get_rot_repr_dim(self.action_type)
        self.act_dim = len(self.env_keep_indices) + self.rot_repr_dim

        expected_obs_dim = get_obs_dim(self.obs_components)
        if self.raw_obs_dim != expected_obs_dim:
            raise ValueError(
                f"DexBench observation dim mismatch: raw_obs_dim={self.raw_obs_dim}, expected={expected_obs_dim}, "
                f"signature={self.obs_component_signature}"
            )

    def _init_obs_stats(self) -> None:
        if not self.normalize_obs:
            return
        if os.path.exists(self.obs_stats_path):
            stats = load_obs_stats(self.obs_stats_path, self.obs_components)
        else:
            mean, std, num_samples = compute_obs_stats_from_hdf5(
                self.hdf5_path,
                self.train_demo_keys,
                self.obs_components,
                std_floor=self.obs_stats_std_floor,
            )
            save_obs_stats(self.obs_stats_path, mean, std, self.obs_components, num_samples)
            stats = {"mean": mean, "std": std, "num_samples": num_samples, "signature": self.obs_component_signature}
        self.obs_mean = np.asarray(stats["mean"], dtype=np.float32)
        self.obs_std = np.asarray(stats["std"], dtype=np.float32)

    def __len__(self):
        return len(self.refs)

    def _encode_action(self, act_seq_raw: np.ndarray) -> np.ndarray:
        rot_euler = act_seq_raw[:, self.rotation_indices]
        rot_repr = encode_euler_xyz_np(rot_euler, self.action_type)
        env_part = act_seq_raw[:, self.env_keep_indices]
        return np.concatenate([env_part, rot_repr], axis=-1).astype(np.float32)

    def __getitem__(self, idx):
        ref = self.refs[idx]
        with h5py.File(self.hdf5_path, "r") as f:
            grp = f["data"][ref.demo_key]
            obs_all = concatenate_hdf5_observations(grp, self.obs_components)
            act_raw_all = grp[self.action_key][...].astype(np.float32)

        if self.normalize_obs:
            obs_all = normalize_obs_np(obs_all, self.obs_mean, self.obs_std)

        T = int(act_raw_all.shape[0])
        obs_seq = obs_all[max(0, ref.start): ref.start + self.obs_horizon]
        act_raw_seq = act_raw_all[max(0, ref.start): ref.end]

        pad_before = 0
        pad_after = 0
        if ref.start < 0:
            pad_before = -ref.start
            obs_seq = np.concatenate([np.tile(obs_seq[0:1], (pad_before, 1)), obs_seq], axis=0)
            act_raw_seq = np.concatenate([np.tile(act_raw_seq[0:1], (pad_before, 1)), act_raw_seq], axis=0)
        if ref.end > T:
            pad_after = ref.end - T
            act_raw_seq = np.concatenate([act_raw_seq, np.tile(act_raw_seq[-1:], (pad_after, 1))], axis=0)

        if (
            self.split == "train"
            and self.policy_actions_dropout_prob > 0.0
            and self.policy_actions_slice is not None
        ):
            keep = (np.random.rand(obs_seq.shape[0], 1) >= self.policy_actions_dropout_prob).astype(np.float32)
            obs_seq = obs_seq.copy()
            obs_seq[:, self.policy_actions_slice] *= keep

        act_seq = self._encode_action(act_raw_seq)

        valid_mask = np.ones((self.pred_horizon,), dtype=np.float32)
        if pad_before > 0:
            valid_mask[:pad_before] = 0.0
        if pad_after > 0:
            valid_mask[-pad_after:] = 0.0

        return {
            "observations": torch.from_numpy(obs_seq),
            "actions": torch.from_numpy(act_seq),
            "valid_mask": torch.from_numpy(valid_mask),
            "meta": {
                "demo": ref.demo_key,
                "action_type": self.action_type,
                "obs_component_signature": self.obs_component_signature,
            },
        }


class DexBenchReplayPTDataset(Dataset):
    def __init__(self, cfg, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.replay_obs_dir = str(cfg.replay_obs_dir)
        self.obs_horizon = int(cfg.obs_horizon)
        self.pred_horizon = int(cfg.pred_horizon)
        self.rotation_indices = list(cfg.rotation_indices)
        self.action_type = str(cfg.action_type)
        self.split_ratio = float(cfg.split_ratio)
        self.obs_groups = canonicalize_deploy_obs_groups(getattr(cfg, "obs_groups", None))
        self.policy_actions_scale = float(getattr(cfg, "policy_actions_scale", 1.0))
        self.policy_actions_dropout_prob = float(getattr(cfg, "policy_actions_dropout_prob", 0.0))
        if self.policy_actions_dropout_prob < 0.0 or self.policy_actions_dropout_prob > 1.0:
            raise ValueError(
                f"policy_actions_dropout_prob must be in [0, 1], got {self.policy_actions_dropout_prob}"
            )
        self.policy_actions_slice = get_policy_actions_slice(self.obs_groups)
        self.obs_component_signature = get_deploy_obs_signature(
            self.obs_groups,
            policy_actions_scale=self.policy_actions_scale,
        )
        self.normalize_obs = bool(getattr(cfg, "normalize_obs", True))
        self.obs_stats_std_floor = float(getattr(cfg, "obs_stats_std_floor", 1.0e-6))
        self.obs_stats_path = derive_replay_obs_stats_path(
            self.replay_obs_dir,
            self.obs_groups,
            self.split_ratio,
            policy_actions_scale=self.policy_actions_scale,
            explicit_path=str(getattr(cfg, "obs_stats_path", "") or ""),
        )

        if not os.path.isdir(self.replay_obs_dir):
            raise FileNotFoundError(f"Replay obs dir not found: {self.replay_obs_dir}")

        self.obs_mean: np.ndarray | None = None
        self.obs_std: np.ndarray | None = None

        self._load_index()
        self._init_obs_stats()

    def _list_episode_entries(self) -> list[tuple[str, str]]:
        manifest_path = os.path.join(self.replay_obs_dir, "_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            entries = []
            for item in manifest.get("saved_episodes", []):
                episode_name = str(item.get("episode_name", ""))
                file_name = str(item.get("file", ""))
                if not episode_name or not file_name:
                    continue
                path = os.path.join(self.replay_obs_dir, file_name)
                if os.path.exists(path):
                    entries.append((episode_name, path))
            if entries:
                return sorted(entries, key=lambda x: int(x[0].split("_")[-1]))

        files = sorted(glob(os.path.join(self.replay_obs_dir, "demo_*.pt")), key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]))
        return [(os.path.basename(p).replace(".pt", ""), p) for p in files]

    @staticmethod
    def _to_numpy_obs_sequence(
        obs_steps: list,
        obs_groups: list[str],
        policy_actions_scale: float,
    ) -> np.ndarray:
        rows = []
        for step_obs in obs_steps:
            row = concat_deploy_obs_dict_np(
                step_obs,
                obs_groups,
                policy_actions_scale=policy_actions_scale,
            )
            if row.shape[0] != 1:
                raise ValueError(f"Unexpected deploy obs row shape: {row.shape}")
            rows.append(row[0])
        if not rows:
            return np.zeros((0, get_deploy_obs_dim(obs_groups)), dtype=np.float32)
        return np.stack(rows, axis=0).astype(np.float32)

    def _load_episode_arrays(self, episode_name: str) -> tuple[np.ndarray, np.ndarray]:
        path = self.episode_path_map[episode_name]
        payload = torch.load(path, map_location="cpu")
        obs_steps = payload.get("obs_steps", [])
        actions = payload.get("actions", None)
        if actions is None:
            raise KeyError(f"Replay payload missing 'actions': {path}")
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32)
        act_raw = actions.detach().cpu().numpy().astype(np.float32)
        obs_all = self._to_numpy_obs_sequence(
            obs_steps,
            self.obs_groups,
            policy_actions_scale=self.policy_actions_scale,
        )
        if obs_all.shape[0] != act_raw.shape[0]:
            min_len = min(int(obs_all.shape[0]), int(act_raw.shape[0]))
            if min_len <= 0:
                raise ValueError(
                    f"Replay payload has empty aligned trajectory: obs_len={obs_all.shape[0]}, act_len={act_raw.shape[0]}, path={path}"
                )
            obs_all = obs_all[:min_len]
            act_raw = act_raw[:min_len]
        return obs_all, act_raw

    def _load_index(self):
        entries = self._list_episode_entries()
        if not entries:
            raise RuntimeError(f"No replay episode files found in: {self.replay_obs_dir}")

        self.episode_path_map = {name: path for name, path in entries}
        demo_keys = [name for name, _ in entries]

        n_total = len(demo_keys)
        n_train = max(1, int(n_total * self.split_ratio))
        self.train_demo_keys = demo_keys[:n_train]
        self.val_demo_keys = demo_keys[n_train:] if n_train < n_total else demo_keys[-1:]
        self.demo_keys = self.train_demo_keys if self.split == "train" else self.val_demo_keys

        self.episode_len_map: dict[str, int] = {}
        for key in demo_keys:
            _, act_raw = self._load_episode_arrays(key)
            self.episode_len_map[key] = int(act_raw.shape[0])

        self.refs: list[EpisodeRef] = []
        pad_before = self.obs_horizon - 1
        pad_after = self.pred_horizon - self.obs_horizon
        for key in self.demo_keys:
            T = int(self.episode_len_map[key])
            for s in range(-pad_before, T - self.pred_horizon + pad_after + 1):
                self.refs.append(EpisodeRef(demo_key=key, start=s, end=s + self.pred_horizon))

        sample_key = self.demo_keys[0]
        obs0, act0 = self._load_episode_arrays(sample_key)

        self.raw_obs_dim = int(obs0.shape[1])
        self.raw_act_dim = int(act0.shape[1])
        self.env_keep_indices = [i for i in range(self.raw_act_dim) if i not in self.rotation_indices]
        self.rot_repr_dim = get_rot_repr_dim(self.action_type)
        self.act_dim = len(self.env_keep_indices) + self.rot_repr_dim

        expected_obs_dim = get_deploy_obs_dim(self.obs_groups)
        if self.raw_obs_dim != expected_obs_dim:
            raise ValueError(
                f"Replay obs dim mismatch: raw_obs_dim={self.raw_obs_dim}, expected={expected_obs_dim}, "
                f"signature={self.obs_component_signature}"
            )

    def _init_obs_stats(self) -> None:
        if not self.normalize_obs:
            return
        if os.path.exists(self.obs_stats_path):
            stats = load_replay_obs_stats(
                self.obs_stats_path,
                self.obs_groups,
                policy_actions_scale=self.policy_actions_scale,
            )
        else:
            total_sum = np.zeros((self.raw_obs_dim,), dtype=np.float64)
            total_sq_sum = np.zeros((self.raw_obs_dim,), dtype=np.float64)
            total_count = 0
            for key in self.train_demo_keys:
                obs_all, _ = self._load_episode_arrays(key)
                obs64 = obs_all.astype(np.float64)
                total_sum += obs64.sum(axis=0)
                total_sq_sum += np.square(obs64).sum(axis=0)
                total_count += int(obs64.shape[0])
            if total_count <= 0:
                raise RuntimeError("Cannot compute replay obs stats from empty train split")
            mean = total_sum / float(total_count)
            var = np.maximum(total_sq_sum / float(total_count) - np.square(mean), 0.0)
            std = np.sqrt(var)
            std = np.maximum(std, float(self.obs_stats_std_floor))
            save_replay_obs_stats(
                self.obs_stats_path,
                mean,
                std,
                self.obs_groups,
                total_count,
                policy_actions_scale=self.policy_actions_scale,
            )
            stats = {
                "mean": np.asarray(mean, dtype=np.float32),
                "std": np.asarray(std, dtype=np.float32),
                "signature": self.obs_component_signature,
                "num_samples": int(total_count),
            }
        self.obs_mean = np.asarray(stats["mean"], dtype=np.float32)
        self.obs_std = np.asarray(stats["std"], dtype=np.float32)

    def __len__(self):
        return len(self.refs)

    def _encode_action(self, act_seq_raw: np.ndarray) -> np.ndarray:
        rot_euler = act_seq_raw[:, self.rotation_indices]
        rot_repr = encode_euler_xyz_np(rot_euler, self.action_type)
        env_part = act_seq_raw[:, self.env_keep_indices]
        return np.concatenate([env_part, rot_repr], axis=-1).astype(np.float32)

    def __getitem__(self, idx):
        ref = self.refs[idx]
        obs_all, act_raw_all = self._load_episode_arrays(ref.demo_key)

        if self.normalize_obs:
            obs_all = normalize_obs_np(obs_all, self.obs_mean, self.obs_std)

        T = int(act_raw_all.shape[0])
        obs_seq = obs_all[max(0, ref.start): ref.start + self.obs_horizon]
        act_raw_seq = act_raw_all[max(0, ref.start): ref.end]

        pad_before = 0
        pad_after = 0
        if ref.start < 0:
            pad_before = -ref.start
            obs_seq = np.concatenate([np.tile(obs_seq[0:1], (pad_before, 1)), obs_seq], axis=0)
            act_raw_seq = np.concatenate([np.tile(act_raw_seq[0:1], (pad_before, 1)), act_raw_seq], axis=0)
        if ref.end > T:
            pad_after = ref.end - T
            act_raw_seq = np.concatenate([act_raw_seq, np.tile(act_raw_seq[-1:], (pad_after, 1))], axis=0)

        act_seq = self._encode_action(act_raw_seq)

        valid_mask = np.ones((self.pred_horizon,), dtype=np.float32)
        if pad_before > 0:
            valid_mask[:pad_before] = 0.0
        if pad_after > 0:
            valid_mask[-pad_after:] = 0.0

        return {
            "observations": torch.from_numpy(obs_seq),
            "actions": torch.from_numpy(act_seq),
            "valid_mask": torch.from_numpy(valid_mask),
            "meta": {
                "demo": ref.demo_key,
                "action_type": self.action_type,
                "obs_component_signature": self.obs_component_signature,
                "obs_source": "deploy_dict",
            },
        }


def _collate(batch):
    return {
        "observations": torch.stack([b["observations"] for b in batch], dim=0),
        "actions": torch.stack([b["actions"] for b in batch], dim=0),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch], dim=0),
        "meta": [b["meta"] for b in batch],
    }


def create_dataloader(cfg):
    dataset_format = str(getattr(cfg, "dataset_format", "hdf5")).lower()
    if dataset_format == "replay_pt" or bool(getattr(cfg, "replay_obs_dir", "")):
        ds_cls = DexBenchReplayPTDataset
    else:
        ds_cls = DexBenchHDF5Dataset

    train_ds = ds_cls(cfg, split="train")
    val_ds = ds_cls(cfg, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        shuffle=True,
        drop_last=bool(cfg.drop_last),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.eval_batch_size),
        num_workers=int(cfg.num_workers),
        shuffle=False,
        drop_last=False,
        collate_fn=_collate,
    )
    return train_loader, val_loader, train_ds, val_ds
