from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.action_utils import absolute_rot_to_relative, convert_q
from utils.hand_model import create_hand_model


@dataclass
class DexGraspMeta:
    target_q: torch.Tensor
    object_name: str
    robot_name: str


def _load_dro_create_hand_model(dro_root: str):
    """
    Load DRO-Grasp create_hand_model() directly from
    <dro_root>/utils/hand_model.py without permanently changing main imports.
    """
    dro_root = os.path.abspath(os.path.expanduser(str(dro_root)))
    dro_hand_model_path = os.path.join(dro_root, "utils", "hand_model.py")
    if not os.path.exists(dro_hand_model_path):
        raise FileNotFoundError(f"DRO hand_model.py not found: {dro_hand_model_path}")

    backup_modules = {
        name: sys.modules.get(name)
        for name in ("utils", "utils.func_utils", "utils.mesh_utils", "utils.rotation")
    }
    old_sys_path = list(sys.path)
    inserted_path = False
    try:
        # Ensure DRO-Grasp's namespace package `utils` wins during import time.
        cleaned = []
        root_abs = os.path.abspath(ROOT_DIR)
        cwd_abs = os.path.abspath(os.getcwd())
        for p in old_sys_path:
            if p == "":
                continue
            p_abs = os.path.abspath(p)
            if p_abs in {root_abs, cwd_abs}:
                continue
            cleaned.append(p)
        sys.path = [dro_root] + cleaned
        inserted_path = True
        for name in ("utils", "utils.func_utils", "utils.mesh_utils", "utils.rotation"):
            sys.modules.pop(name, None)

        spec = importlib.util.spec_from_file_location("_dexgrasp_dro_hand_model", dro_hand_model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load spec for DRO hand_model: {dro_hand_model_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, mod in backup_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        if inserted_path:
            sys.path = old_sys_path

    if not hasattr(module, "create_hand_model"):
        raise RuntimeError(f"DRO hand_model missing create_hand_model(): {dro_hand_model_path}")
    return module.create_hand_model


def _stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def _load_object_pc(dro_root: str, object_name: str) -> torch.Tensor:
    dataset_type, obj = object_name.split("+")
    pc_path = os.path.join(dro_root, "data", "PointCloud", "object", dataset_type, f"{obj}.pt")
    if not os.path.exists(pc_path):
        raise FileNotFoundError(f"Object point cloud not found: {pc_path}")
    pc = torch.load(pc_path, map_location="cpu")
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise RuntimeError(f"Invalid object point cloud shape at {pc_path}: {tuple(pc.shape)}")
    return pc[:, :3].to(torch.float32)


def _resolve_cache_path(cache_path: str, num_points: int, seed: int) -> str:
    if cache_path.endswith(".pt"):
        return cache_path
    return os.path.join(cache_path, f"dexgrasp_pc_indices_{num_points}_{seed}.pt")


def _build_or_load_pc_index_cache(dro_root: str, object_names: list[str], num_points: int, seed: int, cache_path: str) -> dict[str, torch.Tensor]:
    cache_file = _resolve_cache_path(cache_path, num_points, seed)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    if os.path.exists(cache_file):
        data = torch.load(cache_file, map_location="cpu")
        if data.get("num_points") == num_points and data.get("seed") == seed:
            index_map = data.get("index_map", {})
            if all(name in index_map for name in object_names):
                return {k: v.to(torch.long) for k, v in index_map.items()}

    index_map: dict[str, torch.Tensor] = {}
    for object_name in sorted(object_names):
        obj_pc = _load_object_pc(dro_root, object_name)
        if obj_pc.shape[0] < num_points:
            raise RuntimeError(
                f"Object {object_name} has {obj_pc.shape[0]} points < required {num_points}."
            )
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + _stable_hash_int(object_name))
        idx = torch.randperm(obj_pc.shape[0], generator=gen)[:num_points].to(torch.long)
        index_map[object_name] = idx

    torch.save(
        {
            "seed": seed,
            "num_points": num_points,
            "index_map": index_map,
        },
        cache_file,
    )
    return index_map


def _fps_indices(pc: torch.Tensor, k: int) -> torch.Tensor:
    """Greedy farthest-point sampling. pc: (N, 3). Returns indices (k,)."""
    N = pc.shape[0]
    if k >= N:
        return torch.arange(N, dtype=torch.long)
    dists = torch.full((N,), float("inf"), dtype=pc.dtype)
    sel = torch.zeros(k, dtype=torch.long)
    cur = 0
    for i in range(k):
        sel[i] = cur
        d = ((pc - pc[cur]) ** 2).sum(dim=1)
        dists = torch.minimum(dists, d)
        cur = int(dists.argmax().item())
    return sel


def _extract_metadata_items(metadata_item) -> DexGraspMeta:
    # Expected tuple in DRO: (target_q, object_name, robot_name)
    if not isinstance(metadata_item, (list, tuple)) or len(metadata_item) < 3:
        raise RuntimeError(f"Unexpected metadata item format: {type(metadata_item)}")
    target_q, object_name, robot_name = metadata_item[0], metadata_item[1], metadata_item[2]

    if isinstance(target_q, np.ndarray):
        target_q = torch.from_numpy(target_q)
    elif not isinstance(target_q, torch.Tensor):
        target_q = torch.as_tensor(target_q)

    return DexGraspMeta(
        target_q=target_q.to(torch.float32),
        object_name=str(object_name),
        robot_name=str(robot_name),
    )


class DexGraspDataset(Dataset):
    def __init__(self, cfg, is_train: bool):
        self.cfg = cfg
        self.is_train = is_train

        self.dro_root = cfg.dro_root
        self.num_points = int(cfg.pc_sampling.num_points)
        self.pc_sampling_mode = str(getattr(cfg.pc_sampling, "mode", "cached_indices"))
        self.pc_randomize_train_only = bool(getattr(cfg.pc_sampling, "randomize_train_only", True))
        self.use_random_sampling = self.pc_sampling_mode == "random_per_call" and (
            self.is_train or not self.pc_randomize_train_only
        )
        if self.pc_sampling_mode not in {"cached_indices", "random_per_call"}:
            raise ValueError(
                f"Unsupported pc_sampling.mode={self.pc_sampling_mode}. "
                "Use 'cached_indices' or 'random_per_call'."
            )
        self.obs_horizon = int(cfg.obs_horizon)
        self.pred_horizon = int(cfg.pred_horizon)
        self.action_type = str(cfg.action_type)
        self.action_mode = str(cfg.action_mode)
        self.env_act_dim = int(cfg.env_act_dim)
        self.num_joints = int(cfg.num_joints)
        self.obs_dim = int(cfg.obs_dim)
        self.obs_encoder = str(cfg.observation.encoder)
        self.hand_points_source = str(getattr(cfg.observation, "hand_points_source", "tip"))
        self.hand_num_points = int(getattr(cfg.observation, "hand_num_points", 512))
        self.use_contact_heatmap = bool(getattr(cfg.observation, "use_contact_heatmap", False))
        self.object_encoder = str(getattr(cfg.observation, "object_encoder", "stats"))
        self.fps_k = int(getattr(cfg.observation, "fps_k", 40))
        self.topk_k = int(getattr(cfg.observation, "topk_k", 20))
        self.pointnet_out_dim = int(getattr(cfg.observation, "pointnet_out_dim", 64))
        self.interp_mode = str(cfg.trajectory.interp)
        self.robot_names = list(cfg.robot_names)
        self.q_dof_mismatch = str(getattr(cfg, "q_dof_mismatch", "tail"))
        self._warned_q_mismatch = False
        self._canonical_full_hand_points: dict[str, torch.Tensor] = {}
        aug_cfg = getattr(cfg.pc_sampling, "augmentation", None)
        self.enable_pc_aug = bool(getattr(aug_cfg, "enable_train", False)) and self.is_train
        self.pc_aug_jitter_std = float(getattr(aug_cfg, "jitter_std", 0.0))
        self.pc_aug_jitter_clip = float(getattr(aug_cfg, "jitter_clip", 0.0))
        self.pc_aug_dropout = float(getattr(aug_cfg, "dropout_ratio", 0.0))
        self.pc_aug_rotate_rad = float(getattr(aug_cfg, "rotate_deg", 0.0)) * (math.pi / 180.0)

        if self.interp_mode not in {"static_pose", "joint_target", "linear"}:
            raise ValueError(f"Unsupported trajectory interp: {self.interp_mode}")

        split_path = os.path.join(self.dro_root, "data", "CMapDataset_filtered", "split_train_validate_objects.json")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")
        split_data = json.load(open(split_path, "r"))
        object_names = split_data["train"] if is_train else split_data["validate"]
        self.object_names = set(object_names)

        dataset_path = os.path.join(self.dro_root, "data", "CMapDataset_filtered", "cmap_dataset.pt")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        raw = torch.load(dataset_path, map_location="cpu")
        metadata = raw.get("metadata", None)
        if metadata is None:
            raise RuntimeError("cmap_dataset.pt missing `metadata`")

        self.samples: list[DexGraspMeta] = []
        self._sample_raw_indices: list[int] = []
        for raw_idx, m in enumerate(metadata):
            item = _extract_metadata_items(m)
            if item.object_name not in self.object_names:
                continue
            if item.robot_name not in self.robot_names:
                continue
            self.samples.append(item)
            self._sample_raw_indices.append(raw_idx)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for split={'train' if is_train else 'validate'} and robot_names={self.robot_names}"
            )

        unique_object_names = sorted({s.object_name for s in self.samples})
        self.object_pc_full = {
            obj_name: _load_object_pc(self.dro_root, obj_name).to(torch.float32)
            for obj_name in unique_object_names
        }
        if self.use_random_sampling:
            self.index_cache: dict[str, torch.Tensor] = {}
        else:
            self.index_cache = _build_or_load_pc_index_cache(
                dro_root=self.dro_root,
                object_names=unique_object_names,
                num_points=self.num_points,
                seed=int(cfg.pc_sampling.seed),
                cache_path=str(cfg.pc_sampling.cache_path),
            )

        # Fix C+D: load heatmap and 2048-pt object PCs, precompute FPS indices per object
        self.heatmaps: list[torch.Tensor] = []
        self.object_pc_2048: dict[str, torch.Tensor] = {}
        self.fps_cache: dict[str, torch.Tensor] = {}
        _need_heatmap = self.use_contact_heatmap or self.object_encoder in ("fps", "topk_contact", "pointnet")
        if _need_heatmap:
            heatmap_path = os.path.join(self.dro_root, "data", "CMapDataset_filtered", "cmap_dataset_heatmap.pt")
            if not os.path.exists(heatmap_path):
                raise FileNotFoundError(f"Heatmap dataset not found: {heatmap_path}")
            hm_raw = torch.load(heatmap_path, map_location="cpu")
            hm_metadata = hm_raw["metadata"]
            self.heatmaps = [hm_metadata[i][0].to(torch.float32) for i in self._sample_raw_indices]

            opc_path = os.path.join(self.dro_root, "data", "CMapDataset_filtered", "object_point_clouds.pt")
            if not os.path.exists(opc_path):
                raise FileNotFoundError(f"Object point cloud (2048-pt) not found: {opc_path}")
            self.object_pc_2048 = torch.load(opc_path, map_location="cpu")

            if self.object_encoder == "fps":
                unique_objects = sorted({s.object_name for s in self.samples})
                for obj_name in unique_objects:
                    pc = self.object_pc_2048[obj_name].to(torch.float32)
                    self.fps_cache[obj_name] = _fps_indices(pc, self.fps_k)

        self.hand_models = {name: create_hand_model(name, device="cpu") for name in sorted(set(self.robot_names))}
        self._dro_create_hand_model = _load_dro_create_hand_model(self.dro_root)
        self.dro_hand_models = {
            name: self._dro_create_hand_model(name, device=torch.device("cpu"))
            for name in sorted(set(self.robot_names))
        }
        if self.hand_points_source == "full_hand":
            for name, hand_model in self.hand_models.items():
                q_init_dro = self.dro_hand_models[name].get_canonical_q().to(torch.float32)
                q_init = self._adapt_target_q_dof(
                    q_init_dro,
                    dof=len(hand_model.joint_orders),
                    robot_name=name,
                    object_name="__canonical__",
                )
                mesh = hand_model.get_trimesh_q(q_init)["visual"]
                pts_np = mesh.sample(self.hand_num_points)
                self._canonical_full_hand_points[name] = torch.from_numpy(pts_np).to(torch.float32)
        elif self.hand_points_source != "tip":
            raise ValueError(
                f"Unsupported observation.hand_points_source={self.hand_points_source}. Use 'tip' or 'full_hand'."
            )

    def __len__(self):
        return len(self.samples)

    def _sample_object_pc(self, object_name: str) -> torch.Tensor:
        pc_full = self.object_pc_full[object_name]
        n = int(pc_full.shape[0])
        if self.use_random_sampling:
            if n >= self.num_points:
                idx = torch.randperm(n, device=pc_full.device)[: self.num_points]
            else:
                idx = torch.randint(0, n, (self.num_points,), device=pc_full.device)
        else:
            idx = self.index_cache[object_name]
        pts = pc_full[idx].to(torch.float32).clone()
        return self._augment_points(pts)

    def _random_small_rotation(self, pts: torch.Tensor) -> torch.Tensor:
        if self.pc_aug_rotate_rad <= 0.0:
            return pts
        axis = torch.randn((3,), dtype=pts.dtype, device=pts.device)
        axis = axis / axis.norm().clamp_min(1e-8)
        theta = (torch.rand((), dtype=pts.dtype, device=pts.device) * 2.0 - 1.0) * self.pc_aug_rotate_rad
        kx, ky, kz = axis.unbind()
        K = torch.tensor(
            [
                [0.0, -kz, ky],
                [kz, 0.0, -kx],
                [-ky, kx, 0.0],
            ],
            dtype=pts.dtype,
            device=pts.device,
        )
        I = torch.eye(3, dtype=pts.dtype, device=pts.device)
        R = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)
        center = pts.mean(dim=0, keepdim=True)
        return (pts - center) @ R.T + center

    def _apply_point_dropout(self, pts: torch.Tensor) -> torch.Tensor:
        if self.pc_aug_dropout <= 0.0 or pts.shape[0] < 4:
            return pts
        keep_mask = torch.rand((pts.shape[0],), device=pts.device) > self.pc_aug_dropout
        keep_count = int(keep_mask.sum().item())
        if keep_count < 2:
            keep_idx = torch.randperm(pts.shape[0], device=pts.device)[:2]
            keep_mask = torch.zeros((pts.shape[0],), dtype=torch.bool, device=pts.device)
            keep_mask[keep_idx] = True
            keep_count = 2
        kept = pts[keep_mask]
        resample_idx = torch.randint(0, keep_count, (pts.shape[0],), device=pts.device)
        return kept[resample_idx]

    def _apply_jitter(self, pts: torch.Tensor) -> torch.Tensor:
        if self.pc_aug_jitter_std <= 0.0:
            return pts
        noise = torch.randn_like(pts) * self.pc_aug_jitter_std
        if self.pc_aug_jitter_clip > 0.0:
            noise = noise.clamp(-self.pc_aug_jitter_clip, self.pc_aug_jitter_clip)
        return pts + noise

    def _augment_points(self, pts: torch.Tensor) -> torch.Tensor:
        if not self.enable_pc_aug:
            return pts
        out = self._random_small_rotation(pts)
        out = self._apply_point_dropout(out)
        out = self._apply_jitter(out)
        return out

    def _augment_points_with_heatmap(self, pc: torch.Tensor, heatmap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enable_pc_aug:
            return pc, heatmap
        pts = self._random_small_rotation(pc.clone())
        h = heatmap.clone()
        if self.pc_aug_dropout > 0.0 and pts.shape[0] >= 4:
            keep_mask = torch.rand((pts.shape[0],), device=pts.device) > self.pc_aug_dropout
            keep_count = int(keep_mask.sum().item())
            if keep_count < 2:
                keep_idx = torch.randperm(pts.shape[0], device=pts.device)[:2]
                keep_mask = torch.zeros((pts.shape[0],), dtype=torch.bool, device=pts.device)
                keep_mask[keep_idx] = True
                keep_count = 2
            pts_kept = pts[keep_mask]
            h_kept = h[keep_mask]
            resample_idx = torch.randint(0, keep_count, (pts.shape[0],), device=pts.device)
            pts = pts_kept[resample_idx]
            h = h_kept[resample_idx]
        pts = self._apply_jitter(pts)
        return pts, h

    def _encode_stats(self, robot_pts: torch.Tensor, object_pts: torch.Tensor) -> torch.Tensor:
        def _stats(x: torch.Tensor) -> torch.Tensor:
            x_mean = x.mean(dim=0)
            x_std = x.std(dim=0).clamp_min(1e-6)
            x_min = x.min(dim=0).values
            x_max = x.max(dim=0).values
            return torch.cat([x_mean, x_std, x_min, x_max], dim=0)

        robot_stats = _stats(robot_pts)
        object_stats = _stats(object_pts)
        centroid_delta = robot_pts.mean(dim=0) - object_pts.mean(dim=0)
        feat = torch.cat([robot_stats, object_stats, centroid_delta], dim=0)

        if feat.numel() >= self.obs_dim:
            return feat[: self.obs_dim]
        pad = torch.zeros(self.obs_dim - feat.numel(), dtype=feat.dtype)
        return torch.cat([feat, pad], dim=0)

    def _encode_pointnet_global(self, robot_pts: torch.Tensor, object_pts: torch.Tensor) -> torch.Tensor:
        def _point_feat(x: torch.Tensor) -> torch.Tensor:
            x2 = x * x
            xy = torch.stack([x[:, 0] * x[:, 1], x[:, 1] * x[:, 2], x[:, 2] * x[:, 0]], dim=1)
            ones = torch.ones((x.shape[0], 1), dtype=x.dtype)
            per_point = torch.cat([x, x2, xy, ones], dim=1)
            return torch.cat([per_point.mean(dim=0), per_point.max(dim=0).values], dim=0)

        feat = torch.cat([_point_feat(robot_pts), _point_feat(object_pts)], dim=0)
        if feat.numel() >= self.obs_dim:
            return feat[: self.obs_dim]
        pad = torch.zeros(self.obs_dim - feat.numel(), dtype=feat.dtype)
        return torch.cat([feat, pad], dim=0)

    def _encode_fps_heatmap(self, object_name: str, heatmap: torch.Tensor) -> torch.Tensor:
        """FPS object encoding + heatmap-weighted centroid. Returns (fps_k*3 + 3,) tensor."""
        pc_2048 = self.object_pc_2048[object_name].to(torch.float32)  # (2048, 3)
        fps_idx = self.fps_cache[object_name]
        fps_feat = pc_2048[fps_idx].reshape(-1)  # (fps_k*3,)
        h = heatmap.to(torch.float32).reshape(-1)  # (2048,)
        h_norm = h / h.sum().clamp_min(1e-6)
        heatmap_centroid = (h_norm.unsqueeze(1) * pc_2048).sum(0)  # (3,)
        return torch.cat([fps_feat, heatmap_centroid], dim=0)  # (fps_k*3 + 3,)

    def _encode_topk_contact(self, object_name: str, heatmap: torch.Tensor) -> torch.Tensor:
        """Top-k heatmap-weighted contact points. Returns (topk_k*3,) tensor."""
        pc_2048 = self.object_pc_2048[object_name].to(torch.float32)  # (2048, 3)
        h = heatmap.to(torch.float32).reshape(-1)  # (2048,)
        topk_idx = h.topk(self.topk_k).indices  # (topk_k,)
        return pc_2048[topk_idx].reshape(-1)  # (topk_k*3,)

    def _build_aux_obs(self, robot_pts: torch.Tensor, object_pts: torch.Tensor) -> torch.Tensor:
        """robot_stats(12) + object_stats(12) + centroid_delta(3) = 27 dims, no padding."""
        def _stats(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x.mean(0), x.std(0).clamp_min(1e-6), x.min(0).values, x.max(0).values])
        return torch.cat([_stats(robot_pts), _stats(object_pts), robot_pts.mean(0) - object_pts.mean(0)])

    def _build_obs(self, hand_model, q_init: torch.Tensor, object_pts: torch.Tensor,
                   heatmap: torch.Tensor | None = None, object_name: str = "") -> torch.Tensor:
        if self.hand_points_source == "tip":
            robot_pts = hand_model.compute_tip_positions(q_init.unsqueeze(0))[0].to(torch.float32)
        elif self.hand_points_source == "full_hand":
            robot_pts = self._canonical_full_hand_points[hand_model.robot_name]
        else:
            raise ValueError(
                f"Unsupported observation.hand_points_source={self.hand_points_source}. Use 'tip' or 'full_hand'."
            )
        if self.object_encoder == "pointnet" and heatmap is not None:
            # PointNet path: return zeros placeholder; pl_module applies ContactPointNet at batch level
            obs_vec = torch.zeros(self.obs_dim, dtype=torch.float32)
        elif self.object_encoder in ("fps", "topk_contact") and heatmap is not None:
            def _stats(x: torch.Tensor) -> torch.Tensor:
                return torch.cat([x.mean(0), x.std(0).clamp_min(1e-6), x.min(0).values, x.max(0).values])
            robot_stats = _stats(robot_pts)    # 12
            object_stats = _stats(object_pts)  # 12
            centroid_delta = robot_pts.mean(0) - object_pts.mean(0)  # 3
            if self.object_encoder == "fps":
                # fps_object(fps_k*3) + heatmap_centroid(3) — unique: 2.2%
                unique_feat = self._encode_fps_heatmap(object_name, heatmap)  # fps_k*3 + 3
                obs_vec = torch.cat([robot_stats, unique_feat, centroid_delta], dim=0)
            else:
                # topk_contact(topk_k*3) — unique: 69% of 87 dims
                topk_feat = self._encode_topk_contact(object_name, heatmap)  # topk_k*3 = 60
                obs_vec = torch.cat([robot_stats, object_stats, topk_feat, centroid_delta], dim=0)
        elif self.obs_encoder == "stats":
            obs_vec = self._encode_stats(robot_pts, object_pts)
        elif self.obs_encoder == "pointnet_global":
            obs_vec = self._encode_pointnet_global(robot_pts, object_pts)
        else:
            raise ValueError(f"Unsupported observation encoder: {self.obs_encoder}")
        return obs_vec.unsqueeze(0).repeat(self.obs_horizon, 1)

    def _adapt_target_q_dof(self, target_q: torch.Tensor, dof: int, robot_name: str, object_name: str) -> torch.Tensor:
        q_numel = int(target_q.numel())
        if q_numel == dof:
            return target_q

        mode = self.q_dof_mismatch
        if q_numel > dof and mode in {"tail", "head"}:
            if mode == "tail":
                out = target_q[-dof:]
            else:
                out = target_q[:dof]
            if not self._warned_q_mismatch:
                print(
                    f"[DexGraspDataset] q dof mismatch detected (got={q_numel}, expected={dof}). "
                    f"Applying q_dof_mismatch='{mode}' (robot={robot_name}, object={object_name})."
                )
                self._warned_q_mismatch = True
            return out

        raise RuntimeError(
            f"target_q dof mismatch: got {q_numel}, expected {dof}. "
            f"Set dataset.q_dof_mismatch to 'tail' or 'head' if this is expected."
        )

    def _build_action_seq(self, hand_model, target_q: torch.Tensor, robot_name: str, object_name: str) -> torch.Tensor:
        target_q_full = target_q
        dof = len(hand_model.joint_orders)
        if self.env_act_dim > 0:
            expected_numel = int(self.env_act_dim + dof)
            if int(target_q.numel()) != expected_numel:
                raise RuntimeError(
                    f"target_q size mismatch: expected {expected_numel} (= env_act_dim {self.env_act_dim} + dof {dof}), "
                    f"got {target_q.numel()}."
                )
            root_target_q = target_q[: self.env_act_dim]
            joint_target_q_src = target_q[self.env_act_dim :]
        else:
            root_target_q = torch.zeros((0,), dtype=torch.float32)
            joint_target_q_src = target_q

        target_q = self._adapt_target_q_dof(
            joint_target_q_src,
            dof=dof,
            robot_name=robot_name,
            object_name=object_name,
        )

        # Build joint trajectory according to interpolation mode.
        if self.interp_mode == "static_pose":
            q_seq = target_q.unsqueeze(0).repeat(self.pred_horizon, 1)
        elif self.interp_mode == "joint_target":
            # Keep a full-horizon sequence; current dataset provides a terminal target.
            q_seq = target_q.unsqueeze(0).repeat(self.pred_horizon, 1)
        elif self.interp_mode == "linear":
            q_init_joint = self._build_init_q_joint(
                hand_model=hand_model,
                robot_name=robot_name,
                object_name=object_name,
                target_q_raw=target_q_full,
            )
            alpha = torch.linspace(0.0, 1.0, self.pred_horizon, dtype=torch.float32).unsqueeze(1)
            q_seq = (1.0 - alpha) * q_init_joint.unsqueeze(0) + alpha * target_q.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported trajectory interp: {self.interp_mode}")

        if self.action_type == "joint_value":
            rot_seq = q_seq
        else:
            rot_seq = convert_q(hand_model, q_seq, output_q_type=self.action_type)

        if self.action_mode == "relative" and self.action_type != "joint_value":
            rot_seq = absolute_rot_to_relative(rot_seq, self.action_type, J=self.num_joints)

        if self.env_act_dim > 0:
            if self.interp_mode == "linear":
                alpha = torch.linspace(0.0, 1.0, self.pred_horizon, dtype=torch.float32).unsqueeze(1)
                root_init_q = torch.zeros((1, self.env_act_dim), dtype=torch.float32)
                root_seq = (1.0 - alpha) * root_init_q + alpha * root_target_q.unsqueeze(0)
            else:
                root_seq = root_target_q.unsqueeze(0).repeat(self.pred_horizon, 1)
            return torch.cat([root_seq.to(torch.float32), rot_seq.to(torch.float32)], dim=1)
        return rot_seq.to(torch.float32)

    def _build_init_q_joint(
        self,
        hand_model,
        robot_name: str,
        object_name: str,
        target_q_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Initialize q by directly using DRO-Grasp hand_model.py.
        Returns joint-only q with local hand_model dof.
        """
        dro_hand = self.dro_hand_models[robot_name]

        if target_q_raw is None:
            q_init_dro = dro_hand.get_canonical_q().to(torch.float32)
        else:
            q_in = target_q_raw.to(torch.float32)
            if int(q_in.numel()) != int(dro_hand.dof):
                # If local target is joint-only, prepend zero root (env_act_dim) for DRO init.
                if int(q_in.numel()) + int(self.env_act_dim) == int(dro_hand.dof):
                    q_in = torch.cat(
                        [torch.zeros((self.env_act_dim,), dtype=torch.float32), q_in],
                        dim=0,
                    )
                elif int(q_in.numel()) > int(dro_hand.dof):
                    q_in = q_in[-int(dro_hand.dof):]
                else:
                    q_in = dro_hand.get_canonical_q().to(torch.float32)
            q_init_dro = dro_hand.get_initial_q(q_in).to(torch.float32)

        dof = len(hand_model.joint_orders)
        q_init_joint = self._adapt_target_q_dof(
            q_init_dro,
            dof=dof,
            robot_name=robot_name,
            object_name=object_name,
        )
        if int(q_init_joint.numel()) != dof:
            raise RuntimeError(
                f"Init q dof mismatch after DRO init: expected {dof}, got {int(q_init_joint.numel())} "
                f"(robot={robot_name}, object={object_name})."
            )
        return q_init_joint

    def __getitem__(self, idx):
        sample = self.samples[idx]
        hand_model = self.hand_models[sample.robot_name]

        object_pc = self._sample_object_pc(sample.object_name)

        target_q = sample.target_q.to(torch.float32)
        q_init = self._build_init_q_joint(
            hand_model=hand_model,
            robot_name=sample.robot_name,
            object_name=sample.object_name,
            target_q_raw=target_q,
        )

        heatmap = self.heatmaps[idx] if self.heatmaps else None
        obs_seq = self._build_obs(hand_model, q_init, object_pc,
                                  heatmap=heatmap, object_name=sample.object_name)
        action_seq = self._build_action_seq(
            hand_model,
            target_q,
            robot_name=sample.robot_name,
            object_name=sample.object_name,
        )

        expected_rot_dim = action_seq.shape[1] - self.env_act_dim
        if expected_rot_dim <= 0:
            raise RuntimeError("Invalid action dimension after assembly")
        if obs_seq.shape != (self.obs_horizon, self.obs_dim):
            raise RuntimeError(f"Unexpected observation shape: {tuple(obs_seq.shape)}")
        if action_seq.shape[0] != self.pred_horizon:
            raise RuntimeError(f"Unexpected action horizon: {tuple(action_seq.shape)}")
        if hasattr(self.cfg, "act_dim") and int(self.cfg.act_dim) != int(action_seq.shape[1]):
            raise RuntimeError(
                f"Action dim mismatch: cfg.act_dim={int(self.cfg.act_dim)}, built={int(action_seq.shape[1])}"
            )

        out = {
            "observations": obs_seq,
            "actions": action_seq,
            "valid_mask": torch.ones((self.pred_horizon,), dtype=torch.float32),
            "meta": {
                "robot_name": sample.robot_name,
                "object_name": sample.object_name,
                "sample_id": idx,
            },
        }

        # PointNet path: also return raw pc+heatmap and pre-computed stats for pl_module
        if self.object_encoder == "pointnet" and heatmap is not None:
            pc_2048 = self.object_pc_2048[sample.object_name].to(torch.float32)  # (2048, 3)
            h = heatmap.to(torch.float32).reshape(-1, 1)  # (2048, 1)
            pc_2048, h = self._augment_points_with_heatmap(pc_2048, h)
            out["pc_heatmap"] = torch.cat([pc_2048, h], dim=1)  # (2048, 4)

            if self.hand_points_source == "tip":
                robot_pts = hand_model.compute_tip_positions(q_init.unsqueeze(0))[0].to(torch.float32)
            else:
                robot_pts = self._canonical_full_hand_points[hand_model.robot_name]
            aux = self._build_aux_obs(robot_pts, object_pc)  # (27,)
            out["aux_obs"] = aux.unsqueeze(0).repeat(self.obs_horizon, 1)  # (T, 27)

        return out


def _collate_meta(batch):
    out = {}
    for key in ["observations", "actions", "valid_mask"]:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    out["meta"] = [item["meta"] for item in batch]
    for key in ["pc_heatmap", "aux_obs"]:
        if key in batch[0]:
            out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out


def create_dataloader(cfg, is_train: bool):
    train_dataset = DexGraspDataset(cfg=cfg, is_train=True)
    val_dataset = DexGraspDataset(cfg=cfg, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        shuffle=True,
        drop_last=bool(cfg.drop_last),
        collate_fn=_collate_meta,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.eval_batch_size),
        num_workers=int(cfg.num_workers),
        shuffle=False,
        drop_last=False,
        collate_fn=_collate_meta,
    )
    return train_loader, val_loader
