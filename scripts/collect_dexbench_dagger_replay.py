from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils.dexbench_replay_obs import (
    canonicalize_deploy_obs_groups,
    concat_deploy_obs_dict_torch,
    derive_replay_obs_stats_path,
    load_replay_obs_stats,
)
from utils.dexbench_rotations import decode_to_euler_xyz_np, get_rot_repr_dim


def _parse_args():
    parser = argparse.ArgumentParser(description="Collect DexBench DAgger replay dataset.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to student checkpoint used for rollout.",
    )
    parser.add_argument(
        "--train_cfg_path",
        type=str,
        default="",
        help="Optional explicit train config path. If empty, infer from checkpoint.",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="Expert HDF5 demonstrations file (for labels/initial states).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for replay .pt files and manifest.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="DexbenchLite-Relocate-FloatingShadowRight-v0",
        help="Isaac task name.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--repeat_per_demo",
        type=int,
        default=4,
        help="How many DAgger rollouts to collect per expert demo.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Only num_envs=1 is supported.",
    )
    parser.add_argument(
        "--max_demos",
        type=int,
        default=0,
        help="If > 0, only use first N demos from dataset.",
    )
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=0,
        help="If > 0, truncate rollout at this many steps.",
    )
    parser.add_argument(
        "--exec_horizon",
        type=int,
        default=1,
        help="Number of predicted chunk steps executed before re-planning.",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=10,
        help="DDIM denoising steps used by student policy.",
    )
    parser.add_argument(
        "--action_scale",
        type=float,
        default=1.0,
        help="Global scale applied to student raw actions before env.step.",
    )
    parser.add_argument(
        "--exploration_noise_std",
        type=float,
        default=0.0,
        help="Gaussian exploration noise std added to student raw actions.",
    )
    parser.add_argument(
        "--bootstrap_replay_dir",
        type=str,
        default="",
        help="Optional existing replay dir to copy as seed data before DAgger samples.",
    )
    return parser.parse_args()


def _infer_train_cfg_path(checkpoint_path: str) -> str:
    run_root = os.path.abspath(os.path.join(os.path.dirname(checkpoint_path), ".."))
    return os.path.join(run_root, "log", "hydra", ".hydra", "config.yaml")


def _extract_obs_from_reset(reset_out):
    if isinstance(reset_out, tuple):
        if len(reset_out) >= 1:
            return reset_out[0]
        raise RuntimeError("env.reset() returned empty tuple.")
    return reset_out


def _extract_step(step_out, device: torch.device):
    if not isinstance(step_out, tuple):
        raise RuntimeError(f"Unsupported env.step return type: {type(step_out)}")
    if len(step_out) == 5:
        obs, reward, terminated, truncated, _ = step_out
    elif len(step_out) == 4:
        obs, reward, terminated, _ = step_out
        truncated = torch.zeros_like(torch.as_tensor(terminated), dtype=torch.bool)
    else:
        raise RuntimeError(f"Unsupported env.step output length: {len(step_out)}")

    reward = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
    terminated = torch.as_tensor(terminated, device=device, dtype=torch.bool).view(-1)
    truncated = torch.as_tensor(truncated, device=device, dtype=torch.bool).view(-1)
    done = terminated | truncated
    return obs, reward, terminated, truncated, done


def _slice_first_env(obs_obj):
    if isinstance(obs_obj, torch.Tensor):
        if obs_obj.ndim == 0:
            return obs_obj.detach().cpu()
        return obs_obj[0].detach().cpu()
    if isinstance(obs_obj, np.ndarray):
        if obs_obj.ndim == 0:
            return np.asarray(obs_obj)
        return np.asarray(obs_obj[0])
    if isinstance(obs_obj, dict):
        return {k: _slice_first_env(v) for k, v in obs_obj.items()}
    if isinstance(obs_obj, (list, tuple)):
        return type(obs_obj)(_slice_first_env(v) for v in obs_obj)
    return obs_obj


def _get_obs_after_reset_to(env):
    if hasattr(env, "get_observations"):
        return env.get_observations()
    if hasattr(env, "observation_manager") and hasattr(env.observation_manager, "compute"):
        return env.observation_manager.compute()
    raise RuntimeError("Could not fetch observation after reset_to().")


def _restore_goal_pose_from_episode(env, episode_data) -> None:
    if "goal_pose" not in episode_data.data or not hasattr(env, "command_manager"):
        return
    goal_pose = episode_data.data["goal_pose"]
    goal_pose_tensor = None
    if isinstance(goal_pose, torch.Tensor):
        goal_pose_tensor = goal_pose.to(env.device)
    elif isinstance(goal_pose, list) and len(goal_pose) > 0:
        goal_pose_tensor = goal_pose[0].to(env.device)
    if goal_pose_tensor is None:
        return
    if goal_pose_tensor.dim() == 1:
        goal_pose_tensor = goal_pose_tensor.unsqueeze(0)
    command_term = env.command_manager.get_term("object_pose")
    if hasattr(command_term, "pose_command_b"):
        command_term.pose_command_b[0] = goal_pose_tensor[0]
        if hasattr(command_term, "_update_metrics"):
            command_term._update_metrics()


def _decode_raw_action(compact_action: torch.Tensor, dataset_cfg) -> torch.Tensor:
    raw_act_dim = int(dataset_cfg.raw_act_dim)
    rotation_indices = list(dataset_cfg.rotation_indices)
    env_keep_indices = [i for i in range(raw_act_dim) if i not in rotation_indices]
    env_act_dim = int(dataset_cfg.env_act_dim)
    rot_dim = int(dataset_cfg.rot_dim)
    action_type = str(dataset_cfg.action_type)

    raw_action = torch.zeros((compact_action.shape[0], raw_act_dim), device=compact_action.device, dtype=compact_action.dtype)
    raw_action[:, env_keep_indices] = compact_action[:, :env_act_dim]
    rot_repr = compact_action[:, env_act_dim: env_act_dim + rot_dim]
    euler_np = decode_to_euler_xyz_np(rot_repr.detach().cpu().numpy(), action_type)
    raw_action[:, rotation_indices] = torch.from_numpy(euler_np).to(device=compact_action.device, dtype=compact_action.dtype)
    return raw_action


def _clamp_to_action_space(raw_action: torch.Tensor, action_space) -> torch.Tensor:
    if not hasattr(action_space, "low") or not hasattr(action_space, "high"):
        return raw_action
    low = torch.as_tensor(action_space.low, device=raw_action.device, dtype=raw_action.dtype)
    high = torch.as_tensor(action_space.high, device=raw_action.device, dtype=raw_action.dtype)
    return torch.minimum(torch.maximum(raw_action, low), high)


def _iter_replay_entries(replay_dir: str):
    manifest_path = os.path.join(replay_dir, "_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        entries = []
        for item in manifest.get("saved_episodes", []):
            ep_name = str(item.get("episode_name", ""))
            file_name = str(item.get("file", ""))
            if not ep_name or not file_name:
                continue
            src = os.path.join(replay_dir, file_name)
            if os.path.exists(src):
                entries.append((ep_name, src))
        if entries:
            entries.sort(key=lambda x: int(x[0].split("_")[-1]))
            return entries

    files = []
    for file_name in os.listdir(replay_dir):
        if file_name.startswith("demo_") and file_name.endswith(".pt"):
            files.append(file_name)
    files.sort(key=lambda x: int(x.split("_")[-1].replace(".pt", "")))
    return [(f.replace(".pt", ""), os.path.join(replay_dir, f)) for f in files]


def _copy_bootstrap_replay(bootstrap_dir: str, output_dir: str, start_index: int) -> tuple[int, list[dict[str, Any]]]:
    if not bootstrap_dir:
        return start_index, []
    entries = _iter_replay_entries(bootstrap_dir)
    manifest_entries = []
    next_index = int(start_index)
    for src_name, src_path in entries:
        dst_episode_name = f"demo_{next_index}"
        dst_file = os.path.join(output_dir, f"{dst_episode_name}.pt")
        shutil.copy2(src_path, dst_file)
        manifest_entries.append(
            {
                "episode_index": int(next_index),
                "episode_name": dst_episode_name,
                "num_steps": -1,
                "file": os.path.basename(dst_file),
                "source": "bootstrap_replay",
                "source_episode_name": str(src_name),
            }
        )
        next_index += 1
    return next_index, manifest_entries


def main():
    args = _parse_args()
    if int(args.num_envs) != 1:
        raise ValueError("collect_dexbench_dagger_replay currently supports --num_envs=1 only.")
    if int(args.repeat_per_demo) <= 0:
        raise ValueError("--repeat_per_demo must be > 0.")
    if float(args.action_scale) <= 0.0:
        raise ValueError("--action_scale must be > 0.")
    if float(args.exploration_noise_std) < 0.0:
        raise ValueError("--exploration_noise_std must be >= 0.")

    if not os.path.exists(args.dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_file}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    train_cfg_path = str(args.train_cfg_path) if str(args.train_cfg_path) else _infer_train_cfg_path(str(args.checkpoint_path))
    if not os.path.exists(train_cfg_path):
        raise FileNotFoundError(f"Training config not found: {train_cfg_path}")
    train_cfg = OmegaConf.load(train_cfg_path)

    dataset_format = str(getattr(train_cfg.dataset, "dataset_format", "hdf5")).lower()
    obs_source = str(getattr(train_cfg.dataset, "obs_source", "component_map")).lower()
    use_deploy_obs = dataset_format == "replay_pt" or obs_source == "deploy_dict"
    if not use_deploy_obs:
        raise RuntimeError(
            "DAgger collector currently supports deploy-format observation models only "
            "(dataset_format=replay_pt or obs_source=deploy_dict)."
        )

    train_cfg.dataset.rot_dim = get_rot_repr_dim(str(train_cfg.dataset.action_type))
    train_cfg.dataset.env_act_dim = int(train_cfg.dataset.raw_act_dim) - len(train_cfg.dataset.rotation_indices)
    train_cfg.dataset.act_dim = int(train_cfg.dataset.env_act_dim + train_cfg.dataset.rot_dim)

    obs_groups = canonicalize_deploy_obs_groups(getattr(train_cfg.dataset, "obs_groups", None))
    policy_actions_scale = float(getattr(train_cfg.dataset, "policy_actions_scale", 1.0))
    obs_stats_path = derive_replay_obs_stats_path(
        str(train_cfg.dataset.replay_obs_dir),
        obs_groups,
        float(train_cfg.dataset.split_ratio),
        policy_actions_scale=policy_actions_scale,
        explicit_path=str(getattr(train_cfg.dataset, "obs_stats_path", "") or ""),
    )
    obs_stats = load_replay_obs_stats(
        obs_stats_path,
        obs_groups,
        policy_actions_scale=policy_actions_scale,
    )
    obs_mean = torch.from_numpy(np.asarray(obs_stats["mean"], dtype=np.float32))
    obs_std = torch.from_numpy(np.asarray(obs_stats["std"], dtype=np.float32))

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        {
            "headless": bool(args.headless),
            "device": str(args.device),
        }
    )
    simulation_app = app_launcher.app

    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab.utils.datasets import HDF5DatasetFileHandler
    from dexbench_lite.tasks.utils import parse_env_cfg
    import dexbench_lite.tasks  # noqa: F401

    from model.dexbench.pl_module import DexBenchTrainingModule

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    env_cfg = parse_env_cfg(
        str(args.task),
        device=str(args.device),
        num_envs=1,
        use_fabric=True,
    )
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "object_pose"):
        env_cfg.commands.object_pose.debug_vis = False
    env = gym.make(str(args.task), cfg=env_cfg).unwrapped
    _ = env.reset()

    device = torch.device(str(args.device))
    obs_mean = obs_mean.to(device=device, dtype=torch.float32)
    obs_std = obs_std.to(device=device, dtype=torch.float32)

    module = DexBenchTrainingModule.load_from_checkpoint(str(args.checkpoint_path), map_location=device, cfg=train_cfg)
    module = module.to(device).eval()

    obs_horizon = int(train_cfg.model.obs_horizon)
    model_act_horizon = int(train_cfg.model.act_horizon)
    exec_horizon = int(args.exec_horizon)
    if exec_horizon <= 0:
        raise ValueError("--exec_horizon must be > 0")
    exec_horizon = min(exec_horizon, model_act_horizon)

    def _normalize_obs(obs_tensor: torch.Tensor) -> torch.Tensor:
        return (obs_tensor - obs_mean) / torch.clamp(obs_std, min=1.0e-6)

    def _runtime_obs_from_env_obs(env_obs_obj) -> torch.Tensor:
        obs_now = concat_deploy_obs_dict_torch(
            env_obs_obj,
            obs_groups,
            policy_actions_scale=policy_actions_scale,
        ).to(device=device, dtype=torch.float32)
        return _normalize_obs(obs_now)

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_entries: list[dict[str, Any]] = []
    next_episode_index = 0

    if str(args.bootstrap_replay_dir).strip():
        next_episode_index, bootstrap_entries = _copy_bootstrap_replay(
            str(args.bootstrap_replay_dir),
            str(args.output_dir),
            start_index=next_episode_index,
        )
        manifest_entries.extend(bootstrap_entries)
        print(f"[bootstrap] copied {len(bootstrap_entries)} episodes from {args.bootstrap_replay_dir}")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args.dataset_file)
    episode_names = list(dataset_file_handler.get_episode_names())
    if int(args.max_demos) > 0:
        episode_names = episode_names[: int(args.max_demos)]
    if len(episode_names) == 0:
        raise RuntimeError("No episodes available for DAgger collection.")

    total_target = int(len(episode_names) * int(args.repeat_per_demo))
    collected = 0
    with torch.inference_mode():
        for repeat_idx in range(int(args.repeat_per_demo)):
            for episode_name in episode_names:
                episode_data = dataset_file_handler.load_episode(episode_name, env.device)

                obs_reset_api = _slice_first_env(_extract_obs_from_reset(env.reset()))
                initial_state = episode_data.get_initial_state()
                env.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=True)
                _restore_goal_pose_from_episode(env, episode_data)
                obs_reset_replay = _slice_first_env(_get_obs_after_reset_to(env))

                obs_now = _runtime_obs_from_env_obs(obs_reset_replay)
                obs_hist = [obs_now.clone() for _ in range(obs_horizon)]

                action_list = []
                student_action_list = []
                obs_step_list = []
                reward_list = []
                terminated_list = []
                truncated_list = []

                pred_seq = None
                chunk_k = 0
                step_idx = 0
                while True:
                    teacher_action = episode_data.get_next_action()
                    if teacher_action is None:
                        break
                    if not isinstance(teacher_action, torch.Tensor):
                        teacher_action = torch.as_tensor(teacher_action, device=env.device, dtype=torch.float32)
                    else:
                        teacher_action = teacher_action.to(device=env.device, dtype=torch.float32)
                    if teacher_action.ndim == 1:
                        teacher_action = teacher_action.unsqueeze(0)

                    if pred_seq is None or chunk_k >= exec_horizon:
                        obs_seq = torch.stack(obs_hist, dim=1)
                        pred_seq = module.model.get_action(obs_seq, num_inference_steps=int(args.inference_steps))
                        chunk_k = 0
                    compact_action = pred_seq[:, chunk_k]
                    student_raw_action = _decode_raw_action(compact_action, train_cfg.dataset)
                    if float(args.action_scale) != 1.0:
                        student_raw_action = student_raw_action * float(args.action_scale)
                    if float(args.exploration_noise_std) > 0.0:
                        student_raw_action = student_raw_action + torch.randn_like(student_raw_action) * float(
                            args.exploration_noise_std
                        )
                    student_raw_action = _clamp_to_action_space(student_raw_action, env.action_space)

                    step_out = env.step(student_raw_action)
                    step_obs, reward_step, terminated_step, truncated_step, done_step = _extract_step(step_out, device=device)

                    action_list.append(teacher_action[0].detach().cpu())
                    student_action_list.append(student_raw_action[0].detach().cpu())
                    obs_step_list.append(_slice_first_env(step_obs))
                    reward_list.append(float(reward_step[0].item()))
                    terminated_list.append(bool(terminated_step[0].item()))
                    truncated_list.append(bool(truncated_step[0].item()))

                    obs_now = _runtime_obs_from_env_obs(step_obs)
                    obs_hist = obs_hist[1:] + [obs_now]
                    chunk_k += 1
                    step_idx += 1

                    if int(args.max_steps_per_episode) > 0 and step_idx >= int(args.max_steps_per_episode):
                        break
                    if bool(done_step[0].item()):
                        break

                if len(action_list) == 0:
                    continue

                dst_episode_name = f"demo_{next_episode_index}"
                dst_file = os.path.join(args.output_dir, f"{dst_episode_name}.pt")
                payload = {
                    "episode_index": int(next_episode_index),
                    "episode_name": dst_episode_name,
                    "num_steps": int(len(action_list)),
                    "obs_reset_api": obs_reset_api,
                    "obs_reset_replay": obs_reset_replay,
                    "obs_steps": obs_step_list,
                    "actions": torch.stack(action_list, dim=0),
                    "student_actions": torch.stack(student_action_list, dim=0),
                    "rewards": np.asarray(reward_list, dtype=np.float32),
                    "terminated": np.asarray(terminated_list, dtype=np.bool_),
                    "truncated": np.asarray(truncated_list, dtype=np.bool_),
                    "source_episode_name": str(episode_name),
                    "repeat_idx": int(repeat_idx),
                    "collector": "dagger_student_rollout_teacher_label",
                }
                torch.save(payload, dst_file)
                manifest_entries.append(
                    {
                        "episode_index": int(next_episode_index),
                        "episode_name": dst_episode_name,
                        "num_steps": int(len(action_list)),
                        "file": os.path.basename(dst_file),
                        "source": "dagger",
                        "source_episode_name": str(episode_name),
                        "repeat_idx": int(repeat_idx),
                    }
                )
                next_episode_index += 1
                collected += 1
                print(
                    f"[{collected:4d}/{total_target}] saved {dst_episode_name}.pt "
                    f"(src={episode_name}, repeat={repeat_idx}, steps={len(action_list)})"
                )

    manifest = {
        "dataset_file": os.path.abspath(str(args.dataset_file)),
        "task": str(args.task),
        "checkpoint_path": os.path.abspath(str(args.checkpoint_path)),
        "train_cfg_path": os.path.abspath(str(train_cfg_path)),
        "seed": int(args.seed),
        "repeat_per_demo": int(args.repeat_per_demo),
        "exec_horizon": int(exec_horizon),
        "action_scale": float(args.action_scale),
        "exploration_noise_std": float(args.exploration_noise_std),
        "obs_groups": list(obs_groups),
        "policy_actions_scale_for_student_obs": float(policy_actions_scale),
        "saved_episodes": manifest_entries,
    }
    manifest_path = os.path.join(args.output_dir, "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")
    print(f"Total episodes saved: {len(manifest_entries)}")

    env.close()
    dataset_file_handler.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
