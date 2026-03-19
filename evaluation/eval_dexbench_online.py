from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils.dexbench_observation import (
    canonicalize_obs_components,
    concatenate_component_dict_torch,
    derive_obs_stats_path,
    get_obs_component_signature,
    get_obs_dim,
    load_obs_stats,
    make_runtime_obs_component_map,
    normalize_obs_torch,
)
from utils.dexbench_replay_obs import (
    canonicalize_deploy_obs_groups,
    concat_deploy_obs_dict_torch,
    derive_replay_obs_stats_path,
    get_deploy_obs_dim,
    get_deploy_obs_signature,
    load_replay_obs_stats,
)
from utils.dexbench_rotations import decode_to_euler_xyz_np, get_rot_repr_dim


def _parse_args():
    parser = argparse.ArgumentParser(description="Online rollout evaluation for dexbench diffusion checkpoints.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(ROOT_DIR, "configs", "eval_dexbench_online.yaml"),
        help="Path to eval config yaml.",
    )
    parser.add_argument("--headless", action="store_true", default=None, help="Force headless mode.")
    parser.add_argument("--device", type=str, default=None, help="Override simulation device, e.g. cuda:0.")
    args, unknown = parser.parse_known_args()
    return args, unknown


def _load_cfg(args, unknown):
    cfg = OmegaConf.load(args.config)
    if unknown:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(unknown))
    if args.headless is not None:
        cfg.headless = bool(args.headless)
    if args.device is not None:
        cfg.device = str(args.device)
    return cfg


def _infer_train_cfg_path(checkpoint_path: str) -> str:
    run_root = os.path.abspath(os.path.join(os.path.dirname(checkpoint_path), ".."))
    return os.path.join(run_root, "log", "hydra", ".hydra", "config.yaml")


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


def _extract_obs_from_reset(reset_out):
    if isinstance(reset_out, tuple):
        if len(reset_out) >= 1:
            return reset_out[0]
        raise RuntimeError("env.reset() returned an empty tuple.")
    return reset_out


def _extract_step(step_out, device):
    if isinstance(step_out, tuple):
        if len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
        elif len(step_out) == 4:
            obs, reward, terminated, _ = step_out
            truncated = torch.zeros_like(terminated)
        else:
            raise RuntimeError(f"Unsupported env.step output length: {len(step_out)}")
    else:
        raise RuntimeError("Unsupported env.step return type.")

    reward = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
    terminated = torch.as_tensor(terminated, device=device, dtype=torch.bool).view(-1)
    truncated = torch.as_tensor(truncated, device=device, dtype=torch.bool).view(-1)
    done = terminated | truncated
    return obs, reward, done


def _safe_stats(arr: list[float]) -> dict[str, float]:
    if not arr:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0}
    x = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
    }


def _episode_len_histogram(lens: list[int]) -> dict[str, int]:
    out = {
        "1": 0,
        "2-5": 0,
        "6-10": 0,
        "11-20": 0,
        "21-50": 0,
        "51-100": 0,
        "101-200": 0,
        "201-400": 0,
        "401-600": 0,
        ">600": 0,
    }
    for length in lens:
        x = int(length)
        if x <= 1:
            out["1"] += 1
        elif x <= 5:
            out["2-5"] += 1
        elif x <= 10:
            out["6-10"] += 1
        elif x <= 20:
            out["11-20"] += 1
        elif x <= 50:
            out["21-50"] += 1
        elif x <= 100:
            out["51-100"] += 1
        elif x <= 200:
            out["101-200"] += 1
        elif x <= 400:
            out["201-400"] += 1
        elif x <= 600:
            out["401-600"] += 1
        else:
            out[">600"] += 1
    return out


def main():
    args, unknown = _parse_args()
    cfg = _load_cfg(args, unknown)

    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path is required.")

    train_cfg_path = str(cfg.train_cfg_path) if str(cfg.train_cfg_path) else _infer_train_cfg_path(str(cfg.checkpoint_path))
    if not os.path.exists(train_cfg_path):
        raise FileNotFoundError(f"Training config not found: {train_cfg_path}")
    train_cfg = OmegaConf.load(train_cfg_path)

    train_cfg.dataset.action_type = str(cfg.dataset.action_type)
    rot_dim = get_rot_repr_dim(str(train_cfg.dataset.action_type))
    train_cfg.dataset.rot_dim = rot_dim
    train_cfg.dataset.env_act_dim = int(train_cfg.dataset.raw_act_dim) - len(train_cfg.dataset.rotation_indices)
    train_cfg.dataset.act_dim = int(train_cfg.dataset.env_act_dim + rot_dim)
    dataset_format = str(getattr(train_cfg.dataset, "dataset_format", "hdf5")).lower()
    obs_source = str(getattr(train_cfg.dataset, "obs_source", "component_map")).lower()
    use_deploy_obs = dataset_format == "replay_pt" or obs_source == "deploy_dict"

    obs_components = None
    obs_groups = None
    policy_actions_scale = 1.0
    if use_deploy_obs:
        obs_groups = canonicalize_deploy_obs_groups(getattr(train_cfg.dataset, "obs_groups", None))
        policy_actions_scale = float(getattr(train_cfg.dataset, "policy_actions_scale", 1.0))
        obs_signature = get_deploy_obs_signature(
            obs_groups,
            policy_actions_scale=policy_actions_scale,
        )
        train_cfg.dataset.obs_dim = get_deploy_obs_dim(obs_groups)
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
        obs_mean = obs_stats["mean"]
        obs_std = obs_stats["std"]
    else:
        obs_components = canonicalize_obs_components(getattr(train_cfg.dataset, "obs_components", None))
        obs_signature = get_obs_component_signature(obs_components)
        train_cfg.dataset.obs_dim = get_obs_dim(obs_components)
        obs_stats_path = derive_obs_stats_path(
            str(train_cfg.dataset.hdf5_path),
            obs_components,
            float(train_cfg.dataset.split_ratio),
            explicit_path=str(getattr(train_cfg.dataset, "obs_stats_path", "") or ""),
        )
        obs_stats = load_obs_stats(obs_stats_path, obs_components)
        obs_mean = obs_stats["mean"]
        obs_std = obs_stats["std"]

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(
        {
            "headless": bool(cfg.headless),
            "device": str(cfg.device),
        }
    )
    simulation_app = app_launcher.app

    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab.managers import SceneEntityCfg
    from dexbench_lite.tasks.manager_based.table_top_manipulation import mdp as dexbench_mdp
    from dexbench_lite.tasks.utils import parse_env_cfg
    import dexbench_lite.tasks  # noqa: F401

    from model.dexbench.pl_module import DexBenchTrainingModule

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    env_cfg = parse_env_cfg(
        str(cfg.task),
        device=str(cfg.device),
        num_envs=int(cfg.num_envs),
        use_fabric=not bool(cfg.disable_fabric),
    )
    # Disable debug_vis to avoid downloading frame_prim.usd from Omniverse CDN (no internet)
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "object_pose"):
        env_cfg.commands.object_pose.debug_vis = False
    env = gym.make(str(cfg.task), cfg=env_cfg).unwrapped
    reset_out = env.reset()
    reset_obs = _extract_obs_from_reset(reset_out)

    device = torch.device(str(cfg.device))
    module = DexBenchTrainingModule.load_from_checkpoint(str(cfg.checkpoint_path), map_location=device, cfg=train_cfg)
    module = module.to(device).eval()

    obs_horizon = int(train_cfg.model.obs_horizon)
    max_episode_steps = int(cfg.max_episode_steps)
    num_target_episodes = int(cfg.num_episodes)
    exclude_success_within_steps = int(getattr(cfg, "exclude_success_within_steps", 1))
    report_episode_lists = bool(getattr(cfg, "report_episode_lists", False))

    def current_obs_from_env_obs(env_obs_obj) -> torch.Tensor:
        obs_now = concat_deploy_obs_dict_torch(
            env_obs_obj,
            obs_groups,
            policy_actions_scale=policy_actions_scale,
        ).to(device=device, dtype=torch.float32)
        if obs_now.shape[-1] != int(train_cfg.dataset.obs_dim):
            raise RuntimeError(
                f"Runtime observation dim mismatch: got {obs_now.shape[-1]}, expected {int(train_cfg.dataset.obs_dim)} "
                f"for signature {obs_signature}"
            )
        return normalize_obs_torch(obs_now, obs_mean, obs_std)

    def current_obs_from_scene() -> torch.Tensor:
        component_map = make_runtime_obs_component_map(env, dexbench_mdp, SceneEntityCfg)
        obs_now = concatenate_component_dict_torch(component_map, obs_components).to(device=device, dtype=torch.float32)
        if obs_now.shape[-1] != int(train_cfg.dataset.obs_dim):
            raise RuntimeError(
                f"Runtime observation dim mismatch: got {obs_now.shape[-1]}, expected {int(train_cfg.dataset.obs_dim)} "
                f"for signature {obs_signature}"
            )
        return normalize_obs_torch(obs_now, obs_mean, obs_std)

    if use_deploy_obs:
        obs_now = current_obs_from_env_obs(reset_obs)
    else:
        obs_now = current_obs_from_scene()
    obs_hist = [obs_now.clone() for _ in range(obs_horizon)]

    num_envs = int(cfg.num_envs)
    episode_returns = torch.zeros(num_envs, device=device, dtype=torch.float32)
    episode_lens = torch.zeros(num_envs, device=device, dtype=torch.long)
    success_flags = torch.zeros(num_envs, device=device, dtype=torch.bool)
    episode_action_l2_sum = torch.zeros(num_envs, device=device, dtype=torch.float32)
    episode_action_abs_sum = torch.zeros(num_envs, device=device, dtype=torch.float32)

    collected_returns_raw: list[float] = []
    collected_lens_raw: list[int] = []
    collected_success_raw: list[float] = []
    collected_returns: list[float] = []
    collected_lens: list[int] = []
    collected_success: list[float] = []
    collected_action_l2_mean: list[float] = []
    collected_action_abs_mean: list[float] = []
    excluded_step1_success = 0

    global_action_l2_sum = 0.0
    global_action_l2_sq_sum = 0.0
    global_action_abs_sum = 0.0
    global_action_abs_sq_sum = 0.0
    global_action_count = 0

    max_total_steps = int(getattr(cfg, "max_total_steps", 0))
    if max_total_steps <= 0:
        max_total_steps = max(1, num_target_episodes * max_episode_steps * 2)

    act_horizon = int(train_cfg.model.act_horizon)
    exec_horizon = int(getattr(cfg, "exec_horizon", act_horizon))
    if exec_horizon <= 0:
        raise ValueError(f"exec_horizon must be > 0, got {exec_horizon}.")
    exec_horizon = min(exec_horizon, act_horizon)
    action_scale = float(getattr(cfg, "action_scale", 1.0))
    if action_scale <= 0.0:
        raise ValueError(f"action_scale must be > 0, got {action_scale}.")

    total_steps = 0
    while len(collected_success_raw) < num_target_episodes and total_steps < max_total_steps:
        # Plan once, then execute a configurable number of chunked actions before re-planning.
        with torch.inference_mode():
            obs_seq = torch.stack(obs_hist, dim=1)
            pred_seq = module.model.get_action(obs_seq)  # (num_envs, act_horizon, act_dim)

        for k in range(exec_horizon):
            if len(collected_success_raw) >= num_target_episodes or total_steps >= max_total_steps:
                break

            compact_action = pred_seq[:, k]
            raw_action = _decode_raw_action(compact_action, train_cfg.dataset)
            if action_scale != 1.0:
                raw_action = raw_action * action_scale
            raw_action = _clamp_to_action_space(raw_action, env.action_space)

            step_action_l2 = torch.linalg.norm(raw_action, dim=-1)
            step_action_abs = raw_action.abs().mean(dim=-1)
            episode_action_l2_sum += step_action_l2
            episode_action_abs_sum += step_action_abs
            global_action_l2_sum += float(step_action_l2.sum().item())
            global_action_l2_sq_sum += float((step_action_l2 ** 2).sum().item())
            global_action_abs_sum += float(step_action_abs.sum().item())
            global_action_abs_sq_sum += float((step_action_abs ** 2).sum().item())
            global_action_count += int(step_action_l2.numel())

            step_out = env.step(raw_action)
            step_obs, reward, done = _extract_step(step_out, device=device)
            episode_returns += reward
            episode_lens += 1

            try:
                success_step = env.termination_manager.get_term("success").to(device=device, dtype=torch.bool).view(-1)
            except Exception:
                success_step = dexbench_mdp.object_at_goal_position(
                    env,
                    command_name="object_pose",
                    threshold=0.03,
                ).to(device=device, dtype=torch.bool).view(-1)
            success_flags |= success_step

            timeout = episode_lens >= max_episode_steps
            done = done | timeout

            done_ids = torch.nonzero(done, as_tuple=False).view(-1)
            for done_id in done_ids.tolist():
                ep_return = float(episode_returns[done_id].item())
                ep_len = int(episode_lens[done_id].item())
                ep_success = float(success_flags[done_id].item())
                ep_action_l2 = float((episode_action_l2_sum[done_id] / max(ep_len, 1)).item())
                ep_action_abs = float((episode_action_abs_sum[done_id] / max(ep_len, 1)).item())

                collected_returns_raw.append(ep_return)
                collected_lens_raw.append(ep_len)
                collected_success_raw.append(ep_success)

                if not (ep_success > 0.5 and ep_len <= exclude_success_within_steps):
                    collected_returns.append(ep_return)
                    collected_lens.append(ep_len)
                    collected_success.append(ep_success)
                    collected_action_l2_mean.append(ep_action_l2)
                    collected_action_abs_mean.append(ep_action_abs)
                else:
                    excluded_step1_success += 1

                episode_returns[done_id] = 0.0
                episode_lens[done_id] = 0
                success_flags[done_id] = False
                episode_action_l2_sum[done_id] = 0.0
                episode_action_abs_sum[done_id] = 0.0
                if len(collected_success_raw) >= num_target_episodes:
                    break

            if use_deploy_obs:
                obs_now = current_obs_from_env_obs(step_obs)
            else:
                obs_now = current_obs_from_scene()
            obs_hist = obs_hist[1:] + [obs_now]
            total_steps += 1

    if not collected_success_raw:
        env.close()
        simulation_app.close()
        raise RuntimeError(
            f"No episode was collected during online evaluation "
            f"(total_steps={total_steps}, max_total_steps={max_total_steps})."
        )

    fallback_to_raw = False
    if not collected_success:
        fallback_to_raw = True
        collected_returns = list(collected_returns_raw)
        collected_lens = list(collected_lens_raw)
        collected_success = list(collected_success_raw)

    len_stats = _safe_stats([float(x) for x in collected_lens])
    action_l2_stats = _safe_stats(collected_action_l2_mean)
    action_abs_stats = _safe_stats(collected_action_abs_mean)
    if fallback_to_raw and global_action_count > 0:
        mean_l2 = global_action_l2_sum / global_action_count
        mean_abs = global_action_abs_sum / global_action_count
        var_l2 = max(0.0, global_action_l2_sq_sum / global_action_count - mean_l2 * mean_l2)
        var_abs = max(0.0, global_action_abs_sq_sum / global_action_count - mean_abs * mean_abs)
        action_l2_stats["mean"] = float(mean_l2)
        action_l2_stats["std"] = float(np.sqrt(var_l2))
        action_abs_stats["mean"] = float(mean_abs)
        action_abs_stats["std"] = float(np.sqrt(var_abs))

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "task": str(cfg.task),
        "action_type": str(train_cfg.dataset.action_type),
        "seed": int(cfg.seed),
        "num_envs": int(cfg.num_envs),
        "obs_component_signature": obs_signature,
        "obs_stats_path": obs_stats_path,
        "policy_actions_scale": float(policy_actions_scale),
        "exclude_success_within_steps": int(exclude_success_within_steps),
        "model_act_horizon": int(act_horizon),
        "exec_horizon": int(exec_horizon),
        "action_scale": float(action_scale),
        "num_episodes_raw": int(len(collected_success_raw)),
        "num_excluded_step1_success": int(excluded_step1_success),
        "num_episodes": int(len(collected_success)),
        "num_episodes_effective": int(len(collected_success)),
        "used_raw_fallback_after_exclusion": bool(fallback_to_raw),
        "success_rate_raw": float(np.mean(collected_success_raw)),
        "success_rate": float(np.mean(collected_success)),
        "episode_return_mean_raw": float(np.mean(collected_returns_raw)),
        "episode_return_mean": float(np.mean(collected_returns)),
        "avg_episode_len_raw": float(np.mean(collected_lens_raw)),
        "avg_episode_len": float(np.mean(collected_lens)),
        "episode_len_p50": float(len_stats["p50"]),
        "episode_len_p90": float(len_stats["p90"]),
        "episode_len_min": float(len_stats["min"]),
        "episode_len_max": float(len_stats["max"]),
        "episode_len_histogram": _episode_len_histogram(collected_lens),
        "action_l2_mean": float(action_l2_stats["mean"]),
        "action_l2_std": float(action_l2_stats["std"]),
        "action_abs_mean": float(action_abs_stats["mean"]),
        "action_abs_std": float(action_abs_stats["std"]),
        "checkpoint_path": str(cfg.checkpoint_path),
    }
    if report_episode_lists:
        summary["episode_lens_raw"] = [int(x) for x in collected_lens_raw]
        summary["episode_lens_effective"] = [int(x) for x in collected_lens]
        summary["episode_success_raw"] = [float(x) for x in collected_success_raw]
        summary["episode_success_effective"] = [float(x) for x in collected_success]
        summary["episode_action_l2_mean_effective"] = [float(x) for x in collected_action_l2_mean]
        summary["episode_action_abs_mean_effective"] = [float(x) for x in collected_action_abs_mean]

    out_dir = str(cfg.eval.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"online_metrics_{train_cfg.dataset.action_type}.csv")
    json_path = os.path.join(out_dir, f"online_metrics_{train_cfg.dataset.action_type}.json")

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(summary)
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
