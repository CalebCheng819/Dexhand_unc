#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil

import h5py
import numpy as np
import torch

from isaaclab.app import AppLauncher

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from utils.dexbench_observation import DEFAULT_OBS_COMPONENTS, get_obs_component_signature, make_runtime_obs_component_map


parser = argparse.ArgumentParser(description="Rebuild DexBench HDF5 with augmented observation components.")
parser.add_argument(
    "--src_hdf5",
    type=str,
    default=os.path.join(ROOT_DIR, "dexbench_lite", "dexbench_lite", "relocate_no_conflict.hdf5"),
    help="Source DexBench HDF5 file.",
)
parser.add_argument(
    "--dst_hdf5",
    type=str,
    default=os.path.join(ROOT_DIR, "dexbench_lite", "dexbench_lite", "relocate_no_conflict_augmented.hdf5"),
    help="Destination augmented DexBench HDF5 file.",
)
parser.add_argument("--task", type=str, default=None, help="Override task name. Defaults to dataset env name.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric.")
parser.add_argument("--force", action="store_true", help="Overwrite destination file if it exists.")
parser.add_argument("--max_episodes", type=int, default=0, help="Optional cap for smoke rebuilds.")
parser.add_argument(
    "--validate_states",
    action="store_true",
    default=False,
    help="Validate recorded states against runtime states after each replay step.",
)
parser.add_argument(
    "--state_tolerance",
    type=float,
    default=1.0e-2,
    help="Absolute tolerance for state validation.",
)
parser.add_argument(
    "--log_state_mismatch_limit",
    type=int,
    default=3,
    help="Maximum number of per-episode state mismatch details to print.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.datasets import HDF5DatasetFileHandler

import dexbench_lite.tasks  # noqa: F401
from dexbench_lite.tasks.manager_based.table_top_manipulation import mdp as dexbench_mdp
from dexbench_lite.tasks.utils import parse_env_cfg


def _extract_goal_pose(episode_data, device: torch.device) -> torch.Tensor | None:
    if "goal_pose" not in episode_data.data:
        return None
    goal_pose = episode_data.data["goal_pose"]
    if isinstance(goal_pose, torch.Tensor):
        out = goal_pose.to(device)
    elif isinstance(goal_pose, list) and len(goal_pose) > 0:
        out = goal_pose[0].to(device)
    else:
        return None
    if out.ndim == 1:
        out = out.unsqueeze(0)
    return out


def _set_goal_pose(env, goal_pose: torch.Tensor | None) -> None:
    if goal_pose is None or not hasattr(env, "command_manager"):
        return
    command_term = env.command_manager.get_term("object_pose")
    command_term.pose_command_b[0] = goal_pose[0]
    if hasattr(command_term, "_update_metrics"):
        command_term._update_metrics()


def _write_obs_dataset(obs_group: h5py.Group, key: str, arr: np.ndarray) -> None:
    if key in obs_group:
        del obs_group[key]
    obs_group.create_dataset(key, data=arr.astype(np.float32), compression="gzip")


def _normalize_task_name(task_name: str) -> str:
    name = str(task_name)
    if name.startswith("Dexbench-"):
        return "DexbenchLite-" + name[len("Dexbench-") :]
    return name


def _compare_states(
    state_from_dataset: dict,
    runtime_state: dict,
    runtime_env_index: int,
    atol: float,
    max_logs: int,
) -> tuple[bool, float, list[str]]:
    states_matched = True
    max_abs_err = 0.0
    logs: list[str] = []

    for asset_type in ("articulation", "rigid_object"):
        runtime_assets = runtime_state.get(asset_type, {})
        dataset_assets = state_from_dataset.get(asset_type, {})
        for asset_name, runtime_asset_states in runtime_assets.items():
            dataset_asset_states = dataset_assets.get(asset_name)
            if dataset_asset_states is None:
                states_matched = False
                if len(logs) < max_logs:
                    logs.append(f"missing dataset key: states/{asset_type}/{asset_name}")
                continue
            for state_name, runtime_tensor in runtime_asset_states.items():
                if state_name not in dataset_asset_states:
                    states_matched = False
                    if len(logs) < max_logs:
                        logs.append(f"missing dataset key: states/{asset_type}/{asset_name}/{state_name}")
                    continue
                runtime_vec = runtime_tensor[runtime_env_index].detach().cpu().numpy().astype(np.float32).reshape(-1)
                dataset_vec = np.asarray(dataset_asset_states[state_name], dtype=np.float32).reshape(-1)
                if runtime_vec.shape != dataset_vec.shape:
                    states_matched = False
                    if len(logs) < max_logs:
                        logs.append(
                            "shape mismatch at "
                            f"states/{asset_type}/{asset_name}/{state_name}: "
                            f"dataset={dataset_vec.shape}, runtime={runtime_vec.shape}"
                        )
                    continue
                diff = np.abs(runtime_vec - dataset_vec)
                if diff.size > 0:
                    max_abs = float(diff.max())
                    max_abs_err = max(max_abs_err, max_abs)
                    if max_abs > float(atol):
                        states_matched = False
                        if len(logs) < max_logs:
                            worst_i = int(diff.argmax())
                            logs.append(
                                "value mismatch at "
                                f"states/{asset_type}/{asset_name}/{state_name}[{worst_i}] "
                                f"(dataset={float(dataset_vec[worst_i]):.6f}, runtime={float(runtime_vec[worst_i]):.6f}, "
                                f"|diff|={max_abs:.6f})"
                            )
    return states_matched, max_abs_err, logs


def main() -> None:
    src_hdf5 = os.path.abspath(args_cli.src_hdf5)
    dst_hdf5 = os.path.abspath(args_cli.dst_hdf5)
    obs_components = list(DEFAULT_OBS_COMPONENTS)
    obs_signature = get_obs_component_signature(obs_components)
    obs_components_json = json.dumps(obs_components)

    if not os.path.exists(src_hdf5):
        raise FileNotFoundError(f"Source HDF5 not found: {src_hdf5}")
    if os.path.exists(dst_hdf5):
        if not args_cli.force:
            raise FileExistsError(f"Destination already exists: {dst_hdf5}. Use --force to overwrite.")
        os.remove(dst_hdf5)
    os.makedirs(os.path.dirname(dst_hdf5), exist_ok=True)
    shutil.copy2(src_hdf5, dst_hdf5)

    dataset_handler = HDF5DatasetFileHandler()
    dataset_handler.open(src_hdf5)
    env_name = _normalize_task_name(args_cli.task if args_cli.task else dataset_handler.get_env_name())
    if env_name is None:
        raise ValueError("Could not determine DexBench task name from HDF5. Pass --task explicitly.")

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1, use_fabric=not bool(args_cli.disable_fabric))
    env_cfg.recorders = {}
    env_cfg.terminations = {}
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "debug_reset"):
        env_cfg.events.debug_reset = None

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    env.reset()

    with h5py.File(src_hdf5, "r") as src_file, h5py.File(dst_hdf5, "r+") as dst_file, torch.inference_mode():
        episode_names = list(dataset_handler.get_episode_names())
        if int(args_cli.max_episodes) > 0:
            episode_names = episode_names[: int(args_cli.max_episodes)]

        total = len(episode_names)
        total_state_checks = 0
        total_state_mismatch_steps = 0
        total_state_max_abs_err = 0.0
        episodes_without_states = 0
        for idx, episode_name in enumerate(episode_names, start=1):
            episode_data = dataset_handler.load_episode(episode_name, env.device)
            initial_state = episode_data.get_initial_state()
            env.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=True)
            goal_pose = _extract_goal_pose(episode_data, env.device)
            _set_goal_pose(env, goal_pose)

            src_grp = src_file["data"][episode_name]
            dst_grp = dst_file["data"][episode_name]
            processed_actions = src_grp["processed_actions"][...].astype(np.float32)
            steps = int(processed_actions.shape[0])

            records: dict[str, list[np.ndarray]] = {name: [] for name in obs_components if name != "goal_pose"}
            episode_state_checks = 0
            episode_state_mismatch_steps = 0
            episode_state_max_abs_err = 0.0
            episode_state_log_budget = max(0, int(args_cli.log_state_mismatch_limit))
            for step_idx in range(steps):
                component_map = make_runtime_obs_component_map(env, dexbench_mdp, SceneEntityCfg)
                for name in records:
                    records[name].append(component_map[name][0].detach().cpu().numpy().astype(np.float32))
                action = torch.from_numpy(processed_actions[step_idx : step_idx + 1]).to(env.device)
                env.step(action)
                if bool(args_cli.validate_states):
                    state_from_dataset = episode_data.get_next_state()
                    if state_from_dataset is None:
                        continue
                    runtime_state = env.scene.get_state(is_relative=True)
                    states_matched, max_abs_err, logs = _compare_states(
                        state_from_dataset=state_from_dataset,
                        runtime_state=runtime_state,
                        runtime_env_index=0,
                        atol=float(args_cli.state_tolerance),
                        max_logs=episode_state_log_budget,
                    )
                    episode_state_checks += 1
                    episode_state_max_abs_err = max(episode_state_max_abs_err, max_abs_err)
                    if not states_matched:
                        episode_state_mismatch_steps += 1
                        if episode_state_log_budget > 0 and logs:
                            print(
                                f"[{idx}/{total}] state mismatch in {episode_name} at step={step_idx}",
                                flush=True,
                            )
                            for line in logs:
                                print(f"    - {line}", flush=True)
                            episode_state_log_budget = 0

            obs_group = dst_grp.require_group("obs")
            for name, values in records.items():
                key = name.split("/", 1)[1]
                arr = np.stack(values, axis=0).astype(np.float32)
                _write_obs_dataset(obs_group, key, arr)

            dst_grp.attrs["obs_component_signature"] = obs_signature
            dst_grp.attrs["obs_components_json"] = obs_components_json
            dst_file.attrs["dexbench_obs_component_signature"] = obs_signature
            dst_file.attrs["dexbench_obs_components_json"] = obs_components_json

            print(f"[{idx}/{total}] rebuilt {episode_name}: steps={steps}", flush=True)
            if bool(args_cli.validate_states):
                if episode_state_checks == 0:
                    episodes_without_states += 1
                    print(
                        f"[{idx}/{total}] state validation skipped for {episode_name}: no recorded states found.",
                        flush=True,
                    )
                else:
                    total_state_checks += episode_state_checks
                    total_state_mismatch_steps += episode_state_mismatch_steps
                    total_state_max_abs_err = max(total_state_max_abs_err, episode_state_max_abs_err)
                    print(
                        f"[{idx}/{total}] state validation {episode_name}: "
                        f"mismatched_steps={episode_state_mismatch_steps}/{episode_state_checks}, "
                        f"max_abs_err={episode_state_max_abs_err:.6f}",
                        flush=True,
                    )

    env.close()
    dataset_handler.close()
    simulation_app.close()
    print(f"Saved augmented DexBench HDF5: {dst_hdf5}", flush=True)
    print(f"Observation signature: {obs_signature}", flush=True)
    if bool(args_cli.validate_states):
        if total_state_checks > 0:
            mismatch_ratio = float(total_state_mismatch_steps) / float(total_state_checks)
            print(
                "State validation summary: "
                f"mismatched_steps={total_state_mismatch_steps}/{total_state_checks} "
                f"({mismatch_ratio:.2%}), "
                f"max_abs_err={total_state_max_abs_err:.6f}, "
                f"episodes_without_states={episodes_without_states}",
                flush=True,
            )
        else:
            print(
                "State validation summary: no comparable states were found in the replayed episodes.",
                flush=True,
            )


if __name__ == "__main__":
    main()
