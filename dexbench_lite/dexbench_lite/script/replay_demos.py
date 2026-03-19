# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to replay demonstrations with DexBench environments.

This script allows users to replay demonstrations recorded with record_demos.py.
The demonstrations are loaded from an hdf5 file and replayed in the simulation.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import json
import os
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay demonstrations in DexBench environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to be replayed. Keep empty to replay all in the dataset file.",
)
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="Dataset file to be replayed.")
parser.add_argument(
    "--save_obs_dir",
    type=str,
    default="",
    help=(
        "If set, save env.reset/env.step observation trajectories (deploy-format obs) "
        "for each replayed demo into this directory as .pt files."
    ),
)
parser.add_argument(
    "--validate_states",
    action="store_true",
    default=False,
    help=(
        "Validate if the states, if available, match between loaded from datasets and replayed. Only valid if"
        " --num_envs is 1."
    ),
)
parser.add_argument(
    "--validate_success_rate",
    action="store_true",
    default=False,
    help="Validate the replay success rate using the task environment termination criteria",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio (required for dex-retargeting and some IK controllers).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the dex-retargeting utilities
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

import isaaclab_tasks  # noqa: F401
from dexbench_lite.tasks.utils import parse_env_cfg

import dexbench_lite.tasks  # noqa: F401

is_paused = False

# Create marker for visualizing recorded goal poses
# Use a yellow sphere to distinguish from the live goal (which is typically red/green)
RECORDED_GOAL_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Replay/recorded_goal",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.03,  # Slightly larger than default to make it visible
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),  # Yellow
        ),
    },
)
recorded_goal_marker = None


def play_cb():
    """Callback to resume playback."""
    global is_paused
    is_paused = False


def pause_cb():
    """Callback to pause playback."""
    global is_paused
    is_paused = True


def compare_states(state_from_dataset, runtime_state, runtime_env_index) -> tuple[bool, str]:
    """Compare states from dataset and runtime.

    Args:
        state_from_dataset: State from dataset.
        runtime_state: State from runtime.
        runtime_env_index: Index of the environment in the runtime states to be compared.

    Returns:
        tuple[bool, str]: A tuple containing:
            - True if states match, False otherwise
            - Log message if states don't match
    """
    states_matched = True
    output_log = ""
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name][runtime_env_index]
                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name]
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(f"State shape of {state_name} for asset {asset_name} don't match")
                for i in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[i] - runtime_asset_state[i]) > 0.01:
                        states_matched = False
                        output_log += f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n'
                        output_log += f"\t  Dataset:\t{dataset_asset_state[i]}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[i]}\r\n"
    return states_matched, output_log


def _slice_first_env(obs_obj):
    """Slice vectorized observation object to env_0 while preserving structure."""
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


def _extract_obs_from_reset(reset_out):
    """Extract observation payload from env.reset(...) return."""
    if isinstance(reset_out, tuple):
        if len(reset_out) >= 1:
            return reset_out[0]
        raise RuntimeError("env.reset() returned an empty tuple.")
    return reset_out


def _extract_obs_from_step(step_out):
    """Extract observation payload from env.step(...) return."""
    if not isinstance(step_out, tuple):
        raise RuntimeError(f"Unsupported env.step return type: {type(step_out)}")
    if len(step_out) == 5:
        return step_out[0], step_out[1], step_out[2], step_out[3], step_out[4]
    if len(step_out) == 4:
        obs, reward, terminated, info = step_out
        truncated = torch.zeros_like(torch.as_tensor(terminated), dtype=torch.bool)
        return obs, reward, terminated, truncated, info
    raise RuntimeError(f"Unsupported env.step output length: {len(step_out)}")


def _get_obs_after_reset_to(env):
    """Try to fetch current observation after env.reset_to(...) without stepping."""
    if hasattr(env, "get_observations"):
        return env.get_observations()
    if hasattr(env, "observation_manager"):
        manager = getattr(env, "observation_manager")
        if hasattr(manager, "compute"):
            return manager.compute()
    raise RuntimeError(
        "Could not fetch observation after reset_to(). "
        "Expected env.get_observations() or env.observation_manager.compute()."
    )


def main():
    """Replay episodes loaded from a file."""
    global is_paused

    # Load dataset
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    episode_indices_to_replay = args_cli.select_episodes
    if len(episode_indices_to_replay) == 0:
        episode_indices_to_replay = list(range(episode_count))

    # Use task from CLI if provided, otherwise use env_name from dataset
    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    num_envs = args_cli.num_envs

    # Parse environment configuration
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)

    # Disable goal pose resampling during replay to match recorded demos
    # This prevents the goal from changing during replay, which would cause mismatches
    if "TopDownGrasp" in env_name or "Lift" in env_name:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "object_pose"):
            env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)

    # extract success checking function to invoke in the main loop
    success_term = None
    if args_cli.validate_success_rate:
        if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None
        else:
            print(
                "No success termination term was found in the environment."
                " Will not be able to validate success rate."
            )

    # Disable all recorders, terminations, and debug events for replay
    env_cfg.recorders = {}
    env_cfg.terminations = {}
    # Disable debug_reset event to avoid "no termination terms active" messages during replay
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "debug_reset"):
        env_cfg.events.debug_reset = None

    # create environment from loaded config
    task_name = args_cli.task if args_cli.task else env_name
    env = gym.make(task_name, cfg=env_cfg).unwrapped

    # Only set up keyboard interface if not in headless mode
    teleop_interface = None
    if not args_cli.headless:
        teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
        teleop_interface.add_callback("N", play_cb)
        teleop_interface.add_callback("B", pause_cb)
        print('Press "B" to pause and "N" to resume the replayed actions.')
    else:
        print("Running in headless mode. Pause/resume controls disabled.")

    # Determine if state validation should be conducted
    state_validation_enabled = False
    if args_cli.validate_states and num_envs == 1:
        state_validation_enabled = True
    elif args_cli.validate_states and num_envs > 1:
        print("Warning: State validation is only supported with a single environment. Skipping state validation.")

    # Get idle action (idle actions are applied to envs without next action)
    if hasattr(env_cfg, "idle_action"):
        idle_action = env_cfg.idle_action.repeat(num_envs, 1)
    else:
        idle_action = torch.zeros(env.action_space.shape)

    # Initialize recorded goal marker visualization
    global recorded_goal_marker
    recorded_goal_marker = VisualizationMarkers(RECORDED_GOAL_MARKER_CFG)
    recorded_goal_marker.set_visibility(False)  # Hide initially

    # reset before starting
    env.reset()
    if teleop_interface is not None:
        teleop_interface.reset()

    # Optional export mode: save deploy-format obs from env.reset/env.step.
    save_obs_enabled = bool(str(args_cli.save_obs_dir).strip())
    if save_obs_enabled:
        if num_envs != 1:
            raise ValueError("--save_obs_dir currently supports --num_envs=1 only.")

        save_obs_dir = os.path.abspath(str(args_cli.save_obs_dir))
        os.makedirs(save_obs_dir, exist_ok=True)
        print(f"Saving replay observations to: {save_obs_dir}")

        episode_names = list(dataset_file_handler.get_episode_names())
        selected_episode_indices = [idx for idx in episode_indices_to_replay if idx < episode_count]
        if len(selected_episode_indices) == 0:
            print("No valid episodes selected. Nothing to save.")
            env.close()
            dataset_file_handler.close()
            return

        manifest_entries = []
        with torch.inference_mode():
            for replayed_episode_count, episode_index in enumerate(selected_episode_indices, start=1):
                episode_name = episode_names[episode_index]
                print(f"{replayed_episode_count:4}: Loading #{episode_index} ({episode_name})")
                episode_data = dataset_file_handler.load_episode(episode_name, env.device)

                # Keep the literal reset output format (deploy-style API), then reset to demo state for replay.
                obs_reset_api = _slice_first_env(_extract_obs_from_reset(env.reset()))
                initial_state = episode_data.get_initial_state()
                env.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=True)

                # Restore goal pose command if present in dataset.
                if "goal_pose" in episode_data.data and hasattr(env, "command_manager"):
                    goal_pose = episode_data.data["goal_pose"]
                    goal_pose_tensor = None
                    if isinstance(goal_pose, torch.Tensor):
                        goal_pose_tensor = goal_pose.to(env.device)
                    elif isinstance(goal_pose, list) and len(goal_pose) > 0:
                        goal_pose_tensor = goal_pose[0].to(env.device)
                    if goal_pose_tensor is not None:
                        if goal_pose_tensor.dim() == 1:
                            goal_pose_tensor = goal_pose_tensor.unsqueeze(0)
                        try:
                            command_term = env.command_manager.get_term("object_pose")
                            if hasattr(command_term, "pose_command_b"):
                                command_term.pose_command_b[0] = goal_pose_tensor[0]
                                if hasattr(command_term, "_update_metrics"):
                                    command_term._update_metrics()
                        except Exception as exc:  # pragma: no cover - best-effort restore
                            print(f"Warning: Could not restore goal pose in save_obs mode: {exc}")

                # Observation right after reset_to (aligns with replay initial state).
                obs_reset_replay = _slice_first_env(_get_obs_after_reset_to(env))

                action_list = []
                obs_step_list = []
                reward_list = []
                terminated_list = []
                truncated_list = []
                step_idx = 0
                while True:
                    env_next_action = episode_data.get_next_action()
                    if env_next_action is None:
                        break
                    if not isinstance(env_next_action, torch.Tensor):
                        env_next_action = torch.as_tensor(env_next_action, device=env.device, dtype=torch.float32)
                    else:
                        env_next_action = env_next_action.to(device=env.device)
                    if env_next_action.ndim == 1:
                        env_next_action = env_next_action.unsqueeze(0)

                    step_out = env.step(env_next_action)
                    obs_step, reward_step, terminated_step, truncated_step, _ = _extract_obs_from_step(step_out)

                    action_list.append(env_next_action[0].detach().cpu())
                    obs_step_list.append(_slice_first_env(obs_step))
                    reward_list.append(float(torch.as_tensor(reward_step).view(-1)[0].item()))
                    terminated_list.append(bool(torch.as_tensor(terminated_step).view(-1)[0].item()))
                    truncated_list.append(bool(torch.as_tensor(truncated_step).view(-1)[0].item()))

                    if state_validation_enabled:
                        state_from_dataset = episode_data.get_next_state()
                        if state_from_dataset is not None:
                            current_runtime_state = env.scene.get_state(is_relative=True)
                            states_matched, comparison_log = compare_states(state_from_dataset, current_runtime_state, 0)
                            if not states_matched:
                                print(f"State mismatch at step={step_idx} for episode={episode_name}")
                                print(comparison_log)
                    step_idx += 1

                actions_tensor = torch.stack(action_list, dim=0) if action_list else torch.empty((0,), dtype=torch.float32)
                payload = {
                    "episode_index": int(episode_index),
                    "episode_name": episode_name,
                    "num_steps": int(len(action_list)),
                    "obs_reset_api": obs_reset_api,
                    "obs_reset_replay": obs_reset_replay,
                    "obs_steps": obs_step_list,  # deploy-format obs, one item per env.step
                    "actions": actions_tensor,
                    "rewards": np.asarray(reward_list, dtype=np.float32),
                    "terminated": np.asarray(terminated_list, dtype=np.bool_),
                    "truncated": np.asarray(truncated_list, dtype=np.bool_),
                }

                episode_file = os.path.join(save_obs_dir, f"{episode_name}.pt")
                torch.save(payload, episode_file)
                manifest_entries.append(
                    {
                        "episode_index": int(episode_index),
                        "episode_name": episode_name,
                        "num_steps": int(len(action_list)),
                        "file": os.path.basename(episode_file),
                    }
                )
                print(f"Saved {episode_file} (steps={len(action_list)})")

        manifest = {
            "dataset_file": os.path.abspath(args_cli.dataset_file),
            "task": str(task_name),
            "num_envs": int(num_envs),
            "saved_episodes": manifest_entries,
        }
        manifest_file = os.path.join(save_obs_dir, "_manifest.json")
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest: {manifest_file}")

        env.close()
        dataset_file_handler.close()
        return

    # simulate environment -- run everything in inference mode
    episode_names = list(dataset_file_handler.get_episode_names())
    replayed_episode_count = 0
    recorded_episode_count = 0

    # Track current episode indices for each environment
    current_episode_indices = [None] * num_envs

    # Track failed demo IDs
    failed_demo_ids = []

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
            first_loop = True
            has_next_action = True
            episode_ended = [False] * num_envs
            while has_next_action:
                # initialize actions with idle action so those without next action will not move
                actions = idle_action.clone()
                has_next_action = False
                for env_id in range(num_envs):
                    env_next_action = env_episode_data_map[env_id].get_next_action()
                    if env_next_action is None:
                        # check if the episode is successful after the whole episode_data is
                        if (
                            (success_term is not None)
                            and (current_episode_indices[env_id]) is not None
                            and (not episode_ended[env_id])
                        ):
                            if bool(success_term.func(env, **success_term.params)[env_id]):
                                recorded_episode_count += 1
                                plural_trailing_s = "s" if recorded_episode_count > 1 else ""

                                print(
                                    f"Successfully replayed {recorded_episode_count} episode{plural_trailing_s} out"
                                    f" of {replayed_episode_count} demos."
                                )
                            else:
                                # if not successful, add to failed demo IDs list
                                if (
                                    current_episode_indices[env_id] is not None
                                    and current_episode_indices[env_id] not in failed_demo_ids
                                ):
                                    failed_demo_ids.append(current_episode_indices[env_id])

                            episode_ended[env_id] = True

                        next_episode_index = None
                        while episode_indices_to_replay:
                            next_episode_index = episode_indices_to_replay.pop(0)

                            if next_episode_index < episode_count:
                                episode_ended[env_id] = False
                                break
                            next_episode_index = None

                        if next_episode_index is not None:
                            replayed_episode_count += 1
                            current_episode_indices[env_id] = next_episode_index
                            print(f"{replayed_episode_count :4}: Loading #{next_episode_index} episode to env_{env_id}")
                            episode_data = dataset_file_handler.load_episode(
                                episode_names[next_episode_index], env.device
                            )
                            env_episode_data_map[env_id] = episode_data
                            # Set initial state for the new episode
                            initial_state = episode_data.get_initial_state()
                            env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)
                            
                            # Restore goal pose from dataset if available
                            if "goal_pose" in episode_data.data:
                                goal_pose = episode_data.data["goal_pose"]
                                if isinstance(goal_pose, torch.Tensor):
                                    # If goal_pose is a tensor, use it directly
                                    goal_pose_tensor = goal_pose.to(env.device)
                                elif isinstance(goal_pose, list) and len(goal_pose) > 0:
                                    # If it's a list (from recorder), take the first element
                                    goal_pose_tensor = goal_pose[0].to(env.device)
                                else:
                                    goal_pose_tensor = None
                                
                                if goal_pose_tensor is not None and hasattr(env, "command_manager"):
                                    try:
                                        # Get the command term and set the goal pose
                                        command_term = env.command_manager.get_term("object_pose")
                                        if hasattr(command_term, "pose_command_b"):
                                            # Set the goal pose command (shape should be (num_envs, 7))
                                            if goal_pose_tensor.dim() == 1:
                                                goal_pose_tensor = goal_pose_tensor.unsqueeze(0)
                                            command_term.pose_command_b[env_id] = goal_pose_tensor[0]
                                            
                                            # Update world frame command for visualization
                                            # The command term updates pose_command_w from pose_command_b
                                            if hasattr(command_term, "_update_metrics"):
                                                command_term._update_metrics()
                                            
                                            # Visualize the recorded goal pose using world frame coordinates
                                            if hasattr(command_term, "pose_command_w") and command_term.pose_command_w is not None:
                                                # Use world frame command (already in world coordinates)
                                                goal_pos_w = command_term.pose_command_w[env_id, :3].cpu().numpy()
                                                goal_quat_w = command_term.pose_command_w[env_id, 3:7].cpu().numpy()
                                            else:
                                                # Fallback: approximate transformation from base to world frame
                                                robot = env.scene["robot"]
                                                root_pos_w = robot.data.root_pos_w[env_id].cpu().numpy()
                                                goal_pos_b = goal_pose_tensor[0, :3].cpu().numpy()
                                                goal_quat_b = goal_pose_tensor[0, 3:7].cpu().numpy()
                                                
                                                # Simple transformation: add root position (approximate)
                                                goal_pos_w = root_pos_w + goal_pos_b
                                                goal_quat_w = goal_quat_b
                                            
                                            # Update marker visualization with yellow sphere at goal position
                                            recorded_goal_marker.set_visibility(True)
                                            recorded_goal_marker.visualize(
                                                translations=np.array([goal_pos_w]),
                                                orientations=np.array([goal_quat_w])
                                            )
                                    except (AttributeError, KeyError) as e:
                                        print(f"Warning: Could not restore goal pose: {e}")
                                        recorded_goal_marker.set_visibility(False)
                                else:
                                    # Hide marker if no goal pose available
                                    recorded_goal_marker.set_visibility(False)
                            else:
                                # Hide marker if no goal pose in dataset
                                recorded_goal_marker.set_visibility(False)
                            
                            # Get the first action for the new episode
                            env_next_action = env_episode_data_map[env_id].get_next_action()
                            has_next_action = True
                        else:
                            continue
                    else:
                        has_next_action = True
                    actions[env_id] = env_next_action
                if first_loop:
                    first_loop = False
                else:
                    while is_paused:
                        # Only render if not in headless mode
                        if not args_cli.headless:
                            env.sim.render()
                        continue
                env.step(actions)

                if state_validation_enabled:
                    state_from_dataset = env_episode_data_map[0].get_next_state()
                    if state_from_dataset is not None:
                        print(
                            f"Validating states at action-index: {env_episode_data_map[0].next_state_index - 1 :4}",
                            end="",
                        )
                        current_runtime_state = env.scene.get_state(is_relative=True)
                        states_matched, comparison_log = compare_states(state_from_dataset, current_runtime_state, 0)
                        if states_matched:
                            print("\t- matched.")
                        else:
                            print("\t- mismatched.")
                            print(comparison_log)
            break
    # Close environment after replay in complete
    plural_trailing_s = "s" if replayed_episode_count > 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.")

    # Print success statistics only if validation was enabled
    if success_term is not None:
        print(f"Successfully replayed: {recorded_episode_count}/{replayed_episode_count}")

        # Print failed demo IDs if any
        if failed_demo_ids:
            print(f"\nFailed demo IDs ({len(failed_demo_ids)} total):")
            print(f"  {sorted(failed_demo_ids)}")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
