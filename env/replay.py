import os
import sys
import imageio
import numpy as np
import torch
import minari
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model
from utils.action_utils import convert_q, projection_q
from dataset.AdroitDataset import AdroitDataset


if __name__ == "__main__":
    dataset_name = "D4RL/door/expert-v2"
    dataset = minari.load_dataset(dataset_name)
    env = dataset.recover_environment(max_episode_steps=200, render_mode="rgb_array")

    action_type = 'rot_quat'
    dataset = AdroitDataset(
        dataset_name="D4RL/door/expert-v2",
        num_demos=10,
        action_type=action_type,
        obs_horizon=2,
        pred_horizon=16,
        is_train=False,
    )
    # print(dataset.episodes[0]["actions"][:, -4:])
    # print(np.linalg.norm(dataset.episodes[0]["actions"][:, -4:], axis=-1))

    episode = dataset.episodes[0]
    success_idx_list = np.where(episode['infos']['success'])[0]
    episode_len = success_idx_list[0] + 1
    observations = episode['observations'][:episode_len].astype(np.float32)
    actions = episode['actions'][:episode_len].astype(np.float32)

    pos_delta_frame2handle = np.array([0.29, -0.15, -0.025])
    init_state_door = observations[0, 32:35] - pos_delta_frame2handle

    hand_model = create_hand_model('shadowhand', device='cpu')

    obs, info = env.reset(options={
        "initial_state_dict": {
            "qpos": np.zeros(30),
            "qvel": np.zeros(30),
            "door_body_pos": init_state_door
        }
    })
    frames = [env.render()]
    error = 0

    actions = torch.from_numpy(actions)
    for idx, action in enumerate(actions):
        # original_action = action.copy()  ##
        env_action = action[:-96]
        q = action[-96:]
        new_q = projection_q(hand_model, q, input_q_type=action_type)
        new_q = hand_model.q_real2control(new_q)
        joint_values = new_q[0].numpy()
        action = np.concatenate([env_action, joint_values], axis=-1)

        episode_obs = observations[idx]
        # action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # print(obs, reward, terminated, truncated, info)
        # print(reward, episode.rewards[idx], np.abs(reward - episode.rewards[idx]))  ##
        error += abs(reward - episode['rewards'][idx])

        frame = env.render()
        frames.append(frame)
    print("Error:", error)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"door_{timestamp}.gif"
    os.makedirs(f"{ROOT_DIR}/env_video/replay", exist_ok=True)
    imageio.mimsave(f"{ROOT_DIR}/env_video/replay/{video_filename}", frames)
    print(f"Video saved as {video_filename}")

    env.close()

    # q = torch.tensor(actions[:, -24:])
    # q = hand_model.q_control2real(q)
    # convert_q_value = convert_q(hand_model, q, output_q_type=action_type).numpy()
    # num_dims = convert_q_value.shape[1]  # Number of dimensions in convert_qq
    #
    # # Plot every 10 dimensions in a separate plot
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # os.makedirs(f"{ROOT_DIR}/env_video/replay", exist_ok=True)
    #
    # chunk_size = 4
    # for start_dim in range(0, num_dims, chunk_size):
    #     end_dim = min(start_dim + chunk_size, num_dims)
    #     plt.figure(figsize=(12, 8))
    #     for dim in range(start_dim, end_dim):
    #         plt.plot(convert_q_value[:, dim], label=f'Dimension {dim + 1}')
    #     plt.title(f'Trends of q: Dimensions {start_dim + 1}-{end_dim}')
    #     plt.xlabel('Time Step')
    #     plt.ylabel('Value')
    #     plt.legend()
    #     plt.grid(True)
    #
    #     # Save the plot
    #     plot_filename = f"{action_type}_{timestamp}_dims_{start_dim + 1}-{end_dim}.png"
    #     plt.savefig(f"{ROOT_DIR}/env_video/replay/{plot_filename}")
    #     plt.close()
    #     print(f"Plot saved as {plot_filename}")
    # env.close()
    # exit()


    # for idx, action in enumerate(actions):
    #     original_action = action.copy()  ##
    #     env_action = action[:-24]
    #     q = torch.tensor(action[-24:])
    #     qq = hand_model.q_control2real(q)
    #     convert_qq = convert_q(hand_model, qq, output_q_type=action_type, prev_q=prev_q)
    #     prev_q = convert_qq.numpy()
    #
    #     new_qq = projection_q(hand_model, convert_qq, input_q_type=action_type)
    #     new_q = hand_model.q_real2control(new_qq)
    #     joint_values = new_q[0].numpy()
    #     action = np.concatenate([env_action, joint_values], axis=-1)
    #
    #     episode_obs = observations[idx]
    #     # action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     # print(obs, reward, terminated, truncated, info)
    #     # print(reward, episode.rewards[idx], np.abs(reward - episode.rewards[idx]))  ##
    #     error += abs(reward - episode['rewards'][idx])
    #
    #     frame = env.render()
    #     frames.append(frame)
    # print("Error:", error)
    #
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # video_filename = f"door_{timestamp}.gif"
    # os.makedirs(f"{ROOT_DIR}/env_video/replay", exist_ok=True)
    # imageio.mimsave(f"{ROOT_DIR}/env_video/replay/{video_filename}", frames)
    # print(f"Video saved as {video_filename}")
    #
    # env.close()
