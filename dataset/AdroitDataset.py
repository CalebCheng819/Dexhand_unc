import os
import sys
import time

import hydra
import minari
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model
from utils.action_utils import convert_q
from utils.action_utils import temporal_sign_align

rewards_threshold = {
    "D4RL/door/expert-v2": 10,
    "D4RL/hammer/expert-v2": 75,
    "D4RL/pen/expert-v2": 40,
    "D4RL/relocate/expert-v2": 20,
}


class AdroitDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        num_demos: int,
        action_type: str,
        obs_horizon: int,
        pred_horizon: int,
        is_train: bool,
    ):
        self.dataset_name = dataset_name
        self.is_train = is_train
        self.action_type = action_type
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        start_time = time.time()

        #原来：dataset = minari.load_dataset(dataset_name)
        # 改成带兜底的版本：
        try:
            dataset = minari.load_dataset(dataset_name)
        except FileNotFoundError:
            dataset = minari.load_dataset(dataset_name, download=True)  # 自动下载
        dataset = dataset.filter_episodes(lambda episode: episode.rewards.mean() > rewards_threshold[dataset_name])
        self.num_demos = max(num_demos, dataset.total_episodes)
        episodes_origin = dataset.sample_episodes(num_demos)
        self.episodes = []
        self.slices = []

        hand_model = create_hand_model('shadowhand', device='cpu')
        total_transitions = 0
        for episode_idx, episode in tqdm(enumerate(episodes_origin), total=len(episodes_origin), desc="Processing episodes"):
            success_idx_list = np.where(episode.infos['success'])[0]
            assert len(success_idx_list), f"This episode should be filtered out! Reward: {episode.rewards.mean():.2f}"
            episode_len = success_idx_list[0] + 1  # truncate the episode at the first success

            actions = episode.actions[:episode_len].astype(np.float32)
            if self.action_type != 'joint_value':
                q = hand_model.q_control2real(torch.from_numpy(actions[:, -24:]))
                q = convert_q(hand_model, q, output_q_type=self.action_type).numpy()
                # 加在这里（仅当 rot_vec 时）：
                if self.action_type == 'rot_vec':
                    K = 3
                    q_t = torch.from_numpy(q).view(-1, 24, K)    # (T, 24, 3)
                    q_t = temporal_sign_align(q_t)               # 时序对齐
                    q   = q_t.view(-1, 24*K).numpy()                
                actions = np.concatenate([actions[:, :-24], q], axis=-1)

            episode_truncated = {
                'id': episode.id,
                'total_steps': episode_len,
                'observations': episode.observations[:episode_len].astype(np.float32),
                'actions': actions,
                'rewards': episode.rewards[:episode_len].astype(np.float32),
                'terminations': episode.terminations[:episode_len],  # ndarray, dtype bool
                'truncations': episode.truncations[:episode_len],  # ndarray, dtype bool
                'infos': {
                    'success': episode.infos['success'][:episode_len],  # ndarray, dtype bool
                }
            }
            self.episodes.append(episode_truncated)
            total_transitions += episode_len

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1  # pad before the trajectory, so the first action of an episode is in "actions executed"
            pad_after = pred_horizon - obs_horizon  # pad after the trajectory, so all the observations are utilized in training
            self.slices += [
                (episode_idx, start, start + pred_horizon)
                for start in range(-pad_before, episode_len - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print("******************************** Dataset initialized ********************************")
        print(f"Total transitions: {total_transitions}, "
              f"Total obs sequences: {len(self.slices)}, "
              f"Initialization time: {time.time() - start_time:.2f}s")
        print("******************************** Dataset initialized ********************************")

    def __getitem__(self, index):
        episode_idx, start, end = self.slices[index]
        episode = self.episodes[episode_idx]
        episode_len, act_dim = episode['actions'].shape

        obs_seq = episode['observations'][max(0, start):start + self.obs_horizon]  # start + self.obs_horizon is at least 1
        act_seq = episode['actions'][max(0, start):end]
        if start < 0:  # pad before the trajectory
            obs_seq = np.concatenate([np.tile(obs_seq[0], (-start, 1)), obs_seq], axis=0)
            act_seq = np.concatenate([np.tile(act_seq[0], (-start, 1)), act_seq], axis=0)
        if end > episode_len:  # pad after the trajectory
            act_seq = np.concatenate([act_seq, np.tile(act_seq[-1], (end - episode_len, 1))], axis=0)
        assert obs_seq.shape[0] == self.obs_horizon, f"obs_seq.shape[0] ({obs_seq.shape[0]} != obs_horizon ({self.obs_horizon}))"
        assert act_seq.shape[0] == self.pred_horizon, f"act_seq.shape[0] ({act_seq.shape[0]} != pred_horizon ({self.pred_horizon}))"
        assert len(obs_seq.shape) == len(act_seq.shape) == 2
        return {
            'observations': torch.from_numpy(obs_seq),
            'actions': torch.from_numpy(act_seq),
        }

    def __len__(self):
        return len(self.slices)


class ValDataset(Dataset):
    def __init__(self, eval_episodes):
        super().__init__()
        self.eval_episodes = eval_episodes

    def __getitem__(self, index):
        return {"dummy": 0}

    def __len__(self):
        return self.eval_episodes


def create_dataloader(cfg, is_train):
    train_dataset = AdroitDataset(
        dataset_name=cfg.dataset_name,
        num_demos=cfg.num_demos,
        action_type=cfg.action_type,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        is_train=is_train,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=cfg.drop_last if is_train else False,
        shuffle=is_train
    )
    val_dataset = ValDataset(eval_episodes=cfg.eval_episodes)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    np.random.seed(0)
    train_dataset = AdroitDataset(
        dataset_name='D4RL/door/expert-v2',
        num_demos=1,
        action_type='rot_vec',
        obs_horizon=2,
        pred_horizon=16,
        is_train=True,
    )