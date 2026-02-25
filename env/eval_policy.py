import os
import sys
import time
import torch
from hydra import initialize, compose
from omegaconf import OmegaConf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from model.pl_module import TrainingModule
from env.adroit_env import AdroitEnvWrapper
from utils.action_utils import ROT_DIMS


def eval_policy(cfg, eval_times, device, ckpt_name, ckpt_state_dict):
    cfg.dataset.act_dim += 24 * (ROT_DIMS[cfg.dataset.action_type] - 1)
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    env = AdroitEnvWrapper(
        env_name=cfg.dataset.env_name,
        action_type=cfg.dataset.action_type,
        obs_horizon=cfg.model.obs_horizon,
        act_horizon=cfg.model.act_horizon,
        device=device,
        render_mode="rgb_array",
        action_mode=cfg.training.action_mode,  # 新增

    )

    module = TrainingModule.load_from_checkpoint(
        checkpoint_path=f"{ROOT_DIR}/output/{ckpt_name}/state_dict/{ckpt_state_dict}.ckpt",
        map_location=device,
        cfg=cfg,
    )
    module.eval()

    succ_count = 0
    for idx in range(eval_times):
        start_time = time.time()
        env.reset()
        while not env.is_done:
            obs_seq = env.get_obs_seq()
            action_seq = module.model.get_action(obs_seq)
            env.step(action_seq)
        env.save_gif(save_dir=f"{ROOT_DIR}/env_video", file_name=f"{ckpt_name}-{ckpt_state_dict}")

        if env.is_success:
            succ_count += 1
        print(f"[{idx + 1}/{eval_times}] "
              f"Env: {cfg.dataset.env_name}, "
              f"Success: {env.is_success}, "
              f"Time: {time.time() - start_time:.2f}s")

    print('=' * 64)
    print(f"[{ckpt_name}/{ckpt_state_dict}] Success Rate: {succ_count / eval_times * 100:.2f}%")
    env.close()


if __name__ == "__main__":
    eval_times = 5
    device = torch.device("cuda:4")
    ckpt_name = "25demo_door_quat_dppromax_lr-4"
    ckpt_state_dict = "epoch=49999"

    with initialize(version_base="1.2", config_path=f"../output/{ckpt_name}/log/hydra/.hydra"):
        cfg = compose(config_name="config")
        eval_policy(cfg, eval_times, device, ckpt_name, ckpt_state_dict)#修改这里
