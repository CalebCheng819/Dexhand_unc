"""Relocate task configurations for table top manipulation."""

import gymnasium as gym


gym.register(
    id="DexbenchLite-Relocate-FloatingShadowRight-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.floating_shadow_right_cfg:RelocateEnvFloatingShadowRightCfg",
        # RL config entry points can be added here if needed for training
        # "rl_games_cfg_entry_point": "...",
        # "rsl_rl_cfg_entry_point": "...",
        # "skrl_cfg_entry_point": "...",
    },
)
