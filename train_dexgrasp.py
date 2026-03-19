import os
import sys
import warnings

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from dataset.DexGraspDataset import create_dataloader
from model.isaac_small_eval_callback import IsaacSmallEvalCallback
from model.pl_module import TrainingModule
from utils.action_schema import compute_action_schema


def _normalize_optional_path(path_like):
    if path_like is None:
        return None
    path_str = str(path_like).strip()
    if path_str == "" or path_str.lower() in {"none", "null"}:
        return None
    return path_str


@hydra.main(version_base="1.2", config_path="configs", config_name="train_dexgrasp")
def main(cfg):
    schema = compute_action_schema(
        env_act_dim=int(cfg.dataset.env_act_dim),
        num_joints=int(cfg.dataset.num_joints),
        action_type=str(cfg.dataset.action_type),
    )
    cfg.dataset.rot_dim = schema.rot_dim
    cfg.dataset.rot_act_dim = schema.rot_act_dim
    cfg.dataset.act_dim = schema.act_dim

    print("******************************** [DexGrasp Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [DexGrasp Config] ********************************")
    print(
        f"[ActionSchema] env_act_dim={schema.env_act_dim}, rot_dim={schema.rot_dim}, "
        f"rot_act_dim={schema.rot_act_dim}, act_dim={schema.act_dim}"
    )

    pl.seed_everything(cfg.seed)

    wandb_enabled = bool(getattr(cfg.wandb, "enable", True))
    wandb_mode = str(getattr(cfg.wandb, "mode", "online"))
    env_wandb_mode = os.getenv("WANDB_MODE")
    if env_wandb_mode:
        wandb_mode = env_wandb_mode
    if wandb_mode.lower() == "disabled":
        wandb_enabled = False

    if wandb_enabled:
        logger = WandbLogger(
            name=cfg.name,
            save_dir=cfg.wandb.save_dir,
            project=cfg.wandb.project,
            mode=wandb_mode,
        )
    else:
        logger = CSVLogger(
            save_dir=cfg.log_dir,
            name="csv",
        )
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.save_dir,
        filename="{epoch}",
        every_n_epochs=cfg.training.save_every_n_epoch,
        save_top_k=-1,
        save_last=True,
        monitor=None,
        save_weights_only=bool(getattr(cfg.training, "save_weights_only", True)),
    )
    isaac_eval_cfg = getattr(cfg.training, "isaac_small_eval", None)
    isaac_small_eval_enabled = bool(getattr(isaac_eval_cfg, "enable", False)) if isaac_eval_cfg is not None else False
    if isaac_small_eval_enabled:
        best_monitor = "val_isaac_success_rate_small"
        best_mode = "max"
        best_filename = "best_isaac_small"
    else:
        best_monitor = "val_final_step_q_l1"
        best_mode = "min"
        best_filename = "best_val"

    best_val_callback = ModelCheckpoint(
        dirpath=cfg.training.save_dir,
        filename=best_filename,
        monitor=best_monitor,
        mode=best_mode,
        save_top_k=1,
        save_last=False,
        save_weights_only=bool(getattr(cfg.training, "save_weights_only", True)),
    )
    callbacks = []
    if isaac_small_eval_enabled:
        callbacks.append(IsaacSmallEvalCallback(cfg))
    callbacks.extend([checkpoint_callback, best_val_callback, LearningRateMonitor(logging_interval="step")])

    early_stop_cfg = getattr(cfg.training, "early_stopping", None)
    if early_stop_cfg is not None and bool(getattr(early_stop_cfg, "enable", False)):
        callbacks.append(
            EarlyStopping(
                monitor=str(getattr(early_stop_cfg, "monitor", best_monitor)),
                mode=str(getattr(early_stop_cfg, "mode", best_mode)),
                patience=int(getattr(early_stop_cfg, "patience", 12)),
                min_delta=float(getattr(early_stop_cfg, "min_delta", 1e-4)),
                check_finite=True,
            )
        )

    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",
        strategy="auto",
        devices=cfg.gpu,
        callbacks=callbacks,
        log_every_n_steps=cfg.log_every_n_steps,
        max_epochs=cfg.training.total_iters,
        check_val_every_n_epoch=cfg.training.val_every_n_epoch,
        gradient_clip_val=float(getattr(cfg.training, "gradient_clip_val", 1.0)),
        gradient_clip_algorithm=str(getattr(cfg.training, "gradient_clip_algorithm", "norm")),
    )

    train_loader, val_loader = create_dataloader(cfg.dataset, is_train=True)

    module = TrainingModule(cfg=cfg)
    module.train()

    resume_ckpt_path = _normalize_optional_path(getattr(cfg.training, "resume_ckpt_path", None))
    resume_mode = str(getattr(cfg.training, "resume_mode", "auto")).strip().lower()
    if resume_mode not in {"auto", "fit", "weights"}:
        raise ValueError(f"Unknown training.resume_mode={resume_mode}, expected one of: auto|fit|weights")

    fit_ckpt_path = None
    if resume_ckpt_path is not None:
        if not os.path.isabs(resume_ckpt_path):
            resume_ckpt_path = os.path.join(ROOT_DIR, resume_ckpt_path)
        resume_ckpt_path = os.path.abspath(os.path.expanduser(resume_ckpt_path))
        if not os.path.exists(resume_ckpt_path):
            raise FileNotFoundError(f"resume checkpoint not found: {resume_ckpt_path}")

        ckpt_data = torch.load(resume_ckpt_path, map_location="cpu")
        if not isinstance(ckpt_data, dict):
            raise RuntimeError(f"unsupported checkpoint format for resume: {type(ckpt_data)}")
        has_optimizer_states = bool(ckpt_data.get("optimizer_states"))
        has_loop_state = bool(ckpt_data.get("loops"))

        if resume_mode == "fit" or (resume_mode == "auto" and has_optimizer_states and has_loop_state):
            fit_ckpt_path = resume_ckpt_path
            print(f"[Resume] trainer.fit will restore from checkpoint: {resume_ckpt_path}")
        else:
            state_dict = ckpt_data.get("state_dict")
            if not isinstance(state_dict, dict):
                raise RuntimeError("checkpoint missing `state_dict`, cannot warm-start weights")
            missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)
            print(
                f"[Resume] Warm-started model weights from: {resume_ckpt_path} "
                f"(missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)})"
            )
            if has_loop_state and not has_optimizer_states:
                print("[Resume] Checkpoint has loop state but no optimizer state (save_weights_only), using warm-start mode.")
            print(
                f"[Resume] Reference checkpoint epoch={ckpt_data.get('epoch', 'NA')}, "
                f"global_step={ckpt_data.get('global_step', 'NA')}"
            )

    if bool(cfg.training.enable_validation):
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=fit_ckpt_path)
    else:
        trainer.fit(module, train_dataloaders=train_loader, ckpt_path=fit_ckpt_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    warnings.simplefilter(action="ignore", category=FutureWarning)
    main()
