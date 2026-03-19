import os
import sys
import warnings

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from dataset.DexBenchHDF5Dataset import create_dataloader
from model.dexbench.pl_module import DexBenchTrainingModule
from utils.dexbench_rotations import get_rot_repr_dim


@hydra.main(version_base="1.2", config_path="configs", config_name="train_dexbench")
def main(cfg):
    rot_dim = get_rot_repr_dim(str(cfg.dataset.action_type))
    cfg.dataset.rot_dim = rot_dim
    cfg.dataset.env_act_dim = int(cfg.dataset.raw_act_dim) - len(cfg.dataset.rotation_indices)
    cfg.dataset.act_dim = int(cfg.dataset.env_act_dim + rot_dim)

    print("******************************** [DexBench Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [DexBench Config] ********************************")

    pl.seed_everything(cfg.seed)

    train_loader, val_loader, train_ds, _ = create_dataloader(cfg.dataset)
    cfg.dataset.obs_dim = int(train_ds.raw_obs_dim)

    logger = WandbLogger(
        name=cfg.name,
        save_dir=cfg.wandb.save_dir,
        project=cfg.wandb.project,
        offline=str(getattr(cfg.wandb, "mode", "online")) == "offline",
    )
    ckpt = ModelCheckpoint(
        dirpath=cfg.training.save_dir,
        filename="{epoch}",
        every_n_epochs=cfg.training.save_every_n_epoch,
        save_top_k=-1,
        save_last=True,
    )

    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",
        strategy="auto",
        devices=cfg.gpu,
        callbacks=[ckpt],
        log_every_n_steps=cfg.log_every_n_steps,
        max_epochs=cfg.training.total_iters,
        check_val_every_n_epoch=cfg.training.val_every_n_epoch,
    )

    module = DexBenchTrainingModule(cfg)
    if bool(cfg.training.enable_validation):
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        trainer.fit(module, train_dataloaders=train_loader)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    warnings.simplefilter(action="ignore", category=FutureWarning)
    main()
