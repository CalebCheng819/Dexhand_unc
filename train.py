import os
import sys
import hydra
import shutil
import warnings
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from dataset.AdroitDataset import create_dataloader
from model.pl_module import TrainingModule
from utils.action_utils import ROT_DIMS


@hydra.main(version_base="1.2", config_path="configs", config_name="train")
def main(cfg):
    cfg.dataset.act_dim += 24 * (ROT_DIMS[cfg.dataset.action_type] - 1)
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    pl.seed_everything(cfg.seed)

    # Copy code and related files to the output folder for reproducibility
    #权限不足，进行修改
    # os.makedirs(cfg.file_dir, exist_ok=True)
    # ignore_dirs = ['third_party', 'output', 'outputs', 'env_video']
    
    # for file_or_dir in os.listdir(ROOT_DIR):
    #     if file_or_dir in ignore_dirs:
    #         continue
    #     source_path = os.path.join(ROOT_DIR, file_or_dir)
    #     target_path = os.path.join(cfg.file_dir, file_or_dir)

    #     if os.path.isfile(source_path):
    #         shutil.copy2(source_path, target_path)
    #     elif os.path.isdir(source_path):
    #         shutil.copytree(source_path, target_path, dirs_exist_ok=True)
    # ---------------- Safe copy code snapshot ----------------
    os.makedirs(cfg.file_dir, exist_ok=True)

    # 顶层忽略的目录（包括 .git）
    skip_top = {'.git', 'third_party', 'output', 'outputs', 'env_video'}

    # 定义一个安全的忽略函数，递归忽略 __pycache__、.git 等
    def ignore_func(src, names):
        ignore_set = {'__pycache__', '.mypy_cache', '.pytest_cache'}
        # 如果递归进入到 .git 目录，直接忽略全部
        if os.path.basename(src) == '.git':
            return set(names)
        # 忽略 .pyc / .pyo 文件
        ignore_set.update({n for n in names if n.endswith(('.pyc', '.pyo'))})
        return ignore_set

    # 顶层遍历
    for name in os.listdir(ROOT_DIR):
        if name in skip_top:
            continue  # 不复制这些目录

        src = os.path.join(ROOT_DIR, name)
        dst = os.path.join(cfg.file_dir, name)

        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore_func)
    # ----------------------------------------------------------
    logger = WandbLogger(
        name=cfg.name,
        save_dir=cfg.wandb.save_dir,
        project=cfg.wandb.project
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.save_dir,
        filename="{epoch}",
        every_n_epochs=cfg.training.save_every_n_epoch,
        save_top_k=-1,
        save_last=True,
        monitor=None,
        save_weights_only=False,
    )
    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        strategy='auto',
        devices=cfg.gpu,
        callbacks=[checkpoint_callback],
        log_every_n_steps=cfg.log_every_n_steps,
        max_epochs=cfg.training.total_iters,
        check_val_every_n_epoch=cfg.training.val_every_n_epoch,
    )

    train_dataloader, val_dataloader = create_dataloader(cfg.dataset, is_train=True,cfg_train=cfg)

    training_module = TrainingModule(cfg=cfg)
    training_module.train()

    trainer.fit(
        training_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy("file_system")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()
