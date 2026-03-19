from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, open_dict


class IsaacSmallEvalCallback(pl.Callback):
    """Run DRO Isaac eval during training and optionally build feedback pseudo labels."""

    metric_name = "val_isaac_success_rate_small"
    diversity_metric_name = "val_isaac_diversity_small"
    full_metric_name = "val_isaac_success_rate_full"
    full_diversity_metric_name = "val_isaac_diversity_full"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.eval_cfg = getattr(getattr(cfg, "training", None), "isaac_small_eval", None)
        self.feedback_cfg = getattr(getattr(cfg, "training", None), "isaac_feedback", None)
        self.enabled = bool(getattr(self.eval_cfg, "enable", False)) if self.eval_cfg is not None else False
        self.feedback_enabled = bool(getattr(self.feedback_cfg, "enable", False)) if self.feedback_cfg is not None else False

        self.root_dir = Path(__file__).resolve().parents[1]
        self.eval_root = Path(str(cfg.output_dir)) / "isaac_small_eval"
        self.eval_root.mkdir(parents=True, exist_ok=True)
        self.history_csv = self.eval_root / "history.csv"

        self.feedback_root = Path(str(cfg.output_dir)) / "isaac_feedback"
        self.feedback_root.mkdir(parents=True, exist_ok=True)

        self._ensure_history_header()

    def _ensure_history_header(self) -> None:
        if self.history_csv.exists():
            return
        with self.history_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "global_step",
                    "success_rate_percent",
                    "diversity_rad",
                    "checkpoint_path",
                    "export_index",
                    "summary_json",
                    "status",
                    "note",
                ],
            )
            writer.writeheader()

    def _eval_bool(self, key: str, default: bool = False) -> bool:
        if self.eval_cfg is None:
            return default
        return bool(getattr(self.eval_cfg, key, default))

    def _eval_int(self, key: str, default: int) -> int:
        if self.eval_cfg is None:
            return default
        return int(getattr(self.eval_cfg, key, default))

    def _eval_str(self, key: str, default: str) -> str:
        if self.eval_cfg is None:
            return default
        return str(getattr(self.eval_cfg, key, default))

    def _feedback_int(self, key: str, default: int) -> int:
        if self.feedback_cfg is None:
            return default
        return int(getattr(self.feedback_cfg, key, default))

    def _feedback_str(self, key: str, default: str) -> str:
        if self.feedback_cfg is None:
            return default
        return str(getattr(self.feedback_cfg, key, default))

    def _feedback_objects(self) -> list[str]:
        if self.feedback_cfg is None:
            return []
        objects = getattr(self.feedback_cfg, "objects", [])
        return [str(x) for x in objects]

    def _score_weights(self) -> tuple[float, float]:
        rerank_cfg = getattr(getattr(self.cfg, "eval", None), "rerank", None)
        if rerank_cfg is None:
            return 0.7, 0.3
        w = getattr(rerank_cfg, "score_weights", [0.7, 0.3])
        if isinstance(w, (list, tuple)) and len(w) >= 2:
            return float(w[0]), float(w[1])
        return 0.7, 0.3

    def _should_run_eval(self, trainer: "pl.Trainer") -> bool:
        if not self.enabled:
            return False
        if trainer.sanity_checking:
            return False
        every_n = max(1, self._eval_int("every_n_epochs", 1))
        return ((int(trainer.current_epoch) + 1) % every_n) == 0

    def _should_run_full_eval(self, trainer: "pl.Trainer") -> bool:
        every_n = max(1, self._eval_int("full_eval_every_n_epochs", 200))
        return ((int(trainer.current_epoch) + 1) % every_n) == 0

    def _should_run_feedback(self, trainer: "pl.Trainer") -> bool:
        if not self.feedback_enabled:
            return False
        if trainer.sanity_checking:
            return False
        every_n = max(1, self._feedback_int("every_n_epochs", 50))
        return ((int(trainer.current_epoch) + 1) % every_n) == 0

    def _run_subprocess(self, cmd: list[str], log_path: Path) -> None:
        with log_path.open("w", encoding="utf-8") as f:
            subprocess.run(
                cmd,
                cwd=str(self.root_dir),
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def _build_runtime_cfg(self, use_ema_for_eval: bool) -> Path:
        hydra_cfg = Path(str(self.cfg.output_dir)) / "log" / "hydra" / ".hydra" / "config.yaml"
        if hydra_cfg.exists():
            runtime_cfg = OmegaConf.load(str(hydra_cfg))
        else:
            runtime_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))

        with open_dict(runtime_cfg):
            if "training" not in runtime_cfg or runtime_cfg.training is None:
                runtime_cfg.training = OmegaConf.create({})
            runtime_cfg.training.use_ema_for_eval = bool(use_ema_for_eval)

        out_path = self.eval_root / "runtime_train_cfg.yaml"
        OmegaConf.save(runtime_cfg, str(out_path))
        return out_path

    def _broadcast_scalar(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", value: float) -> float:
        world_size = int(getattr(trainer, "world_size", 1) or 1)
        if world_size <= 1:
            return value
        t = torch.tensor(value, dtype=torch.float32, device=pl_module.device)
        try:
            t = trainer.strategy.broadcast(t, src=0)
        except Exception:
            pass
        return float(t.item())

    def _record_history(self, row: dict[str, Any]) -> None:
        with self.history_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "global_step",
                    "success_rate_percent",
                    "diversity_rad",
                    "checkpoint_path",
                    "export_index",
                    "summary_json",
                    "status",
                    "note",
                ],
            )
            writer.writerow(row)

    def _build_export_cmd(
        self,
        runtime_cfg_path: Path,
        checkpoint_path: str,
        output_dir: Path,
        inference_seed: int,
    ) -> list[str]:
        export_python = self._eval_str("export_python", sys.executable)
        action_type = str(self.cfg.dataset.action_type)
        export_device = self._eval_str("export_device", "cpu")
        split = self._eval_str("split", "validate")
        max_batches = self._eval_int("max_batches", -1)
        max_samples = self._eval_int("max_samples", -1)

        export_cmd = [
            export_python,
            str(self.root_dir / "evaluation" / "export_dexgrasp_final_q.py"),
            "--checkpoint_path",
            checkpoint_path,
            "--train_cfg_path",
            str(runtime_cfg_path),
            "--output_dir",
            str(output_dir),
            "--device",
            export_device,
            "--split",
            split,
            "--max_batches",
            str(max_batches),
            "--max_samples",
            str(max_samples),
            "--action_type",
            action_type,
            "--inference_seed",
            str(inference_seed),
        ]
        if self._eval_bool("disable_cache", True):
            export_cmd.append("--disable_cache")
        if self._eval_bool("stochastic", False):
            export_cmd.append("--stochastic")

        rerank_cfg = getattr(getattr(self.cfg, "eval", None), "rerank", None)
        rerank_enable = bool(getattr(rerank_cfg, "enable", False)) if rerank_cfg is not None else False
        if rerank_enable:
            num_candidates = int(getattr(rerank_cfg, "num_candidates", 8))
            w0, w1 = self._score_weights()
            export_cmd.extend(
                [
                    "--rerank_enable",
                    "--num_candidates",
                    str(num_candidates),
                    "--score_weights",
                    str(w0),
                    str(w1),
                ]
            )
        return export_cmd

    def _run_isaac_eval(
        self,
        export_index: str,
        output_dir: Path,
        log_path: Path,
        max_groups: int,
    ) -> tuple[float, float, str]:
        isaac_python = self._eval_str("isaac_python", sys.executable)
        summary_json = str(output_dir / "isaac_summary.json")
        isaac_cmd = [
            isaac_python,
            str(self.root_dir / "DRO-Grasp" / "scripts" / "eval_exported_final_q_isaac.py"),
            "--export_index",
            export_index,
            "--gpu",
            str(self._eval_int("isaac_gpu", 0)),
            "--chunk_size",
            str(self._eval_int("chunk_size", 64)),
            "--output_dir",
            str(output_dir),
            "--max_groups",
            str(max_groups),
            "--q_dof_mismatch",
            self._eval_str("q_dof_mismatch", "tail"),
        ]
        self._run_subprocess(isaac_cmd, log_path)

        if not os.path.exists(summary_json):
            raise FileNotFoundError(f"Isaac summary missing: {summary_json}")
        with open(summary_json, "r", encoding="utf-8") as f:
            summary = json.load(f)
        success_rate = float(summary.get("success_rate_percent", -1.0))
        diversity = float(summary.get("diversity_rad", 0.0))
        return success_rate, diversity, summary_json

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._should_run_eval(trainer):
            return

        rank = int(getattr(trainer, "global_rank", 0) or 0)
        epoch = int(trainer.current_epoch)
        global_step = int(trainer.global_step)

        success_rate_small = -1.0
        diversity_small = 0.0
        success_rate_full = -1.0
        diversity_full = 0.0
        status = "ok"
        note = ""
        export_index = ""
        summary_json_small = ""
        ckpt_path = str(self.eval_root / f"tmp_epoch_{epoch:04d}.ckpt")

        if rank == 0:
            eval_epoch_dir = self.eval_root / f"epoch_{epoch:04d}"
            export_dir = eval_epoch_dir / "export"
            isaac_small_dir = eval_epoch_dir / "isaac_small"
            isaac_full_dir = eval_epoch_dir / "isaac_full"
            export_dir.mkdir(parents=True, exist_ok=True)
            isaac_small_dir.mkdir(parents=True, exist_ok=True)
            isaac_full_dir.mkdir(parents=True, exist_ok=True)
            export_log = eval_epoch_dir / "export.log"
            isaac_small_log = eval_epoch_dir / "isaac_small.log"
            isaac_full_log = eval_epoch_dir / "isaac_full.log"

            export_index = str(export_dir / "export_index.json")
            try:
                trainer.save_checkpoint(ckpt_path)
                runtime_cfg_path = self._build_runtime_cfg(use_ema_for_eval=self._eval_bool("use_ema_for_eval", True))
                inference_seed = self._eval_int("inference_seed", int(getattr(self.cfg, "seed", 2025)))

                export_cmd = self._build_export_cmd(
                    runtime_cfg_path=runtime_cfg_path,
                    checkpoint_path=ckpt_path,
                    output_dir=export_dir,
                    inference_seed=inference_seed,
                )
                self._run_subprocess(export_cmd, export_log)

                success_rate_small, diversity_small, summary_json_small = self._run_isaac_eval(
                    export_index=export_index,
                    output_dir=isaac_small_dir,
                    log_path=isaac_small_log,
                    max_groups=self._eval_int("max_groups", 3),
                )
                self._record_history(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "success_rate_percent": success_rate_small,
                        "diversity_rad": diversity_small,
                        "checkpoint_path": ckpt_path,
                        "export_index": export_index,
                        "summary_json": summary_json_small,
                        "status": "ok",
                        "note": "scope=small",
                    }
                )

                if self._should_run_full_eval(trainer):
                    success_rate_full, diversity_full, summary_json_full = self._run_isaac_eval(
                        export_index=export_index,
                        output_dir=isaac_full_dir,
                        log_path=isaac_full_log,
                        max_groups=self._eval_int("full_max_groups", 10),
                    )
                    self._record_history(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "success_rate_percent": success_rate_full,
                            "diversity_rad": diversity_full,
                            "checkpoint_path": ckpt_path,
                            "export_index": export_index,
                            "summary_json": summary_json_full,
                            "status": "ok",
                            "note": "scope=full",
                        }
                    )
            except Exception as e:
                status = "fail"
                note = str(e)
                success_rate_small = -1.0
                diversity_small = 0.0
                self._record_history(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "success_rate_percent": success_rate_small,
                        "diversity_rad": diversity_small,
                        "checkpoint_path": ckpt_path,
                        "export_index": export_index,
                        "summary_json": summary_json_small,
                        "status": status,
                        "note": note,
                    }
                )
            finally:
                try:
                    if os.path.exists(ckpt_path):
                        os.remove(ckpt_path)
                except OSError:
                    pass

        success_rate_small = self._broadcast_scalar(trainer, pl_module, success_rate_small)
        diversity_small = self._broadcast_scalar(trainer, pl_module, diversity_small)

        success_small_t = torch.tensor(success_rate_small, dtype=torch.float32, device=pl_module.device)
        diversity_small_t = torch.tensor(diversity_small, dtype=torch.float32, device=pl_module.device)
        trainer.callback_metrics[self.metric_name] = success_small_t
        trainer.callback_metrics[self.diversity_metric_name] = diversity_small_t

        if self._should_run_full_eval(trainer):
            success_rate_full = self._broadcast_scalar(trainer, pl_module, success_rate_full)
            diversity_full = self._broadcast_scalar(trainer, pl_module, diversity_full)
            success_full_t = torch.tensor(success_rate_full, dtype=torch.float32, device=pl_module.device)
            diversity_full_t = torch.tensor(diversity_full, dtype=torch.float32, device=pl_module.device)
            trainer.callback_metrics[self.full_metric_name] = success_full_t
            trainer.callback_metrics[self.full_diversity_metric_name] = diversity_full_t
        else:
            success_full_t = None
            diversity_full_t = None

        if trainer.logger is not None:
            log_payload = {
                self.metric_name: float(success_rate_small),
                self.diversity_metric_name: float(diversity_small),
            }
            if success_full_t is not None and diversity_full_t is not None:
                log_payload[self.full_metric_name] = float(success_full_t.item())
                log_payload[self.full_diversity_metric_name] = float(diversity_full_t.item())
            trainer.logger.log_metrics(log_payload, step=global_step)

        try:
            pl_module.log(self.metric_name, success_small_t, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            pl_module.log(self.diversity_metric_name, diversity_small_t, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            if success_full_t is not None and diversity_full_t is not None:
                pl_module.log(self.full_metric_name, success_full_t, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
                pl_module.log(self.full_diversity_metric_name, diversity_full_t, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        except Exception:
            pass

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._should_run_feedback(trainer):
            return
        rank = int(getattr(trainer, "global_rank", 0) or 0)
        if rank != 0:
            return

        epoch = int(trainer.current_epoch)
        feedback_epoch_dir = self.feedback_root / f"epoch_{epoch:04d}"
        export_dir = feedback_epoch_dir / "export"
        isaac_dir = feedback_epoch_dir / "isaac"
        export_dir.mkdir(parents=True, exist_ok=True)
        isaac_dir.mkdir(parents=True, exist_ok=True)

        export_log = feedback_epoch_dir / "export.log"
        isaac_log = feedback_epoch_dir / "isaac.log"
        ckpt_path = str(self.feedback_root / f"tmp_feedback_epoch_{epoch:04d}.ckpt")

        try:
            trainer.save_checkpoint(ckpt_path)
            runtime_cfg_path = self._build_runtime_cfg(use_ema_for_eval=True)

            export_python = self._eval_str("export_python", sys.executable)
            isaac_python = self._eval_str("isaac_python", sys.executable)
            action_type = str(self.cfg.dataset.action_type)
            inference_seed = int(getattr(self.cfg, "seed", 2025)) + epoch

            export_cmd = [
                export_python,
                str(self.root_dir / "evaluation" / "export_dexgrasp_feedback_candidates.py"),
                "--checkpoint_path",
                ckpt_path,
                "--train_cfg_path",
                str(runtime_cfg_path),
                "--output_dir",
                str(export_dir),
                "--device",
                self._feedback_str("export_device", self._eval_str("export_device", "cpu")),
                "--split",
                self._feedback_str("split", "train"),
                "--candidates_k",
                str(self._feedback_int("candidates_k", 8)),
                "--max_samples_per_object",
                str(self._feedback_int("max_samples_per_object", 256)),
                "--action_type",
                action_type,
                "--inference_seed",
                str(inference_seed),
            ]
            hard_objects = self._feedback_objects()
            if hard_objects:
                export_cmd.extend(["--objects", *hard_objects])

            self._run_subprocess(export_cmd, export_log)

            score_w0, score_w1 = self._score_weights()
            cache_dir = self._feedback_str("cache_dir", str(Path(str(self.cfg.output_dir)) / "isaac_feedback_cache"))
            isaac_cmd = [
                isaac_python,
                str(self.root_dir / "DRO-Grasp" / "scripts" / "build_feedback_cache_from_export.py"),
                "--export_index",
                str(export_dir / "export_index.json"),
                "--gpu",
                str(self._eval_int("isaac_gpu", 0)),
                "--chunk_size",
                str(self._eval_int("chunk_size", 64)),
                "--output_dir",
                str(isaac_dir),
                "--cache_dir",
                cache_dir,
                "--q_dof_mismatch",
                self._feedback_str("q_dof_mismatch", self._eval_str("q_dof_mismatch", "tail")),
                "--score_weights",
                str(score_w0),
                str(score_w1),
            ]
            self._run_subprocess(isaac_cmd, isaac_log)
        except Exception as e:
            status_path = feedback_epoch_dir / "status.json"
            with status_path.open("w", encoding="utf-8") as f:
                json.dump({"status": "fail", "error": str(e)}, f, indent=2)
        finally:
            try:
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
            except OSError:
                pass
