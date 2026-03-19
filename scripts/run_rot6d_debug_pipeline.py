#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bool_str(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _list_str(values: list[Any]) -> str:
    return " ".join(str(x) for x in values)


def _run_stream(cmd: list[str], env: dict[str, str], cwd: Path) -> int:
    print(f"[CMD] {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    return proc.wait()


def _find_new_eval_run(
    export_root: Path,
    base_name: str,
    epochs: int,
    before: set[str],
    allow_fallback_latest: bool,
) -> Path | None:
    pattern = str(export_root / f"{base_name}_ep{epochs}_*")
    after = {str(Path(p).resolve()) for p in glob.glob(pattern) if Path(p).is_dir()}
    new_dirs = sorted(after - before, key=lambda p: Path(p).stat().st_mtime)
    if new_dirs:
        return Path(new_dirs[-1])
    if allow_fallback_latest and after:
        return Path(sorted(after, key=lambda p: Path(p).stat().st_mtime)[-1])
    return None


def _collect_stage_metrics(summary_tsv: Path, robust_min_grasps: int) -> dict[str, Any]:
    if not summary_tsv.exists():
        raise FileNotFoundError(f"summary.tsv not found: {summary_tsv}")

    total_rows = 0
    success_rows = 0
    robust_success_rows = 0
    best_success = float("-inf")
    best_robust_success = float("-inf")
    robust_scores: list[float] = []
    all_scores: list[float] = []
    missing_summary_json = 0

    with summary_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total_rows += 1
            if row.get("status") != "success":
                continue
            try:
                sr = float(row.get("success_rate_percent", "nan"))
            except Exception:
                continue
            if sr != sr:  # NaN check
                continue
            success_rows += 1
            all_scores.append(sr)
            best_success = max(best_success, sr)

            summary_json = row.get("summary_json", "")
            num_grasps = 0
            if summary_json and os.path.exists(summary_json):
                try:
                    with open(summary_json, "r", encoding="utf-8") as jf:
                        js = json.load(jf)
                    num_grasps = int(js.get("num_grasps_total", 0) or 0)
                except Exception:
                    num_grasps = 0
            else:
                missing_summary_json += 1

            if num_grasps >= robust_min_grasps:
                robust_success_rows += 1
                robust_scores.append(sr)
                best_robust_success = max(best_robust_success, sr)

    mean_success = sum(all_scores) / len(all_scores) if all_scores else None
    mean_robust_success = sum(robust_scores) / len(robust_scores) if robust_scores else None

    return {
        "summary_tsv": str(summary_tsv),
        "total_rows": total_rows,
        "success_rows": success_rows,
        "robust_min_grasps": int(robust_min_grasps),
        "robust_success_rows": robust_success_rows,
        "best_success_percent": None if best_success == float("-inf") else best_success,
        "best_robust_success_percent": None if best_robust_success == float("-inf") else best_robust_success,
        "mean_success_percent": mean_success,
        "mean_robust_success_percent": mean_robust_success,
        "missing_summary_json_rows": missing_summary_json,
    }


def _safe_float(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _analyze_training_run(run_dir: Path) -> dict[str, Any]:
    metrics_csv = run_dir / "log" / "csv" / "version_1" / "metrics.csv"
    out: dict[str, Any] = {
        "run_dir": str(run_dir),
        "metrics_csv": str(metrics_csv),
        "exists": metrics_csv.exists(),
    }
    if not metrics_csv.exists():
        out["error"] = "metrics_csv_missing"
        return out

    rows: list[dict[str, float]] = []
    with metrics_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mse = _safe_float(row.get("val_action_mse"))
            q_l1 = _safe_float(row.get("val_final_step_q_l1"))
            rot = _safe_float(row.get("val_rot_geodesic_deg_mean"))
            ep = _safe_float(row.get("epoch"))
            if mse is None or q_l1 is None or rot is None or ep is None:
                continue
            rows.append(
                {
                    "epoch": ep,
                    "val_action_mse": mse,
                    "val_final_step_q_l1": q_l1,
                    "val_rot_geodesic_deg_mean": rot,
                }
            )

    if not rows:
        out["error"] = "no_validation_rows"
        return out

    best_mse = min(rows, key=lambda r: r["val_action_mse"])
    last = rows[-1]
    mse_ratio = (last["val_action_mse"] / best_mse["val_action_mse"]) if best_mse["val_action_mse"] > 0 else None
    q_ratio = (
        (last["val_final_step_q_l1"] / best_mse["val_final_step_q_l1"]) if best_mse["val_final_step_q_l1"] > 0 else None
    )

    out.update(
        {
            "num_val_rows": len(rows),
            "best_by_mse": best_mse,
            "last": last,
            "ratios_final_over_best": {
                "val_action_mse": mse_ratio,
                "val_final_step_q_l1": q_ratio,
            },
            "clear_late_degradation": bool(
                (mse_ratio is not None and mse_ratio >= 1.25) or (q_ratio is not None and q_ratio >= 1.25)
            ),
        }
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage-wise rot_6d debug pipeline: train -> DRO Isaac eval -> milestone gating."
    )
    parser.add_argument(
        "--plan",
        type=str,
        default=str(ROOT_DIR / "configs" / "ablation" / "rot6d_debug_plan.json"),
        help="Path to rot_6d debug plan JSON.",
    )
    parser.add_argument(
        "--start-stage",
        type=str,
        default="",
        help="Optional stage name to start from (inclusive).",
    )
    args = parser.parse_args()

    plan_path = Path(args.plan).resolve()
    with plan_path.open("r", encoding="utf-8") as f:
        plan = json.load(f)

    runtime = plan["runtime"]
    experiment = plan["experiment"]
    stages = plan["stages"]

    action = str(experiment.get("action", "rot_6d"))
    if action != "rot_6d":
        raise ValueError(f"This pipeline is rot_6d-only, but plan action is {action!r}.")

    base_prefix = str(experiment["base_prefix"])
    common_train = dict(experiment.get("common_train", {}))

    output_root = Path(runtime["output_root"]).resolve()
    export_root = Path(runtime["export_root"]).resolve()
    robust_min_grasps = int(runtime.get("robust_min_grasps", 500))
    gpu_ids = list(runtime["gpu_ids"])
    threads_per_proc = int(runtime.get("threads_per_proc", 4))
    wandb_mode = str(runtime.get("wandb_mode", "disabled"))
    q_dof_mismatch = str(runtime.get("q_dof_mismatch", "tail"))
    ckpt_priority = str(runtime.get("ckpt_priority", "best_val.ckpt last.ckpt"))

    train_python = str(runtime["train_python"])
    export_python = str(runtime["export_python"])
    isaac_python = str(runtime["isaac_python"])
    isaac_chunk = int(runtime.get("isaac_chunk", 64))
    isaac_retries = int(runtime.get("isaac_retries", 2))

    reports_root = export_root / f"{base_prefix}_pipeline_reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    report_path = reports_root / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"

    print("================================ [Rot6D Debug Pipeline] ================================")
    print(f"plan:        {plan_path}")
    print(f"output_root: {output_root}")
    print(f"export_root: {export_root}")
    print(f"gpu_ids:     {gpu_ids}")
    print(f"robust_min_grasps: {robust_min_grasps}")
    print(f"ckpt_priority: {ckpt_priority}")

    if args.start_stage:
        start_idx = None
        for i, st in enumerate(stages):
            if st.get("name") == args.start_stage:
                start_idx = i
                break
        if start_idx is None:
            raise ValueError(f"start-stage={args.start_stage!r} not found in plan.")
        stages = stages[start_idx:]

    stage_results: list[dict[str, Any]] = []
    pipeline_ok = True

    for stage in stages:
        stage_name = str(stage["name"])
        epochs = int(stage["epochs"])
        seeds = [int(s) for s in stage["seeds"]]
        target_success = float(stage["target_success_percent"])
        stage_train = dict(common_train)
        stage_train.update(stage.get("train_overrides", {}))
        stage_base = f"{base_prefix}_{stage_name}"

        print(f"\n---------------- Stage: {stage_name} ----------------")
        print(f"epochs={epochs}, seeds={seeds}, target_success_percent={target_success}")

        train_env = os.environ.copy()
        train_env.update(
            {
                "GPU_IDS": _list_str(gpu_ids),
                "SEEDS": _list_str(seeds),
                "EPOCHS": str(epochs),
                "BASE_NAME": stage_base,
                "OUTPUT_ROOT": str(output_root),
                "THREADS_PER_PROC": str(threads_per_proc),
                "WANDB_MODE": wandb_mode,
                "PYTHON_BIN": train_python,
                "ACTION_TYPE": "rot_6d",
                "LOSS_MODE": str(stage_train.get("loss_mode", "mse")),
                "OBS_HAND_POINTS_SOURCE": str(stage_train.get("obs_hand_points_source", "full_hand")),
                "HAND_NUM_POINTS": str(stage_train.get("hand_num_points", 512)),
                "USE_CONTACT_HEATMAP": _bool_str(stage_train.get("use_contact_heatmap", False)),
                "OBJECT_ENCODER": str(stage_train.get("object_encoder", "stats")),
                "TRAJ_INTERP": str(stage_train.get("traj_interp", "linear")),
                "OBS_DIM": str(stage_train.get("obs_dim", 64)),
                "OBS_HORIZON": str(stage_train.get("obs_horizon", 2)),
                "ACT_HORIZON": str(stage_train.get("act_horizon", 8)),
                "PRED_HORIZON": str(stage_train.get("pred_horizon", 16)),
                "NUM_WORKERS": str(stage_train.get("num_workers", 0)),
                "SCHEDULER_CLIP_SAMPLE": _bool_str(stage_train.get("scheduler_clip_sample", True)),
                "JOINT_ATTN": _bool_str(stage_train.get("joint_attn", False)),
                "USE_EMA": _bool_str(stage_train.get("use_ema", False)),
                "USE_EMA_FOR_EVAL": _bool_str(stage_train.get("use_ema_for_eval", False)),
                "EARLY_STOP_ENABLE": _bool_str(stage_train.get("early_stop_enable", True)),
                "EARLY_STOP_PATIENCE": str(stage_train.get("early_stop_patience", 12)),
                "EARLY_STOP_MIN_DELTA": str(stage_train.get("early_stop_min_delta", "1e-4")),
                "GRAD_CLIP_VAL": str(stage_train.get("grad_clip_val", 1.0)),
                "VAL_EVERY": str(stage_train.get("val_every", 5)),
                "SAVE_EVERY": str(stage_train.get("save_every", 20)),
                "SAVE_WEIGHTS_ONLY": _bool_str(stage_train.get("save_weights_only", True)),
                "ENABLE_VALIDATION": _bool_str(stage_train.get("enable_validation", True)),
            }
        )

        train_cmd = ["bash", "scripts/sweep_rot6d_debug_train_multigpu.sh"]
        train_rc = _run_stream(train_cmd, env=train_env, cwd=ROOT_DIR)
        if train_rc != 0:
            print(f"[Stage {stage_name}] training failed with rc={train_rc}. Stop pipeline.")
            stage_results.append(
                {
                    "stage": stage_name,
                    "started_utc": _utc_now(),
                    "train_rc": train_rc,
                    "eval_rc": None,
                    "run_root": None,
                    "target_success_percent": target_success,
                    "passed": False,
                    "reason": "train_failed",
                }
            )
            pipeline_ok = False
            break

        train_diagnostics: list[dict[str, Any]] = []
        for seed in seeds:
            run_name = f"{stage_base}_rot_6d_seed{seed}_ep{epochs}"
            run_dir = output_root / run_name
            diag = _analyze_training_run(run_dir)
            train_diagnostics.append(diag)
            if "error" in diag:
                print(f"[Stage {stage_name}] train diagnostic missing for {run_name}: {diag['error']}")
            else:
                best = diag["best_by_mse"]
                last = diag["last"]
                ratios = diag["ratios_final_over_best"]
                print(
                    f"[Stage {stage_name}] train diagnostic {run_name}: "
                    f"best_epoch={int(best['epoch'])}, last_epoch={int(last['epoch'])}, "
                    f"mse_ratio={ratios['val_action_mse']:.3f}, q_l1_ratio={ratios['val_final_step_q_l1']:.3f}, "
                    f"late_degradation={diag['clear_late_degradation']}"
                )

        before_runs = {
            str(Path(p).resolve())
            for p in glob.glob(str(export_root / f"{stage_base}_ep{epochs}_*"))
            if Path(p).is_dir()
        }

        eval_env = os.environ.copy()
        eval_env.update(
            {
                "EXPORT_PY": export_python,
                "ISAAC_PY": isaac_python,
                "OUTPUT_ROOT": str(output_root),
                "EXPORT_ROOT": str(export_root),
                "BASE_NAME": stage_base,
                "EPOCHS": str(epochs),
                "GPU_IDS": _list_str(gpu_ids),
                "SEEDS": _list_str(seeds),
                "ACTIONS": "rot_6d",
                "ISAAC_CHUNK": str(isaac_chunk),
                "THREADS_PER_PROC": str(threads_per_proc),
                "ISAAC_RETRIES": str(isaac_retries),
                "Q_DOF_MISMATCH": q_dof_mismatch,
                "CKPT_PRIORITY": ckpt_priority,
                "EXIT_NONZERO_ON_FAIL": "0",
            }
        )
        eval_cmd = ["bash", "scripts/sweep_dexgrasp_to_dro_isaac_multigpu.sh"]
        eval_rc = _run_stream(eval_cmd, env=eval_env, cwd=ROOT_DIR)

        run_root = _find_new_eval_run(
            export_root,
            stage_base,
            epochs,
            before_runs,
            allow_fallback_latest=(eval_rc == 0),
        )
        if run_root is None:
            stage_results.append(
                {
                    "stage": stage_name,
                    "started_utc": _utc_now(),
                    "train_rc": train_rc,
                    "eval_rc": eval_rc,
                    "run_root": None,
                    "target_success_percent": target_success,
                    "passed": False,
                    "reason": "eval_run_root_not_found",
                }
            )
            pipeline_ok = False
            print(f"[Stage {stage_name}] eval run root not found. Stop pipeline.")
            break

        summary_tsv = run_root / "reports" / "summary.tsv"
        metrics = _collect_stage_metrics(summary_tsv, robust_min_grasps=robust_min_grasps)
        best_robust = metrics.get("best_robust_success_percent")
        passed = best_robust is not None and float(best_robust) >= target_success and eval_rc == 0
        reason = "ok" if passed else "target_not_met_or_eval_failed"

        print(
            f"[Stage {stage_name}] best_robust_success={best_robust} "
            f"(target={target_success}), robust_rows={metrics['robust_success_rows']}, eval_rc={eval_rc}"
        )

        stage_results.append(
            {
                "stage": stage_name,
                "started_utc": _utc_now(),
                "train_rc": train_rc,
                "eval_rc": eval_rc,
                "run_root": str(run_root),
                "summary_tsv": str(summary_tsv),
                "target_success_percent": target_success,
                "train_diagnostics": train_diagnostics,
                "metrics": metrics,
                "passed": bool(passed),
                "reason": reason,
            }
        )

        if not passed:
            pipeline_ok = False
            print(f"[Stage {stage_name}] did not pass milestone. Stop pipeline.")
            break

    final_report = {
        "created_utc": _utc_now(),
        "plan_path": str(plan_path),
        "pipeline_ok": bool(pipeline_ok),
        "stage_results": stage_results,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\n================================ [Rot6D Debug Pipeline Done] ================================")
    print(f"pipeline_ok: {pipeline_ok}")
    print(f"report:      {report_path}")

    return 0 if pipeline_ok else 2


if __name__ == "__main__":
    sys.exit(main())
