#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(v: str | None) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def _parse_metrics(metrics_csv: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "metrics_csv": str(metrics_csv),
        "exists": metrics_csv.exists(),
        "max_epoch_seen": None,
        "last_train": None,
        "num_val_points": 0,
        "latest_val": None,
        "best_val_by_mse": None,
    }
    if not metrics_csv.exists():
        return out

    max_epoch = -1
    last_train = None
    vals: list[dict[str, float]] = []
    with metrics_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = _safe_float(row.get("epoch"))
            if epoch is not None:
                e = int(epoch)
                max_epoch = max(max_epoch, e)
                if row.get("loss") not in (None, ""):
                    last_train = {
                        "epoch": e,
                        "loss": _safe_float(row.get("loss")),
                        "step": row.get("step"),
                    }
            vmse = _safe_float(row.get("val_action_mse"))
            vq = _safe_float(row.get("val_final_step_q_l1"))
            vr = _safe_float(row.get("val_rot_geodesic_deg_mean"))
            if vmse is not None and vq is not None and vr is not None and epoch is not None:
                vals.append(
                    {
                        "epoch": int(epoch),
                        "val_action_mse": vmse,
                        "val_final_step_q_l1": vq,
                        "val_rot_geodesic_deg_mean": vr,
                    }
                )

    out["max_epoch_seen"] = None if max_epoch < 0 else max_epoch
    out["last_train"] = last_train
    out["num_val_points"] = len(vals)
    if vals:
        out["latest_val"] = vals[-1]
        out["best_val_by_mse"] = min(vals, key=lambda x: x["val_action_mse"])
    return out


def _count_procs(pattern: str) -> int:
    cmd = f"ps -eo cmd | rg -F \"{pattern}\" | rg -v rg | wc -l"
    res = subprocess.run(["/bin/sh", "-lc", cmd], capture_output=True, text=True)
    if res.returncode != 0:
        return 0
    try:
        return int((res.stdout or "0").strip())
    except Exception:
        return 0


def _find_eval_summary(export_root: Path, eval_prefix: str, epochs: int) -> dict[str, Any]:
    pattern = str(export_root / f"{eval_prefix}_ep{epochs}_*")
    runs = [Path(p) for p in glob.glob(pattern) if Path(p).is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime)
    out: dict[str, Any] = {
        "run_root": None,
        "summary_tsv": None,
        "latest_summary_row": None,
    }
    if not runs:
        return out
    run_root = runs[-1]
    summary_tsv = run_root / "reports" / "summary.tsv"
    out["run_root"] = str(run_root)
    out["summary_tsv"] = str(summary_tsv)
    if not summary_tsv.exists():
        return out

    last_row = None
    try:
        with summary_tsv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                last_row = row
    except Exception:
        last_row = None
    out["latest_summary_row"] = last_row
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Live monitor for rot6d train/eval progress.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--train_match", type=str, required=True, help="Substring to match training process cmdline.")
    parser.add_argument("--export_root", type=str, default="")
    parser.add_argument("--eval_prefix", type=str, default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--interval_sec", type=int, default=30)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--out_log", type=str, required=True)
    parser.add_argument("--max_hours", type=float, default=24.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    metrics_csv = run_dir / "log" / "csv" / "version_0" / "metrics.csv"
    out_json = Path(args.out_json).resolve()
    out_log = Path(args.out_log).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_log.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    max_sec = max(5.0, float(args.max_hours) * 3600.0)

    while True:
        status: dict[str, Any] = {
            "timestamp_utc": _now(),
            "run_dir": str(run_dir),
            "train_proc_count": _count_procs(args.train_match),
            "training": _parse_metrics(metrics_csv),
            "evaluation": None,
        }
        if args.export_root and args.eval_prefix and args.epochs > 0:
            status["evaluation"] = _find_eval_summary(Path(args.export_root), args.eval_prefix, int(args.epochs))

        with out_json.open("w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)

        latest_val = status["training"].get("latest_val")
        best_val = status["training"].get("best_val_by_mse")
        eval_row = (status.get("evaluation") or {}).get("latest_summary_row")
        line = (
            f"{status['timestamp_utc']} "
            f"proc={status['train_proc_count']} "
            f"epoch={status['training'].get('max_epoch_seen')} "
            f"latest_val={latest_val} "
            f"best_val={best_val} "
            f"eval={eval_row}\n"
        )
        with out_log.open("a", encoding="utf-8") as f:
            f.write(line)
        print(line, end="", flush=True)

        train_done = int(status["train_proc_count"]) == 0
        eval_done = False
        if status.get("evaluation") and status["evaluation"].get("latest_summary_row"):
            eval_done = True
        if train_done and (eval_done or not status.get("evaluation")):
            break
        if (time.time() - t0) > max_sec:
            break

        time.sleep(max(5, int(args.interval_sec)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
