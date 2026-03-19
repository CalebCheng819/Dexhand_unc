#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _to_float(v: str | None) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def _load_val_rows(metrics_csv: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with metrics_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mse = _to_float(row.get("val_action_mse"))
            q_l1 = _to_float(row.get("val_final_step_q_l1"))
            rot_deg = _to_float(row.get("val_rot_geodesic_deg_mean"))
            epoch = _to_float(row.get("epoch"))
            if mse is None or q_l1 is None or rot_deg is None or epoch is None:
                continue
            rows.append(
                {
                    "epoch": epoch,
                    "val_action_mse": mse,
                    "val_final_step_q_l1": q_l1,
                    "val_rot_geodesic_deg_mean": rot_deg,
                }
            )
    return rows


def _ratio(num: float, den: float) -> float | None:
    if abs(den) < 1e-12:
        return None
    return num / den


def _fmt_ratio(v: float | None) -> str:
    if v is None:
        return "nan"
    return f"{v:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze DexGrasp training metrics and recommend checkpoint selection."
    )
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory, e.g. /data/.../dexgrasp_xxx")
    parser.add_argument(
        "--degrade_ratio_threshold",
        type=float,
        default=1.25,
        help="If final/best ratio exceeds this threshold, treat as clear late-epoch degradation.",
    )
    parser.add_argument("--out_json", type=str, default="", help="Optional output JSON path.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    metrics_csv = run_dir / "log" / "csv" / "version_1" / "metrics.csv"
    ckpt_dir = run_dir / "state_dict"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_csv}")

    val_rows = _load_val_rows(metrics_csv)
    if not val_rows:
        raise RuntimeError(f"No validation rows found in {metrics_csv}")

    best_by_mse = min(val_rows, key=lambda r: r["val_action_mse"])
    best_by_q = min(val_rows, key=lambda r: r["val_final_step_q_l1"])
    last = val_rows[-1]

    mse_ratio = _ratio(last["val_action_mse"], best_by_mse["val_action_mse"])
    q_ratio = _ratio(last["val_final_step_q_l1"], best_by_q["val_final_step_q_l1"])
    rot_ratio = _ratio(last["val_rot_geodesic_deg_mean"], best_by_mse["val_rot_geodesic_deg_mean"])

    clear_degradation = any(
        r is not None and r >= float(args.degrade_ratio_threshold) for r in [mse_ratio, q_ratio]
    )

    best_ckpt = ckpt_dir / "best_val.ckpt"
    last_ckpt = ckpt_dir / "last.ckpt"
    if best_ckpt.exists() and clear_degradation:
        recommended_ckpt = str(best_ckpt)
        reason = "late_epoch_degradation_detected_use_best_val"
    elif last_ckpt.exists():
        recommended_ckpt = str(last_ckpt)
        reason = "no_clear_degradation_use_last"
    elif best_ckpt.exists():
        recommended_ckpt = str(best_ckpt)
        reason = "last_missing_fallback_best_val"
    else:
        recommended_ckpt = ""
        reason = "no_checkpoint_found"

    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "metrics_csv": str(metrics_csv),
        "num_val_rows": len(val_rows),
        "best_by_mse": best_by_mse,
        "best_by_q_l1": best_by_q,
        "last": last,
        "ratios_final_over_best": {
            "val_action_mse": mse_ratio,
            "val_final_step_q_l1": q_ratio,
            "val_rot_geodesic_deg_mean": rot_ratio,
        },
        "degrade_ratio_threshold": float(args.degrade_ratio_threshold),
        "clear_degradation": clear_degradation,
        "recommended_checkpoint": recommended_ckpt,
        "recommendation_reason": reason,
    }

    print("=== DexGrasp Training Analysis ===")
    print(f"run_dir: {report['run_dir']}")
    print(f"val rows: {report['num_val_rows']}")
    print(
        "best_by_mse: "
        f"epoch={best_by_mse['epoch']:.0f}, mse={best_by_mse['val_action_mse']:.6f}, "
        f"q_l1={best_by_mse['val_final_step_q_l1']:.6f}, rot_deg={best_by_mse['val_rot_geodesic_deg_mean']:.3f}"
    )
    print(
        "best_by_q_l1: "
        f"epoch={best_by_q['epoch']:.0f}, mse={best_by_q['val_action_mse']:.6f}, "
        f"q_l1={best_by_q['val_final_step_q_l1']:.6f}, rot_deg={best_by_q['val_rot_geodesic_deg_mean']:.3f}"
    )
    print(
        "last: "
        f"epoch={last['epoch']:.0f}, mse={last['val_action_mse']:.6f}, "
        f"q_l1={last['val_final_step_q_l1']:.6f}, rot_deg={last['val_rot_geodesic_deg_mean']:.3f}"
    )
    print(
        "ratios(final/best): "
        f"mse={_fmt_ratio(mse_ratio)}, "
        f"q_l1={_fmt_ratio(q_ratio)}, "
        f"rot={_fmt_ratio(rot_ratio)}"
    )
    print(f"clear_degradation: {clear_degradation}")
    print(f"recommended_checkpoint: {recommended_ckpt or 'N/A'}")
    print(f"recommendation_reason: {reason}")

    if args.out_json:
        out_path = Path(args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
