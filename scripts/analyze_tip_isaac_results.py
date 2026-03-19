#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def _read_tsv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _read_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _to_int(x: str) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def _write_csv(path: str, fieldnames: list[str], rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Analyze tip Isaac results and generate success/failure visualizations.")
    parser.add_argument("--run_root", type=str, required=True, help="Run root path, e.g. .../cmp_mse_repr_ablation_tip_ep200_...")
    args = parser.parse_args()

    run_root = os.path.abspath(args.run_root)
    reports_dir = os.path.join(run_root, "reports")
    summary_tsv = os.path.join(reports_dir, "summary.tsv")
    if not os.path.exists(summary_tsv):
        raise FileNotFoundError(f"summary.tsv not found: {summary_tsv}")

    out_dir = os.path.join(reports_dir, "tip_analysis")
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = _read_tsv(summary_tsv)
    summary_rows = [r for r in summary_rows if r.get("status", "") == "success"]
    if len(summary_rows) == 0:
        raise RuntimeError("No successful rows in summary.tsv.")

    action_summary = []
    per_object_by_action: dict[str, dict[str, float]] = {}
    per_object_count_by_action: dict[str, dict[str, tuple[int, int]]] = {}
    all_objects = set()

    for r in summary_rows:
        action = r["action"]
        summary_json = r["summary_json"]
        with open(summary_json, "r", encoding="utf-8") as f:
            js = json.load(f)

        run_dir = os.path.dirname(os.path.dirname(summary_json))
        per_object_csv = os.path.join(run_dir, "isaac_eval", "isaac_per_object.csv")
        per_rows = _read_csv(per_object_csv)

        obj_rate = {}
        obj_counts = {}
        for pr in per_rows:
            obj = pr["object_name"]
            sr = _to_float(pr["success_rate_percent"])
            sn = _to_int(pr["success_num"])
            ng = _to_int(pr["num_grasps"])
            obj_rate[obj] = sr
            obj_counts[obj] = (sn, ng)
            all_objects.add(obj)

        per_object_by_action[action] = obj_rate
        per_object_count_by_action[action] = obj_counts

        success_num_total = _to_int(str(js.get("success_num_total", 0)))
        grasps_total = _to_int(str(js.get("num_grasps_total", 0)))
        fail_num_total = max(0, grasps_total - success_num_total)
        action_summary.append(
            {
                "action": action,
                "success_rate_percent": _to_float(str(js.get("success_rate_percent", "nan"))),
                "diversity_rad": _to_float(str(js.get("diversity_rad", "nan"))),
                "num_grasps_total": grasps_total,
                "success_num_total": success_num_total,
                "fail_num_total": fail_num_total,
                "summary_json": summary_json,
            }
        )

    action_summary = sorted(action_summary, key=lambda x: x["success_rate_percent"], reverse=True)
    actions = [r["action"] for r in action_summary]
    objects = sorted(all_objects)

    # Write summary table
    summary_csv = os.path.join(out_dir, "tip_isaac_action_summary.csv")
    _write_csv(
        summary_csv,
        [
            "action",
            "success_rate_percent",
            "diversity_rad",
            "num_grasps_total",
            "success_num_total",
            "fail_num_total",
            "summary_json",
        ],
        action_summary,
    )

    # Build object matrix table
    matrix_rows = []
    for obj in objects:
        row = {"object_name": obj}
        for a in actions:
            row[a] = per_object_by_action.get(a, {}).get(obj, float("nan"))
        row["mean_success_rate_percent"] = float(
            np.nanmean([row[a] for a in actions]) if len(actions) > 0 else np.nan
        )
        matrix_rows.append(row)

    matrix_csv = os.path.join(out_dir, "tip_isaac_object_success_matrix.csv")
    _write_csv(matrix_csv, ["object_name", *actions, "mean_success_rate_percent"], matrix_rows)

    # Success/failure cases
    case_rows = []
    for a in actions:
        pairs = []
        for obj in objects:
            sr = per_object_by_action.get(a, {}).get(obj, float("nan"))
            sn, ng = per_object_count_by_action.get(a, {}).get(obj, (0, 0))
            pairs.append((obj, sr, sn, ng))
        pairs = [p for p in pairs if not np.isnan(p[1])]
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        top3 = pairs_sorted[:3]
        bot3 = list(reversed(pairs_sorted[-3:]))
        for rank, (obj, sr, sn, ng) in enumerate(top3, 1):
            case_rows.append(
                {"action": a, "case_type": "top_success", "rank": rank, "object_name": obj, "success_rate_percent": sr, "success_num": sn, "num_grasps": ng}
            )
        for rank, (obj, sr, sn, ng) in enumerate(bot3, 1):
            case_rows.append(
                {"action": a, "case_type": "top_failure", "rank": rank, "object_name": obj, "success_rate_percent": sr, "success_num": sn, "num_grasps": ng}
            )

    cases_csv = os.path.join(out_dir, "tip_isaac_success_failure_cases.csv")
    _write_csv(
        cases_csv,
        ["action", "case_type", "rank", "object_name", "success_rate_percent", "success_num", "num_grasps"],
        case_rows,
    )

    # Plot 1: success rate by action
    fig1 = plt.figure(figsize=(9, 4), dpi=160)
    xs = np.arange(len(actions))
    ys = np.array([r["success_rate_percent"] for r in action_summary], dtype=float)
    plt.bar(xs, ys, color="#3B82F6")
    plt.xticks(xs, actions, rotation=25, ha="right")
    plt.ylabel("Success Rate (%)")
    plt.title("TIP Isaac Success Rate by Representation")
    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "tip_isaac_success_rate_bar.png")
    fig1.savefig(fig1_path)
    plt.close(fig1)

    # Plot 2: success/failure stacked counts
    fig2 = plt.figure(figsize=(9, 4), dpi=160)
    succ = np.array([r["success_num_total"] for r in action_summary], dtype=float)
    fail = np.array([r["fail_num_total"] for r in action_summary], dtype=float)
    plt.bar(xs, succ, color="#10B981", label="Success")
    plt.bar(xs, fail, bottom=succ, color="#EF4444", label="Fail")
    plt.xticks(xs, actions, rotation=25, ha="right")
    plt.ylabel("Num Grasps")
    plt.title("TIP Isaac Success / Failure Counts by Representation")
    plt.legend()
    plt.tight_layout()
    fig2_path = os.path.join(out_dir, "tip_isaac_success_failure_stacked.png")
    fig2.savefig(fig2_path)
    plt.close(fig2)

    # Plot 3: object-level heatmap
    heat = np.full((len(objects), len(actions)), np.nan, dtype=float)
    for i, obj in enumerate(objects):
        for j, a in enumerate(actions):
            heat[i, j] = per_object_by_action.get(a, {}).get(obj, np.nan)

    fig3 = plt.figure(figsize=(11, max(4, 0.35 * len(objects))), dpi=160)
    cmap = plt.cm.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="#f2f2f2")
    plt.imshow(heat, aspect="auto", cmap=cmap, vmin=0.0, vmax=max(1.0, float(np.nanmax(heat))))
    plt.colorbar(label="Success Rate (%)")
    plt.xticks(np.arange(len(actions)), actions, rotation=25, ha="right")
    plt.yticks(np.arange(len(objects)), objects)
    plt.title("TIP Isaac Object-level Success Rate Heatmap")
    plt.tight_layout()
    fig3_path = os.path.join(out_dir, "tip_isaac_object_heatmap.png")
    fig3.savefig(fig3_path)
    plt.close(fig3)

    print("================================ [TIP Isaac Analysis] ================================")
    print(f"run_root: {run_root}")
    print(f"out_dir:  {out_dir}")
    print(f"summary_csv: {summary_csv}")
    print(f"matrix_csv:  {matrix_csv}")
    print(f"cases_csv:   {cases_csv}")
    print(f"plot_bar:    {fig1_path}")
    print(f"plot_stack:  {fig2_path}")
    print(f"plot_heat:   {fig3_path}")


if __name__ == "__main__":
    main()
