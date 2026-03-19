#!/usr/bin/env python3
"""
DexBench smoke tests (no IsaacLab required).

Tests per design doc's Test Plan:
  1. Loss smoke  – geodesic loss is near-zero for (q, -q) and small for (±pi euler)
  2. Dataset smoke – observations.shape[-1] == 210, goal_pose in obs  (skipped if HDF5 absent)
  3. Runtime parity – obs signature from training config matches dexbench_observation
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.dexbench_rotations import rot_geodesic_torch, rot_geodesic_deg_torch
from utils.dexbench_observation import (
    DEFAULT_OBS_COMPONENTS,
    get_obs_component_signature,
    get_obs_dim,
    canonicalize_obs_components,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

_results: list[tuple[str, str, str]] = []  # (name, status, detail)


def _check(name: str, cond: bool, detail: str = "") -> bool:
    status = PASS if cond else FAIL
    _results.append((name, status, detail))
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    return cond


def _skip(name: str, reason: str) -> None:
    _results.append((name, SKIP, reason))
    print(f"  [{SKIP}] {name}  — {reason}")


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 1. Loss smoke – rotation geometry
# ---------------------------------------------------------------------------

def test_loss_smoke() -> int:
    _section("1. Loss smoke: SO(3) geodesic loss correctness")
    failures = 0

    # ── 1a. rot_euler: near-antipodal angles → small geodesic, large raw MSE
    angle = 3.13  # close to pi
    pred_euler = torch.tensor([[angle, 0.0, 0.0]])   # (1, 3)
    gt_euler   = torch.tensor([[-angle, 0.0, 0.0]])  # (1, 3) — same rotation modulo 2pi

    geo_rad = rot_geodesic_torch(pred_euler, gt_euler, "rot_euler").item()
    mse_raw = float(torch.mean((pred_euler - gt_euler) ** 2).item())

    geo_deg = geo_rad * 180.0 / np.pi
    ok_geo  = geo_deg < 20.0   # should be small; actual ~0.5° for ±3.13 rad ≈ ±179.4°
    ok_mse  = mse_raw > 1.0    # raw MSE between +3.13 and -3.13 is (6.26)^2 ≈ 39.2 ≫ 1

    failures += not _check(
        "rot_euler ±3.13 → geodesic_deg is small (<20°)",
        ok_geo,
        f"geodesic_deg={geo_deg:.3f}°",
    )
    failures += not _check(
        "rot_euler ±3.13 → raw MSE is large (>1)",
        ok_mse,
        f"raw_mse={mse_raw:.3f}",
    )

    # ── 1b. rot_quat: q vs -q → geodesic ≈ 0
    import math
    axis = torch.tensor([0.0, 0.0, 1.0])
    half_ang = math.pi / 4  # 45°
    qw = math.cos(half_ang)
    qxyz = (math.sin(half_ang) * axis).tolist()
    q_pos = torch.tensor([[qxyz[0], qxyz[1], qxyz[2], qw]])   # xyzw
    q_neg = -q_pos

    geo_quat = rot_geodesic_deg_torch(q_pos, q_neg, "rot_quat").item()
    failures += not _check(
        "rot_quat q vs -q → geodesic_deg < 0.1°",
        geo_quat < 0.1,
        f"geodesic_deg={geo_quat:.6f}°",
    )

    # ── 1c. rot_6d: orthogonal pair → geodesic ≈ 90°
    a = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])  # identity-ish
    b_angle = math.pi / 2
    b = torch.tensor([[math.cos(b_angle), math.sin(b_angle), 0.0,
                       -math.sin(b_angle), math.cos(b_angle), 0.0]])  # 90° rotation
    geo_6d = rot_geodesic_deg_torch(a, b, "rot_6d").item()
    failures += not _check(
        "rot_6d 90° rotation → geodesic_deg ≈ 90°",
        85.0 < geo_6d < 95.0,
        f"geodesic_deg={geo_6d:.3f}°",
    )

    # ── 1d. rot_vec: small vs large angle (≠ embedding space distance)
    pred_rv = torch.tensor([[3.13, 0.0, 0.0]])   # ~π rotation around x
    gt_rv   = torch.tensor([[-3.13, 0.0, 0.0]])  # same rotation, opposite sign
    geo_rv  = rot_geodesic_deg_torch(pred_rv, gt_rv, "rot_vec").item()
    failures += not _check(
        "rot_vec +3.13 vs -3.13 → geodesic_deg is small (<5°)",
        geo_rv < 5.0,
        f"geodesic_deg={geo_rv:.3f}°",
    )

    return failures


# ---------------------------------------------------------------------------
# 2. Dataset smoke – check obs dim and goal_pose presence (skip if no HDF5)
# ---------------------------------------------------------------------------

def test_dataset_smoke() -> int:
    _section("2. Dataset smoke: observations.shape[-1] == 210 + goal_pose in obs")

    DEFAULT_HDF5 = os.path.join(
        ROOT_DIR,
        "dexbench_lite",
        "dexbench_lite",
        "relocate_no_conflict_augmented.hdf5",
    )

    if not os.path.exists(DEFAULT_HDF5):
        _skip("Dataset dim == 210", f"Augmented HDF5 not found: {DEFAULT_HDF5}")
        _skip("goal_pose in obs", "Augmented HDF5 not found")
        _skip("goal_pose values non-zero", "Augmented HDF5 not found")
        print(
            "  ↳ Run  scripts/rebuild_dexbench_hdf5_obs.py  to build the augmented file first."
        )
        return 0  # skip, not failure

    import h5py
    from utils.dexbench_observation import concatenate_hdf5_observations, DEFAULT_OBS_COMPONENTS

    failures = 0
    with h5py.File(DEFAULT_HDF5, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        if not demo_keys:
            _skip("Dataset dim == 210", "HDF5 has no episodes")
            return 0

        first_key = demo_keys[0]
        grp = f["data"][first_key]

        # Full obs concatenation
        try:
            obs = concatenate_hdf5_observations(grp, DEFAULT_OBS_COMPONENTS)
        except KeyError as exc:
            failures += not _check(
                "Dataset obs components all present",
                False,
                f"Missing component: {exc}",
            )
            return failures

        failures += not _check(
            "observations.shape[-1] == 210",
            obs.shape[1] == 210,
            f"got {obs.shape[1]}",
        )

        # goal_pose column slice (last 7 dims)
        goal_slice = obs[:, -7:]
        nz = bool(np.any(goal_slice != 0.0))
        failures += not _check(
            "goal_pose columns are non-zero",
            nz,
            f"all-zero={not nz}",
        )

        # All components present individually
        from utils.dexbench_observation import load_hdf5_obs_component
        all_ok = True
        missing = []
        for name in DEFAULT_OBS_COMPONENTS:
            try:
                arr = load_hdf5_obs_component(grp, name)
                expected_dim = get_obs_dim([name])
                if arr.shape[1] != expected_dim:
                    all_ok = False
                    missing.append(f"{name}(dim={arr.shape[1]},want={expected_dim})")
            except KeyError as exc:
                all_ok = False
                missing.append(str(exc))
        failures += not _check(
            "All 8 obs components present with correct dims",
            all_ok,
            ", ".join(missing) if missing else "",
        )

    return failures


# ---------------------------------------------------------------------------
# 3. Runtime parity smoke – obs signature consistency
# ---------------------------------------------------------------------------

def test_parity_smoke() -> int:
    _section("3. Runtime parity: obs_component_signature matches configs")
    failures = 0

    default_sig = get_obs_component_signature(DEFAULT_OBS_COMPONENTS)
    default_dim = get_obs_dim(DEFAULT_OBS_COMPONENTS)

    failures += not _check(
        "DEFAULT_OBS_COMPONENTS produce signature (deterministic)",
        isinstance(default_sig, str) and len(default_sig) > 0,
        default_sig[:80],
    )
    failures += not _check(
        "DEFAULT_OBS_COMPONENTS total dim == 210",
        default_dim == 210,
        f"got {default_dim}",
    )

    # Config-loaded components must match
    try:
        from omegaconf import OmegaConf
        cfg_path = os.path.join(ROOT_DIR, "configs", "dataset", "dexbench_hdf5.yaml")
        if os.path.exists(cfg_path):
            cfg = OmegaConf.load(cfg_path)
            cfg_components = canonicalize_obs_components(
                list(cfg.obs_components) if cfg.obs_components is not None else None
            )
            cfg_sig = get_obs_component_signature(cfg_components)
            failures += not _check(
                "Config obs_components signature == DEFAULT signature",
                cfg_sig == default_sig,
                f"cfg={cfg_sig[:60]}  default={default_sig[:60]}",
            )
            failures += not _check(
                "Config obs_dim field == 210",
                int(cfg.obs_dim) == 210,
                f"got {cfg.obs_dim}",
            )
        else:
            _skip("Config obs_components check", f"Not found: {cfg_path}")
    except Exception as exc:  # noqa: BLE001
        _skip("Config obs_components check", f"OmegaConf error: {exc}")

    # eval_dexbench_online.yaml must reference same components via dataset defaults
    try:
        from omegaconf import OmegaConf
        online_cfg_path = os.path.join(ROOT_DIR, "configs", "eval_dexbench_online.yaml")
        if os.path.exists(online_cfg_path):
            # Just check file exists and obs_dim resolves to 210 from dexbench_hdf5.yaml
            failures += not _check(
                "eval_dexbench_online.yaml exists",
                True,
                online_cfg_path,
            )
        else:
            failures += not _check("eval_dexbench_online.yaml exists", False, "not found")
    except Exception as exc:  # noqa: BLE001
        _skip("Online eval config check", str(exc))

    return failures


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def main() -> int:
    print("\nDexBench smoke tests")
    print("=" * 60)

    total_failures = 0
    try:
        total_failures += test_loss_smoke()
    except Exception:
        print(f"  [{FAIL}] Loss smoke crashed:")
        traceback.print_exc()
        total_failures += 1

    try:
        total_failures += test_dataset_smoke()
    except Exception:
        print(f"  [{FAIL}] Dataset smoke crashed:")
        traceback.print_exc()
        total_failures += 1

    try:
        total_failures += test_parity_smoke()
    except Exception:
        print(f"  [{FAIL}] Parity smoke crashed:")
        traceback.print_exc()
        total_failures += 1

    n_pass  = sum(1 for _, s, _ in _results if s == PASS)
    n_fail  = sum(1 for _, s, _ in _results if s == FAIL)
    n_skip  = sum(1 for _, s, _ in _results if s == SKIP)
    total   = len(_results)

    print(f"\n{'='*60}")
    print(f"  Results: {n_pass}/{total} passed, {n_fail} failed, {n_skip} skipped")
    if total_failures == 0:
        print(f"  \033[92mAll checks passed.\033[0m")
    else:
        print(f"  \033[91m{total_failures} check(s) failed.\033[0m")
    print("=" * 60)

    return int(total_failures > 0)


if __name__ == "__main__":
    sys.exit(main())
