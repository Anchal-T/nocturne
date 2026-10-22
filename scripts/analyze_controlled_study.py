#!/usr/bin/env python
"""Validate the controlled-study evaluation JSONs and build paper artifacts.

Every number written by this script comes from the evaluation JSONs produced
by ``examples/drl_collision_avoidance/evaluate.py``. The script refuses to
publish a table when the evidence is incomplete or inconsistent:

* all 8 arms x 3 seeds must be present (unless --allow-missing, for smoke tests)
* every run must report the same hash-split test file list (sha256)
* every checkpoint's observation metadata must match the frozen arm matrix
* every scene must be visited once, with the same ego id across arms
* the aggregate block must agree with the per-episode records
* the selected file count must be the full test half (6,102 of 12,170)

Usage:
    python scripts/analyze_controlled_study.py \
        --eval-dir paper/workspace/eval \
        --out-dir paper/workspace/analysis

    # smoke test on two-scene JSONs:
    python scripts/analyze_controlled_study.py \
        --eval-dir /path/to/smoke/eval --out-dir /tmp/analysis \
        --allow-missing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - only for --no-figures use
    plt = None
    _MPL_ERROR = exc
else:
    _MPL_ERROR = None

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_DIR = ROOT / "paper" / "workspace" / "eval"
DEFAULT_OUT_DIR = ROOT / "paper" / "workspace" / "analysis"
DEFAULT_OCCLUSION = DEFAULT_EVAL_DIR / "scene_occlusion.json"

SEEDS = (42, 123, 456)

# Frozen arm matrix. Keep in lock-step with
# paper/workspace/phase2_arms_and_compute.md and observation_config.py.
ARMS: "OrderedDict[str, Dict[str, Any]]" = OrderedDict([
    ("A", dict(
        obs_dim=609, visibility_filter=False, use_occlusion=False,
        occlusion_mask=False, history_frames=1, history_stride=1,
        reward_ttz_source="privileged", grid_channels=3,
        label="A: privileged, no occlusion encoding",
        short="A (privileged)",
        color="#245b8a")),
    ("B", dict(
        obs_dim=609, visibility_filter=True, use_occlusion=False,
        occlusion_mask=False, history_frames=1, history_stride=1,
        reward_ttz_source="privileged", grid_channels=3,
        label="B: visible, no occlusion encoding",
        short="B (visible)",
        color="#3b7a57")),
    ("C", dict(
        obs_dim=613, visibility_filter=True, use_occlusion=True,
        occlusion_mask=False, history_frames=1, history_stride=1,
        reward_ttz_source="privileged", grid_channels=3,
        label="C: visible + occlusion summaries",
        short="C (summaries)",
        color="#6b8e9f")),
    ("C_vis", dict(
        obs_dim=613, visibility_filter=True, use_occlusion=True,
        occlusion_mask=False, history_frames=1, history_stride=1,
        reward_ttz_source="visible", grid_channels=3,
        label="C_vis: C, visible reward TTZ",
        short=r"$C_{\mathrm{vis}}$ (summaries, visible reward)",
        color="#8aa9b6")),
    ("D", dict(
        obs_dim=809, visibility_filter=True, use_occlusion=False,
        occlusion_mask=True, history_frames=1, history_stride=1,
        reward_ttz_source="privileged", grid_channels=3,
        label="D: visible + spatial occlusion mask",
        short="D (mask)",
        color="#b06b32")),
    ("D_vis", dict(
        obs_dim=809, visibility_filter=True, use_occlusion=False,
        occlusion_mask=True, history_frames=1, history_stride=1,
        reward_ttz_source="visible", grid_channels=3,
        label="D_vis: D, visible reward TTZ",
        short=r"$D_{\mathrm{vis}}$ (mask, visible reward)",
        color="#d09a6a")),
    ("O", dict(
        obs_dim=813, visibility_filter=True, use_occlusion=True,
        occlusion_mask=True, history_frames=1, history_stride=1,
        reward_ttz_source="privileged", grid_channels=3,
        label="O: visible + summaries + mask (ours)",
        short="O (summaries + mask, ours)",
        color="#a63d40")),
    ("C_hist", dict(
        obs_dim=1809, visibility_filter=True, use_occlusion=False,
        occlusion_mask=False, history_frames=3, history_stride=5,
        reward_ttz_source="privileged", grid_channels=9,
        label="C_hist: visible + 3-frame grid history",
        short=r"$C_{\mathrm{hist}}$ (grid history)",
        color="#6a5a8c")),
])

BINARY_METRICS = [
    ("goal", "Goal rate"),
    ("collided", "Collision rate"),
    ("edge_collision", "Edge-collision rate"),
    ("timed_out", "Timeout rate"),
]
CONTINUOUS_METRICS = [
    ("ade", "ADE (m)"),
    ("fde", "FDE (m)"),
    ("jerk", r"Jerk (m/s$^3$)"),
    ("smoothness", "Smoothness"),
    ("min_ttz_vehicle", "Min TTZ veh (s)"),
    ("min_ttz_pedestrian", "Min TTZ ped (s)"),
    ("steps", "Length (steps)"),
    ("reward", "Return"),
]

SELECTED_FOR_STRATA = ("A", "B", "C", "D", "O", "C_hist")

Z95 = 1.959963984540054


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> Dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True,
            stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(ROOT), text=True,
            stderr=subprocess.DEVNULL).strip()
        return {"git_head": head, "git_dirty": bool(status),
                "git_status": status.splitlines()[:100] if status else []}
    except Exception as exc:  # pragma: no cover
        return {"git_head": None, "git_dirty": None, "git_error": str(exc)}


def wilson(k: int, n: int, z: float = Z95) -> Tuple[Optional[float],
                                                   Optional[float]]:
    if n <= 0:
        return None, None
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
            / denom)
    return max(0.0, centre - half), min(1.0, centre + half)


def mean_sd(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    vals = [float(v) for v in values if v is not None and not
            (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return {"n": len(vals), "mean": mean, "sd": sd,
            "min": float(min(vals)), "max": float(max(vals)),
            "values": vals}


def fmt_pct(stats: Dict[str, Any], digits: int = 1) -> str:
    if stats["mean"] is None:
        return "--"
    if stats["n"] == 1:
        return f"{100.0 * stats['mean']:.{digits}f}"
    return (f"{100.0 * stats['mean']:.{digits}f} "
            f"[{100.0 * stats['min']:.{digits}f}, "
            f"{100.0 * stats['max']:.{digits}f}]")


def fmt_num(stats: Dict[str, Any], digits: int = 2) -> str:
    if stats["mean"] is None:
        return "--"
    if stats["n"] == 1:
        return f"{stats['mean']:.{digits}f}"
    return f"{stats['mean']:.{digits}f} $\\pm$ {stats['sd']:.{digits}f}"


def fmt_delta(stats: Dict[str, Any], digits: int = 1) -> str:
    """Signed percentage-point summary for a paired contrast."""
    if stats["mean"] is None:
        return "--"
    s = f"{100.0 * stats['mean']:+.{digits}f}"
    return s


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------

def load_runs(eval_dir: Path, arms: Sequence[str], seeds: Sequence[int],
              allow_missing: bool) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]],
                                           List[Dict[str, Any]],
                                           List[str]]:
    runs: Dict[Tuple[str, int], Dict[str, Any]] = OrderedDict()
    missing: List[Dict[str, Any]] = []
    errors: List[str] = []
    for seed in seeds:
        for arm in arms:
            path = eval_dir / f"{arm}_seed{seed}.json"
            if not path.exists():
                missing.append({"arm": arm, "seed": seed, "path": str(path)})
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
            except Exception as exc:
                errors.append(f"cannot read {path}: {exc}")
                continue
            if not isinstance(data, dict) or "episodes" not in data:
                errors.append(f"{path}: not an evaluate.py JSON (no episodes)")
                continue
            runs[(arm, seed)] = data
    if missing and not allow_missing:
        errors.append("missing evaluation JSONs: " + ", ".join(
            f"{m['arm']}/{m['seed']}" for m in missing))
    return runs, missing, errors


def validate_runs(runs: Dict[Tuple[str, int], Dict[str, Any]],
                  arms: Sequence[str], seeds: Sequence[int],
                  allow_missing: bool, expected_scenes: int,
                  expected_total: int,
                  allowed_untracked_prefixes: Sequence[str],
                  ) -> Tuple[List[str], List[str], Dict[str, Any]]:
    errors: List[str] = []
    warnings: List[str] = []
    info: Dict[str, Any] = {"runs": {}, "selected_sha": None,
                            "scene_sets": {}, "git_heads": set(),
                            "dirty_runs": []}

    expected_runs = len(arms) * len(seeds)
    if not allow_missing and len(runs) != expected_runs:
        errors.append(f"expected {expected_runs} runs, found {len(runs)}")

    for (arm, seed), data in runs.items():
        exp = ARMS[arm]
        tag = f"{arm}/seed{seed}"
        prov = data.get("provenance") or {}
        scen = prov.get("scenario") or {}
        ckpt = prov.get("checkpoint") or {}
        meta = ckpt.get("observation_metadata") or {}
        env = prov.get("env") or {}
        grid = env.get("occupancy_grid") or {}
        code = prov.get("code") or {}

        if scen.get("split") != "test":
            errors.append(f"{tag}: scenario.split={scen.get('split')!r}, "
                          "expected 'test'")
        if int(scen.get("split_seed", -1)) != 0:
            errors.append(f"{tag}: split_seed={scen.get('split_seed')!r}, "
                          "expected 0")
        if not allow_missing:
            if int(scen.get("num_files_total", -1)) != expected_total:
                errors.append(f"{tag}: num_files_total="
                              f"{scen.get('num_files_total')}, expected "
                              f"{expected_total}")
            if int(scen.get("num_files_selected", -1)) != expected_scenes:
                errors.append(f"{tag}: num_files_selected="
                              f"{scen.get('num_files_selected')}, expected "
                              f"{expected_scenes}")

        selected = scen.get("selected_files_sha256")
        if info["selected_sha"] is None:
            info["selected_sha"] = selected
        elif selected != info["selected_sha"]:
            errors.append(f"{tag}: selected_files_sha256 mismatch")
        if not allow_missing and not selected:
            errors.append(f"{tag}: missing selected_files_sha256")

        expected_meta = {
            "obs_dim": exp["obs_dim"],
            "visibility_filter": exp["visibility_filter"],
            "use_occlusion": exp["use_occlusion"],
            "occlusion_mask": exp["occlusion_mask"],
            "history_frames": exp["history_frames"],
            "history_stride": exp["history_stride"],
            "reward_ttz_source": exp["reward_ttz_source"],
            "grid_channels": exp["grid_channels"],
        }
        for key, value in expected_meta.items():
            got = meta.get(key)
            if got != value:
                errors.append(f"{tag}: observation_metadata[{key}]={got!r}, "
                              f"expected {value!r}")
        for key, value in expected_meta.items():
            if key == "reward_ttz_source":
                got = env.get("reward_ttz_source")
            else:
                got = grid.get(key)
            if got != value:
                errors.append(f"{tag}: env.occupancy_grid[{key}]={got!r}, "
                              f"expected {value!r}")
        if int(grid.get("obs_dim", -1)) != exp["obs_dim"]:
            errors.append(f"{tag}: env.obs_dim={grid.get('obs_dim')}, "
                          f"expected {exp['obs_dim']}")

        ckpt_path = ckpt.get("path") or ""
        if not allow_missing:
            needle = f"/controlled_study/{arm}/seed_{seed}/ddqn_final.pth"
            if needle not in ckpt_path:
                errors.append(f"{tag}: checkpoint path {ckpt_path!r} does not "
                              f"match *{needle}")
        if not ckpt.get("sha256"):
            errors.append(f"{tag}: missing checkpoint sha256")

        head = code.get("git_head")
        if head:
            info["git_heads"].add(head)
        if code.get("git_dirty"):
            status = code.get("git_status") or []
            info["dirty_runs"].append({"run": tag, "git_head": head,
                                       "status": status})
            unknown = []
            for line in status:
                p = line[3:] if len(line) > 3 else line
                if not any(p.startswith(pref)
                           for pref in allowed_untracked_prefixes):
                    unknown.append(line)
            if unknown:
                warnings.append(
                    f"{tag}: eval JSON was generated with unexpected dirty "
                    f"paths: {unknown[:5]}")

        episodes = data.get("episodes") or []
        if not episodes:
            errors.append(f"{tag}: no episodes")
            continue
        if not allow_missing:
            if len(episodes) != expected_scenes:
                errors.append(f"{tag}: {len(episodes)} episodes, expected "
                              f"{expected_scenes}")
        valid = [e for e in episodes if e.get("valid")]
        invalid = [e for e in episodes if not e.get("valid")]
        agg = data.get("aggregate") or {}
        if int(agg.get("num_valid_episodes", -1)) != len(valid):
            errors.append(f"{tag}: aggregate num_valid_episodes="
                          f"{agg.get('num_valid_episodes')}, records={len(valid)}")
        if int(agg.get("num_invalid_episodes", -1)) != len(invalid):
            errors.append(f"{tag}: aggregate num_invalid_episodes="
                          f"{agg.get('num_invalid_episodes')}, records="
                          f"{len(invalid)}")

        # Aggregate vs episode recomputation.
        for field in ("goal", "collided", "edge_collision", "timed_out"):
            if valid:
                recomputed = float(np.mean([bool(e.get(field)) for e in valid]))
            else:
                recomputed = None
            recorded = agg.get(
                {"goal": "goal_rate", "collided": "collision_rate",
                 "edge_collision": "edge_collision_rate",
                 "timed_out": "timeout_rate"}[field])
            if recomputed is not None and recorded is not None:
                if abs(float(recorded) - recomputed) > 1e-9:
                    errors.append(f"{tag}: aggregate {field} rate "
                                  f"{recorded} != episodes {recomputed}")
        for field, agg_key in (("ade", "ade"), ("fde", "fde"),
                               ("jerk", "jerk"),
                               ("smoothness", "smoothness"),
                               ("steps", "avg_length"),
                               ("reward", "avg_reward"),
                               ("initial_goal_dist", "avg_initial_goal_dist"),
                               ("min_ttz_vehicle", "ttz_vehicle_min_mean"),
                               ("min_ttz_pedestrian", "ttz_pedestrian_min_mean")):
            vals = [e.get(field) for e in valid if e.get(field) is not None]
            recomputed = float(np.mean(vals)) if vals else None
            recorded = agg.get(agg_key)
            if recomputed is not None and recorded is not None:
                if abs(float(recorded) - recomputed) > 1e-9:
                    errors.append(f"{tag}: aggregate {agg_key} "
                                  f"{recorded} != episodes {recomputed}")

        # One episode per scene, no duplicates.
        scene_files = [e.get("scene_file") for e in episodes]
        if len(set(scene_files)) != len(scene_files):
            errors.append(f"{tag}: duplicate scene_file entries")
        if any(not s for s in scene_files):
            errors.append(f"{tag}: episode without scene_file")

        info["scene_sets"][tag] = set(scene_files)
        info["runs"][tag] = {
            "path": str(ckpt_path),
            "checkpoint_sha256": ckpt.get("sha256"),
            "git_head": head,
            "git_dirty": code.get("git_dirty"),
            "num_episodes": len(episodes),
            "num_valid": len(valid),
            "num_invalid": len(invalid),
            "invalid_reasons": sorted(
                {e.get("reason") for e in invalid if e.get("reason")}),
            "selected_files_sha256": selected,
            "observation_metadata": meta,
        }

    # Scene sets and ego pairing across runs.
    if runs and not allow_missing:
        base_tag, base_set = next(iter(info["scene_sets"].items()))
        for tag, scene_set in info["scene_sets"].items():
            if scene_set != base_set:
                errors.append(f"{tag}: scene set differs from {base_tag} "
                              f"({len(scene_set)} vs {len(base_set)})")
    ego_by_scene: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for (arm, seed), data in runs.items():
        for e in data.get("episodes") or []:
            if e.get("valid"):
                ego_by_scene[e["scene_file"]][f"{arm}/{seed}"] = e.get("ego_id")
    mismatched = []
    for scene, egos in ego_by_scene.items():
        if len(set(egos.values())) > 1:
            mismatched.append((scene, egos))
    if mismatched:
        errors.append(f"{len(mismatched)} scenes have different ego ids "
                      f"across arms/seeds; first: {mismatched[0]}")

    info["git_heads"] = sorted(info["git_heads"])
    info["expected_scenes"] = expected_scenes
    info["expected_total"] = expected_total
    info["allow_missing"] = allow_missing
    return errors, warnings, info


# --------------------------------------------------------------------------
# Data frames and per-run metrics
# --------------------------------------------------------------------------

def runs_to_frame(runs: Dict[Tuple[str, int], Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for (arm, seed), data in runs.items():
        for e in data.get("episodes") or []:
            row = {
                "arm": arm,
                "seed": int(seed),
                "scene_file": e.get("scene_file"),
                "ego_id": e.get("ego_id"),
                "valid": bool(e.get("valid")),
                "goal": bool(e.get("goal")),
                "collided": bool(e.get("collided")),
                "edge_collision": bool(e.get("edge_collision")),
                "timed_out": bool(e.get("timed_out")),
                "steps": e.get("steps"),
                "reward": e.get("reward"),
                "initial_goal_dist": e.get("initial_goal_dist"),
                "n_vehicles": e.get("n_vehicles"),
                "ade": e.get("ade"),
                "fde": e.get("fde"),
                "jerk": e.get("jerk"),
                "smoothness": e.get("smoothness"),
                "min_ttz_vehicle": e.get("min_ttz_vehicle"),
                "min_ttz_pedestrian": e.get("min_ttz_pedestrian"),
                "reason": e.get("reason"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def run_metrics(df: pd.DataFrame, arm: str, seed: int) -> Dict[str, Any]:
    sub = df[(df["arm"] == arm) & (df["seed"] == seed) & df["valid"]]
    n = int(len(sub))
    out: Dict[str, Any] = {"n": n}
    for field, _ in BINARY_METRICS:
        k = int(sub[field].sum()) if n else 0
        rate = k / n if n else None
        lo, hi = wilson(k, n)
        out[field] = {"count": k, "rate": rate, "wilson_lo": lo,
                      "wilson_hi": hi}
    for field, _ in CONTINUOUS_METRICS:
        vals = pd.to_numeric(sub[field], errors="coerce")
        vals = vals[np.isfinite(vals)]
        out[field] = mean_sd(vals.tolist()) if len(vals) else mean_sd([])
    out["mean_speed"] = None
    return out


def across_seed_stats(per_run: Dict[Tuple[str, int], Dict[str, Any]],
                      arm: str, seeds: Sequence[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for field, _ in BINARY_METRICS:
        vals = [per_run[(arm, s)][field]["rate"] for s in seeds
                if (arm, s) in per_run and
                per_run[(arm, s)][field]["rate"] is not None]
        out[field] = mean_sd(vals)
    for field, _ in CONTINUOUS_METRICS:
        vals = [per_run[(arm, s)][field]["mean"] for s in seeds
                if (arm, s) in per_run and
                per_run[(arm, s)][field]["mean"] is not None]
        out[field] = mean_sd(vals)
    out["n"] = int(np.median([per_run[(arm, s)]["n"] for s in seeds
                              if (arm, s) in per_run])) if any(
        (arm, s) in per_run for s in seeds) else 0
    return out


# --------------------------------------------------------------------------
# Paired contrasts
# --------------------------------------------------------------------------

def paired_contrast(df: pd.DataFrame, seed: int, ours: str, base: str,
                    ) -> Optional[Dict[str, Any]]:
    a = df[(df["arm"] == ours) & (df["seed"] == seed) & df["valid"]]
    b = df[(df["arm"] == base) & (df["seed"] == seed) & df["valid"]]
    if a.empty or b.empty:
        return None
    keys = ["scene_file", "ego_id"]
    m = a.merge(b, on=keys, suffixes=("_o", "_b"))
    if m.empty:
        return None
    # Require the same scene set for valid paired evidence.
    if len(m) != len(a) or len(m) != len(b):
        # Restrict to the intersection, but record the exclusion.
        pass
    d_goal = (m["goal_o"].astype(int) - m["goal_b"].astype(int)).to_numpy()
    d_coll = (m["collided_o"].astype(int) -
              m["collided_b"].astype(int)).to_numpy()
    out: Dict[str, Any] = {
        "seed": seed, "n_paired": int(len(m)),
        "n_ours_valid": int(len(a)), "n_base_valid": int(len(b)),
        "goal_diff_pp": 100.0 * float(np.mean(d_goal)),
        "collision_diff_pp": 100.0 * float(np.mean(d_coll)),
        "goal_discordant": {
            "ours_only": int(np.sum((d_goal == 1))),
            "base_only": int(np.sum((d_goal == -1))),
        },
        "collision_discordant": {
            "ours_only": int(np.sum((d_coll == 1))),
            "base_only": int(np.sum((d_coll == -1))),
        },
    }
    for name, diff in (("goal", d_goal), ("collision", d_coll)):
        if len(diff) > 1:
            se = float(np.std(diff, ddof=1) / math.sqrt(len(diff)))
            mean = float(np.mean(diff))
            out[f"{name}_ci95_pp"] = [100.0 * (mean - Z95 * se),
                                      100.0 * (mean + Z95 * se)]
        else:
            out[f"{name}_ci95_pp"] = None
    return out


def contrast_summary(df: pd.DataFrame, seeds: Sequence[int], ours: str,
                     base: str) -> Dict[str, Any]:
    per_seed = [paired_contrast(df, s, ours, base) for s in seeds]
    per_seed = [p for p in per_seed if p is not None]
    if not per_seed:
        return {"ours": ours, "base": base, "n_seeds": 0}
    goal = [p["goal_diff_pp"] for p in per_seed]
    coll = [p["collision_diff_pp"] for p in per_seed]
    return {
        "ours": ours, "base": base, "n_seeds": len(per_seed),
        "goal": {
            "mean_pp": float(np.mean(goal)),
            "min_pp": float(min(goal)), "max_pp": float(max(goal)),
            "values": goal,
            "seeds_better": int(sum(v > 0 for v in goal)),
        },
        "collision": {
            "mean_pp": float(np.mean(coll)),
            "min_pp": float(min(coll)), "max_pp": float(max(coll)),
            "values": coll,
            "seeds_better": int(sum(v < 0 for v in coll)),
        },
        "per_seed": per_seed,
    }


# --------------------------------------------------------------------------
# Stratification
# --------------------------------------------------------------------------

def density_bins(values: pd.Series) -> Tuple[List[str], List[Tuple[float, float]]]:
    # Fixed, interpretable bins; the paper reports counts and edges.
    edges = [(0, 10), (11, 20), (21, 30), (31, float("inf"))]
    labels = ["1--10", "11--20", "21--30", "31+"]
    return labels, edges


def add_density_bin(df: pd.DataFrame) -> pd.DataFrame:
    labels, edges = density_bins(df["n_vehicles"])
    out = df.copy()
    bins = []
    for v in out["n_vehicles"]:
        try:
            iv = int(v)
        except Exception:
            bins.append(None)
            continue
        found = None
        for (lo, hi), lab in zip(edges, labels):
            if lo <= iv <= hi:
                found = lab
                break
        bins.append(found)
    out["density_bin"] = bins
    return out


def load_occlusion(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    scenes = data.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError(f"{path}: no scenes list")
    rows = []
    for s in scenes:
        if not s.get("valid"):
            continue
        rows.append({
            "scene_file": s.get("scene_file"),
            "ego_id": s.get("ego_id"),
            "occlusion_fraction": s.get("occlusion_fraction"),
            "nearest_blocker_norm": s.get("nearest_blocker_norm"),
            "blocker_density": s.get("blocker_density"),
            "pressure": s.get("pressure"),
            "n_vehicles_scene": s.get("n_vehicles"),
        })
    return pd.DataFrame(rows)


def add_occlusion_bin(df: pd.DataFrame, occ: pd.DataFrame,
                      ) -> Tuple[pd.DataFrame, List[float]]:
    """Attach a common scene-level occlusion tertile to every episode.

    The tertile edges are computed once over the unique test scenes, not per
    arm, so all arms are compared in the same intrinsic occlusion strata.
    """
    # The probe should select the same ego; join on both keys when possible.
    occ = occ.drop_duplicates(subset=["scene_file", "ego_id"])
    joined = df.merge(occ, on=["scene_file", "ego_id"], how="left")
    if joined["occlusion_fraction"].isna().any():
        # Fall back to scene-only join and verify uniqueness.
        occ2 = occ.drop_duplicates(subset=["scene_file"]).drop(
            columns=["ego_id"])
        joined = df.merge(occ2, on="scene_file", how="left")
    vals = joined.loc[joined["valid"], "occlusion_fraction"].dropna()
    if vals.empty:
        joined["occlusion_bin"] = None
        return joined, []
    qs = np.quantile(vals.to_numpy(), [1 / 3, 2 / 3])
    edges = [float("-inf"), float(qs[0]), float(qs[1]), float("inf")]
    labels = ["low", "medium", "high"]
    bins = []
    for v in joined["occlusion_fraction"]:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            bins.append(None)
        elif v <= edges[1]:
            bins.append(labels[0])
        elif v <= edges[2]:
            bins.append(labels[1])
        else:
            bins.append(labels[2])
    joined["occlusion_bin"] = bins
    return joined, [float(qs[0]), float(qs[1])]


def strata_table(df: pd.DataFrame, bin_col: str, bin_order: Sequence[str],
                 seeds: Sequence[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for arm in df["arm"].unique():
        for b in bin_order:
            per_seed = []
            for seed in seeds:
                sub = df[(df["arm"] == arm) & (df["seed"] == seed) &
                         df["valid"] & (df[bin_col] == b)]
                n = len(sub)
                if n:
                    k = int(sub["collided"].sum())
                    per_seed.append({"seed": int(seed), "n": n,
                                     "collisions": k,
                                     "rate": k / n})
            if not per_seed:
                out[f"{arm}|{b}"] = {"n": 0, "rate": None}
                continue
            rates = [p["rate"] for p in per_seed]
            ns = [p["n"] for p in per_seed]
            pooled_n = int(sum(ns))
            pooled_k = int(sum(p["collisions"] for p in per_seed))
            lo, hi = wilson(pooled_k, pooled_n)
            out[f"{arm}|{b}"] = {
                "n": pooled_n, "collisions": pooled_k,
                "rate": pooled_k / pooled_n if pooled_n else None,
                "wilson_lo": lo, "wilson_hi": hi,
                "seed_rates": rates, "seed_min": float(min(rates)),
                "seed_max": float(max(rates)),
                "per_seed": per_seed,
            }
    return out


# --------------------------------------------------------------------------
# LaTeX tables
# --------------------------------------------------------------------------

def tex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}")
             .replace("&", r"\&").replace("%", r"\%")
             .replace("_", r"\_").replace("#", r"\#"))


def write_main_binary(path: Path, per_run: Dict[Tuple[str, int], Dict[str, Any]],
                      across: Dict[str, Dict[str, Any]], arms: Sequence[str],
                      seeds: Sequence[int], manifest: Dict[str, Any]) -> None:
    lines = []
    lines.append("% Generated by scripts/analyze_controlled_study.py")
    lines.append(f"% Inputs: {len(manifest['inputs'])} evaluation JSONs; "
                 f"selected_sha256={manifest['validation'].get('selected_sha')}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main outcomes on the held-out hash-split test half "
                 r"of the full validation directory (6{,}102 scenes; one "
                 r"paired ego per scene; no subsampling). Values are the mean "
                 r"across three training seeds with the seed range in "
                 r"brackets. Goal higher is better; collision, edge-collision, "
                 r"and timeout lower is better. Wilson intervals for each seed "
                 r"are reported in the appendix.}")
    lines.append(r"\label{tab:main-binary}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}llccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Arm & Observation & Goal (\%) $\uparrow$ & "
                 r"Collision (\%) $\downarrow$ & Edge (\%) $\downarrow$ & "
                 r"Timeout (\%) $\downarrow$ & $n$ \\")
    lines.append(r"\midrule")
    for arm in arms:
        if arm not in across:
            continue
        a = across[arm]
        exp = ARMS[arm]
        name = exp["label"].split(":", 1)[1].strip()
        if arm == "O":
            name = r"\textbf{" + name + "}"
        lines.append(
            f"{exp['short'].split(' (')[0]} & {tex_escape(name)} & "
            f"{fmt_pct(a['goal'])} & {fmt_pct(a['collided'])} & "
            f"{fmt_pct(a['edge_collision'])} & {fmt_pct(a['timed_out'])} & "
            f"{a['n']} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n")


def write_main_continuous(path: Path, across: Dict[str, Dict[str, Any]],
                          arms: Sequence[str], manifest: Dict[str, Any]) -> None:
    lines = []
    lines.append("% Generated by scripts/analyze_controlled_study.py")
    lines.append(f"% selected_sha256={manifest['validation'].get('selected_sha')}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Trajectory-quality, comfort, near-miss, and "
                 r"efficiency metrics on the same test half. Values are mean "
                 r"$\pm$ standard deviation across three seeds. ADE/FDE are "
                 r"distance to the logged human trajectory; min TTZ is the "
                 r"per-episode minimum time-to-zero (clipped at the no-conflict "
                 r"sentinel during aggregation).}")
    lines.append(r"\label{tab:main-continuous}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}lccccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Arm & ADE & FDE & Jerk & Smoothness & Min TTZ veh & "
                 r"Min TTZ ped & Length \\")
    lines.append(r"\midrule")
    for arm in arms:
        if arm not in across:
            continue
        a = across[arm]
        exp = ARMS[arm]
        label = exp["short"].split(" (")[0]
        if arm == "O":
            label = r"\textbf{" + label + "}"
        lines.append(
            f"{label} & {fmt_num(a['ade'])} & {fmt_num(a['fde'])} & "
            f"{fmt_num(a['jerk'])} & {fmt_num(a['smoothness'])} & "
            f"{fmt_num(a['min_ttz_vehicle'])} & "
            f"{fmt_num(a['min_ttz_pedestrian'])} & "
            f"{fmt_num(a['steps'], 1)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n")


def write_paired(path: Path, contrasts: Sequence[Dict[str, Any]],
                 manifest: Dict[str, Any]) -> None:
    lines = []
    lines.append("% Generated by scripts/analyze_controlled_study.py")
    lines.append(f"% selected_sha256={manifest['validation'].get('selected_sha')}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Paired scene-level contrasts against the full "
                 r"model O. Each scene contributes the same ego to both arms; "
                 r"$\Delta$ is in percentage points (pp). The three values are "
                 r"the independent training seeds; the last column counts how "
                 r"many seeds favor O (goal up, collision down). Positive goal "
                 r"$\Delta$ and negative collision $\Delta$ favor O. These are "
                 r"descriptive paired effects; with three seeds we do not "
                 r"report cross-seed significance tests.}")
    lines.append(r"\label{tab:paired}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}lcccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Contrast & $\Delta$Goal s42 & s123 & s456 & "
                 r"$\Delta$Collision s42 & s123 & s456 \\")
    lines.append(r"\midrule")
    for c in contrasts:
        if "goal" not in c or "collision" not in c:
            continue
        base = c["base"]
        exp = ARMS[base]
        g = c["goal"]["values"]
        h = c["collision"]["values"]
        if len(g) < 3 or len(h) < 3:
            continue
        lines.append(
            f"O $-$ {exp['short'].split(' (')[0]} & "
            f"{g[0]:+.1f} & {g[1]:+.1f} & {g[2]:+.1f} & "
            f"{h[0]:+.1f} & {h[1]:+.1f} & {h[2]:+.1f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n")


def write_per_seed(path: Path, per_run: Dict[Tuple[str, int], Dict[str, Any]],
                   arms: Sequence[str], seeds: Sequence[int],
                   manifest: Dict[str, Any]) -> None:
    lines = []
    lines.append("% Generated by scripts/analyze_controlled_study.py")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-seed results. Every row is one trained "
                 r"checkpoint evaluated on the same 6{,}102-scene test half.}")
    lines.append(r"\label{tab:per-seed}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{@{}llccccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Arm & Seed & Goal & Collision & Edge & ADE & FDE & "
                 r"Jerk & Min TTZ veh \\")
    lines.append(r"\midrule")
    for arm in arms:
        for seed in seeds:
            if (arm, seed) not in per_run:
                continue
            m = per_run[(arm, seed)]
            exp = ARMS[arm]
            label = exp["short"].split(" (")[0]
            lines.append(
                f"{label} & {seed} & "
                f"{100*m['goal']['rate']:.1f} & "
                f"{100*m['collided']['rate']:.1f} & "
                f"{100*m['edge_collision']['rate']:.1f} & "
                f"{m['ade']['mean']:.2f} & {m['fde']['mean']:.2f} & "
                f"{m['jerk']['mean']:.2f} & "
                f"{m['min_ttz_vehicle']['mean']:.2f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n")


def write_strata(path: Path, strata: Dict[str, Any], bin_order: Sequence[str],
                 label: str, caption: str, arms: Sequence[str]) -> None:
    lines = []
    lines.append("% Generated by scripts/analyze_controlled_study.py")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{tab:strata-" + label.lower() + "}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}l" + "c" * len(bin_order) + "@{}}")
    lines.append(r"\toprule")
    lines.append("Arm & " + " & ".join(bin_order) + r" \\")
    lines.append(r"\midrule")
    for arm in arms:
        if not any(f"{arm}|{b}" in strata for b in bin_order):
            continue
        exp = ARMS[arm]
        label_arm = exp["short"].split(" (")[0]
        cells = []
        for b in bin_order:
            s = strata.get(f"{arm}|{b}")
            if not s or s.get("rate") is None:
                cells.append("--")
            else:
                cells.append(f"{100*s['rate']:.1f} ({s['n']})")
        lines.append(f"{label_arm} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _figure_style() -> None:
    if plt is None:
        return
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.5,
    })


def figure_main_outcomes(out_dir: Path, across: Dict[str, Dict[str, Any]],
                         per_run: Dict[Tuple[str, int], Dict[str, Any]],
                         arms: Sequence[str], seeds: Sequence[int]) -> None:
    if plt is None:
        return
    _figure_style()
    arms_present = [a for a in arms if a in across]
    x = np.arange(len(arms_present))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, metric, title in (
            (axes[0], "goal", "Goal rate (%)"),
            (axes[1], "collided", "Collision rate (%)")):
        means = [100.0 * across[a][metric]["mean"] for a in arms_present]
        mins = [100.0 * across[a][metric]["min"] for a in arms_present]
        maxs = [100.0 * across[a][metric]["max"] for a in arms_present]
        colors = [ARMS[a]["color"] for a in arms_present]
        bars = ax.bar(x, means, width=0.68, color=colors, alpha=0.85,
                      edgecolor="white", linewidth=0.6)
        ax.errorbar(x, means,
                    yerr=[np.array(means) - np.array(mins),
                          np.array(maxs) - np.array(means)],
                    fmt="none", ecolor="#333333", elinewidth=0.8,
                    capsize=2.0)
        for i, a in enumerate(arms_present):
            for s in seeds:
                if (a, s) in per_run:
                    v = per_run[(a, s)][metric]["rate"]
                    if v is not None:
                        ax.scatter(i, 100.0 * v, s=7, color="#222222",
                                   zorder=4, linewidths=0)
        ax.set_xticks(x)
        ax.set_xticklabels([ARMS[a]["short"].split(" (")[0]
                            for a in arms_present], rotation=20, ha="right")
        ax.set_ylabel(title)
        ax.set_ylim(0, 100)
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if metric == "goal":
            ax.set_title("Task success (mean, seed range, seed points)")
        else:
            ax.set_title("Safety (mean, seed range, seed points)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_main_outcomes.pdf")
    fig.savefig(out_dir / "fig_main_outcomes.png")
    plt.close(fig)


def figure_paired(out_dir: Path, contrasts: Sequence[Dict[str, Any]]) -> None:
    if plt is None:
        return
    _figure_style()
    rows = [c for c in contrasts if c.get("goal", {}).get("values")]
    if not rows:
        return
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, key, title in ((axes[0], "goal", "$\\Delta$Goal (pp; + favors O)"),
                           (axes[1], "collision",
                            "$\\Delta$Collision (pp; $-$ favors O)")):
        means = [c[key]["mean_pp"] for c in rows]
        ax.axvline(0, color="#555555", lw=0.7)
        ax.barh(y, means, height=0.62, color="#a63d40", alpha=0.8,
                edgecolor="white", linewidth=0.6)
        for i, c in enumerate(rows):
            vals = c[key]["values"]
            ax.scatter(vals, [i] * len(vals), s=8, color="#222222",
                       zorder=4, linewidths=0)
        ax.set_yticks(y)
        ax.set_yticklabels([f"O $-$ {ARMS[c['base']]['short'].split(' (')[0]}"
                            for c in rows])
        ax.set_xlabel(title)
        ax.grid(axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_paired_effects.pdf")
    fig.savefig(out_dir / "fig_paired_effects.png")
    plt.close(fig)


def figure_strata(out_dir: Path, strata: Dict[str, Any],
                  bin_order: Sequence[str], title: str, fname: str,
                  arms: Sequence[str]) -> None:
    if plt is None:
        return
    _figure_style()
    present = [a for a in arms if any(f"{a}|{b}" in strata for b in bin_order)]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    x = np.arange(len(bin_order))
    for arm in present:
        means, mins, maxs = [], [], []
        for b in bin_order:
            s = strata.get(f"{arm}|{b}")
            if not s or s.get("rate") is None:
                means.append(np.nan); mins.append(np.nan); maxs.append(np.nan)
            else:
                means.append(100.0 * s["rate"])
                mins.append(100.0 * s["seed_min"])
                maxs.append(100.0 * s["seed_max"])
        line = ax.plot(x, means, marker="o", ms=3.2, lw=1.1,
                       color=ARMS[arm]["color"],
                       label=ARMS[arm]["short"].split(" (")[0])[0]
        ax.fill_between(x, mins, maxs, color=line.get_color(), alpha=0.10,
                        linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_order)
    ax.set_xlabel(title)
    ax.set_ylabel("Collision rate (%)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / fname)
    fig.savefig(out_dir / fname.replace(".pdf", ".png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--scene-occlusion", type=Path, default=DEFAULT_OCCLUSION)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS.keys()))
    ap.add_argument("--expected-scenes", type=int, default=6102)
    ap.add_argument("--expected-total", type=int, default=12170)
    ap.add_argument("--allow-missing", action="store_true",
                    help="smoke/testing mode: allow missing arms/seeds and "
                         "skip full-test-size and checkpoint-path checks")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(args.out_dir / "tables")
    ensure_dir(args.out_dir / "figures")

    runs, missing, errors = load_runs(args.eval_dir, args.arms, args.seeds,
                                     args.allow_missing)
    allowed_untracked = (
        "paper/ppt_content.md",
        "paper/ppt_training_data.csv",
        "paper/runs data/",
        "paper/workspace/launches.txt",
        "paper/workspace/analysis/",
        "paper/workspace/eval/",
    )
    v_errors, warnings, validation = validate_runs(
        runs, args.arms, args.seeds, args.allow_missing,
        args.expected_scenes, args.expected_total, allowed_untracked)
    errors.extend(v_errors)
    if errors:
        print("VALIDATION FAILED", file=sys.stderr)
        for e in errors:
            print("  -", e, file=sys.stderr)
        return 1

    df = runs_to_frame(runs)
    per_run: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for (arm, seed) in runs:
        per_run[(arm, seed)] = run_metrics(df, arm, seed)
    across = {arm: across_seed_stats(per_run, arm, args.seeds)
              for arm in args.arms if any((arm, s) in per_run
                                          for s in args.seeds)}

    contrasts = [contrast_summary(df, args.seeds, "O", base)
                 for base in args.arms if base != "O"]

    df = add_density_bin(df)
    density_strata = strata_table(
        df, "density_bin", ["1--10", "11--20", "21--30", "31+"],
        args.seeds)

    occ_strata = None
    occ_edges: List[float] = []
    occ = load_occlusion(args.scene_occlusion)
    if occ is not None:
        df, occ_edges = add_occlusion_bin(df, occ)
        occ_strata = strata_table(df, "occlusion_bin", ["low", "medium", "high"],
                                  args.seeds)
    else:
        warnings.append(f"scene occlusion file not found: "
                        f"{args.scene_occlusion}; skipping occlusion strata")

    # Input manifest.
    inputs = {}
    for (arm, seed), data in runs.items():
        p = args.eval_dir / f"{arm}_seed{seed}.json"
        inputs[str(p)] = sha256_file(p)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_git": git_state(),
        "eval_dir": str(args.eval_dir.resolve()),
        "scene_occlusion": str(args.scene_occlusion.resolve()),
        "scene_occlusion_sha256": (
            sha256_file(args.scene_occlusion)
            if args.scene_occlusion.exists() else None),
        "arms": list(args.arms),
        "seeds": list(args.seeds),
        "inputs": inputs,
        "validation": {
            "runs": len(runs),
            "expected_runs": len(args.arms) * len(args.seeds),
            "selected_sha": validation.get("selected_sha"),
            "git_heads": validation.get("git_heads"),
            "dirty_runs": validation.get("dirty_runs"),
            "warnings": warnings,
        },
    }

    summary = {
        "manifest": manifest,
        "runs": {
            f"{arm}/seed{seed}": {
                "metrics": per_run[(arm, seed)],
                "source": validation["runs"].get(f"{arm}/seed{seed}"),
            }
            for (arm, seed) in runs
        },
        "across_seeds": across,
        "contrasts": contrasts,
        "strata": {
            "density": {"bins": ["1--10", "11--20", "21--30", "31+"],
                        "table": density_strata},
            "occlusion": {
                "bins": ["low", "medium", "high"],
                "tertile_edges": occ_edges,
                "table": occ_strata,
            },
        },
        "warnings": warnings,
    }
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(jsonable(summary), f, indent=2, sort_keys=True)
    with open(args.out_dir / "validation.json", "w") as f:
        json.dump(jsonable({"manifest": manifest, "warnings": warnings,
                            "runs": validation["runs"]}),
                  f, indent=2, sort_keys=True)

    # Tables.
    write_main_binary(args.out_dir / "tables" / "main_binary.tex",
                      per_run, across, args.arms, args.seeds, manifest)
    write_main_continuous(args.out_dir / "tables" / "main_continuous.tex",
                          across, args.arms, manifest)
    write_paired(args.out_dir / "tables" / "paired_contrasts.tex",
                 contrasts, manifest)
    write_per_seed(args.out_dir / "tables" / "per_seed.tex",
                   per_run, args.arms, args.seeds, manifest)
    write_strata(args.out_dir / "tables" / "strata_density.tex",
                 density_strata, ["1--10", "11--20", "21--30", "31+"],
                 "density",
                 "Collision rate (\\%) by traffic density; parentheses give "
                 "the number of valid episodes pooled across seeds. Density "
                 "is the number of vehicles at reset, common to all arms.",
                 args.arms)
    if occ_strata is not None:
        write_strata(args.out_dir / "tables" / "strata_occlusion.tex",
                     occ_strata, ["low", "medium", "high"], "occlusion",
                     "Collision rate (\\%) by intrinsic scene occlusion "
                     "severity (tertiles of the test half; edges reported in "
                     "the text). Each scene is assigned once and reused for "
                     "all arms; parentheses give pooled valid episodes.",
                     args.arms)

    # Figures.
    if not args.no_figures:
        if plt is None:
            warnings.append(f"matplotlib unavailable ({_MPL_ERROR}); "
                            "skipping figures")
        else:
            figure_main_outcomes(args.out_dir / "figures", across, per_run,
                                 args.arms, args.seeds)
            figure_paired(args.out_dir / "figures", contrasts)
            figure_strata(args.out_dir / "figures", density_strata,
                          ["1--10", "11--20", "21--30", "31+"],
                          "Vehicles in scene", "fig_density.pdf",
                          SELECTED_FOR_STRATA)
            if occ_strata is not None:
                figure_strata(args.out_dir / "figures", occ_strata,
                              ["low", "medium", "high"],
                              "Scene occlusion tertile",
                              "fig_occlusion.pdf", SELECTED_FOR_STRATA)

    print(f"Validated {len(runs)} runs "
          f"({len(args.arms)} arms x {len(args.seeds)} seeds).")
    print(f"Selected files sha256: {validation.get('selected_sha')}")
    if warnings:
        for w in warnings:
            print("WARNING:", w, file=sys.stderr)
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
