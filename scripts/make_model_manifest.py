#!/usr/bin/env python3
"""Phase-0 model manifest generator.

Audit of every checkpoint referenced by the paper draft. For each file it
records existence, size, mtime, sha256, and (for torch checkpoints)
architecture/training metadata. Groups carry the audit status:
trained / untrained startup save / smoke / archived pilot / other family.

Output: archive/phase0/model_manifest.json (created next to this tree).

Run with the nocturne environment python (needs torch + tensorboard):

  cd /home/anchal/anchal/nocturne
  /home/anchal/miniconda3/envs/nocturne-k80/bin/python scripts/make_model_manifest.py
"""

import glob
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_PATH = os.path.join(REPO, "archive", "phase0", "model_manifest.json")

META_KEYS = (
    "obs_dim", "grid_size", "grid_channels", "grid_rows", "grid_cols",
    "hidden_layers", "dueling", "noisy", "mlp_depth", "train_steps",
    "epsilon", "use_muon", "world_size", "episodes_completed", "env_steps",
    "global_step", "update_step", "training_iteration", "timesteps_total",
)

STATUS_LEGEND = {
    "archived_pilot": "Historical run behind a removed result row; evaluated outside any held-out split.",
    "trained": "Completed training with saved weights; budget recorded in metadata/TensorBoard.",
    "untrained_startup_save": "Weights saved at process start (train_steps=0, epsilon=1); the run never trained to a checkpoint.",
    "failed_attempt": "Launch attempt that saved only startup weights.",
    "smoke": "Smoke/validation artifact; not a completed training run.",
    "aborted_no_weights": "Aborted run; TensorBoard logs only, no weight file saved.",
    "full_training_other_family": "Completed training run outside the DDQN family (retired from the paper).",
    "other_family": "Checkpoint family outside the DDQN family (retired from the paper).",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(value):
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def torch_metadata(path):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        return {"_error": f"torch unavailable: {exc}"}
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}
    if isinstance(obj, dict):
        meta = {k: _clean(obj[k]) for k in META_KEYS if k in obj}
        if meta:
            if isinstance(meta.get("epsilon"), float):
                meta["epsilon"] = round(meta["epsilon"], 6)
            return meta
        n_params = 0
        has_tensor = False
        for value in obj.values():
            if hasattr(value, "numel"):
                n_params += int(value.numel())
                has_tensor = True
        if has_tensor:
            return {"content": "state_dict", "n_params": n_params}
    return {"content": type(obj).__name__}


def tb_summary(tb_dir):
    if not tb_dir or not os.path.isdir(tb_dir):
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:  # pragma: no cover
        return {"error": f"tensorboard unavailable: {exc}"}
    out = {"event_files": [], "last_episode_completed": None, "n_episode_points": 0}
    for event_file in sorted(glob.glob(os.path.join(tb_dir, "events*"))):
        out["event_files"].append(os.path.basename(event_file))
        try:
            acc = EventAccumulator(event_file, size_guidance={"scalars": 0})
            acc.Reload()
            if "train/episode_reward" in acc.Tags().get("scalars", []):
                vals = acc.Scalars("train/episode_reward")
                if vals:
                    out["n_episode_points"] += len(vals)
                    last = vals[-1].step
                    if out["last_episode_completed"] is None or last > out["last_episode_completed"]:
                        out["last_episode_completed"] = last
        except Exception as exc:
            out.setdefault("errors", []).append(f"{os.path.basename(event_file)}: {exc}")
    return out


def file_record(path):
    absolute = path if os.path.isabs(path) else os.path.join(REPO, path)
    record = {
        "path": path,
        "external": os.path.isabs(path),
        "exists": os.path.exists(absolute),
    }
    if not record["exists"]:
        return record
    st = os.stat(absolute)
    record["size_bytes"] = st.st_size
    record["mtime"] = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    record["sha256"] = sha256(absolute)
    if path.endswith((".pth", ".pt")) or path.endswith("checkpoint-1"):
        record["metadata"] = torch_metadata(absolute)
    return record


def group(id_, dir_path, family, role, status, status_note, paper_refs,
          files, tb_logs=None, external_dir=None):
    entry = {
        "id": id_,
        "family": family,
        "role": role,
        "status": status,
        "status_note": status_note,
        "paper_references": paper_refs,
        "dir": dir_path,
        "external": bool(external_dir),
        "files": [],
    }
    base = external_dir or REPO
    for name in sort_episodes(files):
        full = os.path.join(base, dir_path, name) if dir_path else os.path.join(base, name)
        entry["files"].append(file_record(full))
    tb = tb_summary(os.path.join(base, dir_path, tb_logs) if tb_logs else None)
    if tb:
        entry["tb_logs"] = tb
    return entry


def sort_episodes(files):
    def key(name):
        stem = os.path.basename(name)
        digits = "".join(ch for ch in stem if ch.isdigit())
        return (0, int(digits)) if stem.startswith("ddqn_ep") and digits else (1, stem)
    return sorted(files, key=key)


DDQN_BIG = "ddqn_ep10000.pth ddqn_ep20000.pth ddqn_ep30000.pth ddqn_ep40000.pth " \
           "ddqn_ep50000.pth ddqn_ep60000.pth ddqn_final.pth".split()

GROUPS = [
    group(
        "ddqn_archived_march_run",
        "outputs/2026-03-11/20-29-53/checkpoints/drl_collision_avoidance",
        "DDQN (archived, 600-D observation)",
        "Source run of the archived 95.33%/5.13% result, evaluated on one repeated fixture scene.",
        "archived_pilot",
        "Old 20x10-grid observation without target/TTZ/occlusion extras, hidden [256,256,256,64]. "
        "The training transcript was moved out of paper/ on 2026-10-17.",
        [
            "main.tex tab:main row 'Archived DDQN result' (removed 2026-10-17)",
            "paper/runs data/eval_baseline_fixed.json",
            "paper/runs data/evaluation_summary.json",
        ],
        ["ddqn_ep100000.pth", "ddqn_ep200000.pth", "ddqn_ep300000.pth",
         "ddqn_ep400000.pth", "ddqn_ep500000.pth", "ddqn_final.pth"],
        tb_logs="tb_logs",
    ),
    group(
        "ddqn_with_occlusion_1gpu_resume",
        "checkpoints/ddqn_with_occlusion_1gpu_resume",
        "DDQN",
        "Trained with-occlusion model; matched-eval row 'DDQN + occlusion (resume)'.",
        "trained",
        "613-D observation with occlusion features. Last TensorBoard episode 63,334; "
        "final checkpoint train_steps=157,675, epsilon=0.05.",
        [
            "main.tex tab:main row 'DDQN + occlusion (resume)' via matched_eval (removed 2026-10-17)",
            "paper/runs data/matched_eval/ddqn_with_occlusion_resume.json",
        ],
        DDQN_BIG,
        tb_logs="tb_logs",
    ),
    group(
        "ddqn_with_occlusion_3gpu_async",
        "checkpoints/ddqn_with_occlusion_3gpu_async",
        "DDQN",
        "Trained with-occlusion model; matched-eval row 'DDQN + occlusion (async 70k)'.",
        "trained",
        "613-D observation with occlusion features. Last TensorBoard episode 76,604; "
        "ep70000 checkpoint train_steps=135,704.",
        [
            "main.tex tab:main row 'DDQN + occlusion (async 70k)' via matched_eval (removed 2026-10-17)",
            "paper/runs data/matched_eval/ddqn_with_occlusion_async70k.json",
        ],
        ["ddqn_ep10000.pth", "ddqn_ep20000.pth", "ddqn_ep30000.pth", "ddqn_ep40000.pth",
         "ddqn_ep50000.pth", "ddqn_ep60000.pth", "ddqn_ep70000.pth"],
        tb_logs="tb_logs",
    ),
    group(
        "ddqn_with_occlusion_startup",
        "checkpoints/ddqn_with_occlusion",
        "DDQN",
        "Source of matched-eval row 'DDQN + occlusion (Sep18)' — actually an untrained startup save.",
        "untrained_startup_save",
        "ddqn_final.pth has train_steps=0, epsilon=1.0 (weights saved at process start). "
        "The run was retried and aborted after ~1,136 episodes; no trained weights were saved.",
        [
            "main.tex tab:main row 'DDQN + occlusion (Sep18)' via matched_eval (removed 2026-10-17)",
            "paper/runs data/matched_eval/ddqn_with_occlusion_sep18.json",
        ],
        ["ddqn_emergency.pth", "ddqn_final.pth"],
        tb_logs="tb_logs",
    ),
    group(
        "ddqn_without_occlusion_startup",
        "checkpoints/ddqn_without_occlusion",
        "DDQN",
        "Source of matched-eval row 'minus occlusion' — the ablation was never trained.",
        "untrained_startup_save",
        "ddqn_final.pth has train_steps=0, epsilon=1.0 (weights saved at process start). "
        "No trained no-occlusion checkpoint exists anywhere in the workspace.",
        [
            "main.tex tab:main row 'minus occlusion' via matched_eval (removed 2026-10-17)",
            "paper/runs data/matched_eval/ddqn_without_occlusion.json",
        ],
        ["ddqn_emergency.pth", "ddqn_final.pth"],
        tb_logs="tb_logs",
    ),
    group(
        "ddqn_with_occlusion_3gpu_failed",
        "checkpoints/ddqn_with_occlusion_3gpu",
        "DDQN",
        "Failed 3-GPU launch attempt.",
        "failed_attempt",
        "ddqn_final.pth world_size=3, train_steps=0; only startup weights were saved.",
        ["unreferenced by the paper"],
        ["ddqn_emergency.pth", "ddqn_final.pth"],
        tb_logs="tb_logs",
    ),
    group(
        "ddqn_with_occlusion_retry_aborted",
        "checkpoints/ddqn_with_occlusion_retry",
        "DDQN",
        "Aborted retry of the with-occlusion run.",
        "aborted_no_weights",
        "Only TensorBoard logs (about 1,136 episodes); no weight file saved.",
        ["unreferenced by the paper"],
        [],
        tb_logs="tb_logs",
    ),
    group(
        "ddqn_without_occlusion_retry_aborted",
        "checkpoints/ddqn_without_occlusion_retry",
        "DDQN",
        "Aborted retry of the no-occlusion run.",
        "aborted_no_weights",
        "Only TensorBoard logs; no weight file saved. No trained no-occlusion checkpoint exists.",
        ["unreferenced by the paper"],
        [],
        tb_logs="tb_logs",
    ),
    group(
        "ddqn_validation_smoke",
        "checkpoints/drl_collision_avoidance",
        "DDQN",
        "Validation smoke checkpoint; matched-eval / audit row 'DDQN validation'.",
        "smoke",
        "ddqn_final.pth train_steps=3, noisy=False; not a completed training run.",
        [
            "main.tex tab:main row 'DDQN validation' (removed 2026-10-17)",
            "paper/runs data/matched_eval/ddqn_ddp_validation.json",
        ],
        ["ddqn_emergency.pth", "ddqn_final.pth"],
        tb_logs="tb_logs",
    ),
    group(
        "ddqn_ddp_validation_copy",
        ".",
        "DDQN",
        "External copy of the DDQN validation smoke checkpoint referenced by the audit JSON.",
        "smoke",
        "Same artifact as checkpoints/drl_collision_avoidance/ddqn_final.pth by role; "
        "referred to by paper/runs data/evaluation_summary.json.",
        ["paper/runs data/evaluation_summary.json"],
        ["ddqn_final.pth"],
        external_dir="/home/anchal/tmp/ddqn_ddp_validation",
    ),
    group(
        "bc_ddp_validation_smoke",
        ".",
        "behavior cloning",
        "BC smoke checkpoint behind the audit row 'BC'.",
        "smoke",
        "train_steps=1 (validation smoke artifact).",
        [
            "main.tex tab:main row 'BC' (removed 2026-10-17)",
            "paper/runs data/matched_eval/bc_validation.json",
        ],
        ["bc_ddp_validation.pth"],
        external_dir="/home/anchal/tmp",
    ),
    group(
        "crl_large_full",
        "checkpoints",
        "contrastive RL",
        "Completed CRL checkpoint behind the audit row 'CRL'.",
        "full_training_other_family",
        "episodes_completed=400001, train_steps=648241; retired from the paper's table "
        "(other family, different split semantics).",
        [
            "main.tex tab:main row 'CRL' (removed 2026-10-17)",
            "paper/runs data/matched_eval/crl_large.json",
        ],
        ["crl_final.pth"],
        external_dir="/home/anchal/tmp/crl_large",
    ),
    group(
        "waymo_bc_epoch100",
        "train_logs/2026_09_05_19_41_29",
        "Waymo imitation learning",
        "Completed Waymo behavior-cloning checkpoint behind the audit row 'Waymo BC'.",
        "full_training_other_family",
        "Epoch-100 checkpoint; retired from the paper's table (other family, full-split "
        "scenario-level aggregation).",
        ["main.tex tab:main row 'Waymo BC' (removed 2026-10-17)"],
        ["model_100.pth"],
        external_dir="/home/anchal/tmp/imitation_resume",
    ),
    group(
        "rmappo_multi_gpu_long",
        ".",
        "PPO",
        "R-MAPPO checkpoints behind the audit row 'R-MAPPO'.",
        "other_family",
        "Training length not encoded in the result artifact; run2/models is empty. "
        "Row retired from the paper on 2026-10-17.",
        ["main.tex tab:main row 'R-MAPPO' (removed 2026-10-17)"],
        [
            "run1/models/actor.pt", "run1/models/critic.pt", "run1/models/trainer_state.pt",
            "run3/models/actor.pt", "run3/models/critic.pt", "run3/models/trainer_state.pt",
            "run4/models/actor.pt", "run4/models/critic.pt", "run4/models/trainer_state.pt",
        ],
        external_dir="/home/anchal/tmp/ppo_multi_gpu_long_v2",
    ),
    group(
        "rllib_ppo_smoke",
        "PPO/PPO_nocturne_a424a_00000_0_2026-09-01_18-19-16/checkpoint_000001",
        "RLlib PPO",
        "Framework smoke checkpoint behind the audit row 'RLlib PPO'.",
        "smoke",
        "training_iteration=1, timesteps_total=512; retired from the paper's table.",
        ["main.tex tab:main row 'RLlib PPO' (removed 2026-10-17)"],
        ["checkpoint-1", "checkpoint-1.tune_metadata"],
        external_dir="/home/anchal/tmp/rllib_ddp_results",
    ),
    group(
        "sf_appo_smoke",
        "checkpoint_p0",
        "Sample Factory/APPO",
        "Framework smoke checkpoint behind the audit row 'Sample Factory APPO'.",
        "smoke",
        "Checkpoint suffix records 480 environment steps; retired from the paper's table.",
        ["main.tex tab:main row 'Sample Factory APPO' (removed 2026-10-17)"],
        ["checkpoint_000000012_480.pth"],
        external_dir="/home/anchal/tmp/sf_results/sf_ddp_smoke2",
    ),
]


def main():
    try:
        commit = subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = None

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/make_model_manifest.py",
        "repo": REPO,
        "repo_commit": commit,
        "purpose": (
            "Audit of every model checkpoint referenced by the paper draft, created during "
            "the 2026-10-17 purge of invalid evidence. Maps each checkpoint family to the "
            "audit rows it backed and records its true training state."
        ),
        "status_legend": STATUS_LEGEND,
        "groups": GROUPS,
    }

    total_files = 0
    total_bytes = 0
    status_counts = {}
    for grp in GROUPS:
        status_counts[grp["status"]] = status_counts.get(grp["status"], 0) + 1
        for rec in grp["files"]:
            if rec["exists"]:
                total_files += 1
                total_bytes += rec["size_bytes"]
    manifest["summary"] = {
        "groups": len(GROUPS),
        "model_files_present": total_files,
        "model_total_bytes": total_bytes,
        "groups_by_status": status_counts,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"wrote {OUT_PATH}")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
