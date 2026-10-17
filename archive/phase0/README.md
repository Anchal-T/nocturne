# Phase 0 archive (created 2026-10-17)

Evidence retired from the paper draft before the controlled study. Nothing
here is a valid benchmark result; it is kept for provenance only.

## Contents

- `model_manifest.json` — audit of every checkpoint referenced by the draft:
  path, sha256, size, checkpoint metadata (observation dimension, learner
  steps, epsilon, world size), TensorBoard episode counts, and the paper rows
  each model backed. Regenerate with:
  `/home/anchal/miniconda3/envs/nocturne-k80/bin/python scripts/make_model_manifest.py`
- `agent_results_transcript.txt` — console transcript of the archived March
  DDQN run (moved out of `paper/runs data/`).
- `figures/training_curves.png`, `figures/evaluation_summary.png` — the two
  figures rendered from the retired evidence (removed from `paper/figures/`).

## Why these were retired

- The archived DDQN result (95.33% goal) was evaluated by repeating one
  fixture scene, not a held-out split.
- Both sides of the occlusion ablation (`checkpoints/ddqn_with_occlusion`,
  `checkpoints/ddqn_without_occlusion`) are untrained startup saves
  (`train_steps=0`); no trained no-occlusion checkpoint exists.
- The other comparison rows came from smoke checkpoints or other-family
  checkpoints evaluated with heterogeneous splits and aggregation.
- All prior evaluations used a nondeterministic evaluator; the replacement
  evaluator is enumerated, seeded, and paired across arms.

## Still unresolved in the workspace (not touched)

- `paper/runs data/matched_eval/` (untracked) — source JSONs for the removed
  matched comparison.
- `paper/ppt_content.md`, `paper/ppt_training_data.csv` (untracked) — quote
  the removed numbers.
- `paper/runs data/eval_baseline_fixed.json` and
  `paper/runs data/framework_rollout_results.json` — retained as provenance
  for the audit rows recorded in `paper/runs data/evaluation_summary.json`.
