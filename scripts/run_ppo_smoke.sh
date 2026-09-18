#!/usr/bin/env bash
# Conservative 400k-step R-MAPPO smoke run.
#
# Does not change global PPO defaults. Extra Hydra overrides pass through.
#
# Single GPU (recommended for this small model):
#   CUDA_VISIBLE_DEVICES=1 NPROC=1 ./scripts/run_ppo_smoke.sh
#
# Two GPU:
#   CUDA_VISIBLE_DEVICES=1,2 NPROC=2 ./scripts/run_ppo_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TMPDIR="${TMPDIR:-${HOME}/tmp}"
mkdir -p "$TMPDIR"
export TMPDIR
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$TMPDIR/torchinductor}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

export NPROC="${NPROC:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$TMPDIR/ppo_smoke/$STAMP}"

exec "$ROOT/scripts/run_ppo_ddp.sh" \
  algorithm.num_env_steps=400000 \
  algorithm.n_rollout_threads=4 \
  algorithm.n_eval_rollout_threads=1 \
  algorithm.ppo_epoch=3 \
  algorithm.num_mini_batch=2 \
  algorithm.save_interval=100 \
  algorithm.eval_interval=100 \
  algorithm.eval_episodes=20 \
  algorithm.find_unused_parameters=False \
  algorithm.use_centralized_V=False \
  scenario_cache_size=32 \
  scenario_pool_size=16 \
  resample_pool_interval=0 \
  scenario.max_visible_road_points=8 \
  subscriber.use_occlusion_features=False \
  subscriber.n_frames_stacked=1 \
  hydra.run.dir="$RUN_DIR" \
  "$@"
