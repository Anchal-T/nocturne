#!/usr/bin/env bash
# Launch R-MAPPO / MAPPO / Lagrangian with one process per GPU.
#
# Two-GPU validation (does not touch GPUs 0 or 3):
#   CUDA_VISIBLE_DEVICES=1,2 ./scripts/run_ppo_ddp.sh
#
# Four-GPU (when 0 and 3 are free):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 ./scripts/run_ppo_ddp.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NPROC="${NPROC:-2}"
VISIBLE="${CUDA_VISIBLE_DEVICES:-1,2}"
MASTER_PORT="${MASTER_PORT:-29501}"
EXTRA_ARGS=("$@")

TMPDIR="${TMPDIR:-${HOME}/tmp}"
mkdir -p "$TMPDIR"
export TMPDIR
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$TMPDIR/torchinductor}"
export CUDA_VISIBLE_DEVICES="$VISIBLE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec python -m torch.distributed.run \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  examples/on_policy_files/nocturne_runner.py \
  algorithm.distributed=True \
  "${EXTRA_ARGS[@]}"
