#!/usr/bin/env bash
# Launch Waymo behavioral cloning with one process per GPU.
#
#   CUDA_VISIBLE_DEVICES=1,2 ./scripts/run_imitation_ddp.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 ./scripts/run_imitation_ddp.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NPROC="${NPROC:-2}"
VISIBLE="${CUDA_VISIBLE_DEVICES:-1,2}"
MASTER_PORT="${MASTER_PORT:-29502}"
EXTRA_ARGS=("$@")

TMPDIR="${TMPDIR:-${HOME}/tmp}"
mkdir -p "$TMPDIR"
export TMPDIR
export CUDA_VISIBLE_DEVICES="$VISIBLE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec python -m torch.distributed.run \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  examples/imitation_learning/train.py \
  "${EXTRA_ARGS[@]}"
