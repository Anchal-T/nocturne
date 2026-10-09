#!/usr/bin/env bash
# Launch DDQN with one in-process learner per GPU.
# Background ProcessLearner is disabled under torchrun.
#
#   CUDA_VISIBLE_DEVICES=1,2 ./scripts/run_ddqn_ddp.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 ./scripts/run_ddqn_ddp.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NPROC="${NPROC:-2}"
VISIBLE="${CUDA_VISIBLE_DEVICES:-1,2}"
MASTER_PORT="${MASTER_PORT:-29504}"
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="$VISIBLE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec python -m torch.distributed.run \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  -m examples.drl_collision_avoidance.train \
  drl.train_in_background=false \
  "${EXTRA_ARGS[@]}"
