#!/usr/bin/env bash
# Launch Sample Factory APPO. SF 1.123 has one learner; extra GPUs are optional
# actor-side inference via algorithm.actor_worker_gpus, not extra SGD learners.
#
#   CUDA_VISIBLE_DEVICES=1,2 ./scripts/run_sample_factory_multi_gpu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VISIBLE="${CUDA_VISIBLE_DEVICES:-1,2}"
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="$VISIBLE"

exec python examples/sample_factory_files/run_sample_factory.py \
  algorithm=APPO \
  "${EXTRA_ARGS[@]}"
