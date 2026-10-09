#!/usr/bin/env bash
# Launch RLlib PPO with Ray 1.11 legacy num_gpus (one shared policy).
#
# Two-GPU validation (does not touch GPUs 0 or 3):
#   CUDA_VISIBLE_DEVICES=1,2 ./scripts/run_rllib_multi_gpu.sh
#
# Four-GPU:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/run_rllib_multi_gpu.sh rllib.num_gpus=4
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VISIBLE="${CUDA_VISIBLE_DEVICES:-1,2}"
IFS=',' read -r -a GPU_IDS <<< "$VISIBLE"
NUM_GPUS="${#GPU_IDS[@]}"
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="$VISIBLE"

exec python examples/rllib_files/run_rllib.py \
  "rllib.num_gpus=${NUM_GPUS}" \
  "${EXTRA_ARGS[@]}"
