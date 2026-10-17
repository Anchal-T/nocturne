#!/usr/bin/env bash
# Launch one controlled-study training run.
#
#   scripts/launch_controlled_study.sh <ARM> <SEED> <GPU>
#
# Arms: A B C C_vis D D_vis O C_hist (see
# paper/workspace/phase2_arms_and_compute.md). The exact command and the git
# sha are appended to paper/workspace/launches.txt before exec.
#
# Everything here is pinned. Changes require a new commit and a new sha;
# do not edit flags in place once a wave has started.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <ARM> <SEED> <GPU>" >&2
  exit 2
fi

ARM="$1"
SEED="$2"
GPU="$3"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/anchal/miniconda3/envs/nocturne-k80/bin/python}"
TRAIN_SCENARIO_PATH="${TRAIN_SCENARIO_PATH:-$ROOT/dataset/formatted_json_v2_no_tl_train}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoints/controlled_study}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/controlled_study}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-10000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
LAUNCHES_FILE="${LAUNCHES_FILE:-$ROOT/paper/workspace/launches.txt}"

case "$ARM" in
  A)        VIS=false; STATS=false; MASK=false; HF=1; HS=1; REW=privileged ;;
  B)        VIS=true;  STATS=false; MASK=false; HF=1; HS=1; REW=privileged ;;
  C)        VIS=true;  STATS=true;  MASK=false; HF=1; HS=1; REW=privileged ;;
  C_vis)    VIS=true;  STATS=true;  MASK=false; HF=1; HS=1; REW=visible ;;
  D)        VIS=true;  STATS=false; MASK=true;  HF=1; HS=1; REW=privileged ;;
  D_vis)    VIS=true;  STATS=false; MASK=true;  HF=1; HS=1; REW=visible ;;
  O)        VIS=true;  STATS=true;  MASK=true;  HF=1; HS=1; REW=privileged ;;
  C_hist)   VIS=true;  STATS=false; MASK=false; HF=3; HS=5; REW=privileged ;;
  *)
    echo "unknown arm: $ARM" >&2
    exit 2
    ;;
esac

case "$SEED" in
  42|123|456) ;;
  *) echo "warning: seed $SEED is outside the pre-registered {42,123,456}" >&2 ;;
esac

CKPT_DIR="$CHECKPOINT_ROOT/$ARM/seed_$SEED"
HYDRA_DIR="$LOG_ROOT/$ARM/seed_$SEED"
mkdir -p "$CKPT_DIR" "$HYDRA_DIR" "$(dirname "$LAUNCHES_FILE")"

GIT_SHA="$(git -C "$ROOT" rev-parse HEAD)"
CMD=(
  "$PYTHON_BIN" -m examples.drl_collision_avoidance.train
  "seed=$SEED"
  "scenario_path=$TRAIN_SCENARIO_PATH"
  "scenario_split=train"
  "drl.checkpoint_dir=$CKPT_DIR"
  "drl.max_env_steps=$MAX_ENV_STEPS"
  "drl.min_runtime_seconds=0"
  "drl.num_workers=4"
  "drl.num_envs_per_worker=12"
  "drl.num_envs=48"
  "drl.vec_env_mode=sync"
  "drl.train_freq=128"
  "drl.batch_size=512"
  "drl.save_interval=$SAVE_INTERVAL"
  "occupancy_grid.visibility_filter=$VIS"
  "occupancy_grid.use_occlusion=$STATS"
  "occupancy_grid.occlusion_mask=$MASK"
  "occupancy_grid.history_frames=$HF"
  "occupancy_grid.history_stride=$HS"
  "reward.reward_ttz_source=$REW"
  "hydra.run.dir=$HYDRA_DIR"
)

{
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GIT_SHA" "$ARM" "$SEED" "$GPU" \
    >> "$LAUNCHES_FILE"
  printf 'CMD: %s\n' "${CMD[*]}" >> "$LAUNCHES_FILE"
} 9>>"$LAUNCHES_FILE"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $ARM seed=$SEED gpu=$GPU sha=$GIT_SHA"
echo "  ckpt: $CKPT_DIR"
echo "  log : $HYDRA_DIR"

exec env CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 "${CMD[@]}"
