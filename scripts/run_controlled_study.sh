#!/usr/bin/env bash
# Drive the controlled study: 8 arms x 3 seeds, 10M env steps each.
#
#   JOBS=2 nohup setsid scripts/run_controlled_study.sh >/dev/null 2>&1 &
#
# JOBS=2 uses two GPUs and 12 waves; JOBS=4 uses four GPUs and 6 waves.
# Choose based on the last measured aggregate throughput (see
# paper/workspace/phase2_arms_and_compute.md §3.4).
#
# The driver is resumable: it skips arms whose final checkpoint exists and
# skips evaluations whose JSON already exists. A failed run is retried once;
# a second failure is logged and does not block the remaining waves.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/anchal/miniconda3/envs/nocturne-k80/bin/python}"
JOBS="${JOBS:-2}"
SEEDS=(42 123 456)
ALL_ARMS=(A B C C_vis D D_vis O C_hist)

if [ "$JOBS" -eq 2 ]; then
  WAVES=("A B" "C C_vis" "D D_vis" "O C_hist")
elif [ "$JOBS" -eq 4 ]; then
  WAVES=("A B C C_vis" "D D_vis O C_hist")
else
  echo "JOBS must be 2 or 4" >&2
  exit 2
fi

TRAIN_SCENARIO_PATH="${TRAIN_SCENARIO_PATH:-$ROOT/dataset/formatted_json_v2_no_tl_train}"
EVAL_SCENARIO_PATH="${EVAL_SCENARIO_PATH:-$ROOT/dataset/full_valid/formatted_json_v2_no_tl_valid}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoints/controlled_study}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/controlled_study}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/paper/workspace/eval}"
DRIVER_LOG="$LOG_ROOT/driver.log"
LAUNCHES_FILE="${LAUNCHES_FILE:-$ROOT/paper/workspace/launches.txt}"
export LAUNCHES_FILE
export MAX_ENV_STEPS="${MAX_ENV_STEPS:-10000000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
EVAL_NUM_FILES="${EVAL_NUM_FILES:--1}"
export PYTHON_BIN CHECKPOINT_ROOT LOG_ROOT TRAIN_SCENARIO_PATH
mkdir -p "$LOG_ROOT" "$EVAL_ROOT" "$(dirname "$LAUNCHES_FILE")"
echo "$$" > "$LOG_ROOT/driver.pid"

# Test/resume overrides. WAVES_OVERRIDE uses '|' between waves, e.g.
#   WAVES_OVERRIDE='A B|C C_vis' ARMS_OVERRIDE='A B C C_vis'
if [ -n "${SEEDS_OVERRIDE:-}" ]; then
  read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
fi
if [ -n "${WAVES_OVERRIDE:-}" ]; then
  IFS='|' read -r -a WAVES <<< "$WAVES_OVERRIDE"
fi
if [ -n "${ARMS_OVERRIDE:-}" ]; then
  read -r -a ALL_ARMS <<< "$ARMS_OVERRIDE"
fi

GIT_SHA="$(git -C "$ROOT" rev-parse HEAD)"
if [ ! -s "$LAUNCHES_FILE" ]; then
  printf '# controlled study launches\n# driver git sha: %s\n# arms: %s\n# seeds: %s\n# jobs: %s\n' \
    "$GIT_SHA" "${ALL_ARMS[*]}" "${SEEDS[*]}" "$JOBS" > "$LAUNCHES_FILE"
fi

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$DRIVER_LOG"
}

free_gb() {
  df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9'
}

eval_pids=()

launch_eval() {
  local arm="$1" seed="$2"
  local json="$EVAL_ROOT/${arm}_seed${seed}.json"
  if [ -s "$json" ]; then
    log "eval exists, skip $arm/$seed"
    return 0
  fi
  log "eval start $arm/$seed"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 "$PYTHON_BIN" \
    "$ROOT/examples/drl_collision_avoidance/evaluate.py" \
    --checkpoint "$CHECKPOINT_ROOT/$arm/seed_$seed/ddqn_final.pth" \
    --scenario_path "$EVAL_SCENARIO_PATH" \
    --split test --split_seed 0 --num_files "$EVAL_NUM_FILES" \
    --output_json "$json" --log_steps \
    > "$LOG_ROOT/eval_${arm}_seed${seed}.log" 2>&1 &
  eval_pids+=($!)
}

run_train() {
  local arm="$1" seed="$2" gpu="$3" attempt="$4"
  local ckpt="$CHECKPOINT_ROOT/$arm/seed_$seed/ddqn_final.pth"
  if [ -s "$ckpt" ]; then
    log "train exists, skip $arm/$seed"
    return 0
  fi
  local free
  free="$(free_gb)"
  if [ -n "$free" ] && [ "$free" -lt 8 ]; then
    log "ABORT wave: only ${free}G free on $ROOT (need >= 8G)"
    return 3
  fi
  log "train start $arm/$seed gpu=$gpu attempt=$attempt (free=${free}G)"
  local started elapsed
  started="$(date +%s)"
  "$ROOT/scripts/launch_controlled_study.sh" "$arm" "$seed" "$gpu" \
    > "$LOG_ROOT/console_${arm}_seed${seed}_attempt${attempt}.log" 2>&1
  local rc=$?
  elapsed=$(( $(date +%s) - started ))
  if [ "$rc" -eq 0 ] && [ -s "$ckpt" ]; then
    log "train done $arm/$seed wall=${elapsed}s"
    return 0
  fi
  log "train FAIL $arm/$seed rc=$rc attempt=$attempt wall=${elapsed}s"
  return 1
}

run_wave() {
  local seed="$1" wave="$2"
  read -r -a arms <<< "$wave"
  local pids=()
  local i arm

  for i in "${!arms[@]}"; do
    arm="${arms[$i]}"
    run_train "$arm" "$seed" "$i" 1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done

  # Retry missing final checkpoints once, still within the wave's GPU slots.
  local retry_pids=()
  for i in "${!arms[@]}"; do
    arm="${arms[$i]}"
    if [ ! -s "$CHECKPOINT_ROOT/$arm/seed_$seed/ddqn_final.pth" ]; then
      run_train "$arm" "$seed" "$i" 2 &
      retry_pids+=($!)
    fi
  done
  for pid in "${retry_pids[@]}"; do wait "$pid"; done

  for arm in "${arms[@]}"; do
    if [ -s "$CHECKPOINT_ROOT/$arm/seed_$seed/ddqn_final.pth" ]; then
      launch_eval "$arm" "$seed"
    else
      log "NO CHECKPOINT after retry: $arm/$seed"
    fi
  done
}

log "study start sha=$GIT_SHA jobs=$JOBS waves=${#WAVES[@]} seeds=${SEEDS[*]}"
log "train data: $TRAIN_SCENARIO_PATH"
log "eval data : $EVAL_SCENARIO_PATH (hash-split test half, split_seed=0)"
log "checkpoints: $CHECKPOINT_ROOT"
log "evals: $EVAL_ROOT"

for seed in "${SEEDS[@]}"; do
  for wave in "${WAVES[@]}"; do
    log "wave start seed=$seed arms=[$wave]"
    run_wave "$seed" "$wave"
    log "wave end   seed=$seed arms=[$wave]"
  done
done

log "training waves complete; waiting for ${#eval_pids[@]} eval job(s)"
for pid in "${eval_pids[@]}"; do
  if ! wait "$pid"; then
    log "eval job pid=$pid FAILED"
  fi
done

missing=0
for seed in "${SEEDS[@]}"; do
  for arm in "${ALL_ARMS[@]}"; do
    ckpt="$CHECKPOINT_ROOT/$arm/seed_$seed/ddqn_final.pth"
    json="$EVAL_ROOT/${arm}_seed${seed}.json"
    if [ ! -s "$ckpt" ]; then
      log "MISSING checkpoint $arm/$seed"
      missing=$((missing + 1))
    fi
    if [ ! -s "$json" ]; then
      log "MISSING eval $arm/$seed"
      missing=$((missing + 1))
    fi
  done
done

log "study complete sha=$GIT_SHA missing=$missing"
exit "$([ "$missing" -eq 0 ] && echo 0 || echo 1)"
