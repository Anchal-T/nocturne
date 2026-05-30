#!/bin/bash
# Multi-seed training for statistical evaluation.
# Usage: bash scripts/multi_seed_train.sh [--method ddqn|ddqn_occ] [--dry-run]
#
# Trains 5 seeds, then evaluate each with the unified eval script.

set -e

SEEDS=(42 123 456 789 1024)
METHOD="${1:-ddqn}"
DRY_RUN=false
if [[ "$2" == "--dry-run" ]] || [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    METHOD="${METHOD/--dry-run/}"
    METHOD="${METHOD:-ddqn}"
fi

# Method-specific config overrides
case "$METHOD" in
    ddqn)
        EXTRA_ARGS="occupancy_grid.use_occlusion=false"
        ;;
    ddqn_occ)
        EXTRA_ARGS="occupancy_grid.use_occlusion=true"
        ;;
    *)
        echo "Unknown method: $METHOD (use ddqn or ddqn_occ)"
        exit 1
        ;;
esac

echo "=== Multi-seed training: method=${METHOD} ==="
for seed in "${SEEDS[@]}"; do
    CKPT_DIR="checkpoints/${METHOD}/seed_${seed}"
    CMD="python -m examples.drl_collision_avoidance.train \
        seed=${seed} \
        drl.checkpoint_dir=${CKPT_DIR} \
        ${EXTRA_ARGS}"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] $CMD"
    else
        echo "--- Seed ${seed} ---"
        $CMD
    fi
done

echo ""
echo "=== Training complete. Evaluate all seeds: ==="
for seed in "${SEEDS[@]}"; do
    echo "  python -m examples.drl_collision_avoidance.evaluate --checkpoint checkpoints/${METHOD}/seed_${seed}/ddqn_final.pth --num_episodes 500"
done
