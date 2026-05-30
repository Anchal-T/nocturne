#!/bin/bash
# Multi-seed training for statistical evaluation.
# Usage: bash scripts/multi_seed_train.sh [METHOD] [--dry-run]
#   METHOD: ddqn (default) or ddqn_occ
# Examples:
#   bash scripts/multi_seed_train.sh                  # trains 5 seeds of ddqn
#   bash scripts/multi_seed_train.sh ddqn_occ         # trains 5 seeds of ddqn_occ
#   bash scripts/multi_seed_train.sh ddqn_occ --dry-run

set -e

SEEDS=(42 123 456 789 1024)
METHOD="ddqn"
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        ddqn|ddqn_occ) METHOD="$arg" ;;
        *) echo "Unknown arg: $arg (use ddqn, ddqn_occ, or --dry-run)"; exit 1 ;;
    esac
done

case "$METHOD" in
    ddqn) USE_OCC=false ;;
    ddqn_occ) USE_OCC=true ;;
esac

echo "=== Multi-seed training: method=${METHOD} ==="
for seed in "${SEEDS[@]}"; do
    CKPT_DIR="checkpoints/${METHOD}/seed_${seed}"
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] python -m examples.drl_collision_avoidance.train seed=${seed} drl.checkpoint_dir=${CKPT_DIR} occupancy_grid.use_occlusion=${USE_OCC}"
    else
        echo "--- Seed ${seed} ---"
        python -m examples.drl_collision_avoidance.train \
            seed=${seed} \
            drl.checkpoint_dir=${CKPT_DIR} \
            occupancy_grid.use_occlusion=${USE_OCC}
    fi
done

echo ""
echo "=== Training complete. Evaluate all seeds: ==="
for seed in "${SEEDS[@]}"; do
    echo "  python -m examples.drl_collision_avoidance.evaluate --checkpoint checkpoints/${METHOD}/seed_${seed}/ddqn_final.pth --num_episodes 500"
done
