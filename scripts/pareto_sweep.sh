#!/bin/bash
# Pareto trade-off sweep: train with different collision penalties.
# Usage: bash scripts/pareto_sweep.sh [--dry-run]
#
# Produces checkpoints at checkpoints/pareto/penalty_<value>/ddqn_final.pth
# Then evaluate each with: python -m examples.drl_collision_avoidance.evaluate \
#   --checkpoint checkpoints/pareto/penalty_<value>/ddqn_final.pth --num_episodes 500

set -e

PENALTIES=(10 25 50 100 250 500 1000)
SEED=42
NUM_EPISODES=100000  # shorter runs for sweep
DRY_RUN=false

if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

for penalty in "${PENALTIES[@]}"; do
    CKPT_DIR="checkpoints/pareto/penalty_${penalty}"
    echo "=== Training with collision_penalty=${penalty} ==="
    CMD="python -m examples.drl_collision_avoidance.train \
        seed=${SEED} \
        reward.collision_penalty=${penalty} \
        drl.num_episodes=${NUM_EPISODES} \
        drl.checkpoint_dir=${CKPT_DIR}"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] $CMD"
    else
        $CMD
    fi
done

echo ""
echo "=== Sweep complete. Evaluate with: ==="
for penalty in "${PENALTIES[@]}"; do
    echo "  python -m examples.drl_collision_avoidance.evaluate --checkpoint checkpoints/pareto/penalty_${penalty}/ddqn_final.pth --num_episodes 500"
done
