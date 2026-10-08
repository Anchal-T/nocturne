#!/usr/bin/env bash
# Controlled ablation for the reviewer response.
# This starts training. It refuses to run unless NOCTURNE_ALLOW_TRAINING=1,
# which should be set only on the training machine.
set -euo pipefail

if [[ "${NOCTURNE_ALLOW_TRAINING:-}" != "1" ]]; then
  echo "Refusing to train on this machine."
  echo "Set NOCTURNE_ALLOW_TRAINING=1 on the training machine, then re-run."
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${ROOT}/paper/runs data/ablation"
mkdir -p "$OUT"

# Every arm keeps the ray-cast visibility filter except the full-observability
# oracle. reward_ttz_source=ground_truth is shared, so the TTZ penalty is not
# a second difference between arms. Seeds differ only by the training seed.
train_arm() {
  local arm="$1"
  local seed="$2"
  shift 2
  local ckpt="checkpoints/ablation/${arm}/seed_${seed}"
  python examples/drl_collision_avoidance/train.py \
    seed="$seed" \
    drl.checkpoint_dir="$ckpt" \
    occupancy_grid.use_occlusion=false \
    reward.reward_ttz_source=ground_truth \
    "$@"
}

eval_arm() {
  local arm="$1"
  local seed="$2"
  local split="$3"
  local ckpt="checkpoints/ablation/${arm}/seed_${seed}/ddqn_final.pth"
  local dest="${OUT}/${arm}/seed_${seed}"
  mkdir -p "$dest"
  python examples/drl_collision_avoidance/evaluate.py \
    --checkpoint "$ckpt" \
    --scenario_split valid \
    --heldout_split "$split" \
    --num_episodes 1500 \
    --num_files -1 \
    --method "$arm" \
    --output_json "${dest}/eval_${split}.json"
}

SEEDS=(42 43 44)

for seed in "${SEEDS[@]}"; do
  train_arm A_no_occ_no_risk "$seed" \
    occupancy_grid.visibility_filter=true \
    occupancy_grid.occlusion_features=false \
    reward.ttz_reward_scale=0
  train_arm B_occ_no_risk "$seed" \
    occupancy_grid.visibility_filter=true \
    occupancy_grid.occlusion_features=true \
    reward.ttz_reward_scale=0
  train_arm C_no_occ_risk "$seed" \
    occupancy_grid.visibility_filter=true \
    occupancy_grid.occlusion_features=false \
    reward.ttz_reward_scale=0.5
  train_arm D_occ_risk "$seed" \
    occupancy_grid.visibility_filter=true \
    occupancy_grid.occlusion_features=true \
    reward.ttz_reward_scale=0.5
  train_arm O_full_obs_risk "$seed" \
    occupancy_grid.visibility_filter=false \
    occupancy_grid.occlusion_features=false \
    reward.ttz_reward_scale=0.5
done

# Threshold sweep on the proposed observation, one seed. tau=3 is arm D.
for tau in 1 2 4 5; do
  train_arm "D_tau_${tau}" 42 \
    occupancy_grid.visibility_filter=true \
    occupancy_grid.occlusion_features=true \
    reward.ttz_reward_scale=0.5 \
    reward.ttz_safe_threshold="$tau"
done

for scale in 0.25 1.0; do
  train_arm "D_scale_${scale}" 42 \
    occupancy_grid.visibility_filter=true \
    occupancy_grid.occlusion_features=true \
    reward.ttz_reward_scale="$scale" \
    reward.ttz_safe_threshold=3
done

for seed in "${SEEDS[@]}"; do
  for arm in A_no_occ_no_risk B_occ_no_risk C_no_occ_risk D_occ_risk O_full_obs_risk; do
    eval_arm "$arm" "$seed" val
    eval_arm "$arm" "$seed" test
  done
done

# Select the proposed checkpoint on val before quoting test numbers.
# Robustness probes reuse the seed-42 proposed checkpoint and change only eval.
D42="checkpoints/ablation/D_occ_risk/seed_42/ddqn_final.pth"
ROB="${OUT}/D_occ_risk/seed_42"
for p in 0.25 0.5; do
  python examples/drl_collision_avoidance/evaluate.py \
    --checkpoint "$D42" --scenario_split valid --heldout_split test \
    --num_episodes 1500 --method "dropout_${p}" \
    --occlusion_dropout_p "$p" \
    --output_json "${ROB}/eval_test_dropout_${p}.json"
done
for sigma in 0.05 0.1; do
  python examples/drl_collision_avoidance/evaluate.py \
    --checkpoint "$D42" --scenario_split valid --heldout_split test \
    --num_episodes 1500 --method "noise_${sigma}" \
    --obs_noise_std "$sigma" \
    --output_json "${ROB}/eval_test_noise_${sigma}.json"
done
python examples/drl_collision_avoidance/evaluate.py \
  --checkpoint "$D42" --scenario_split valid --heldout_split test \
  --num_episodes 1500 --method fov_120 \
  --occlusion_view_angle 2.094395 \
  --trace_dir "${ROB}/traces" --trace_max 2 \
  --output_json "${ROB}/eval_test_fov120.json"

python examples/drl_collision_avoidance/evaluate.py \
  --checkpoint "$D42" --scenario_split valid --heldout_split test \
  --num_episodes 1500 --method D_occ_risk \
  --trace_dir "${ROB}/traces" --trace_max 2 \
  --output_json "${ROB}/eval_test.json"

python paper/analysis/make_tables.py
python paper/analysis/plot_traces.py --trace_dir "${ROB}/traces"
python scripts/count_q_parameters.py
