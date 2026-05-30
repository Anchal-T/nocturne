"""Interpretability: occlusion-conditioned behavior analysis.

Runs evaluation and logs (occlusion_fraction, ego_speed) at every timestep.
Computes Pearson correlation and produces a binned plot showing that the
occlusion-aware agent slows down when it can't see.

Usage:
    python scripts/interpretability/occlusion_behavior.py \
        --checkpoint checkpoints/ddqn_occ/seed_42/ddqn_final.pth \
        --num_episodes 200 --output plots/occlusion_speed.png
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cfgs.config import set_display_window
from examples.drl_collision_avoidance.agent_utils import build_cpu_agent
from examples.drl_collision_avoidance.collision_avoidance_env import CollisionAvoidanceEnv
from examples.drl_collision_avoidance.scenario_utils import load_config


def collect_occlusion_speed_data(checkpoint_path, scenario_path, num_episodes, use_occlusion):
    set_display_window()
    cfg = load_config(scenario_path=scenario_path, scenario_split='valid', num_files=-1)
    cfg['occupancy_grid']['use_occlusion'] = use_occlusion

    env = CollisionAvoidanceEnv(cfg)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = build_cpu_agent(cfg, obs_dim, n_actions)
    agent.load(checkpoint_path)
    agent.epsilon = 0.0

    occlusion_fracs = []
    speeds = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        for t in range(cfg['drl']['max_episode_steps']):
            ego_veh = env._get_ego_vehicle()
            if ego_veh is None:
                break

            # Compute occlusion fraction
            vis_objects = env._query_visible_objects(ego_veh)
            occ_features = env._compute_occlusion_features(vis_objects)
            occlusion_fracs.append(float(occ_features[0]))
            speeds.append(float(ego_veh.speed))

            action = agent.select_action(state)
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    return np.array(occlusion_fracs), np.array(speeds)


def analyze(occ_fracs, speeds, output_path=None):
    from scipy import stats

    # Pearson correlation
    r, p = stats.pearsonr(occ_fracs, speeds)
    print(f"Pearson r = {r:.4f}, p = {p:.2e}")
    print(f"  {'Significant' if p < 0.01 else 'Not significant'} (p < 0.01)")

    # Binned analysis
    bins = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    print(f"\n{'Occlusion Bin':<15} {'N':>6} {'Mean Speed':>12} {'Std':>8}")
    print("-" * 45)
    for i in range(len(bins) - 1):
        mask = (occ_fracs >= bins[i]) & (occ_fracs < bins[i + 1])
        n = mask.sum()
        if n > 0:
            mean_speed = speeds[mask].mean()
            std_speed = speeds[mask].std()
            print(f"[{bins[i]:.1f}, {bins[i+1]:.1f}){'':<5} {n:>6} {mean_speed:>12.2f} {std_speed:>8.2f}")

    if output_path:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            bin_centers = []
            bin_means = []
            bin_stds = []
            for i in range(len(bins) - 1):
                mask = (occ_fracs >= bins[i]) & (occ_fracs < bins[i + 1])
                if mask.sum() > 10:
                    bin_centers.append((bins[i] + bins[i + 1]) / 2)
                    bin_means.append(speeds[mask].mean())
                    bin_stds.append(speeds[mask].std() / np.sqrt(mask.sum()))

            ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', capsize=4)
            ax.set_xlabel('Occlusion Fraction')
            ax.set_ylabel('Mean Ego Speed (m/s)')
            ax.set_title(f'Speed vs Occlusion (r={r:.3f}, p={p:.2e})')
            ax.grid(True, alpha=0.3)

            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to {output_path}")
        except ImportError:
            print("\nmatplotlib not available, skipping plot")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--scenario_path', type=str, default=None)
    parser.add_argument('--num_episodes', type=int, default=200)
    parser.add_argument('--use_occlusion', action='store_true', default=True)
    parser.add_argument('--output', type=str, default='plots/occlusion_speed.png')
    args = parser.parse_args()

    print(f"Collecting data (use_occlusion={args.use_occlusion})...")
    occ_fracs, speeds = collect_occlusion_speed_data(
        args.checkpoint, args.scenario_path, args.num_episodes, args.use_occlusion
    )
    print(f"Collected {len(occ_fracs)} timesteps")
    analyze(occ_fracs, speeds, args.output)


if __name__ == '__main__':
    main()
