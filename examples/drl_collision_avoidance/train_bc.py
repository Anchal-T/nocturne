"""Behavioral Cloning baseline using the same observation format as DDQN.

Collects (obs, expert_action) pairs by running scenarios with expert control,
then trains a classifier with cross-entropy loss.

Usage:
    python -m examples.drl_collision_avoidance.train_bc \
        --scenario_path /path/to/train_scenarios \
        --num_files 100 --num_episodes 5000 --epochs 50
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cfgs.config import set_display_window
from examples.drl_collision_avoidance.collision_avoidance_env import CollisionAvoidanceEnv
from examples.drl_collision_avoidance.dqn_modules.q_network import QNetwork
from examples.drl_collision_avoidance.scenario_utils import load_config


def discretize_expert_action(accel, steer, action_table):
    """Map continuous (accel, steer) to nearest discrete action index."""
    best_idx = 0
    best_dist = float('inf')
    for i, (t, s) in enumerate(action_table):
        d = (accel - t) ** 2 + (steer - s) ** 2
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def collect_data(cfg, num_episodes):
    """Run scenarios with expert actions, collect (obs, action) pairs."""
    env = CollisionAvoidanceEnv(cfg)
    action_table = env.action_table
    max_steps = cfg['drl']['max_episode_steps']
    start_time = cfg.get('scenario', {}).get('start_time', 0)

    observations = []
    actions = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ego_veh = env._get_ego_vehicle()
        if ego_veh is None:
            continue

        for t in range(max_steps):
            # Get expert action at current timestep
            expert_action = env.base_env.scenario.getExpertAction(ego_veh, start_time + t)
            if expert_action is None:
                break

            accel = expert_action.acceleration
            steer = expert_action.steering
            if accel is None or steer is None:
                break

            discrete_action = discretize_expert_action(accel, steer, action_table)
            observations.append(obs.copy())
            actions.append(discrete_action)

            # Step with expert action
            obs, _, terminated, truncated, _ = env.step(discrete_action)
            if terminated or truncated:
                break
            ego_veh = env._get_ego_vehicle()
            if ego_veh is None:
                break

        if (ep + 1) % 500 == 0:
            print(f'  Collected {ep + 1}/{num_episodes} episodes, {len(observations)} samples')

    return np.array(observations, dtype=np.float32), np.array(actions, dtype=np.int64)


def train_bc(observations, actions, obs_dim, n_actions, grid_size, grid_channels,
             grid_rows, grid_cols, mlp_depth, epochs, batch_size, lr, save_path):
    """Train BC classifier."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Same architecture as DDQN Q-network
    model = QNetwork(
        obs_dim=obs_dim, n_actions=n_actions, grid_size=grid_size,
        grid_channels=grid_channels, grid_rows=grid_rows, grid_cols=grid_cols,
        dueling=False, noisy=False, mlp_depth=mlp_depth,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset_size = len(observations)
    indices = np.arange(dataset_size)

    print(f'\nTraining BC: {dataset_size} samples, {epochs} epochs, device={device}')
    print(f'  Model params: {sum(p.numel() for p in model.parameters()):,}')

    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0.0
        correct = 0
        n_batches = 0

        for start in range(0, dataset_size, batch_size):
            batch_idx = indices[start:start + batch_size]
            obs_batch = torch.FloatTensor(observations[batch_idx]).to(device)
            act_batch = torch.LongTensor(actions[batch_idx]).to(device)

            logits = model(obs_batch)
            loss = F.cross_entropy(logits, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == act_batch).sum().item()
            n_batches += 1

        acc = correct / dataset_size
        avg_loss = total_loss / n_batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'  Epoch {epoch + 1:3d}/{epochs} | loss={avg_loss:.4f} | acc={acc:.3f}')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save({'model': model.state_dict(), 'obs_dim': obs_dim, 'n_actions': n_actions,
                'grid_size': grid_size, 'grid_channels': grid_channels,
                'grid_rows': grid_rows, 'grid_cols': grid_cols, 'mlp_depth': mlp_depth}, save_path)
    print(f'\nSaved BC model to {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Train BC baseline')
    parser.add_argument('--scenario_path', type=str, default=None)
    parser.add_argument('--scenario_split', type=str, default='train')
    parser.add_argument('--num_files', type=int, default=-1)
    parser.add_argument('--num_episodes', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save_path', type=str, default='checkpoints/bc/bc_final.pth')
    args = parser.parse_args()

    set_display_window()
    cfg = load_config(scenario_path=args.scenario_path, scenario_split=args.scenario_split,
                      num_files=args.num_files)

    print('Collecting expert data...')
    observations, actions = collect_data(cfg, args.num_episodes)
    print(f'Collected {len(observations)} samples')

    grid_cfg = cfg['occupancy_grid']
    grid_channels = 3
    grid_rows = int(grid_cfg['rows'])
    grid_cols = int(grid_cfg['cols'])
    grid_size = grid_channels * grid_rows * grid_cols
    obs_dim = observations.shape[1]
    n_actions = len(CollisionAvoidanceEnv(cfg).action_table)

    train_bc(observations, actions, obs_dim, n_actions, grid_size, grid_channels,
             grid_rows, grid_cols, int(cfg['drl']['mlp_depth']),
             args.epochs, args.batch_size, args.lr, args.save_path)


if __name__ == '__main__':
    main()
