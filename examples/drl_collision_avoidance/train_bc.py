"""Behavioral Cloning baseline using the same observation format as DDQN.

Collects (obs, discretized expert_action) pairs while advancing scenarios with
the continuous expert action, then trains a classifier with cross-entropy loss.

Usage:
    python -m examples.drl_collision_avoidance.train_bc \
        --scenario_path /path/to/train_scenarios \
        --num_files 100 --num_episodes 5000 --epochs 50
"""
import argparse
import os
import sys

import numpy as np
import torch
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
    """Run scenarios with continuous expert actions, collect discrete labels."""
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
            accel = float(accel)
            steer = float(steer)
            if not np.isfinite(accel) or not np.isfinite(steer):
                break

            discrete_action = discretize_expert_action(accel, steer, action_table)
            observations.append(obs.copy())
            actions.append(discrete_action)

            obs, terminated, truncated = step_continuous_action(env, accel, steer)
            if terminated or truncated:
                break
            ego_veh = env._get_ego_vehicle()
            if ego_veh is None:
                break

        if (ep + 1) % 500 == 0:
            print(f'  Collected {ep + 1}/{num_episodes} episodes, {len(observations)} samples')

    return np.array(observations, dtype=np.float32), np.array(actions, dtype=np.int64)


def step_continuous_action(env, accel, steer):
    """Advance the wrapped Nocturne env with a continuous expert action."""
    action_dict = {env._ego_id: [accel, steer, 0.0]}
    _, _, done_dict, truncated_dict, _ = env.base_env.step(action_dict)
    env._step_count += 1
    obs = env._build_observation()
    ego_done = done_dict.get(env._ego_id, False)
    all_done = done_dict.get("__all__", False)
    terminated = ego_done or all_done
    truncated = truncated_dict.get(env._ego_id, False) or env._step_count >= env._max_steps
    return obs, terminated, truncated


def train_bc(observations, actions, obs_dim, n_actions, grid_size, grid_channels,
             grid_rows, grid_cols, hidden_layers, mlp_depth, epochs, batch_size,
             lr, save_path):
    """Train BC classifier."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Same architecture as DDQN Q-network
    model = QNetwork(
        obs_dim=obs_dim, n_actions=n_actions, grid_size=grid_size,
        hidden_layers=hidden_layers,
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
    state_dict = model.state_dict()
    torch.save({
        'online_net': state_dict,
        'target_net': state_dict,
        'optimizer': optimizer.state_dict(),
        'train_steps': epochs,
        'epsilon': 0.0,
        'obs_dim': obs_dim,
        'n_actions': n_actions,
        'hidden_layers': model.hidden_layers,
        'grid_size': grid_size,
        'grid_channels': grid_channels,
        'grid_rows': grid_rows,
        'grid_cols': grid_cols,
        'dueling': False,
        'noisy': False,
        'mlp_depth': mlp_depth,
        'use_muon': False,
        'bc_model': True,
    }, save_path)
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
    if len(observations) == 0:
        raise RuntimeError('No BC samples collected; check scenario validity and expert actions.')

    grid_cfg = cfg['occupancy_grid']
    grid_channels = 3
    grid_rows = int(grid_cfg['rows'])
    grid_cols = int(grid_cfg['cols'])
    grid_size = grid_channels * grid_rows * grid_cols
    obs_dim = observations.shape[1]
    n_actions = len(CollisionAvoidanceEnv(cfg).action_table)

    hidden_layers = list(cfg['drl']['hidden_layers'])
    train_bc(observations, actions, obs_dim, n_actions, grid_size, grid_channels,
             grid_rows, grid_cols, hidden_layers, int(cfg['drl']['mlp_depth']),
             args.epochs, args.batch_size, args.lr, args.save_path)


if __name__ == '__main__':
    main()
