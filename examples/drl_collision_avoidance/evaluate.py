import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cfgs.config import set_display_window
from examples.drl_collision_avoidance.agent_utils import build_cpu_agent
from examples.drl_collision_avoidance.scenario_utils import load_config


def _checkpoint_obs_dim(checkpoint_path: str) -> Optional[int]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        return None
    if checkpoint.get('obs_dim') is not None:
        return int(checkpoint['obs_dim'])

    state_dict = checkpoint.get('online_net') or checkpoint.get('model')
    hidden_layers = checkpoint.get('hidden_layers')
    grid_size = checkpoint.get('grid_size')
    if not state_dict or not hidden_layers or grid_size is None:
        return None

    encoder_dim = int(hidden_layers[1])
    head_weight_names = {
        'advantage_head.initial_layer.0.weight',
        'advantage_head.initial_layer.0.weight_mu',
        'head.initial_layer.0.weight',
        'head.initial_layer.0.weight_mu',
    }
    for key, value in state_dict.items():
        normalized_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ('module.', '_orig_mod.'):
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix):]
                    changed = True
        if normalized_key in head_weight_names:
            head_input_dim = int(value.shape[1])
            return int(grid_size) + head_input_dim - encoder_dim
    return None


def _apply_checkpoint_observation_settings(cfg, checkpoint_path, use_occlusion):
    grid_cfg = cfg['occupancy_grid']
    if use_occlusion is not None:
        grid_cfg['use_occlusion'] = bool(use_occlusion)
        return

    ckpt_obs_dim = _checkpoint_obs_dim(checkpoint_path)
    if ckpt_obs_dim is None:
        return

    grid_size = 3 * int(grid_cfg['rows']) * int(grid_cfg['cols'])
    no_occ_dim = grid_size + 4 + 2 + 3  # ego state + traffic light state + turn indicator = 4 + 2 + 3 = 9
    occ_dim = no_occ_dim + 4  # occlusion adds 4 channels of grid input
    if ckpt_obs_dim == occ_dim:
        grid_cfg['use_occlusion'] = True
    elif ckpt_obs_dim == no_occ_dim:
        grid_cfg['use_occlusion'] = False
    else:
        print(
            f'WARNING: checkpoint obs_dim={ckpt_obs_dim} does not match '
            f'expected no-occlusion ({no_occ_dim}) or occlusion ({occ_dim}) dimensions.'
        )


def _finite_ttz_min(values):
    finite = [v for v in values if v < 100]
    return min(finite) if finite else None


def _valid_numbers(values):
    return [v for v in values if v is not None]


def evaluate(checkpoint_path: str, scenario_path: Optional[str] = None,
             scenario_split: str = 'valid', num_episodes: int = 10,
             num_files: int = 1, render: bool = False,
             use_occlusion: Optional[bool] = None):
    set_display_window()

    from examples.drl_collision_avoidance.collision_avoidance_env import CollisionAvoidanceEnv

    cfg = load_config(scenario_path=scenario_path, scenario_split=scenario_split, num_files=num_files)
    _apply_checkpoint_observation_settings(cfg, checkpoint_path, use_occlusion)
    print(f'Using scenario split={scenario_split} path={cfg["scenario_path"]}')
    print(f'Using occlusion={cfg["occupancy_grid"].get("use_occlusion", False)}')

    env = CollisionAvoidanceEnv(cfg)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = build_cpu_agent(cfg, obs_dim, n_actions)
    agent.load(checkpoint_path)
    agent.epsilon = 0.0

    dt = cfg.get('dt', 0.1)
    start_time = cfg.get('scenario', {}).get('start_time', 0)

    results = {
        'rewards': [], 'lengths': [],
        'collisions': 0, 'goals': 0,
        'collided_list': [], 'goal_list': [],
        'ttz_vehicles': [], 'ttz_pedestrians': [],
        'episode_ttz_vehicle': [], 'episode_ttz_pedestrian': [],
        'ade': [], 'fde': [], 'jerk': [], 'smoothness': [],
        'num_vehicles': [],
    }

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        collided = False
        goal = False
        min_goal_dist = float('inf')

        trajectory_pairs = []
        speeds = []
        steerings = []
        episode_ttz_vehicle = []
        episode_ttz_pedestrian = []
        ade = None
        fde = None
        jerk = None
        smoothness = None

        ego_veh = env._get_ego_vehicle()
        n_vehicles = len(env.base_env.scenario.getVehicles())

        for t in range(cfg['drl']['max_episode_steps']):
            ego_veh = env._get_ego_vehicle()
            if ego_veh is not None:
                speeds.append(float(ego_veh.speed))
                try:
                    exp_pos = env.base_env.scenario.expert_position(ego_veh, start_time + t)
                    trajectory_pairs.append((
                        (ego_veh.position.x, ego_veh.position.y),
                        (exp_pos.x, exp_pos.y),
                    ))
                except (IndexError, KeyError, RuntimeError):
                    pass

            action = agent.select_action(state)
            _, steer = env.action_table[action]
            steerings.append(steer)

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            collided = collided or info.get('collided', False)
            goal = goal or info.get('goal_achieved', False)
            gd = info.get('goal_dist', float('inf'))
            min_goal_dist = min(min_goal_dist, gd)

            ttz_vehicle = info.get('ttz_vehicle', 999.0)
            ttz_pedestrian = info.get('ttz_pedestrian', 999.0)
            results['ttz_vehicles'].append(ttz_vehicle)
            results['ttz_pedestrians'].append(ttz_pedestrian)
            episode_ttz_vehicle.append(ttz_vehicle)
            episode_ttz_pedestrian.append(ttz_pedestrian)

            if render:
                env.render()
            if done:
                break

        if len(trajectory_pairs) > 1:
            pos = np.array([p for p, _ in trajectory_pairs])
            exp = np.array([p for _, p in trajectory_pairs])
            displacements = np.linalg.norm(pos - exp, axis=1)
            ade = float(np.mean(displacements))
            fde = float(displacements[-1])
        if len(speeds) > 2 and dt > 0:
            accel = np.diff(np.array(speeds)) / dt
            jerk_signal = np.diff(accel) / dt
            jerk = float(np.sqrt(np.mean(jerk_signal ** 2)))
        if len(steerings) > 1 and dt > 0:
            steer_arr = np.array(steerings)
            steer_rate = np.diff(steer_arr) / dt
            smoothness = float(np.std(steer_rate))

        results['rewards'].append(total_reward)
        results['lengths'].append(t + 1)
        results['ade'].append(ade)
        results['fde'].append(fde)
        results['jerk'].append(jerk)
        results['smoothness'].append(smoothness)
        results['collisions'] += int(collided)
        results['goals'] += int(goal)
        results['collided_list'].append(collided)
        results['goal_list'].append(goal)
        results['num_vehicles'].append(n_vehicles)
        results['episode_ttz_vehicle'].append(_finite_ttz_min(episode_ttz_vehicle))
        results['episode_ttz_pedestrian'].append(_finite_ttz_min(episode_ttz_pedestrian))
        print(f'Episode {ep:3d} | reward={total_reward:8.2f} | len={t+1:4d} | '
              f'collided={collided} | goal={goal} | min_goal_dist={min_goal_dist:.2f}')

    print('\n' + '=' * 60)
    print('EVALUATION SUMMARY')
    print('=' * 60)
    print(f'  Episodes:       {num_episodes}')
    print(f'  Avg reward:     {np.mean(results["rewards"]):.2f} +/- {np.std(results["rewards"]):.2f}')
    print(f'  Avg length:     {np.mean(results["lengths"]):.1f}')
    print(f'  Collision rate:  {results["collisions"] / num_episodes:.1%}')
    print(f'  Goal rate:       {results["goals"] / num_episodes:.1%}')
    ttz_v = [t for t in results['ttz_vehicles'] if t < 100]
    ttz_p = [t for t in results['ttz_pedestrians'] if t < 100]
    if ttz_v:
        print(f'  Avg TTZ (veh):  {np.mean(ttz_v):.2f}s')
    if ttz_p:
        print(f'  Avg TTZ (ped):  {np.mean(ttz_p):.2f}s')
    ade_values = _valid_numbers(results['ade'])
    fde_values = _valid_numbers(results['fde'])
    jerk_values = _valid_numbers(results['jerk'])
    smoothness_values = _valid_numbers(results['smoothness'])
    if ade_values:
        print(f'  ADE:            {np.mean(ade_values):.2f} +/- {np.std(ade_values):.2f} m')
        print(f'  FDE:            {np.mean(fde_values):.2f} +/- {np.std(fde_values):.2f} m')
    if jerk_values:
        print(f'  Jerk:           {np.mean(jerk_values):.2f} +/- {np.std(jerk_values):.2f} m/s^3')
    if smoothness_values:
        print(f'  Smoothness:     {np.mean(smoothness_values):.2f} +/- {np.std(smoothness_values):.2f} rad/s')
    print('=' * 60)
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained DDQN agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint (.pth)')
    parser.add_argument('--scenario_path', type=str,
                        default=None,
                        help='Override scenario directory with valid_files.json')
    parser.add_argument('--scenario_split', type=str,
                        default='valid', choices=['train', 'valid'],
                        help='Dataset split used when --scenario_path is omitted')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--num_files', type=int, default=-1)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--use_occlusion', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Override occupancy_grid.use_occlusion; defaults to checkpoint inference')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Save results to JSON file for downstream analysis')
    parser.add_argument('--method', type=str, default='ddqn',
                        help='Method name label for JSON output')
    args = parser.parse_args()
    results = evaluate(
        args.checkpoint,
        args.scenario_path,
        args.scenario_split,
        args.num_episodes,
        args.num_files,
        args.render,
        args.use_occlusion,
    )
    if args.output_json:
        import json
        n = len(results['rewards'])
        episodes = []
        for i in range(n):
            ep = {
                'reward': results['rewards'][i],
                'length': results['lengths'][i],
                'collided': results['collided_list'][i],
                'goal': results['goal_list'][i],
                'num_vehicles': results['num_vehicles'][i],
                'ttz_vehicle': results['episode_ttz_vehicle'][i],
                'ttz_pedestrian': results['episode_ttz_pedestrian'][i],
            }
            if results['ade'][i] is not None:
                ep['ade'] = results['ade'][i]
                ep['fde'] = results['fde'][i]
            if results['jerk'][i] is not None:
                ep['jerk'] = results['jerk'][i]
            if results['smoothness'][i] is not None:
                ep['smoothness'] = results['smoothness'][i]
            episodes.append(ep)

        output = {
            'method': args.method,
            'checkpoint': args.checkpoint,
            'num_episodes': n,
            'goal_rate': results['goals'] / max(n, 1),
            'collision_rate': results['collisions'] / max(n, 1),
            'avg_reward': float(np.mean(results['rewards'])),
            'ade': float(np.mean(_valid_numbers(results['ade']))) if _valid_numbers(results['ade']) else None,
            'fde': float(np.mean(_valid_numbers(results['fde']))) if _valid_numbers(results['fde']) else None,
            'jerk': float(np.mean(_valid_numbers(results['jerk']))) if _valid_numbers(results['jerk']) else None,
            'smoothness': float(np.mean(_valid_numbers(results['smoothness']))) if _valid_numbers(results['smoothness']) else None,
            'episodes': episodes,
        }
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f'\nResults saved to {args.output_json}')


if __name__ == '__main__':
    main()
