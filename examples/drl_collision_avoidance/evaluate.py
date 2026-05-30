import argparse
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cfgs.config import set_display_window
from examples.drl_collision_avoidance.agent_utils import build_cpu_agent
from examples.drl_collision_avoidance.scenario_utils import load_config


def evaluate(checkpoint_path: str, scenario_path: Optional[str] = None,
             scenario_split: str = 'valid', num_episodes: int = 10,
             num_files: int = 1, render: bool = False):
    set_display_window()

    from examples.drl_collision_avoidance.collision_avoidance_env import CollisionAvoidanceEnv

    cfg = load_config(scenario_path=scenario_path, scenario_split=scenario_split, num_files=num_files)
    print(f'Using scenario split={scenario_split} path={cfg["scenario_path"]}')
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
        'ade': [], 'fde': [], 'jerk': [], 'smoothness': [],
        'num_vehicles': [],
    }

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        collided = False
        goal = False
        min_goal_dist = float('inf')

        # Trajectory tracking
        agent_positions = []
        expert_positions = []
        accelerations = []
        steerings = []

        ego_veh = env._get_ego_vehicle()
        n_vehicles = len(env.base_env.scenario.getVehicles())

        for t in range(cfg['drl']['max_episode_steps']):
            # Record positions before step
            ego_veh = env._get_ego_vehicle()
            if ego_veh is not None:
                agent_positions.append((ego_veh.position.x, ego_veh.position.y))
                try:
                    exp_pos = env.base_env.scenario.expert_position(ego_veh, start_time + t)
                    expert_positions.append((exp_pos.x, exp_pos.y))
                except (IndexError, RuntimeError):
                    expert_positions.append(agent_positions[-1])

            action = agent.select_action(state)
            throttle, steer = env.action_table[action]
            accelerations.append(throttle)
            steerings.append(steer)

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            collided = collided or info.get('collided', False)
            goal = goal or info.get('goal_achieved', False)
            gd = info.get('goal_dist', float('inf'))
            min_goal_dist = min(min_goal_dist, gd)
            results['ttz_vehicles'].append(info.get('ttz_vehicle', 999.0))
            results['ttz_pedestrians'].append(info.get('ttz_pedestrian', 999.0))

            if render:
                env.render()
            if done:
                break

        # Compute trajectory metrics
        if len(agent_positions) > 1 and len(expert_positions) == len(agent_positions):
            pos = np.array(agent_positions)
            exp = np.array(expert_positions)
            displacements = np.linalg.norm(pos - exp, axis=1)
            results['ade'].append(float(np.mean(displacements)))
            results['fde'].append(float(displacements[-1]))
        if len(accelerations) > 1 and dt > 0:
            accel = np.array(accelerations)
            jerk_signal = np.diff(accel) / dt
            results['jerk'].append(float(np.sqrt(np.mean(jerk_signal ** 2))))
        if len(steerings) > 1 and dt > 0:
            steer_arr = np.array(steerings)
            steer_rate = np.diff(steer_arr) / dt
            results['smoothness'].append(float(np.std(steer_rate)))

        results['rewards'].append(total_reward)
        results['lengths'].append(t + 1)
        results['collisions'] += int(collided)
        results['goals'] += int(goal)
        results['collided_list'].append(collided)
        results['goal_list'].append(goal)
        results['num_vehicles'].append(n_vehicles)
        print(f'Episode {ep:3d} | reward={total_reward:8.2f} | len={t+1:4d} | '
              f'collided={collided} | goal={goal} | min_goal_dist={min_goal_dist:.2f}')

    print('\n' + '=' * 60)
    print('EVALUATION SUMMARY')
    print('=' * 60)
    print(f'  Episodes:       {num_episodes}')
    print(f'  Avg reward:     {np.mean(results["rewards"]):.2f} ± {np.std(results["rewards"]):.2f}')
    print(f'  Avg length:     {np.mean(results["lengths"]):.1f}')
    print(f'  Collision rate:  {results["collisions"] / num_episodes:.1%}')
    print(f'  Goal rate:       {results["goals"] / num_episodes:.1%}')
    ttz_v = [t for t in results['ttz_vehicles'] if t < 100]
    ttz_p = [t for t in results['ttz_pedestrians'] if t < 100]
    if ttz_v:
        print(f'  Avg TTZ (veh):  {np.mean(ttz_v):.2f}s')
    if ttz_p:
        print(f'  Avg TTZ (ped):  {np.mean(ttz_p):.2f}s')
    if results['ade']:
        print(f'  ADE:            {np.mean(results["ade"]):.2f} ± {np.std(results["ade"]):.2f} m')
        print(f'  FDE:            {np.mean(results["fde"]):.2f} ± {np.std(results["fde"]):.2f} m')
    if results['jerk']:
        print(f'  Jerk:           {np.mean(results["jerk"]):.2f} ± {np.std(results["jerk"]):.2f} m/s³')
    if results['smoothness']:
        print(f'  Smoothness:     {np.mean(results["smoothness"]):.2f} ± {np.std(results["smoothness"]):.2f} rad/s²')
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
            }
            if i < len(results['ade']):
                ep['ade'] = results['ade'][i]
                ep['fde'] = results['fde'][i]
            if i < len(results['jerk']):
                ep['jerk'] = results['jerk'][i]
            if i < len(results['smoothness']):
                ep['smoothness'] = results['smoothness'][i]
            episodes.append(ep)

        output = {
            'method': args.method,
            'checkpoint': args.checkpoint,
            'num_episodes': n,
            'goal_rate': results['goals'] / max(n, 1),
            'collision_rate': results['collisions'] / max(n, 1),
            'avg_reward': float(np.mean(results['rewards'])),
            'ade': float(np.mean(results['ade'])) if results['ade'] else None,
            'fde': float(np.mean(results['fde'])) if results['fde'] else None,
            'jerk': float(np.mean(results['jerk'])) if results['jerk'] else None,
            'smoothness': float(np.mean(results['smoothness'])) if results['smoothness'] else None,
            'episodes': episodes,
        }
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f'\nResults saved to {args.output_json}')


if __name__ == '__main__':
    main()
