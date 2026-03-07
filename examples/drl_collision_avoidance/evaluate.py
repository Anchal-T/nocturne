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

    results = {
        'rewards': [], 'lengths': [],
        'collisions': 0, 'goals': 0,
        'ttz_vehicles': [], 'ttz_pedestrians': [],
    }

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        collided = False
        goal = False
        min_goal_dist = float('inf')

        for t in range(cfg['drl']['max_episode_steps']):
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
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

        results['rewards'].append(total_reward)
        results['lengths'].append(t + 1)
        results['collisions'] += int(collided)
        results['goals'] += int(goal)
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
    # Ignore large sentinel values (e.g. 999.0) when reporting TTZ averages.
    ttz_v = [t for t in results['ttz_vehicles'] if t < 100]
    ttz_p = [t for t in results['ttz_pedestrians'] if t < 100]
    if ttz_v:
        print(f'  Avg TTZ (veh):  {np.mean(ttz_v):.2f}s')
    if ttz_p:
        print(f'  Avg TTZ (ped):  {np.mean(ttz_p):.2f}s')
    print('=' * 60)


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
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        args.scenario_path,
        args.scenario_split,
        args.num_episodes,
        args.num_files,
        args.render,
    )


if __name__ == '__main__':
    main()
