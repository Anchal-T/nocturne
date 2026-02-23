import argparse
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cfgs.config import set_display_window
from examples.drl_collision_avoidance.scenario_utils import resolve_scenario_path


def make_default_cfg(
    scenario_path: Optional[str] = None,
    scenario_split: str = 'valid',
    num_files: int = 1,
) -> dict:
    resolved_scenario_path = resolve_scenario_path(scenario_path, scenario_split)
    return {
        'single_agent_mode': True,
        'max_num_vehicles': 1,
        'episode_length': 80,
        'num_files': num_files,
        'scenario_path': resolved_scenario_path,
        'dt': 0.1,
        'seed': 42,
        'discretize_actions': False,
        'accel_lower_bound': -3.0,
        'accel_upper_bound': 2.0,
        'steering_lower_bound': -0.7,
        'steering_upper_bound': 0.7,
        'head_angle_lower_bound': 0.0,
        'head_angle_upper_bound': 0.0,
        'remove_at_goal': True,
        'remove_at_collide': True,
        'rew_cfg': {
            'shared_reward': False,
            'goal_tolerance': 2.0,
            'reward_scaling': 1.0,
            'collision_penalty': 50.0,
            'shaped_goal_distance': False,
            'shaped_goal_distance_scaling': 0.0,
            'goal_distance_penalty': False,
            'goal_achieved_bonus': 100.0,
            'position_target': True,
            'position_target_tolerance': 2.0,
            'speed_target': False,
            'speed_target_tolerance': 1.0,
            'heading_target': False,
            'heading_target_tolerance': 0.3,
        },
        'subscriber': {
            'view_angle': 2.1,
            'view_dist': 80,
            'use_ego_state': True,
            'use_observations': True,
            'keep_inactive_agents': False,
            'n_frames_stacked': 1,
            'render_img_width': 640,
            'render_img_height': 480,
            'render_padding': 30.0,
        },
        'scenario': {
            'start_time': 0,
            'allow_non_vehicles': True,
            'moving_threshold': 0.2,
            'speed_threshold': 0.05,
            'max_visible_objects': 16,
            'max_visible_road_points': 1000,
            'max_visible_traffic_lights': 20,
            'max_visible_stop_signs': 4,
            'sample_every_n': 1,
            'road_edge_first': False,
        },
        'occupancy_grid': {
            'rows': 25, 'cols': 14,
            'forward_dist': 20.0, 'backward_dist': 5.0, 'lateral_dist': 7.0,
            'vru_weight': 2.0, 'vehicle_weight': 1.0,
        },
        'reward': {
            'goal_bonus': 100.0, 'collision_penalty': 50.0,
            'step_penalty': 0.1, 'progress_scale': 1.0,
            'ttz_safe_threshold': 4.0, 'ttz_reward_scale': 0.5,
        },
        'action_map': {
            'throttle_levels': [-3.0, 0.0, 2.0],
            'steer_levels': [-0.3, 0.0, 0.3],
        },
        'drl': {'max_episode_steps': 80},
    }


def evaluate(checkpoint_path: str, scenario_path: Optional[str] = None,
             scenario_split: str = 'valid', num_episodes: int = 10,
             num_files: int = 1, render: bool = False):
    set_display_window()

    from examples.drl_collision_avoidance.collision_avoidance_env import CollisionAvoidanceEnv
    from examples.drl_collision_avoidance.ddqn_agent import DDQNAgent

    cfg = make_default_cfg(scenario_path, scenario_split, num_files)
    print(f'Using scenario split={scenario_split} path={cfg["scenario_path"]}')
    env = CollisionAvoidanceEnv(cfg)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    grid_size = cfg['occupancy_grid']['rows'] * cfg['occupancy_grid']['cols']

    agent = DDQNAgent(obs_dim=obs_dim, n_actions=n_actions, grid_size=grid_size, device='cpu')
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

        for t in range(cfg['drl']['max_episode_steps']):
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            collided = collided or info.get('collided', False)
            goal = goal or info.get('goal_achieved', False)
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
              f'collided={collided} | goal={goal}')

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
    parser.add_argument('--num_files', type=int, default=1)
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
