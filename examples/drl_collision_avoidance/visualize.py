import argparse
import os
import sys
from typing import Optional

import imageio
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cfgs.config import PROJECT_PATH, set_display_window
from examples.drl_collision_avoidance.scenario_utils import resolve_scenario_path


def make_default_cfg(scenario_path: Optional[str] = None, scenario_split: str = 'valid',
                     num_files: int = 1, render_width: int = 640,
                     render_height: int = 640) -> dict:
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
            'shared_reward': False, 'goal_tolerance': 2.0,
            'reward_scaling': 1.0, 'collision_penalty': 50.0,
            'shaped_goal_distance': False, 'shaped_goal_distance_scaling': 0.0,
            'goal_distance_penalty': False, 'goal_achieved_bonus': 100.0,
            'position_target': True, 'position_target_tolerance': 2.0,
            'speed_target': False, 'speed_target_tolerance': 1.0,
            'heading_target': False, 'heading_target_tolerance': 0.3,
        },
        'subscriber': {
            'view_angle': 2.1, 'view_dist': 80,
            'use_ego_state': True, 'use_observations': True,
            'keep_inactive_agents': False, 'n_frames_stacked': 1,
            'render_img_width': render_width, 'render_img_height': render_height,
            'render_padding': 30.0,
        },
        'scenario': {
            'start_time': 0, 'allow_non_vehicles': True,
            'moving_threshold': 0.2, 'speed_threshold': 0.05,
            'max_visible_objects': 16, 'max_visible_road_points': 1000,
            'max_visible_traffic_lights': 20, 'max_visible_stop_signs': 4,
            'sample_every_n': 1, 'road_edge_first': False,
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


def visualize(checkpoint_path: str, scenario_path: Optional[str] = None,
              scenario_split: str = 'valid', num_episodes: int = 5,
              num_files: int = 1,
              video_fps: int = 10,
              render_width: int = 640, render_height: int = 640,
              record_ego: bool = False, record_features: bool = False):
    set_display_window()

    from examples.drl_collision_avoidance.collision_avoidance_env import CollisionAvoidanceEnv
    from examples.drl_collision_avoidance.ddqn_agent import DDQNAgent

    cfg = make_default_cfg(
        scenario_path,
        scenario_split,
        num_files,
        render_width,
        render_height,
    )
    print(f'Using scenario split={scenario_split} path={cfg["scenario_path"]}')
    env = CollisionAvoidanceEnv(cfg)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    grid_size = cfg['occupancy_grid']['rows'] * cfg['occupancy_grid']['cols']

    agent = DDQNAgent(obs_dim=obs_dim, n_actions=n_actions, grid_size=grid_size, device='cpu')
    agent.load(checkpoint_path)
    agent.epsilon = 0.0

    writers = {}

    def open_writer(enabled, stream_name, filename):
        if not enabled:
            return
        output_path = os.path.join(PROJECT_PATH, filename)
        try:
            writers[stream_name] = imageio.get_writer(output_path, fps=video_fps)
            print(f'Recording {stream_name} stream to {output_path}')
        except Exception as exc:
            # Continue without this stream so one codec/path issue does not
            # cancel the full rollout capture.
            print(f'Failed to initialize {stream_name} writer: {exc}')

    def append_frame(stream_name, frame):
        if frame is None:
            return
        writer = writers.get(stream_name)
        if writer is None:
            return
        try:
            writer.append_data(frame)
        except Exception as exc:
            # Drop only the failed stream; preserving the main video is usually
            # more valuable than failing the whole run.
            print(f'Failed to append frame to {stream_name}: {exc}')
            try:
                writer.close()
            except Exception:
                pass
            writers.pop(stream_name, None)

    try:
        open_writer(True, 'main', 'animation.mp4')
        open_writer(record_ego, 'ego', 'animation_ego.mp4')
        open_writer(record_features, 'features', 'animation_feature.mp4')

        for ep in range(1, num_episodes + 1):
            state = env.reset()
            total_reward = 0.0
            collided = False
            goal_achieved = False

            max_steps = cfg['drl']['max_episode_steps']
            for t in range(max_steps):
                append_frame('main', env.base_env.render())
                if 'ego' in writers:
                    append_frame('ego', env.base_env.render_ego())
                if 'features' in writers:
                    append_frame('features', env.base_env.render_features())

                action = agent.select_action(state)
                state, reward, done, info = env.step(action)
                total_reward += reward
                collided = collided or info.get('collided', False)
                goal_achieved = goal_achieved or info.get('goal_achieved', False)

                if done:
                    append_frame('main', env.base_env.render())
                    break

            status = 'GOAL' if goal_achieved else ('COLLISION' if collided else 'TIMEOUT')
            print(f'Episode {ep:3d} | {status:9s} | reward={total_reward:8.2f} | len={t+1:4d}')

    finally:
        for stream_name, writer in list(writers.items()):
            try:
                writer.close()
            except Exception as exc:
                print(f'Failed to close {stream_name} writer: {exc}')

    print(f'\nDone. Videos saved to {PROJECT_PATH}/')


def main():
    set_display_window()

    parser = argparse.ArgumentParser(description='Visualize trained DDQN agent as MP4')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--scenario_path', type=str,
                        default=None,
                        help='Override scenario directory with valid_files.json')
    parser.add_argument('--scenario_split', type=str,
                        default='valid', choices=['train', 'valid'],
                        help='Dataset split used when --scenario_path is omitted')
    parser.add_argument('--num_episodes', type=int, default=5)
    parser.add_argument('--num_files', type=int, default=1)
    parser.add_argument('--video_fps', type=int, default=10)
    parser.add_argument('--render_width', type=int, default=640)
    parser.add_argument('--render_height', type=int, default=640)
    parser.add_argument('--record_ego', action='store_true')
    parser.add_argument('--record_features', action='store_true')
    args = parser.parse_args()

    visualize(
        args.checkpoint, args.scenario_path, args.scenario_split, args.num_episodes,
        args.num_files, args.video_fps,
        args.render_width, args.render_height,
        args.record_ego, args.record_features,
    )


if __name__ == '__main__':
    main()
