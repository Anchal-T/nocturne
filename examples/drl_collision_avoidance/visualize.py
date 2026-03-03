import argparse
import os
import sys
from typing import Any, Dict, Optional

import imageio
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cfgs.config import PROJECT_PATH, set_display_window
from examples.drl_collision_avoidance.scenario_utils import load_config


def _build_cpu_agent(cfg: Dict[str, Any], obs_dim: int, n_actions: int):
    from examples.drl_collision_avoidance.dqn_modules.ddqn_agent import DDQNAgent, DDQNAgentConfig

    grid_cfg = cfg.get('occupancy_grid', {})
    grid_rows = int(grid_cfg['rows'])
    grid_cols = int(grid_cfg['cols'])
    grid_size = grid_rows * grid_cols
    drl_cfg = cfg.get('drl', {})

    agent_cfg = DDQNAgentConfig.from_drl_cfg(
        drl_cfg,
        grid_size=grid_size,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        device='cpu',
    )
    return DDQNAgent.from_config(obs_dim=obs_dim, n_actions=n_actions, config=agent_cfg)


def _open_writer(writers: Dict[str, Any], enabled: bool, stream_name: str, filename: str, video_fps: int) -> None:
    if not enabled:
        return
    output_path = os.path.join(PROJECT_PATH, filename)
    try:
        writers[stream_name] = imageio.get_writer(output_path, fps=video_fps)
        print(f'Recording {stream_name} stream to {output_path}')
    except Exception as exc:
        print(f'Failed to initialize {stream_name} writer: {exc}')


def _append_frame(writers: Dict[str, Any], stream_name: str, frame: Optional[np.ndarray]) -> None:
    if frame is None:
        return
    writer = writers.get(stream_name)
    if writer is None:
        return
    try:
        writer.append_data(frame)
    except Exception as exc:
        print(f'Failed to append frame to {stream_name}: {exc}')
        try:
            writer.close()
        except Exception:
            pass
        writers.pop(stream_name, None)


def _capture_step_frames(env, writers: Dict[str, Any]) -> None:
    _append_frame(writers, 'main', env.base_env.render())
    if 'ego' in writers:
        _append_frame(writers, 'ego', env.base_env.render_ego())
    if 'features' in writers:
        _append_frame(writers, 'features', env.base_env.render_features())


def _run_episode(env, agent, max_steps: int, writers: Dict[str, Any]):
    state = env.reset()
    total_reward = 0.0
    collided = False
    goal_achieved = False
    episode_len = 0

    for step in range(max_steps):
        _capture_step_frames(env, writers)
        action = agent.select_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        collided = collided or info.get('collided', False)
        goal_achieved = goal_achieved or info.get('goal_achieved', False)
        episode_len = step + 1

        if done:
            _append_frame(writers, 'main', env.base_env.render())
            break

    return total_reward, collided, goal_achieved, episode_len


def _close_writers(writers: Dict[str, Any]) -> None:
    for stream_name, writer in list(writers.items()):
        try:
            writer.close()
        except Exception as exc:
            print(f'Failed to close {stream_name} writer: {exc}')


def visualize(checkpoint_path: str, scenario_path: Optional[str] = None,
              scenario_split: str = 'valid', num_episodes: int = 5,
              num_files: int = 1,
              video_fps: int = 10,
              render_width: int = 640, render_height: int = 640,
              record_ego: bool = False, record_features: bool = False):
    set_display_window()

    from examples.drl_collision_avoidance.collision_avoidance_env import CollisionAvoidanceEnv

    cfg = load_config(
        scenario_path=scenario_path,
        scenario_split=scenario_split,
        num_files=num_files,
    )
    cfg['subscriber']['render_img_width'] = render_width
    cfg['subscriber']['render_img_height'] = render_height
    print(f'Using scenario split={scenario_split} path={cfg["scenario_path"]}')
    env = CollisionAvoidanceEnv(cfg)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = _build_cpu_agent(cfg, obs_dim, n_actions)
    agent.load(checkpoint_path)
    agent.epsilon = 0.0

    writers: Dict[str, Any] = {}

    try:
        _open_writer(writers, True, 'main', 'animation.mp4', video_fps)
        _open_writer(writers, record_ego, 'ego', 'animation_ego.mp4', video_fps)
        _open_writer(writers, record_features, 'features', 'animation_feature.mp4', video_fps)
        max_steps = cfg['drl']['max_episode_steps']

        for ep in range(1, num_episodes + 1):
            total_reward, collided, goal_achieved, episode_len = _run_episode(
                env=env,
                agent=agent,
                max_steps=max_steps,
                writers=writers,
            )

            status = 'GOAL' if goal_achieved else ('COLLISION' if collided else 'TIMEOUT')
            print(f'Episode {ep:3d} | {status:9s} | reward={total_reward:8.2f} | len={episode_len:4d}')

    finally:
        _close_writers(writers)

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
    parser.add_argument('--video_fps', type=int, default=20)
    parser.add_argument('--render_width', type=int, default=1200)
    parser.add_argument('--render_height', type=int, default=1000)
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
