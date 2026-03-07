import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cfgs.config import PROJECT_PATH, set_display_window


def test_ddqn_forward_and_train():
    from examples.drl_collision_avoidance.dqn_modules.ddqn_agent import DDQNAgent, DDQNAgentConfig
    import torch
    import numpy as np

    config = DDQNAgentConfig(grid_size=350, grid_rows=25, grid_cols=14, device='cpu', batch_size=64)
    agent = DDQNAgent(obs_dim=359, n_actions=15, config=config)
    state = torch.randn(1, 359)
    q = agent.online_net(state)
    assert q.shape == (1, 15), f'Expected (1,15) got {q.shape}'

    for _ in range(100):
        s = np.random.randn(359).astype(np.float32)
        ns = np.random.randn(359).astype(np.float32)
        agent.store_transition(s, np.random.randint(15), np.random.randn(), ns, 0.0)

    loss = agent.train_step()
    assert loss is not None
    print(f'[PASS] DDQN forward + train  (Q shape={q.shape}, loss={loss:.4f})')


def test_env_interface():
    set_display_window()
    from examples.drl_collision_avoidance.collision_avoidance_env import CollisionAvoidanceEnv
    import json
    import shutil
    import numpy as np

    test_dir = str(PROJECT_PATH / 'tests')
    scenario_file = 'large_file_tfrecord.json'

    valid_path = os.path.join(test_dir, 'valid_files.json')
    backup_path = valid_path + '.bak'
    shutil.copy2(valid_path, backup_path)
    try:
        with open(valid_path, 'w') as f:
            json.dump({scenario_file: []}, f)

        cfg = {
            'single_agent_mode': True,
            'max_num_vehicles': 1,
            'episode_length': 80,
            'num_files': 1,
            'scenario_path': test_dir,
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
                'render_img_width': 640, 'render_img_height': 480,
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
                'vru_weight': 2.0, 'vehicle_weight': 1.0, 'road_edge_weight': 3.0,
            },
            'reward': {
                'goal_bonus': 100.0, 'collision_penalty': 0.0,
                'step_penalty': 0.0, 'progress_scale': 0.5,
                'ttz_safe_threshold': 4.0, 'ttz_reward_scale': 0.5,
                'offroad_penalty': 5.0, 'heading_scale': 0.3,
            },
            'action_map': {
                'throttle_levels': [-1.0, 0.0, 2.0],
                'steer_levels': [-0.6, -0.2, 0.0, 0.2, 0.6],
            },
            'drl': {'max_episode_steps': 400},
        }

        env = CollisionAvoidanceEnv(cfg)
        obs = env.reset()
        expected_dim = 25 * 14 + 4 + 2 + 3
        assert obs.shape == (expected_dim,), f'Expected ({expected_dim},) got {obs.shape}'
        assert env.action_space.n == 15

        for i in range(5):
            obs, rew, done, info = env.step(4)
            assert obs.shape == (expected_dim,)
            assert isinstance(rew, float)
            assert isinstance(done, bool)
            if done:
                obs = env.reset()
    finally:
        if os.path.exists(backup_path):
            os.replace(backup_path, valid_path)

    print(f'[PASS] Env interface  (obs_dim={expected_dim}, actions=9, 5 steps OK)')


if __name__ == '__main__':
    test_ddqn_forward_and_train()
    test_env_interface()
    print('\nAll smoke tests passed!')
