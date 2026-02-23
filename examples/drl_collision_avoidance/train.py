import os
import signal
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf

from cfgs.config import set_display_window

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def _make_cfg_dict(cfg):
    if hasattr(cfg, '_content'):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


@hydra.main(config_path="../../cfgs/drl_collision_avoidance", config_name="config")
def main(cfg):
    set_display_window()
    cfg_dict = _make_cfg_dict(cfg)

    from examples.drl_collision_avoidance.collision_avoidance_env import CollisionAvoidanceEnv
    from examples.drl_collision_avoidance.ddqn_agent import DDQNAgent
    from examples.drl_collision_avoidance.scenario_utils import apply_scenario_path_defaults

    cfg_dict = apply_scenario_path_defaults(cfg_dict, default_split='train')
    print(
        f"Using scenario split={cfg_dict['scenario_split']} "
        f"path={cfg_dict['scenario_path']}"
    )

    drl_cfg = cfg_dict.get('drl', {})
    grid_cfg = cfg_dict.get('occupancy_grid', {})
    grid_size = grid_cfg.get('rows', 25) * grid_cfg.get('cols', 14)

    env = CollisionAvoidanceEnv(cfg_dict)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = cfg_dict.get('device', 'cpu')
    if device.startswith('cuda') and not _cuda_available():
        device = 'cpu'

    agent = DDQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        grid_size=grid_size,
        hidden_layers=drl_cfg.get('hidden_layers', [128, 128, 128, 32]),
        lr=drl_cfg.get('lr', 0.01),
        gamma=drl_cfg.get('gamma', 0.9),
        epsilon_start=drl_cfg.get('epsilon_start', 1.0),
        epsilon_end=drl_cfg.get('epsilon_end', 0.05),
        epsilon_decay_steps=drl_cfg.get('epsilon_decay_steps', 200000),
        replay_buffer_size=drl_cfg.get('replay_buffer_size', 100000),
        batch_size=drl_cfg.get('batch_size', 64),
        target_update_freq=drl_cfg.get('target_update_freq', 1000),
        device=device,
    )

    num_episodes = drl_cfg.get('num_episodes', 5000)
    max_steps = drl_cfg.get('max_episode_steps', 400)
    train_freq = drl_cfg.get('train_freq', 4)
    min_replay = drl_cfg.get('min_replay_size', 1000)
    log_interval = drl_cfg.get('log_interval', 10)
    save_interval = drl_cfg.get('save_interval', 100)
    checkpoint_dir = drl_cfg.get('checkpoint_dir', 'checkpoints/drl_collision_avoidance')
    os.makedirs(checkpoint_dir, exist_ok=True)

    writer = _make_writer(checkpoint_dir)

    running = True
    def _signal_handler(sig, frame):
        nonlocal running
        print('\nCaught interrupt — saving checkpoint and exiting...')
        running = False
    signal.signal(signal.SIGINT, _signal_handler)

    global_step = 0
    reward_history = []
    loss_history = []

    for episode in range(1, num_episodes + 1):
        if not running:
            break

        state = env.reset()
        episode_reward = 0.0
        episode_collided = False
        episode_goal = False

        for t in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, float(done))

            global_step += 1
            episode_reward += reward
            episode_collided = episode_collided or info.get('collided', False)
            episode_goal = episode_goal or info.get('goal_achieved', False)

            # DQN-style ratio: collect transitions every step, optimize periodically.
            if global_step % train_freq == 0 and len(agent.replay_buffer) >= min_replay:
                loss = agent.train_step()
                if loss is not None:
                    loss_history.append(loss)

            state = next_state
            if done:
                break

        reward_history.append(episode_reward)

        if writer is not None:
            writer.add_scalar('train/episode_reward', episode_reward, episode)
            writer.add_scalar('train/episode_length', t + 1, episode)
            writer.add_scalar('train/epsilon', agent.epsilon, episode)
            writer.add_scalar('train/collided', float(episode_collided), episode)
            writer.add_scalar('train/goal_achieved', float(episode_goal), episode)
            if loss_history:
                writer.add_scalar('train/loss', loss_history[-1], episode)

        if episode % log_interval == 0:
            avg_rew = np.mean(reward_history[-log_interval:])
            col_rate = sum(1 for r in reward_history[-log_interval:] if r < -20) / log_interval
            print(
                f'Episode {episode:5d} | '
                f'avg_reward={avg_rew:8.2f} | '
                f'eps={agent.epsilon:.3f} | '
                f'buf={len(agent.replay_buffer):6d} | '
                f'steps={global_step}'
            )

        if episode % save_interval == 0:
            path = os.path.join(checkpoint_dir, f'ddqn_ep{episode}.pth')
            agent.save(path)
            print(f'  → Saved checkpoint: {path}')

    final_path = os.path.join(checkpoint_dir, 'ddqn_final.pth')
    agent.save(final_path)
    print(f'Training complete. Final checkpoint: {final_path}')

    if writer is not None:
        writer.close()


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _make_writer(log_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=os.path.join(log_dir, 'tb_logs'))
    except ImportError:
        print('TensorBoard not available — logging to console only.')
        return None


if __name__ == '__main__':
    main()
