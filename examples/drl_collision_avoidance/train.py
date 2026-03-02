"""Training script with Sample Factory-style async architecture.

Architecture:

    ┌─────────────┐                            ┌──────────────────┐
    │  Actor 0    │──pipe──┐                   │   Learner Thread │
    │ (subprocess)│        │    ┌────────--┐   │   (background)   │
    ├─────────────┤        ├───▸│  Main    │──▸│                  │
    │  Actor 1    │──pipe──┤    │ Process  │   │  replay buffer   │
    │ (subprocess)│        │    │          │   │  gradient steps  │
    ├─────────────┤        │    │ batched  │   │  target sync     │
    │    ...      │        │    │ inference│   └──────────────────┘
    ├─────────────┤        │    │ logging  │
    │  Actor N    │──pipe──┘    └────────--┘
    │ (subprocess)│
    └─────────────┘
       ▲ shared memory observations (zero-copy)
       ▲ CPU core affinity
       ▲ experience decorrelation

Usage:
    #parallelism
    python -m examples.drl_collision_avoidance.train drl.num_workers=5 drl.num_envs_per_worker=2

    # Sync (for debugging)
    python -m examples.drl_collision_avoidance.train drl.vec_env_mode=sync drl.num_workers=2 drl.num_envs_per_worker=2

    # Legacy total env count (backward-compatible)
    python -m examples.drl_collision_avoidance.train drl.num_envs=10

    # Single-process sequential (for debugging)
    python -m examples.drl_collision_avoidance.train drl.vec_env_mode=dummy drl.num_envs=1
"""

import os
import signal
import sys
import time
import threading
import queue

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from cfgs.config import set_display_window

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def _make_cfg_dict(cfg):
    if hasattr(cfg, '_content'):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


def _make_env_fn(cfg_dict, env_index):
    """Return a zero-arg factory for a CollisionAvoidanceEnv with a unique seed."""
    def _thunk():
        os.environ['OMP_NUM_THREADS'] = '1'
        torch.set_num_threads(1)
        from examples.drl_collision_avoidance.collision_avoidance_env import CollisionAvoidanceEnv
        env_cfg = dict(cfg_dict)
        env_cfg['seed'] = cfg_dict.get('seed', 42) + env_index * 1000
        return CollisionAvoidanceEnv(env_cfg)
    return _thunk


def _resolve_parallel_env_config(drl_cfg):
    """Resolve SF-style parallel knobs with backward-compatible legacy fallback."""
    legacy_num_envs_cfg = drl_cfg.get('num_envs')
    workers_cfg = drl_cfg.get('num_workers')
    envs_per_worker_cfg = drl_cfg.get('num_envs_per_worker')
    uses_sf_knobs = workers_cfg is not None or envs_per_worker_cfg is not None

    def _to_int(name, value):
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must be an integer, got {value!r}') from exc

    if uses_sf_knobs:
        num_workers = _to_int('drl.num_workers', workers_cfg) if workers_cfg is not None else 1
        num_envs_per_worker = _to_int('drl.num_envs_per_worker', envs_per_worker_cfg) if envs_per_worker_cfg is not None else 1
        total_envs = num_workers * num_envs_per_worker
        if (
            legacy_num_envs_cfg is not None
            and _to_int('drl.num_envs', legacy_num_envs_cfg) != total_envs
        ):
            print(
                '[ParallelConfig] Ignoring drl.num_envs because '
                'drl.num_workers/num_envs_per_worker are set: '
                f'{num_workers} x {num_envs_per_worker} = {total_envs}'
            )
    else:
        total_envs = _to_int('drl.num_envs', legacy_num_envs_cfg) if legacy_num_envs_cfg is not None else 8
        num_workers = total_envs
        num_envs_per_worker = 1

    if num_workers < 1 or num_envs_per_worker < 1:
        raise ValueError('drl.num_workers and drl.num_envs_per_worker must be >= 1')
    if total_envs < 1:
        raise ValueError('Total number of environments must be >= 1')

    return num_workers, num_envs_per_worker, total_envs


# ---------------------------------------------------------------------------
# Background Learner Thread (mirrors SF's train_in_background_thread)
# ---------------------------------------------------------------------------
class BackgroundLearner:
    """Runs gradient updates in a background thread, never blocking env stepping.

    Matches Sample Factory's ``train_in_background_thread=True`` mode: the main
    loop enqueues transition batches and the learner thread continuously drains
    them into the replay buffer + runs training steps.
    """

    def __init__(self, agent, train_freq, min_replay, maxsize=128):
        self.agent = agent
        self.train_freq = max(1, int(train_freq))
        self.min_replay = min_replay
        self._queue = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._steps_ingested = 0
        self._next_train_step = min_replay
        self._train_steps_done = 0
        self._latest_loss = None
        self._last_queue_full_warn = 0.0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True, name="LearnerThread")
        self._thread.start()

    def submit(self, states, actions, rewards, next_states, dones):
        """Backpressure-aware enqueue of a transition batch from the main thread."""
        try:
            self._queue.put((states, actions, rewards, next_states, dones), timeout=0.1)
            return True
        except queue.Full:
            now = time.time()
            if now - self._last_queue_full_warn >= 5.0:
                print("[BackgroundLearner] queue full, dropping transition batch")
                self._last_queue_full_warn = now
            return False

    def _run(self):
        while not self._stop.is_set():
            try:
                batch = self._queue.get(timeout=0.02)
            except queue.Empty:
                continue

            states, actions, rewards, next_states, dones = batch
            self.agent.store_transition_batch(states, actions, rewards, next_states, dones)
            self._steps_ingested += len(states)

            # Run training steps at the configured frequency
            while self._steps_ingested >= self._next_train_step:
                if not self._do_train_step():
                    break
                self._next_train_step += self.train_freq

    def _do_train_step(self):
        loss = self.agent.train_step(env_steps=self._steps_ingested)
        if loss is None:
            return False

        sync_inference = getattr(self.agent, 'sync_inference_net', None)
        if callable(sync_inference):
            sync_inference()

        with self._lock:
            self._latest_loss = loss
            self._train_steps_done += 1
        return True

    @property
    def latest_loss(self):
        with self._lock:
            return self._latest_loss

    @property
    def train_steps(self):
        with self._lock:
            return self._train_steps_done

    @property
    def steps(self):
        return self._steps_ingested

    @property
    def buffer_size(self):
        return len(self.agent.replay_buffer)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
@hydra.main(config_path="../../cfgs/drl_collision_avoidance", config_name="config")
def main(cfg):
    set_display_window()
    # Keep the main-process PyTorch thread pool to 1 so it doesn't fight the
    # env subprocesses for CPU cores. Workers set the same flag via their
    # env-factory thunks.
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    cfg_dict = _make_cfg_dict(cfg)

    from examples.drl_collision_avoidance.dqn_modules.ddqn_agent import DDQNAgent
    from examples.drl_collision_avoidance.scenario_utils import apply_scenario_path_defaults
    from examples.drl_collision_avoidance.vec_env import (
        AsyncSubprocVecEnv, SubprocVecEnv, DummyVecEnv,
    )

    cfg_dict = apply_scenario_path_defaults(cfg_dict, default_split='train')
    drl_cfg = cfg_dict.get('drl', {})
    num_workers, num_envs_per_worker, num_envs = _resolve_parallel_env_config(drl_cfg)
    vec_env_mode = str(drl_cfg.get('vec_env_mode', 'async')).lower()
    train_in_bg = drl_cfg.get('train_in_background', True)

    # --- Build vectorized environment ---
    env_fns = [_make_env_fn(cfg_dict, i) for i in range(num_envs)]

    vec_env_classes = {
        'async': AsyncSubprocVecEnv,
        'sync': SubprocVecEnv,
        'dummy': DummyVecEnv,
    }
    VecEnvCls = vec_env_classes.get(vec_env_mode)
    if VecEnvCls is None:
        raise ValueError(f"Unknown vec_env_mode={vec_env_mode!r}, expected one of {list(vec_env_classes)}")

    vec_env = VecEnvCls(env_fns)

    obs_dim = vec_env.observation_space.shape[0]
    n_actions = vec_env.action_space.n
    bg_label = 'BackgroundLearner' if train_in_bg else 'SyncLearner'
    print(f"\n{'='*60}")
    print(
        f"Training: {num_workers} workers x {num_envs_per_worker} envs/worker "
        f"= {num_envs} envs ({VecEnvCls.__name__}) + {bg_label}"
    )
    print(f"obs_dim={obs_dim} | n_actions={n_actions}")
    print(f"{'='*60}\n")

    grid_cfg = cfg_dict.get('occupancy_grid', {})
    grid_rows = grid_cfg.get('rows', 25)
    grid_cols = grid_cfg.get('cols', 14)
    grid_size = grid_rows * grid_cols

    # --- Build learner agent ---
    device = cfg_dict.get('device', 'cpu')
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'

    agent = DDQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        grid_size=grid_size,
        hidden_layers=drl_cfg['hidden_layers'],
        lr=drl_cfg['lr'],
        gamma=drl_cfg['gamma'],
        epsilon_start=drl_cfg['epsilon_start'],
        epsilon_end=drl_cfg['epsilon_end'],
        epsilon_decay_steps=drl_cfg['epsilon_decay_steps'],
        replay_buffer_size=drl_cfg['replay_buffer_size'],
        batch_size=drl_cfg['batch_size'],
        target_update_freq=drl_cfg['target_update_freq'],
        device=device,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        dueling=bool(drl_cfg.get('dueling', True)),
        alpha=drl_cfg.get('alpha', drl_cfg.get('per_alpha', 0.7)),
        beta_start=drl_cfg.get('beta_start', drl_cfg.get('per_beta_start', 0.4)),
        beta_frames=drl_cfg.get('beta_frames', drl_cfg.get('per_beta_frames', 100000)),
        per_epsilon=drl_cfg.get('per_epsilon', 1e-6),
    )

    # --- Training parameters ---
    num_episodes = drl_cfg['num_episodes']
    train_freq = drl_cfg['train_freq']
    min_replay = drl_cfg['min_replay_size']
    log_interval = drl_cfg['log_interval']
    save_interval = drl_cfg['save_interval']
    checkpoint_dir = drl_cfg.get('checkpoint_dir', 'checkpoints/drl_collision_avoidance')
    os.makedirs(checkpoint_dir, exist_ok=True)

    writer = _make_writer(checkpoint_dir)

    # --- Graceful shutdown ---
    running = True
    def _signal_handler(sig, frame):
        nonlocal running
        print('\nKeyboard interrupt detected, exiting...')
        running = False
    signal.signal(signal.SIGINT, _signal_handler)

    # --- Per-env episode trackers ---
    episode_rewards = np.zeros(num_envs, dtype=np.float64)
    episode_lengths = np.zeros(num_envs, dtype=np.int64)

    episodes_completed = 0
    global_step = 0
    reward_history = []
    start_time = time.time()

    # --- Start background learner if enabled ---
    learner = None
    if train_in_bg:
        learner = BackgroundLearner(agent, train_freq, min_replay)
        print("Background learner thread started.")

    # --- Initial reset ---
    obs = vec_env.reset()  # (num_envs, obs_dim)
    print("Collecting experience...\n")

    try:
        if vec_env_mode == 'async':
            current_obs = obs.copy()
            current_actions = agent.select_action_batch(current_obs)
            vec_env.step_async(current_actions)

        while episodes_completed < num_episodes and running:
            if vec_env_mode == 'async':
                ready_ids, next_obs, rewards, dones, infos = vec_env.step_wait(min_ready=1)
                if len(ready_ids) == 0:
                    continue
                prev_obs = current_obs[ready_ids]
                prev_actions = current_actions[ready_ids]
                # Re-dispatch envs immediately so they step concurrently with
                # the learner submit and episode bookkeeping below.
                current_obs[ready_ids] = next_obs
                next_actions = agent.select_action_batch(next_obs)
                current_actions[ready_ids] = next_actions
                vec_env.step_async(next_actions, env_ids=ready_ids)
                global_step += len(ready_ids)
            else:
                ready_ids = np.arange(num_envs, dtype=np.int64)
                prev_obs = obs
                prev_actions = agent.select_action_batch(obs)
                next_obs, rewards, dones, infos = vec_env.step(prev_actions)
                global_step += num_envs

            dones_f32 = dones.astype(np.float32)
            if learner is not None:
                learner.submit(prev_obs, prev_actions, rewards, next_obs, dones_f32)
            else:
                agent.store_transition_batch(prev_obs, prev_actions, rewards, next_obs, dones_f32)
                if len(agent.replay_buffer) >= min_replay and global_step % train_freq == 0:
                    agent.train_step(env_steps=global_step)

            episode_rewards[ready_ids] += rewards
            episode_lengths[ready_ids] += 1

            for local_idx, env_idx in enumerate(ready_ids):
                if dones[local_idx]:
                    episodes_completed += 1
                    ep_reward = episode_rewards[env_idx]
                    ep_length = int(episode_lengths[env_idx])
                    info = infos[local_idx]

                    reward_history.append(ep_reward)

                    if writer is not None:
                        loss_val = learner.latest_loss if learner else None
                        writer.add_scalar('train/episode_reward', ep_reward, episodes_completed)
                        writer.add_scalar('train/episode_length', ep_length, episodes_completed)
                        writer.add_scalar('train/epsilon', agent.epsilon, episodes_completed)
                        writer.add_scalar('train/collided', float(info.get('collided', False)), episodes_completed)
                        writer.add_scalar('train/goal_achieved', float(info.get('goal_achieved', False)), episodes_completed)
                        if loss_val is not None:
                            writer.add_scalar('train/loss', loss_val, episodes_completed)

                    if episodes_completed % log_interval == 0:
                        avg_rew = np.mean(reward_history[-log_interval:])
                        elapsed = max(time.time() - start_time, 1e-3)
                        fps = int(global_step / elapsed)
                        buf_sz = learner.buffer_size if learner else len(agent.replay_buffer)
                        train_steps = learner.train_steps if learner else agent.train_steps
                        print(
                            f'Episode {episodes_completed:5d} | '
                            f'avg_reward={avg_rew:8.2f} | '
                            f'eps={agent.epsilon:.3f} | '
                            f'goal_dist={info.get("goal_dist", float("inf")):.1f} | '
                            f'buf={buf_sz:6d} | '
                            f'env_steps={global_step} | '
                            f'train_steps={train_steps} | '
                            f'fps={fps}'
                        )

                    if episodes_completed % save_interval == 0:
                        path = os.path.join(checkpoint_dir, f'ddqn_ep{episodes_completed}.pth')
                        agent.save(path)
                        print(f'  → Saved checkpoint: {path}')

                    episode_rewards[env_idx] = 0.0
                    episode_lengths[env_idx] = 0

                    if episodes_completed >= num_episodes:
                        break

            if vec_env_mode != 'async':
                obs = next_obs

    finally:
        elapsed = time.time() - start_time
        print(f'\nTraining finished after {elapsed:.1f}s. Cleaning up...')

        if learner is not None:
            print(f'  Learner: {learner.train_steps} gradient steps, '
                  f'{learner.buffer_size} transitions in buffer')
            learner.stop()

        vec_env.close()

        final_path = os.path.join(checkpoint_dir, 'ddqn_final.pth')
        agent.save(final_path)
        print(f'  Final checkpoint: {final_path}')
        print(f'  Total episodes: {episodes_completed}, env_steps: {global_step}, '
              f'avg fps: {int(global_step / max(elapsed, 1e-3))}')

        if writer is not None:
            writer.close()


def _make_writer(log_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=os.path.join(log_dir, 'tb_logs'))
    except ImportError:
        print('TensorBoard not available — logging to console only.')
        return None


if __name__ == '__main__':
    main()
