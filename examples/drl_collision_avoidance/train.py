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
from typing import Any, Dict, cast

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from cfgs.config import set_display_window

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def _make_cfg_dict(cfg) -> Dict[str, Any]:
    if hasattr(cfg, '_content'):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    else:
        cfg_dict = dict(cfg)

    if not isinstance(cfg_dict, dict):
        raise TypeError(f'Expected mapping-like cfg, got {type(cfg_dict).__name__}')
    return cast(Dict[str, Any], cfg_dict)


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

def _resolve_training_device(cfg_dict, drl_cfg):
    device = cfg_dict.get('device', 'cpu')
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    if device.startswith('cuda'):
        use_tf32 = bool(drl_cfg.get('use_tf32', True))
        cudnn_benchmark = bool(drl_cfg.get('cudnn_benchmark', True))
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return device


def _build_vec_env(cfg_dict, num_envs, vec_env_mode):
    from examples.drl_collision_avoidance.vec_env import (
        AsyncSubprocVecEnv,
        DummyVecEnv,
        SubprocVecEnv,
    )

    env_fns = [_make_env_fn(cfg_dict, i) for i in range(num_envs)]
    vec_env_classes = {
        'async': AsyncSubprocVecEnv,
        'sync': SubprocVecEnv,
        'dummy': DummyVecEnv,
    }
    vec_env_cls = vec_env_classes.get(vec_env_mode)
    if vec_env_cls is None:
        raise ValueError(f"Unknown vec_env_mode={vec_env_mode!r}, expected one of {list(vec_env_classes)}")

    return vec_env_cls(env_fns), vec_env_cls


def _build_agent(cfg_dict, drl_cfg, obs_dim, n_actions):
    from examples.drl_collision_avoidance.dqn_modules.ddqn_agent import DDQNAgent, DDQNAgentConfig

    grid_cfg = cfg_dict.get('occupancy_grid', {})
    grid_rows = grid_cfg.get('rows', 25)
    grid_cols = grid_cfg.get('cols', 14)
    grid_size = grid_rows * grid_cols
    device = _resolve_training_device(cfg_dict, drl_cfg)

    agent_cfg = DDQNAgentConfig.from_drl_cfg(
        drl_cfg,
        grid_size=grid_size,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        device=device,
    )
    return DDQNAgent.from_config(obs_dim=obs_dim, n_actions=n_actions, config=agent_cfg)


def _print_training_header(num_workers, num_envs_per_worker, num_envs, vec_env_cls, train_in_bg, obs_dim, n_actions):
    bg_label = 'BackgroundLearner' if train_in_bg else 'SyncLearner'
    print(f"\n{'='*60}")
    print(
        f"Training: {num_workers} workers x {num_envs_per_worker} envs/worker "
        f"= {num_envs} envs ({vec_env_cls.__name__}) + {bg_label}"
    )
    print(f"obs_dim={obs_dim} | n_actions={n_actions}")
    print(f"{'='*60}\n")


def _collect_transition_batch(
    vec_env,
    vec_env_mode,
    obs,
    current_obs,
    current_actions,
    num_envs,
    agent,
):
    if vec_env_mode == 'async':
        ready_ids, next_obs, rewards, dones, infos = vec_env.step_wait(min_ready=1)
        if len(ready_ids) == 0:
            return None, obs, current_obs, current_actions, 0

        prev_obs = current_obs[ready_ids]
        prev_actions = current_actions[ready_ids]
        current_obs[ready_ids] = next_obs
        next_actions = agent.select_action_batch(next_obs)
        current_actions[ready_ids] = next_actions
        vec_env.step_async(next_actions, env_ids=ready_ids)

        return (
            (ready_ids, prev_obs, prev_actions, next_obs, rewards, dones, infos),
            obs,
            current_obs,
            current_actions,
            len(ready_ids),
        )

    ready_ids = np.arange(num_envs, dtype=np.int64)
    prev_obs = obs
    prev_actions = agent.select_action_batch(obs)
    next_obs, rewards, dones, infos = vec_env.step(prev_actions)
    return (
        (ready_ids, prev_obs, prev_actions, next_obs, rewards, dones, infos),
        next_obs,
        current_obs,
        current_actions,
        num_envs,
    )


def _submit_or_train_batch(learner, agent, prev_obs, prev_actions, rewards, next_obs, dones, min_replay, train_freq, global_step):
    dones_f32 = dones.astype(np.float32)
    if learner is not None:
        learner.submit(prev_obs, prev_actions, rewards, next_obs, dones_f32)
        return

    agent.store_transition_batch(prev_obs, prev_actions, rewards, next_obs, dones_f32)
    if len(agent.replay_buffer) >= min_replay and global_step % train_freq == 0:
        agent.train_step(env_steps=global_step)


def _write_episode_metrics(writer, learner, agent, ep_reward, ep_length, info, episodes_completed):
    if writer is None:
        return
    loss_val = learner.latest_loss if learner else None
    writer.add_scalar('train/episode_reward', ep_reward, episodes_completed)
    writer.add_scalar('train/episode_length', ep_length, episodes_completed)
    writer.add_scalar('train/epsilon', agent.epsilon, episodes_completed)
    writer.add_scalar('train/collided', float(info.get('collided', False)), episodes_completed)
    writer.add_scalar('train/goal_achieved', float(info.get('goal_achieved', False)), episodes_completed)
    if loss_val is not None:
        writer.add_scalar('train/loss', loss_val, episodes_completed)


def _maybe_print_episode_log(
    episodes_completed,
    log_interval,
    reward_history,
    start_time,
    global_step,
    learner,
    agent,
    info,
):
    if episodes_completed % log_interval != 0:
        return

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


def _maybe_save_checkpoint(episodes_completed, save_interval, checkpoint_dir, agent):
    if episodes_completed % save_interval != 0:
        return
    path = os.path.join(checkpoint_dir, f'ddqn_ep{episodes_completed}.pth')
    agent.save(path)
    print(f'  → Saved checkpoint: {path}')


def _process_done_episodes(
    ready_ids,
    dones,
    infos,
    episode_rewards,
    episode_lengths,
    episodes_completed,
    reward_history,
    writer,
    learner,
    agent,
    log_interval,
    save_interval,
    checkpoint_dir,
    global_step,
    start_time,
    num_episodes,
):
    for local_idx, env_idx in enumerate(ready_ids):
        if not dones[local_idx]:
            continue

        episodes_completed += 1
        ep_reward = episode_rewards[env_idx]
        ep_length = int(episode_lengths[env_idx])
        info = infos[local_idx]
        reward_history.append(ep_reward)

        _write_episode_metrics(writer, learner, agent, ep_reward, ep_length, info, episodes_completed)
        _maybe_print_episode_log(
            episodes_completed=episodes_completed,
            log_interval=log_interval,
            reward_history=reward_history,
            start_time=start_time,
            global_step=global_step,
            learner=learner,
            agent=agent,
            info=info,
        )
        _maybe_save_checkpoint(episodes_completed, save_interval, checkpoint_dir, agent)

        episode_rewards[env_idx] = 0.0
        episode_lengths[env_idx] = 0

        if episodes_completed >= num_episodes:
            break

    return episodes_completed


def _run_training_loop(
    vec_env,
    vec_env_mode,
    agent,
    learner,
    num_envs,
    num_episodes,
    min_replay,
    train_freq,
    log_interval,
    save_interval,
    checkpoint_dir,
    writer,
    start_time,
    running_state,
):
    episode_rewards = np.zeros(num_envs, dtype=np.float64)
    episode_lengths = np.zeros(num_envs, dtype=np.int64)
    episodes_completed = 0
    global_step = 0
    reward_history = []

    obs = vec_env.reset()
    print("Collecting experience...\n")

    current_obs = obs.copy()
    current_actions = agent.select_action_batch(current_obs)
    if vec_env_mode == 'async':
        vec_env.step_async(current_actions)

    while episodes_completed < num_episodes and running_state['running']:
        step_batch, obs, current_obs, current_actions, step_size = _collect_transition_batch(
            vec_env=vec_env,
            vec_env_mode=vec_env_mode,
            obs=obs,
            current_obs=current_obs,
            current_actions=current_actions,
            num_envs=num_envs,
            agent=agent,
        )
        if step_batch is None:
            continue

        ready_ids, prev_obs, prev_actions, next_obs, rewards, dones, infos = step_batch
        global_step += step_size

        _submit_or_train_batch(
            learner=learner,
            agent=agent,
            prev_obs=prev_obs,
            prev_actions=prev_actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            min_replay=min_replay,
            train_freq=train_freq,
            global_step=global_step,
        )

        episode_rewards[ready_ids] += rewards
        episode_lengths[ready_ids] += 1
        episodes_completed = _process_done_episodes(
            ready_ids=ready_ids,
            dones=dones,
            infos=infos,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            episodes_completed=episodes_completed,
            reward_history=reward_history,
            writer=writer,
            learner=learner,
            agent=agent,
            log_interval=log_interval,
            save_interval=save_interval,
            checkpoint_dir=checkpoint_dir,
            global_step=global_step,
            start_time=start_time,
            num_episodes=num_episodes,
        )

    return episodes_completed, global_step


@hydra.main(config_path="../../cfgs/drl_collision_avoidance", config_name="config")
def main(cfg):
    set_display_window()
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    cfg_dict = _make_cfg_dict(cfg)

    from examples.drl_collision_avoidance.scenario_utils import apply_scenario_path_defaults

    cfg_dict = apply_scenario_path_defaults(cfg_dict, default_split='train')
    drl_cfg = cfg_dict.get('drl', {})
    num_workers, num_envs_per_worker, num_envs = _resolve_parallel_env_config(drl_cfg)
    vec_env_mode = str(drl_cfg.get('vec_env_mode', 'async')).lower()
    train_in_bg = drl_cfg.get('train_in_background', True)

    vec_env, vec_env_cls = _build_vec_env(cfg_dict, num_envs, vec_env_mode)
    obs_dim = vec_env.observation_space.shape[0]
    n_actions = vec_env.action_space.n
    _print_training_header(
        num_workers,
        num_envs_per_worker,
        num_envs,
        vec_env_cls,
        train_in_bg,
        obs_dim,
        n_actions,
    )

    agent = _build_agent(cfg_dict, drl_cfg, obs_dim, n_actions)

    num_episodes = drl_cfg['num_episodes']
    train_freq = drl_cfg['train_freq']
    min_replay = drl_cfg['min_replay_size']
    log_interval = drl_cfg['log_interval']
    save_interval = drl_cfg['save_interval']
    checkpoint_dir = drl_cfg.get('checkpoint_dir', 'checkpoints/drl_collision_avoidance')
    os.makedirs(checkpoint_dir, exist_ok=True)

    writer = _make_writer(checkpoint_dir)
    running_state = {'running': True}

    def _signal_handler(sig, frame):
        print('\nKeyboard interrupt detected, exiting...')
        running_state['running'] = False

    signal.signal(signal.SIGINT, _signal_handler)

    learner = None
    if train_in_bg:
        learner = BackgroundLearner(agent, train_freq, min_replay)
        print("Background learner thread started.")

    start_time = time.time()
    episodes_completed = 0
    global_step = 0

    try:
        episodes_completed, global_step = _run_training_loop(
            vec_env=vec_env,
            vec_env_mode=vec_env_mode,
            agent=agent,
            learner=learner,
            num_envs=num_envs,
            num_episodes=num_episodes,
            min_replay=min_replay,
            train_freq=train_freq,
            log_interval=log_interval,
            save_interval=save_interval,
            checkpoint_dir=checkpoint_dir,
            writer=writer,
            start_time=start_time,
            running_state=running_state,
        )
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
        from torch.utils.tensorboard.writer import SummaryWriter
        return SummaryWriter(log_dir=os.path.join(log_dir, 'tb_logs'))
    except ImportError:
        print('TensorBoard not available — logging to console only.')
        return None


if __name__ == '__main__':
    main()
