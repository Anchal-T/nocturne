"""Training script with Sample Factory-style async architecture.

Architecture:

    ┌─────────────┐                            ┌──────────────────┐
    │  Actor 0    │──pipe──┐                   │  Learner Process │
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
import multiprocessing as mp
import signal
import sys
import time
import traceback
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


def _clone_transition_batch(states, actions, rewards, next_states, dones):
    return (
        np.asarray(states, dtype=np.float32).copy(),
        np.asarray(actions, dtype=np.int64).copy(),
        np.asarray(rewards, dtype=np.float32).copy(),
        np.asarray(next_states, dtype=np.float32).copy(),
        np.asarray(dones, dtype=np.float32).copy(),
    )


def _safe_queue_size(queue_obj) -> int:
    try:
        return int(queue_obj.qsize())
    except (AttributeError, NotImplementedError, OSError):
        return -1


def _replace_latest_queue_value(queue_obj, value) -> None:
    while True:
        try:
            queue_obj.put_nowait(value)
            return
        except queue.Full:
            try:
                queue_obj.get_nowait()
            except queue.Empty:
                continue


def _scheduled_train_steps(steps_ingested: int, min_replay: int, train_freq: int) -> int:
    if steps_ingested < min_replay:
        return 0
    return 1 + max(0, (steps_ingested - min_replay) // max(1, train_freq))


def _publish_learner_metrics(metrics_queue, transition_queue, agent, steps_ingested, latest_loss, min_replay, train_freq):
    scheduled_steps = _scheduled_train_steps(steps_ingested, min_replay, train_freq)
    _replace_latest_queue_value(
        metrics_queue,
        {
            'latest_loss': latest_loss,
            'train_steps': agent.train_steps,
            'steps_ingested': steps_ingested,
            'buffer_size': len(agent.replay_buffer),
            'queue_size': _safe_queue_size(transition_queue),
            'train_step_lag': max(0, scheduled_steps - agent.train_steps),
            'epsilon': float(agent.epsilon),
        },
    )


def _drain_learner_commands(control_queue, response_queue, agent, shutting_down: bool) -> bool:
    while True:
        try:
            command, payload = control_queue.get_nowait()
        except queue.Empty:
            return shutting_down

        if command == 'shutdown':
            shutting_down = True
            continue

        if command == 'save':
            try:
                agent.save(payload)
                response_queue.put({'type': 'save', 'path': payload, 'ok': True})
            except Exception:
                response_queue.put(
                    {
                        'type': 'save',
                        'path': payload,
                        'ok': False,
                        'error': traceback.format_exc(),
                    }
                )
            continue

        response_queue.put(
            {
                'type': 'fatal',
                'error': f'Unknown learner command: {command!r}',
            }
        )
        return True


def _learner_process_main(
    cfg_dict,
    obs_dim,
    n_actions,
    transition_queue,
    control_queue,
    response_queue,
    metrics_queue,
    weights_queue,
):
    try:
        os.environ['OMP_NUM_THREADS'] = '1'
        torch.set_num_threads(1)

        drl_cfg = cfg_dict['drl']
        train_freq = max(1, int(drl_cfg['train_freq']))
        min_replay = int(drl_cfg['min_replay_size'])
        max_train_steps_per_cycle = max(1, int(drl_cfg.get('learner_max_train_steps_per_cycle', 1)))
        metrics_interval_s = max(0.1, float(drl_cfg.get('learner_metrics_interval_s', 1.0)))
        weight_sync_interval = max(
            1,
            int(drl_cfg.get('learner_weight_sync_interval', drl_cfg.get('inference_sync_interval', 1))),
        )

        agent = _build_agent(cfg_dict, drl_cfg, obs_dim, n_actions)
        _replace_latest_queue_value(weights_queue, agent.export_inference_state(refresh=True))

        latest_loss = None
        steps_ingested = 0
        last_metrics_time = 0.0
        last_exported_train_step = agent.train_steps
        shutting_down = False

        while True:
            shutting_down = _drain_learner_commands(control_queue, response_queue, agent, shutting_down)

            if shutting_down:
                try:
                    batch = transition_queue.get_nowait()
                except queue.Empty:
                    batch = None
            else:
                try:
                    batch = transition_queue.get(timeout=0.05)
                except queue.Empty:
                    batch = None

            batches_ingested = 0
            while batch is not None:
                states, actions, rewards, next_states, dones = batch
                agent.store_transition_batch(states, actions, rewards, next_states, dones)
                steps_ingested += len(states)
                batches_ingested += 1
                try:
                    batch = transition_queue.get_nowait()
                except queue.Empty:
                    batch = None

            scheduled_steps = _scheduled_train_steps(steps_ingested, min_replay, train_freq)
            train_steps_owed = max(0, scheduled_steps - agent.train_steps)
            train_steps_budget = min(train_steps_owed, max_train_steps_per_cycle)
            train_steps_ran = 0

            while train_steps_ran < train_steps_budget:
                loss = agent.train_step(env_steps=steps_ingested)
                if loss is None:
                    break
                latest_loss = loss
                train_steps_ran += 1

            if agent.train_steps > last_exported_train_step:
                next_sync_step = (
                    (last_exported_train_step // weight_sync_interval) + 1
                ) * weight_sync_interval
                if agent.train_steps >= next_sync_step:
                    _replace_latest_queue_value(weights_queue, agent.export_inference_state(refresh=True))
                    last_exported_train_step = agent.train_steps

            now = time.time()
            if batches_ingested or train_steps_ran or (now - last_metrics_time) >= metrics_interval_s:
                _publish_learner_metrics(
                    metrics_queue=metrics_queue,
                    transition_queue=transition_queue,
                    agent=agent,
                    steps_ingested=steps_ingested,
                    latest_loss=latest_loss,
                    min_replay=min_replay,
                    train_freq=train_freq,
                )
                last_metrics_time = now

            if shutting_down and batch is None and train_steps_owed == 0:
                break

        _replace_latest_queue_value(weights_queue, agent.export_inference_state(refresh=True))
        _publish_learner_metrics(
            metrics_queue=metrics_queue,
            transition_queue=transition_queue,
            agent=agent,
            steps_ingested=steps_ingested,
            latest_loss=latest_loss,
            min_replay=min_replay,
            train_freq=train_freq,
        )
        agent.finalize_profiling()
    except Exception:
        try:
            response_queue.put({'type': 'fatal', 'error': traceback.format_exc()})
        except Exception:
            pass


class ProcessLearner:
    def __init__(self, cfg_dict, obs_dim, n_actions, agent):
        drl_cfg = cfg_dict['drl']
        self._ctx = mp.get_context('spawn')
        self._submit_timeout_s = max(0.0, float(drl_cfg.get('learner_queue_put_timeout_s', 0.25)))
        self._startup_timeout_s = max(5.0, float(drl_cfg.get('learner_startup_timeout_s', 60.0)))
        self._overload_policy = str(drl_cfg.get('learner_overload_policy', 'block')).lower()
        if self._overload_policy not in {'block', 'drop_new', 'drop_oldest'}:
            raise ValueError(
                'drl.learner_overload_policy must be one of '
                "{'block', 'drop_new', 'drop_oldest'}"
            )

        queue_size = max(1, int(drl_cfg.get('learner_queue_size', 256)))
        self._transition_queue = self._ctx.Queue(maxsize=queue_size)
        self._control_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._metrics_queue = self._ctx.Queue(maxsize=8)
        self._weights_queue = self._ctx.Queue(maxsize=1)
        self._pending_responses = []
        self._last_queue_full_warn = 0.0

        self._latest_loss = None
        self._train_steps_done = 0
        self._steps_ingested = 0
        self._buffer_size = 0
        self._queue_size = 0
        self._train_step_lag = 0
        self._last_weight_sync_step = 0
        self._dropped_batches = 0

        self._process = self._ctx.Process(
            target=_learner_process_main,
            args=(
                cfg_dict,
                obs_dim,
                n_actions,
                self._transition_queue,
                self._control_queue,
                self._response_queue,
                self._metrics_queue,
                self._weights_queue,
            ),
            daemon=True,
            name='LearnerProcess',
        )
        self._process.start()
        self._wait_for_initial_state(agent)

    def _record_drop(self) -> None:
        self._dropped_batches += 1
        now = time.time()
        if now - self._last_queue_full_warn >= 5.0:
            print('[ProcessLearner] queue saturated, dropping transition batch')
            self._last_queue_full_warn = now

    def _ensure_process_alive(self) -> None:
        if self._process.is_alive():
            return
        self._drain_responses()
        raise RuntimeError('Learner process exited unexpectedly.')

    def _drain_responses(self) -> None:
        while True:
            try:
                response = self._response_queue.get_nowait()
            except queue.Empty:
                return

            if response.get('type') == 'fatal':
                raise RuntimeError(response.get('error', 'Learner process failed.'))
            self._pending_responses.append(response)

    def _wait_for_response(self, response_type: str, path: str, timeout_s: float):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self._drain_responses()
            for index, response in enumerate(self._pending_responses):
                if response.get('type') == response_type and response.get('path') == path:
                    return self._pending_responses.pop(index)
            self._ensure_process_alive()
            time.sleep(0.05)
        raise TimeoutError(f'Timed out waiting for learner response {response_type!r} for {path!r}')

    def _wait_for_initial_state(self, agent) -> None:
        deadline = time.time() + self._startup_timeout_s
        while time.time() < deadline:
            self._drain_responses()
            try:
                state = self._weights_queue.get(timeout=0.2)
            except queue.Empty:
                self._ensure_process_alive()
                continue

            agent.load_inference_state(state)
            self._last_weight_sync_step = int(state.get('train_steps', 0))
            return

        raise TimeoutError('Learner process did not publish initial inference weights in time.')

    def submit(self, states, actions, rewards, next_states, dones):
        self._drain_responses()
        self._ensure_process_alive()
        payload = _clone_transition_batch(states, actions, rewards, next_states, dones)

        if self._overload_policy == 'drop_new':
            try:
                self._transition_queue.put_nowait(payload)
                return True
            except queue.Full:
                self._record_drop()
                return False

        if self._overload_policy == 'drop_oldest':
            try:
                self._transition_queue.put_nowait(payload)
                return True
            except queue.Full:
                try:
                    self._transition_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._transition_queue.put_nowait(payload)
                    self._record_drop()
                    return True
                except queue.Full:
                    self._record_drop()
                    return False

        try:
            self._transition_queue.put(payload, timeout=self._submit_timeout_s)
            return True
        except queue.Full:
            self._record_drop()
            return False

    def poll(self, agent) -> None:
        self._drain_responses()

        latest_metrics = None
        while True:
            try:
                latest_metrics = self._metrics_queue.get_nowait()
            except queue.Empty:
                break

        if latest_metrics is not None:
            self._latest_loss = latest_metrics.get('latest_loss')
            self._train_steps_done = int(latest_metrics.get('train_steps', self._train_steps_done))
            self._steps_ingested = int(latest_metrics.get('steps_ingested', self._steps_ingested))
            self._buffer_size = int(latest_metrics.get('buffer_size', self._buffer_size))
            self._queue_size = int(latest_metrics.get('queue_size', self._queue_size))
            self._train_step_lag = int(latest_metrics.get('train_step_lag', self._train_step_lag))

        latest_state = None
        while True:
            try:
                latest_state = self._weights_queue.get_nowait()
            except queue.Empty:
                break

        if latest_state is not None:
            agent.load_inference_state(latest_state)
            self._last_weight_sync_step = int(latest_state.get('train_steps', self._last_weight_sync_step))

        if self._queue_size < 0:
            self._queue_size = _safe_queue_size(self._transition_queue)

        self._ensure_process_alive()

    @property
    def latest_loss(self):
        return self._latest_loss

    @property
    def train_steps(self):
        return self._train_steps_done

    @property
    def steps(self):
        return self._steps_ingested

    @property
    def buffer_size(self):
        return self._buffer_size

    @property
    def queue_size(self):
        return self._queue_size

    @property
    def train_step_lag(self):
        return self._train_step_lag

    @property
    def last_weight_sync_step(self):
        return self._last_weight_sync_step

    @property
    def dropped_batches(self):
        return self._dropped_batches

    def save(self, path: str, timeout_s: float = 120.0) -> None:
        self._drain_responses()
        self._ensure_process_alive()
        self._control_queue.put(('save', path))
        response = self._wait_for_response('save', path, timeout_s)
        if not response.get('ok', False):
            raise RuntimeError(response.get('error', f'Learner failed to save checkpoint to {path}'))

    def stop(self):
        if not self._process.is_alive():
            return

        try:
            self._control_queue.put(('shutdown', None))
        except Exception:
            pass

        self._process.join(timeout=10.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)

def _resolve_training_device(cfg_dict, drl_cfg):
    device = cfg_dict['device']
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    if device.startswith('cuda'):
        use_tf32 = bool(drl_cfg['use_tf32'])
        cudnn_benchmark = bool(drl_cfg['cudnn_benchmark'])
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

    grid_cfg = cfg_dict['occupancy_grid']
    grid_channels = 3
    grid_rows = grid_cfg['rows']
    grid_cols = grid_cfg['cols']
    grid_size = grid_channels * grid_rows * grid_cols
    device = _resolve_training_device(cfg_dict, drl_cfg)

    agent_cfg = DDQNAgentConfig.from_drl_cfg(
        drl_cfg,
        grid_size=grid_size,
        grid_channels=grid_channels,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        device=device,
    )
    return DDQNAgent(obs_dim=obs_dim, n_actions=n_actions, config=agent_cfg)


def _print_training_header(num_workers, num_envs_per_worker, num_envs, vec_env_cls, train_in_bg, obs_dim, n_actions):
    bg_label = 'ProcessLearner' if train_in_bg else 'SyncLearner'
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
    if learner is not None:
        writer.add_scalar('learner/replay_size', learner.buffer_size, episodes_completed)
        writer.add_scalar('learner/train_steps', learner.train_steps, episodes_completed)
        writer.add_scalar('learner/train_step_lag', learner.train_step_lag, episodes_completed)
        writer.add_scalar('learner/dropped_batches', learner.dropped_batches, episodes_completed)
        if learner.queue_size >= 0:
            writer.add_scalar('learner/queue_size', learner.queue_size, episodes_completed)
        writer.add_scalar('learner/last_weight_sync_step', learner.last_weight_sync_step, episodes_completed)


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
    queue_fragment = ''
    if learner is not None:
        queue_fragment = (
            f' | q={learner.queue_size:4d}'
            f' | lag={learner.train_step_lag:4d}'
            f' | dropped={learner.dropped_batches:4d}'
        )
    print(
        f'Episode {episodes_completed:5d} | '
        f'avg_reward={avg_rew:8.2f} | '
        f'eps={agent.epsilon:.3f} | '
        f'goal_dist={info.get("goal_dist", float("inf")):.1f} | '
        f'buf={buf_sz:6d} | '
        f'env_steps={global_step} | '
        f'train_steps={train_steps}'
        f'{queue_fragment}'
        f' | fps={fps}'
    )


def _maybe_save_checkpoint(episodes_completed, save_interval, checkpoint_dir, agent, learner):
    if episodes_completed % save_interval != 0:
        return
    path = os.path.join(checkpoint_dir, f'ddqn_ep{episodes_completed}.pth')
    if learner is not None:
        learner.save(path)
    else:
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
        _maybe_save_checkpoint(episodes_completed, save_interval, checkpoint_dir, agent, learner)

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
        if learner is not None:
            learner.poll(agent)

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

        if learner is not None:
            agent.update_exploration(global_step)
            learner.poll(agent)

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
    drl_cfg = cfg_dict['drl']
    num_workers, num_envs_per_worker, num_envs = _resolve_parallel_env_config(drl_cfg)
    vec_env_mode = str(drl_cfg['vec_env_mode']).lower()
    train_in_bg = drl_cfg['train_in_background']

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
    checkpoint_dir = drl_cfg['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    writer = _make_writer(checkpoint_dir)
    running_state = {'running': True}

    def _signal_handler(sig, frame):
        print('\nKeyboard interrupt detected, exiting...')
        running_state['running'] = False

    signal.signal(signal.SIGINT, _signal_handler)

    learner = None
    if train_in_bg:
        learner = ProcessLearner(cfg_dict, obs_dim, n_actions, agent)
        print('Learner process started.')

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
            learner.poll(agent)
            print(
                f'  Learner: {learner.train_steps} gradient steps, '
                f'{learner.buffer_size} transitions in buffer, '
                f'dropped_batches={learner.dropped_batches}, '
                f'train_step_lag={learner.train_step_lag}'
            )

        vec_env.close()

        final_path = os.path.join(checkpoint_dir, 'ddqn_final.pth')
        if learner is not None:
            try:
                learner.save(final_path)
                print(f'  Final checkpoint: {final_path}')
            finally:
                learner.stop()
        else:
            agent.finalize_profiling()
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
