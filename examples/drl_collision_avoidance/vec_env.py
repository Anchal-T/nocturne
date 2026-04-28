"""Vectorized environment wrappers for parallel DRL training.

Provides three levels of parallelism, closely modeled on Sample Factory /
PPO ``env_wrappers.py``:

* ``DummyVecEnv``         – sequential, in-process (debugging)
* ``SubprocVecEnv``       – one subprocess per env, synchronous stepping
* ``AsyncSubprocVecEnv``  – fully async actors with experience decorrelation,
                            CPU-core pinning, and shared-memory observation
                            buffers (closest to Sample Factory)
"""

import math
import os
import time
import traceback
from multiprocessing.connection import wait

import numpy as np
import torch
import torch.multiprocessing as mp
import logging

# Module logger
logger = logging.getLogger(__name__)


# Info buffer column order — must match between _write_info_to_buffer and step_wait.
_INFO_KEYS = ("collided", "goal_achieved", "goal_dist")
_INFO_CASTS = (bool, bool, float)


def _probe_spaces_from_env_fn(env_fn):
    """Create one env instance to read observation/action spaces safely."""
    env = env_fn()
    try:
        return env.observation_space, env.action_space
    finally:
        env.close()


# Serialization helper
class CloudpickleWrapper:
    """Serialize env factory functions with cloudpickle for subprocess pickling."""

    def __init__(self, fn):
        self.fn = fn

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.fn)

    def __setstate__(self, ob):
        import pickle

        self.fn = pickle.loads(ob)

    def __call__(self):
        return self.fn()


# DummyVecEnv — sequential, single-process (for debugging)
class DummyVecEnv:
    """Runs all envs sequentially in the main process."""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def step(self, actions):
        obs_list, rew_list, done_list, info_list = [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                obs, _ = env.reset()
            obs_list.append(obs)
            rew_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return (
            np.stack(obs_list),
            np.array(rew_list, dtype=np.float32),
            np.array(done_list, dtype=bool),
            info_list,
        )

    def reset(self):
        return np.stack([env.reset()[0] for env in self.envs])

    def close(self):
        for env in self.envs:
            env.close()


# Worker functions
def _worker(remote, parent_remote, env_fn_wrapper):
    """Basic synchronous subprocess worker (for SubprocVecEnv)."""
    parent_remote.close()
    env = env_fn_wrapper()

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            done = terminated or truncated
            if done:
                obs, _ = env.reset()
            remote.send((obs, reward, done, info))
        elif cmd == "reset":
            obs, _ = env.reset()
            remote.send(obs)
        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.action_space))
        elif cmd == "close":
            env.close()
            remote.close()
            break
        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


def _setup_worker_env(worker_id, cpu_cores):
    """Configures CPU affinity and threading for the async worker."""
    if cpu_cores:
        try:
            os.sched_setaffinity(0, cpu_cores)
            logger.info("[Actor %s] pid %s pinned to cores %s", worker_id, os.getpid(), sorted(cpu_cores))
        except (AttributeError, OSError):
            pass
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _decorrelate_envs(envs, steps, obs_buffer, env_indices, max_consecutive_failures=5):
    """Performs initial random steps to decorrelate environment streams."""
    obs = envs.reset()
    consecutive_failures = 0
    for _ in range(steps):
        actions = np.random.randint(0, envs.action_space.n, size=envs.n_envs)
        try:
            obs, _, _, _ = envs.step(actions)
            consecutive_failures = 0
        except Exception:
            consecutive_failures += 1
            logger.exception("[decorrelate] Exception during random step")
            if consecutive_failures >= max_consecutive_failures:
                logger.warning("[decorrelate] %s consecutive failures, aborting decorrelation.", max_consecutive_failures)
                break
            obs = envs.reset()
    obs_buffer[env_indices].copy_(torch.from_numpy(obs).float())


def _write_info_to_buffer(info_buffer, env_indices, infos):
    """Extracts _INFO_KEYS metrics from info dicts into the shared memory buffer."""
    data = np.array(
        [[float(info.get(k, 0.0)) for k in _INFO_KEYS] for info in infos],
        dtype=np.float32,
    )
    info_buffer[env_indices] = torch.from_numpy(data)


def _handle_worker_step(envs, worker_id, env_indices, buffers):
    """Executes a vectorized step and updates shared memory buffers."""
    obs_buf, act_buf, rew_buf, done_buf, info_buf = buffers
    actions = act_buf[env_indices].numpy()

    try:
        obs, reward, done, infos = envs.step(actions)
    except Exception:
        logger.exception("[Actor %s] Exception in step", worker_id)
        obs = envs.reset()
        reward = np.zeros(envs.n_envs, dtype=np.float32)
        done = np.ones(envs.n_envs, dtype=bool)
        infos = [{} for _ in range(envs.n_envs)]

    rew_buf[env_indices] = torch.from_numpy(reward)
    done_buf[env_indices] = torch.from_numpy(done)
    _write_info_to_buffer(info_buf, env_indices, infos)
    obs_buf[env_indices].copy_(torch.from_numpy(obs).float())


def _async_worker(
    worker_id,
    remote,
    parent_remote,
    env_fn_wrappers,
    obs_buffer,
    action_buffer,
    reward_buffer,
    done_buffer,
    info_buffer,
    decorrelation_steps,
    cpu_cores,
    env_indices,
):
    """Fully-async worker with zero-copy PyTorch shared memory and internal vectorization."""
    parent_remote.close()
    _setup_worker_env(worker_id, cpu_cores)
    envs = DummyVecEnv(env_fn_wrappers)
    _decorrelate_envs(envs, decorrelation_steps, obs_buffer, env_indices)

    buffers = (obs_buffer, action_buffer, reward_buffer, done_buffer, info_buffer)

    try:
        while True:
            cmd, _ = remote.recv()
            if cmd == "step":
                _handle_worker_step(envs, worker_id, env_indices, buffers)
                remote.send("step_done")
            elif cmd == "reset":
                obs = envs.reset()
                obs_buffer[env_indices].copy_(torch.from_numpy(obs).float())
                remote.send("reset_done")
            elif cmd == "get_spaces":
                remote.send((envs.observation_space, envs.action_space))
            elif cmd == "close":
                envs.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except Exception as e:
        logger.exception("[Actor %s] Fatal error", worker_id)
        # Notify the main process so step_wait doesn't hang waiting for this worker.
        try:
            remote.send(e)
        except Exception:
            pass


# SubprocVecEnv — one subprocess per env, synchronous lock-step
class SubprocVecEnv:
    """Vectorized env with one subprocess per env and pipe-based IPC.

    All envs are stepped together in lock-step. Modeled directly on the PPO
    ``SubprocVecEnv`` from ``algos/ppo/env_wrappers.py``.
    """

    def __init__(self, env_fns):
        self.n_envs = len(env_fns)
        self.closed = False

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.n_envs)])

        self.processes = []
        for wr, r, fn in zip(self.work_remotes, self.remotes, env_fns):
            p = mp.Process(
                target=_worker, args=(wr, r, CloudpickleWrapper(fn)), daemon=True
            )
            p.start()
            self.processes.append(p)
        for wr in self.work_remotes:
            wr.close()

        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        return (
            np.stack(obs),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            list(infos),
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.closed = True


# ---------------------------------------------------------------------------
# AsyncSubprocVecEnv — fully async actors, shared-mem obs, decorrelation
# (closest to Sample Factory's actor/runner architecture)
# ---------------------------------------------------------------------------
class AsyncSubprocVecEnv:
    """Fully-async vectorized env matching Sample Factory's worker pattern.

    Features (mirroring Sample Factory):
    * **Zero-copy shared memory** — uses PyTorch tensors for obs, actions,
      rewards, dones, and info, eliminating pipe overhead.
    * **Internal Vectorization** — workers use DummyVecEnv to execute multiple
      environments sequentially, writing batches to shared memory.
    * **Experience decorrelation** — each worker executes a staggered
      number of random steps during init.
    * **CPU core affinity** — workers are pinned to specific cores.
    * **Async stepping** — ``step_async`` / ``step_wait`` split so the
      main thread can overlap inference with env execution.
    """

    def __init__(
        self,
        env_fns,
        num_envs_per_worker=1,
        decorrelation_base=20,
        observation_space=None,
        action_space=None,
    ):
        self.n_envs = len(env_fns)
        if self.n_envs == 0:
            raise ValueError("env_fns must contain at least one env factory")
        if self.n_envs % num_envs_per_worker != 0:
            raise ValueError(
                f"Total envs {self.n_envs} must be divisible by num_envs_per_worker {num_envs_per_worker}"
            )

        self.num_envs_per_worker = num_envs_per_worker
        self.num_workers = self.n_envs // num_envs_per_worker
        self.closed = False
        self._pending_workers = set()

        if (observation_space is None) != (action_space is None):
            raise ValueError(
                "observation_space and action_space must be provided together"
            )

        if observation_space is None:
            # Shared-memory buffers need shapes before worker startup.
            self.observation_space, self.action_space = _probe_spaces_from_env_fn(env_fns[0])
        else:
            self.observation_space = observation_space
            self.action_space = action_space
        obs_shape = self.observation_space.shape

        self._obs_shape = obs_shape

        # Allocate zero-copy shared memory using PyTorch Tensors
        self._obs_buf = torch.zeros(
            (self.n_envs, *obs_shape), dtype=torch.float32
        ).share_memory_()

        if hasattr(self.action_space, "n"):
            self._action_buf = torch.zeros(
                self.n_envs, dtype=torch.int64
            ).share_memory_()
        else:
            self._action_buf = torch.zeros(
                (self.n_envs, *self.action_space.shape), dtype=torch.float32
            ).share_memory_()

        self._reward_buf = torch.zeros(self.n_envs, dtype=torch.float32).share_memory_()
        self._done_buf = torch.zeros(self.n_envs, dtype=torch.bool).share_memory_()
        self._info_buf = torch.zeros(
            (self.n_envs, 3), dtype=torch.float32
        ).share_memory_()  # [collided, goal_achieved, goal_dist]

        # Assign CPU cores round-robin
        try:
            available_cores = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available_cores = list(range(os.cpu_count() or 1))

        # Spawn async workers
        self.remotes, self.work_remotes = zip(
            *[mp.Pipe() for _ in range(self.num_workers)]
        )
        self.processes = []

        logger.info(
            "[AsyncSubprocVecEnv] Initializing %s PyTorch mp workers (each with %s envs)...",
            self.num_workers,
            self.num_envs_per_worker,
        )

        for i in range(self.num_workers):
            core = [available_cores[i % len(available_cores)]]
            decorr_steps = (i * decorrelation_base) % (
                decorrelation_base * self.num_workers
            )
            start_idx = i * self.num_envs_per_worker
            end_idx = start_idx + self.num_envs_per_worker
            worker_env_fns = env_fns[start_idx:end_idx]
            wrappers = [CloudpickleWrapper(fn) for fn in worker_env_fns]
            env_indices = list(range(start_idx, end_idx))

            p = mp.Process(
                target=_async_worker,
                args=(
                    i,
                    self.work_remotes[i],
                    self.remotes[i],
                    wrappers,
                    self._obs_buf,
                    self._action_buf,
                    self._reward_buf,
                    self._done_buf,
                    self._info_buf,
                    decorr_steps,
                    core,
                    env_indices,
                ),
                daemon=True,
            )
            p.start()
            self.processes.append(p)

        for wr in self.work_remotes:
            wr.close()
        self._remote_to_id = {remote: i for i, remote in enumerate(self.remotes)}

        # Wait for all workers to finish init
        for i, remote in enumerate(self.remotes):
            remote.send(("get_spaces", None))
            remote.recv()
            logger.info("  Worker %s initialized (pid %s)", i, self.processes[i].pid)

        logger.info("[AsyncSubprocVecEnv] All %s workers ready.", self.num_workers)

    def get_obs(self, env_ids=None, copy=True):
        """Read current observations from shared memory."""
        if env_ids is None:
            return self._obs_buf.numpy().copy() if copy else self._obs_buf.numpy()
        obs = self._obs_buf[np.asarray(env_ids, dtype=np.int64)].numpy()
        return obs.copy() if copy else obs

    def step_async(self, actions, env_ids=None):
        """Write actions to shared memory and send signal to workers (non-blocking)."""
        if env_ids is None:
            env_ids = np.arange(self.n_envs, dtype=np.int64)
        else:
            env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)

        if env_ids.size == 0:
            return

        unique_positions = np.sort(np.unique(env_ids, return_index=True)[1])
        unique_env_ids = env_ids[unique_positions]
        unique_ids_list = unique_env_ids.tolist()
        worker_ids = sorted(set(eid // self.num_envs_per_worker for eid in unique_ids_list))
        requested_ids = set(unique_ids_list)

        for worker_id in worker_ids:
            start_idx = worker_id * self.num_envs_per_worker
            end_idx = min(start_idx + self.num_envs_per_worker, self.n_envs)
            worker_env_ids = set(range(start_idx, end_idx))
            missing = sorted(worker_env_ids - requested_ids)
            if missing:
                # Each worker atomically steps all its envs together. Allowing
                # partial worker updates would race with the worker's own reset
                # logic and corrupt shared-memory observations.
                raise ValueError(
                    f"Partial worker step is not allowed: worker {worker_id} requires env_ids "
                    f"{list(range(start_idx, end_idx))}, missing {missing}"
                )

        # Write actions to shared memory tensor
        if torch.is_tensor(actions):
            action_values = actions.detach().cpu().to(self._action_buf.dtype)
        else:
            action_values = torch.as_tensor(actions, dtype=self._action_buf.dtype)
        self._action_buf[unique_env_ids] = action_values[unique_positions]

        for worker_id in worker_ids:
            if worker_id in self._pending_workers:
                raise RuntimeError(f"Worker {worker_id} already has a pending step")
            self.remotes[worker_id].send(("step", None))
            self._pending_workers.add(worker_id)

    def step_wait(self, min_ready=1, timeout=None):
        """Wait for signals from workers and read results from shared memory."""
        if min_ready < 1:
            raise ValueError("min_ready must be >= 1")
        if not self._pending_workers:
            empty_ids = np.empty((0,), dtype=np.int64)
            empty_obs = np.empty((0, *self._obs_shape), dtype=np.float32)
            empty_rewards = np.empty((0,), dtype=np.float32)
            empty_dones = np.empty((0,), dtype=bool)
            return empty_ids, empty_obs, empty_rewards, empty_dones, []

        target_ready_workers = min(
            max(1, math.ceil(min_ready / self.num_envs_per_worker)),
            len(self._pending_workers),
        )
        deadline = None if timeout is None else (time.monotonic() + timeout)

        ready_workers = []
        while len(ready_workers) < target_ready_workers and self._pending_workers:
            pending_remotes = [self.remotes[i] for i in self._pending_workers]
            wait_timeout = None
            if deadline is not None:
                wait_timeout = max(0.0, deadline - time.monotonic())
                if wait_timeout == 0.0:
                    break
            ready_remotes = wait(pending_remotes, timeout=wait_timeout)
            if not ready_remotes:
                break
            for remote in ready_remotes:
                worker_id = self._remote_to_id[remote]
                msg = remote.recv()
                self._pending_workers.discard(worker_id)
                if isinstance(msg, Exception):
                    raise RuntimeError(
                        f"Worker {worker_id} died with exception: {msg}"
                    ) from msg
                ready_workers.append(worker_id)

        if not ready_workers:
            empty_ids = np.empty((0,), dtype=np.int64)
            empty_obs = np.empty((0, *self._obs_shape), dtype=np.float32)
            empty_rewards = np.empty((0,), dtype=np.float32)
            empty_dones = np.empty((0,), dtype=bool)
            return empty_ids, empty_obs, empty_rewards, empty_dones, []

        ready_ids = np.concatenate([
            np.arange(w * self.num_envs_per_worker, (w + 1) * self.num_envs_per_worker, dtype=np.int64)
            for w in ready_workers
        ])

        # Zero-copy read from shared memory
        obs = self.get_obs(ready_ids, copy=False)
        rewards = self._reward_buf[ready_ids].numpy().copy()
        dones = self._done_buf[ready_ids].numpy().copy()

        # Reconstruct info dictionaries using the same column order as _write_info_to_buffer
        info_data = self._info_buf[ready_ids].numpy()
        infos = [
            {k: cast(info_data[idx, j]) for j, (k, cast) in enumerate(zip(_INFO_KEYS, _INFO_CASTS))}
            for idx in range(len(ready_ids))
        ]

        return (
            ready_ids,
            obs,
            rewards,
            dones,
            infos,
        )

    def step(self, actions):
        """Convenience: step_async + step_wait."""
        self.step_async(actions)
        env_ids, obs, rewards, dones, infos = self.step_wait(min_ready=self.n_envs)
        order = np.argsort(env_ids)
        return obs[order], rewards[order], dones[order], [infos[i] for i in order]

    def reset(self):
        self._pending_workers.clear()
        for remote in self.remotes:
            remote.send(("reset", None))
        for remote in self.remotes:
            remote.recv()
        return self.get_obs(copy=True)

    def close(self):
        if self.closed:
            return
        self._pending_workers.clear()
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

        for remote in self.remotes:
            try:
                remote.close()
            except Exception:
                pass

        self.closed = True

    def __del__(self):
        if not self.closed:
            self.close()


# ---------------------------------------------------------------------------
# RayAsyncVecEnv — multi-node async actors via Ray (for Anyscale)
# ---------------------------------------------------------------------------

# Module-level cache: @ray.remote is applied only once per process.
_RAY_WORKER_ACTOR_CLS = None


def _get_ray_worker_actor_cls():
    """Return the Ray remote actor class, defining it lazily on first call."""
    global _RAY_WORKER_ACTOR_CLS
    if _RAY_WORKER_ACTOR_CLS is None:
        import ray

        @ray.remote(num_cpus=1, runtime_env={"setup_commands": ["pip install . --no-build-isolation -q"]})
        class _RayEnvWorkerActor:
            """Ray remote actor — runs a DummyVecEnv, returns results via Ray object store."""

            def __init__(self, env_fn_wrappers, decorrelation_steps):
                import os
                import torch

                os.environ["OMP_NUM_THREADS"] = "1"
                torch.set_num_threads(1)
                self._envs = DummyVecEnv(env_fn_wrappers)
                self._envs.reset()
                for _ in range(decorrelation_steps):
                    random_actions = np.asarray(
                        [self._envs.action_space.sample() for _ in range(self._envs.n_envs)]
                    )
                    self._envs.step(random_actions)

            def step(self, actions):
                try:
                    return self._envs.step(actions)
                except Exception as exc:
                    import traceback
                    obs = self._envs.reset()
                    n = self._envs.n_envs
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.exception("[RayEnvWorkerActor] step() error (reset performed): %s", exc)
                    return (
                        obs,
                        np.zeros(n, dtype=np.float32),
                        np.ones(n, dtype=bool),
                        [{} for _ in range(n)],
                    )

            def reset(self):
                try:
                    return self._envs.reset()
                except Exception as exc:
                    import traceback
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.exception("[RayEnvWorkerActor] reset() error: %s", exc)
                    raise

            def get_spaces(self):
                return self._envs.observation_space, self._envs.action_space

        _RAY_WORKER_ACTOR_CLS = _RayEnvWorkerActor
    return _RAY_WORKER_ACTOR_CLS


class RayAsyncVecEnv:
    """Async vectorized env using Ray remote actors (multi-node capable).

    Drop-in replacement for AsyncSubprocVecEnv when vec_env_mode='ray'.
    Exposes the same step_async / step_wait interface. Workers run as
    @ray.remote actors scheduled by Ray across the cluster. Observations
    travel through Ray's object store — no shared memory needed.
    """

    def __init__(
        self,
        env_fns,
        num_envs_per_worker=1,
        decorrelation_base=20,
        observation_space=None,
        action_space=None,
    ):
        import ray as _ray

        self._ray = _ray
        if not _ray.is_initialized():
            raise RuntimeError(
                "ray.init() must be called before constructing RayAsyncVecEnv. "
                "Set vec_env_mode=ray in config — main() calls ray.init() automatically."
            )

        self.n_envs = len(env_fns)
        if self.n_envs == 0:
            raise ValueError("env_fns must contain at least one env factory")
        if self.n_envs % num_envs_per_worker != 0:
            raise ValueError(
                f"Total envs {self.n_envs} must be divisible by "
                f"num_envs_per_worker {num_envs_per_worker}"
            )
        self.num_envs_per_worker = num_envs_per_worker
        self.num_workers = self.n_envs // num_envs_per_worker
        self.closed = False
        self._last_obs = None  # updated by reset() and step_wait(); used by get_obs()

        if (observation_space is None) != (action_space is None):
            raise ValueError(
                "observation_space and action_space must be provided together"
            )

        logger.info(
            "[RayAsyncVecEnv] Initializing %s Ray actors (each with %s envs)...",
            self.num_workers,
            num_envs_per_worker,
        )
        actor_cls = _get_ray_worker_actor_cls()
        self._workers = []
        for i in range(self.num_workers):
            decorr_steps = i * decorrelation_base
            worker_fns = [
                CloudpickleWrapper(fn)
                for fn in env_fns[i * num_envs_per_worker : (i + 1) * num_envs_per_worker]
            ]
            self._workers.append(actor_cls.remote(worker_fns, decorr_steps))

        if observation_space is None:
            self.observation_space, self.action_space = self._ray.get(
                self._workers[0].get_spaces.remote()
            )
        else:
            self.observation_space = observation_space
            self.action_space = action_space
        self._obs_shape = self.observation_space.shape

        self._pending: dict = {}  # worker_id -> ObjectRef
        logger.info("[RayAsyncVecEnv] %s Ray actors submitted.", self.num_workers)

    # ------------------------------------------------------------------
    # Core interface — mirrors AsyncSubprocVecEnv
    # ------------------------------------------------------------------

    def _empty_result(self):
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, *self._obs_shape), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=bool),
            [],
        )

    def step_async(self, actions, env_ids=None):
        """Non-blocking: dispatch step.remote() to relevant workers."""
        if torch.is_tensor(actions):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = np.asarray(actions)

        if env_ids is None:
            worker_ids = range(self.num_workers)
            for wid in worker_ids:
                start = wid * self.num_envs_per_worker
                end = (wid + 1) * self.num_envs_per_worker
                self._pending[wid] = self._workers[wid].step.remote(actions_np[start:end])
            return

        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if env_ids.size == 0:
            return

        unique_positions = np.sort(np.unique(env_ids, return_index=True)[1])
        unique_env_ids = env_ids[unique_positions]
        unique_actions = actions_np[unique_positions]
        worker_ids = sorted(set(int(eid) // self.num_envs_per_worker for eid in unique_env_ids))
        action_by_env = {int(eid): action for eid, action in zip(unique_env_ids, unique_actions)}

        for wid in worker_ids:
            start = wid * self.num_envs_per_worker
            end = min(start + self.num_envs_per_worker, self.n_envs)
            worker_env_ids = list(range(start, end))
            missing = [eid for eid in worker_env_ids if eid not in action_by_env]
            if missing:
                raise ValueError(
                    f"Partial worker step is not allowed: worker {wid} requires env_ids "
                    f"{worker_env_ids}, missing {missing}"
                )
            worker_actions = np.asarray([action_by_env[eid] for eid in worker_env_ids])
            self._pending[wid] = self._workers[wid].step.remote(worker_actions)

    def step_wait(self, min_ready=1, timeout=None):
        """Wait for at least min_ready env results and return a batch.

        Returns (ready_ids, obs, rewards, dones, infos) — same signature as
        AsyncSubprocVecEnv.step_wait().
        """
        if not self._pending:
            return self._empty_result()

        n_workers_needed = min(
            max(1, math.ceil(min_ready / self.num_envs_per_worker)),
            len(self._pending),
        )
        wids = list(self._pending.keys())
        futures = [self._pending[wid] for wid in wids]
        # ray.wait() returns the same ObjectRef objects from the input list, so id() is stable.
        id_to_wid = {id(f): wid for f, wid in zip(futures, wids)}

        ready_futures, _ = self._ray.wait(futures, num_returns=n_workers_needed, timeout=timeout)
        if not ready_futures:
            return self._empty_result()

        ready_wids = sorted(id_to_wid[id(f)] for f in ready_futures)

        all_obs, all_rew, all_done, all_info, ready_env_ids = [], [], [], [], []
        for wid in ready_wids:
            obs, rewards, dones, infos = self._ray.get(self._pending.pop(wid))
            start = wid * self.num_envs_per_worker
            all_obs.append(np.asarray(obs, dtype=np.float32))
            all_rew.append(np.asarray(rewards, dtype=np.float32))
            all_done.append(np.asarray(dones, dtype=bool))
            all_info.extend(infos)
            ready_env_ids.extend(range(start, start + self.num_envs_per_worker))

        obs_arr = np.concatenate(all_obs, axis=0)
        # Keep a local obs cache so get_obs() doesn't need to contact workers.
        if self._last_obs is None:
            self._last_obs = np.zeros((self.n_envs, *self._obs_shape), dtype=np.float32)
        self._last_obs[ready_env_ids] = obs_arr

        return (
            np.array(ready_env_ids, dtype=np.int64),
            obs_arr,
            np.concatenate(all_rew, axis=0),
            np.concatenate(all_done, axis=0),
            all_info,
        )

    def step(self, actions):
        """Synchronous step: step_async + step_wait for all envs."""
        self.step_async(actions)
        env_ids, obs, rew, done, info = self.step_wait(min_ready=self.n_envs)
        order = np.argsort(env_ids)
        return obs[order], rew[order], done[order], [info[i] for i in order]

    def reset(self):
        self._pending.clear()
        obs_list = self._ray.get([w.reset.remote() for w in self._workers])
        self._last_obs = np.concatenate(
            [np.asarray(o, dtype=np.float32) for o in obs_list], axis=0
        )
        return self._last_obs.copy()

    def get_obs(self, env_ids=None, copy=True):
        """Return last-seen observations (updated by reset() and step_wait())."""
        if self._last_obs is None:
            raise RuntimeError("get_obs() called before reset() — call reset() first.")
        obs = self._last_obs if env_ids is None else self._last_obs[np.asarray(env_ids, dtype=np.int64)]
        return obs.copy() if copy else obs

    def close(self):
        if self.closed:
            return
        self._pending.clear()
        for w in self._workers:
            try:
                self._ray.kill(w, no_restart=True)
            except Exception:
                pass
        self.closed = True

    def __del__(self):
        if not self.closed:
            self.close()
