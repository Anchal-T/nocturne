"""Vectorized environment wrappers for parallel DRL training.

Provides three levels of parallelism, closely modeled on Sample Factory /
PPO ``env_wrappers.py``:

* ``DummyVecEnv``         – sequential, in-process (debugging)
* ``SubprocVecEnv``       – one subprocess per env, synchronous stepping
* ``AsyncSubprocVecEnv``  – fully async actors with experience decorrelation,
                            CPU-core pinning, and shared-memory observation
                            buffers (closest to Sample Factory)
"""

import os
import ctypes
import multiprocessing as mp
from multiprocessing import Process, Pipe
from multiprocessing.sharedctypes import RawArray

import numpy as np


# ---------------------------------------------------------------------------
# Serialization helper (identical to PPO env_wrappers.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------
def _worker(remote, parent_remote, env_fn_wrapper):
    """Basic synchronous subprocess worker (for SubprocVecEnv)."""
    parent_remote.close()
    env = env_fn_wrapper()

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            remote.send((obs, reward, done, info))
        elif cmd == 'reset':
            remote.send(env.reset())
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


def _async_worker(
    worker_id,
    remote,
    parent_remote,
    env_fn_wrapper,
    obs_shm_name,
    obs_shape,
    obs_dtype,
    decorrelation_steps,
    cpu_cores,
):
    """Fully-async worker with shared memory, decorrelation, and CPU affinity.

    Each worker runs its own env independently. Observations are written to
    shared memory so the main process can read them without pipe serialization.
    The pipe is still used for (reward, done, info) since those are small.
    """
    parent_remote.close()

    # Pin to CPU cores (like Sample Factory assigning cores per worker)
    if cpu_cores:
        try:
            os.sched_setaffinity(0, cpu_cores)
            print(f"[Actor {worker_id}] pid {os.getpid()} pinned to cores {sorted(cpu_cores)}")
        except (AttributeError, OSError):
            pass  # OS doesn't support affinity (e.g. macOS)

    os.environ['OMP_NUM_THREADS'] = '1'

    env = env_fn_wrapper()

    # Attach to shared memory buffer for this worker's observation
    obs_size = int(np.prod(obs_shape))
    shm_arr = np.frombuffer(
        mp.shared_memory.SharedMemory(name=obs_shm_name).buf,
        dtype=obs_dtype,
    ).reshape(-1)
    worker_slice = slice(worker_id * obs_size, (worker_id + 1) * obs_size)

    def _write_obs(obs):
        flat = obs.astype(obs_dtype).ravel()
        shm_arr[worker_slice] = flat

    # Initial reset + decorrelation
    obs = env.reset()
    for _ in range(decorrelation_steps):
        action = env.action_space.sample()
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    _write_obs(obs)

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            _write_obs(obs)
            remote.send((reward, done, info))
        elif cmd == 'reset':
            obs = env.reset()
            _write_obs(obs)
            remote.send(True)
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


# ---------------------------------------------------------------------------
# DummyVecEnv — sequential, single-process (for debugging)
# ---------------------------------------------------------------------------
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
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
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
        return np.stack([env.reset() for env in self.envs])

    def close(self):
        for env in self.envs:
            env.close()


# ---------------------------------------------------------------------------
# SubprocVecEnv — one subprocess per env, synchronous lock-step
# ---------------------------------------------------------------------------
class SubprocVecEnv:
    """Vectorized env with one subprocess per env and pipe-based IPC.

    All envs are stepped together in lock-step. Modeled directly on the PPO
    ``SubprocVecEnv`` from ``algos/ppo/env_wrappers.py``.
    """

    def __init__(self, env_fns):
        self.n_envs = len(env_fns)
        self.closed = False

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_envs)])

        self.processes = []
        for wr, r, fn in zip(self.work_remotes, self.remotes, env_fns):
            p = Process(target=_worker, args=(wr, r, CloudpickleWrapper(fn)), daemon=True)
            p.start()
            self.processes.append(p)
        for wr in self.work_remotes:
            wr.close()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), list(infos)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(('close', None))
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
    * **Shared memory observations** — workers write obs directly to a
      shared numpy buffer, avoiding pipe serialization of large arrays.
    * **Experience decorrelation** — each worker executes a staggered
      number of random steps during init, just like SF's decorrelation
      phase (``Worker N, sleep for N sec to decorrelate``).
    * **CPU core affinity** — workers are pinned to specific cores to
      prevent thrashing, matching SF's ``Worker N uses CPU cores [N]``.
    * **Async stepping** — ``step_async`` / ``step_wait`` split so the
      main thread can overlap inference with env execution.
    """

    def __init__(self, env_fns, decorrelation_base=20):
        self.n_envs = len(env_fns)
        self.closed = False

        # Probe observation space from a temporary env
        tmp_env = env_fns[0]()
        self.observation_space = tmp_env.observation_space
        self.action_space = tmp_env.action_space
        obs_shape = self.observation_space.shape
        obs_dtype = np.float32
        obs_size = int(np.prod(obs_shape))
        tmp_env.close()

        # Allocate shared memory for all worker observations
        total_floats = self.n_envs * obs_size
        self._shm = mp.shared_memory.SharedMemory(create=True, size=total_floats * 4)
        self._obs_buf = np.frombuffer(self._shm.buf, dtype=obs_dtype).reshape(self.n_envs, *obs_shape)

        # Assign CPU cores round-robin
        try:
            available_cores = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available_cores = list(range(os.cpu_count() or 1))

        # Spawn async workers
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_envs)])
        self.processes = []

        print(f"[AsyncSubprocVecEnv] Initializing {self.n_envs} workers...")

        for i, (wr, r, fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns)):
            core = [available_cores[i % len(available_cores)]]
            decorr_steps = (i * decorrelation_base) % (decorrelation_base * self.n_envs)
            p = Process(
                target=_async_worker,
                args=(i, wr, r, CloudpickleWrapper(fn), self._shm.name,
                      obs_shape, obs_dtype, decorr_steps, core),
                daemon=True,
            )
            p.start()
            self.processes.append(p)

        for wr in self.work_remotes:
            wr.close()

        # Wait for all workers to finish init (they wrote their obs to shared mem)
        for i, remote in enumerate(self.remotes):
            remote.send(('get_spaces', None))
            remote.recv()
            print(f"  Worker {i} initialized (pid {self.processes[i].pid})")

        print(f"[AsyncSubprocVecEnv] All {self.n_envs} workers ready.")

    def get_obs(self):
        """Read current observations from shared memory (zero-copy)."""
        return self._obs_buf.copy()

    def step_async(self, actions):
        """Send actions to all workers without waiting (non-blocking)."""
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

    def step_wait(self):
        """Wait for all workers and return results. Observations come from shared memory."""
        results = [remote.recv() for remote in self.remotes]
        rewards, dones, infos = zip(*results)
        obs = self.get_obs()
        return obs, np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), list(infos)

    def step(self, actions):
        """Convenience: step_async + step_wait."""
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        for remote in self.remotes:
            remote.recv()
        return self.get_obs()

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except BrokenPipeError:
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass
        self.closed = True

    def __del__(self):
        if not self.closed:
            self.close()
