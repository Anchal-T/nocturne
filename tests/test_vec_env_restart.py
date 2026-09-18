"""SubprocVecEnv must survive a worker process dying mid-step."""
import os

import numpy as np
from gymnasium.spaces import Box

from examples.drl_collision_avoidance.vec_env import SubprocVecEnv


class _CrashEnv:
    observation_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def __init__(self):
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return np.zeros(4, dtype=np.float32), {"ego_x": 0.0}

    def step(self, action):
        self.steps += 1
        if self.steps >= 4:
            os._exit(1)
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {"ego_x": 0.0}

    def close(self):
        pass


def test_subproc_vec_env_restarts_dead_worker():
    vec = SubprocVecEnv([_CrashEnv, _CrashEnv])
    try:
        vec.reset()
        actions = np.zeros((2, 2), dtype=np.float32)
        saw_done = False
        for _ in range(8):
            _obs, _rew, dones, _infos = vec.step(actions)
            saw_done = saw_done or bool(np.any(dones))
        assert saw_done
        obs = vec.reset()
        assert obs.shape == (2, 4)
    finally:
        vec.close()
