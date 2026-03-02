from collections import deque
from typing import Deque, Dict, Tuple

import numpy as np


class ReplayBuffer:
    """Numpy replay buffer with optional n-step return aggregation."""

    def __init__(
        self,
        obs_dim: int,
        size: int,
        batch_size: int = 32,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        if n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {n_step}")

        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size,), dtype=np.int64)
        self.rews_buf = np.zeros((size,), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)
        self.max_size = int(size)
        self.batch_size = int(batch_size)
        self.ptr = 0
        self.size = 0

        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.n_step_buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=self.n_step
        )

    def _insert_transition(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def store(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        transition = (
            np.asarray(obs, dtype=np.float32),
            int(act),
            float(rew),
            np.asarray(next_obs, dtype=np.float32),
            bool(done),
        )
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return ()

        n_rew, n_next_obs, n_done = self._get_n_step_info(self.n_step_buffer, self.gamma)
        first_obs, first_act = self.n_step_buffer[0][:2]
        self._insert_transition(first_obs, first_act, n_rew, n_next_obs, n_done)
        return self.n_step_buffer[0]

    def store_batch(self, states, actions, rewards, next_states, dones) -> None:
        for state, action, reward, next_state, done in zip(
            states, actions, rewards, next_states, dones
        ):
            self.store(state, action, reward, next_state, done)

    def push(self, obs, act, rew, next_obs, done):
        return self.store(obs, act, rew, next_obs, done)

    def sample_batch(self, batch_size: int = None) -> Dict[str, np.ndarray]:
        bs = int(batch_size) if batch_size is not None else self.batch_size
        if self.size < bs:
            raise ValueError(
                f"Cannot sample batch of size {bs} from buffer with {self.size} elements"
            )

        indices = np.random.choice(self.size, size=bs, replace=False)
        return {
            "obs": self.obs_buf[indices],
            "next_obs": self.next_obs_buf[indices],
            "acts": self.acts_buf[indices],
            "rews": self.rews_buf[indices],
            "done": self.done_buf[indices],
            "indices": indices,
        }

    def sample(self, batch_size: int = None):
        batch = self.sample_batch(batch_size=batch_size)
        return (
            batch["obs"],
            batch["acts"],
            batch["rews"],
            batch["next_obs"],
            batch["done"],
        )

    def sample_batch_from_idxs(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "obs": self.obs_buf[indices],
            "next_obs": self.next_obs_buf[indices],
            "acts": self.acts_buf[indices],
            "rews": self.rews_buf[indices],
            "done": self.done_buf[indices],
        }

    def _get_n_step_info(
        self, n_step_buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]], gamma: float
    ) -> Tuple[float, np.ndarray, bool]:
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]
            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return float(rew), np.asarray(next_obs, dtype=np.float32), bool(done)

    def __len__(self) -> int:
        return self.size
