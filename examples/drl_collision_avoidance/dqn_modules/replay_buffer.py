from collections import deque
from typing import Deque, Dict, Tuple

import numpy as np

from .sum_tree import SumTree


class ReplayBuffer:
    """Numpy replay buffer with Prioritized Experience Replay and optional n-step returns."""

    def __init__(
        self,
        obs_dim: int,
        size: int,
        batch_size: int = 32,
        n_step: int = 1,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6,
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
        self.sum_tree = SumTree(capacity=size)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = max(float(epsilon), 1e-8)
        self.max_priority = 1.0

        MAX_PRIORITY_CAP = 1e6
        self._max_priority_cap = MAX_PRIORITY_CAP

    def _insert_transition(
        self, obs: np.ndarray, act: int, rew: float, next_obs: np.ndarray, done: bool,
    ) -> None:
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def store(
        self, obs: np.ndarray, act: int, rew: float, next_obs: np.ndarray, done: bool,
    ) -> None:
        transition = (
            np.asarray(obs, dtype=np.float32),
            int(act),
            float(rew),
            np.asarray(next_obs, dtype=np.float32),
            bool(done),
        )
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return

        n_rew, n_next_obs, n_done = self._get_n_step_info(self.n_step_buffer, self.gamma)
        first_obs, first_act = self.n_step_buffer[0][:2]
        self._insert_transition(first_obs, first_act, n_rew, n_next_obs, n_done)
        self.sum_tree.add(self.max_priority)

    def store_batch(self, states, actions, rewards, next_states, dones) -> None:
        for state, action, reward, next_state, done in zip(
            states, actions, rewards, next_states, dones
        ):
            self.store(state, action, reward, next_state, done)

    def sample_batch(self, batch_size: int = None) -> Dict[str, np.ndarray]:
        bs = int(batch_size) if batch_size is not None else self.batch_size
        if self.size < bs:
            raise ValueError(
                f"Cannot sample batch of size {bs} from buffer with {self.size} elements"
            )

        total_priority = float(self.sum_tree.total())
        if not np.isfinite(total_priority) or total_priority <= 0.0:
            indices = np.random.choice(self.size, size=bs, replace=False)
            batch = self._get_batch_from_indices(indices)
            batch["indices"] = np.asarray(indices, dtype=np.int64)
            batch["weights"] = np.ones(bs, dtype=np.float32)
            return batch

        segment = total_priority / bs
        data_indices = []
        weights = np.zeros(bs, dtype=np.float32)
        s_max = np.nextafter(total_priority, 0.0)
        min_prob = self.epsilon / max(total_priority, self.epsilon)

        for i in range(bs):
            s = np.random.uniform(segment * i, segment * (i + 1))
            s = min(max(float(s), 0.0), s_max)
            _, priority, data_idx = self.sum_tree.get(s)
            if data_idx < 0 or data_idx >= self.size:
                data_idx = np.random.randint(0, self.size)
                priority = self.epsilon
            data_indices.append(int(data_idx))

            priority = float(priority)
            if not np.isfinite(priority) or priority <= 0.0:
                priority = self.epsilon

            prob = max(priority / total_priority, min_prob)
            w = (self.size * prob) ** (-self.beta)
            weights[i] = w if np.isfinite(w) else 1.0

        weights_max = float(np.max(weights))
        if np.isfinite(weights_max) and weights_max > 0.0:
            weights /= weights_max
        else:
            weights.fill(1.0)
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)

        indices = np.array(data_indices, dtype=np.int64)
        batch = self._get_batch_from_indices(indices)
        batch["indices"] = indices
        batch["weights"] = weights
        return batch

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        if len(indices) == 0:
            return

        priorities = (np.abs(np.asarray(td_errors, dtype=np.float32)) + self.epsilon) ** self.alpha
        priorities = np.clip(
            np.nan_to_num(priorities, nan=self.max_priority, posinf=self._max_priority_cap, neginf=self.epsilon),
            self.epsilon,
            self._max_priority_cap,
        )

        max_prio = float(np.max(priorities))
        if np.isfinite(max_prio) and max_prio > 0.0:
            self.max_priority = min(max(self.max_priority, max_prio), self._max_priority_cap)

        for idx, prio in zip(indices, priorities):
            if idx < 0 or idx >= self.sum_tree.capacity:
                continue
            tree_idx = idx + self.sum_tree.capacity - 1
            self.sum_tree.update(tree_idx, float(prio))

    def update_beta(self, env_steps: int):
        if self.beta_frames <= 0:
            return
        fraction = min(1.0, env_steps / self.beta_frames)
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)

    def _get_batch_from_indices(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "obs": self.obs_buf[indices],
            "next_obs": self.next_obs_buf[indices],
            "acts": self.acts_buf[indices],
            "rews": self.rews_buf[indices],
            "done": self.done_buf[indices],
        }

    @staticmethod
    def _get_n_step_info(
        n_step_buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]],
        gamma: float,
    ) -> Tuple[float, np.ndarray, bool]:
        rew, next_obs, done = n_step_buffer[-1][-3:]
        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]
            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)
        return float(rew), np.asarray(next_obs, dtype=np.float32), bool(done)

    def __len__(self) -> int:
        return self.size
