from collections import deque
from typing import Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .sum_tree import SumTree


class ReplayBuffer:
    """Prioritized replay with n-step returns.

    When ``device`` is a CUDA device, observations live once on the GPU in
    ``obs_buf`` and ``next_obs`` is derived by index (``next_idx``).  Terminal
    transitions point at themselves — the TD target zeroes ``next_q`` when
    ``done``, so the bootstrap observation is unused.  Non-terminal links are
    filled in lazily when the same env later stores the observation that is
    this transition's n-step next state.
    """

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
        num_envs: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        obs_dtype: str = "float32",
    ):
        if n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {n_step}")

        self.max_size = int(size)
        self.batch_size = int(batch_size)
        self.obs_dim = int(obs_dim)
        self.ptr = 0
        self.size = 0

        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.num_envs = max(1, int(num_envs))
        self._n_step_buffers: Dict[int, Deque[Tuple]] = {
            i: deque(maxlen=self.n_step) for i in range(self.num_envs)
        }

        self.sum_tree = SumTree(capacity=size)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = max(float(epsilon), 1e-8)
        self.max_priority = 1.0
        self._max_priority_cap = 1e6

        # GPU store-once path when device is CUDA; otherwise numpy host buffers.
        self._use_gpu = False
        self.device: Optional[torch.device] = None
        if device is not None:
            dev = torch.device(device)
            if dev.type == "cuda":
                self._use_gpu = True
                self.device = dev

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        self._torch_obs_dtype = dtype_map.get(str(obs_dtype).lower(), torch.float32)

        if self._use_gpu:
            self.obs_buf = torch.zeros(
                (self.max_size, obs_dim),
                dtype=self._torch_obs_dtype,
                device=self.device,
            )
            # Host-side next_idx avoids a D2H sync on every insert.
            # next_idx[i] >= 0 → obs_buf[next_idx[i]]; -1 → not yet linked.
            self.next_idx = np.full((self.max_size,), -1, dtype=np.int64)
            self.acts_buf = torch.zeros(
                (self.max_size,), dtype=torch.int64, device=self.device
            )
            self.rews_buf = torch.zeros(
                (self.max_size,), dtype=torch.float32, device=self.device
            )
            self.done_buf = torch.zeros(
                (self.max_size,), dtype=torch.float32, device=self.device
            )
            # Per-env chronological transition indices for delayed next_obs linking.
            self._env_history: Dict[int, List[int]] = {
                i: [] for i in range(self.num_envs)
            }
            self._history_maxlen = self.n_step + 1
        else:
            self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
            self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
            self.acts_buf = np.zeros((size,), dtype=np.int64)
            self.rews_buf = np.zeros((size,), dtype=np.float32)
            self.done_buf = np.zeros((size,), dtype=np.float32)
            self.next_idx = None
            self._env_history = None

    def _insert_transition(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        env_id: int = 0,
    ) -> None:
        if self._use_gpu:
            self._insert_transition_gpu(obs, act, rew, next_obs, done, env_id)
        else:
            self.obs_buf[self.ptr] = obs
            self.next_obs_buf[self.ptr] = next_obs
            self.acts_buf[self.ptr] = act
            self.rews_buf[self.ptr] = rew
            self.done_buf[self.ptr] = float(done)
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def _insert_transition_gpu(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        env_id: int,
    ) -> None:
        ptr = self.ptr
        obs_t = torch.as_tensor(obs, dtype=self._torch_obs_dtype, device=self.device)
        self.obs_buf[ptr].copy_(obs_t, non_blocking=True)
        self.acts_buf[ptr] = int(act)
        self.rews_buf[ptr] = float(rew)
        self.done_buf[ptr] = float(done)

        if done:
            # TD target ignores next_q when done; self-point is fine.
            self.next_idx[ptr] = ptr
        else:
            self.next_idx[ptr] = -1  # pending link

        # Delayed link: this obs is the n-step next_obs of the transition
        # stored n_step inserts ago for the same env.
        hist = self._env_history[env_id]
        hist.append(ptr)
        if len(hist) > self._history_maxlen:
            hist.pop(0)
        if len(hist) > self.n_step:
            old = hist[-(self.n_step + 1)]
            if self.next_idx[old] == -1:
                self.next_idx[old] = ptr

        self.ptr = (ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _flush_n_step_buffer(self, buf: Deque, env_id: int = 0) -> None:
        """Drain remaining transitions at episode end by shrinking the n-step window."""
        while len(buf) > 0:
            n_rew, n_next_obs, n_done = self._get_n_step_info(buf, self.gamma)
            first_obs, first_act = buf[0][:2]
            self._insert_transition(
                first_obs, first_act, n_rew, n_next_obs, n_done, env_id=env_id
            )
            self.sum_tree.add(self.max_priority)
            buf.popleft()

    def store(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        env_id: int = 0,
    ) -> None:
        env_id = int(env_id) % self.num_envs
        buf = self._n_step_buffers[env_id]

        transition = (
            np.asarray(obs, dtype=np.float32),
            int(act),
            float(rew),
            np.asarray(next_obs, dtype=np.float32),
            bool(done),
        )
        buf.append(transition)

        if done:
            self._flush_n_step_buffer(buf, env_id=env_id)
            if self._use_gpu:
                self._env_history[env_id].clear()
            return

        if len(buf) < self.n_step:
            return

        n_rew, n_next_obs, n_done = self._get_n_step_info(buf, self.gamma)
        first_obs, first_act = buf[0][:2]
        self._insert_transition(
            first_obs, first_act, n_rew, n_next_obs, n_done, env_id=env_id
        )
        self.sum_tree.add(self.max_priority)

    def store_batch(
        self, states, actions, rewards, next_states, dones, env_ids: Optional[np.ndarray] = None
    ) -> None:
        for i, (state, action, reward, next_state, done) in enumerate(
            zip(states, actions, rewards, next_states, dones)
        ):
            env_id = int(env_ids[i]) if env_ids is not None else i % self.num_envs
            self.store(state, action, reward, next_state, done, env_id=env_id)

    def sample_batch(self, batch_size: int = None):
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
            if self._use_gpu:
                batch["weights"] = torch.ones(
                    bs, dtype=torch.float32, device=self.device
                )
            else:
                batch["weights"] = np.ones(bs, dtype=np.float32)
            return batch

        _tree_indices, priorities, data_indices = self.sum_tree.sample_batch(bs)

        valid = (data_indices >= 0) & (data_indices < self.size)
        data_indices = np.where(
            valid, data_indices, np.random.randint(0, self.size, size=bs)
        )
        priorities = np.where(valid, priorities, self.epsilon)

        if self._use_gpu:
            data_indices = self._resolve_linked_indices(data_indices)

        probs = np.maximum(
            priorities / max(total_priority, self.epsilon),
            self.epsilon / max(total_priority, self.epsilon),
        )
        weights = (self.size * probs) ** (-self.beta)
        weights = np.where(np.isfinite(weights), weights, 1.0)
        weights_max = float(weights.max())
        if np.isfinite(weights_max) and weights_max > 0.0:
            weights /= weights_max
        else:
            weights.fill(1.0)
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)

        indices = data_indices.astype(np.int64)
        batch = self._get_batch_from_indices(indices)
        batch["indices"] = indices
        if self._use_gpu:
            batch["weights"] = torch.as_tensor(
                weights, dtype=torch.float32, device=self.device
            )
        else:
            batch["weights"] = weights.astype(np.float32)
        return batch

    def _resolve_linked_indices(self, data_indices: np.ndarray) -> np.ndarray:
        """Replace pending (unlinked) indices with linked ones via resampling."""
        indices = data_indices.copy()
        pending = self.next_idx[indices] < 0
        if not pending.any():
            return indices
        for _ in range(8):
            n_pending = int(pending.sum())
            if n_pending == 0:
                break
            indices[pending] = np.random.randint(0, self.size, size=n_pending)
            pending = self.next_idx[indices] < 0
        if pending.any():
            # Last resort while buffer is young: self-point.
            self.next_idx[indices[pending]] = indices[pending]
        return indices

    def update_priorities(self, indices, td_errors):
        if len(indices) == 0:
            return

        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        if torch.is_tensor(td_errors):
            td_errors = td_errors.detach().cpu().numpy()

        priorities = (
            np.abs(np.asarray(td_errors, dtype=np.float32)) + self.epsilon
        ) ** self.alpha
        priorities = np.clip(
            np.nan_to_num(
                priorities,
                nan=self.max_priority,
                posinf=self._max_priority_cap,
                neginf=self.epsilon,
            ),
            self.epsilon,
            self._max_priority_cap,
        )

        max_prio = float(np.max(priorities))
        if np.isfinite(max_prio) and max_prio > 0.0:
            self.max_priority = min(
                max(self.max_priority, max_prio), self._max_priority_cap
            )

        tree_indices = indices + self.sum_tree.capacity - 1
        valid = (indices >= 0) & (indices < self.sum_tree.capacity)
        if valid.any():
            self.sum_tree.update_batch(
                tree_indices[valid],
                priorities[valid],
            )

    def update_beta(self, env_steps: int):
        if self.beta_frames <= 0:
            return
        fraction = min(1.0, env_steps / self.beta_frames)
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)

    def _get_batch_from_indices(self, indices: np.ndarray):
        if self._use_gpu:
            idx = torch.as_tensor(indices, dtype=torch.int64, device=self.device)
            nxt_np = np.maximum(self.next_idx[indices], 0)
            nxt = torch.as_tensor(nxt_np, dtype=torch.int64, device=self.device)
            return {
                "obs": self.obs_buf[idx],
                "next_obs": self.obs_buf[nxt],
                "acts": self.acts_buf[idx],
                "rews": self.rews_buf[idx],
                "done": self.done_buf[idx],
            }
        return {
            "obs": self.obs_buf[indices],
            "next_obs": self.next_obs_buf[indices],
            "acts": self.acts_buf[indices],
            "rews": self.rews_buf[indices],
            "done": self.done_buf[indices],
        }

    @staticmethod
    def _get_n_step_info(
        n_step_buffer: Deque[Tuple],
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
