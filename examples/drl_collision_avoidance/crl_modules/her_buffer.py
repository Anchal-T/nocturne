"""
Hindsight Experience Replay (HER) buffer for CRL.

Stores complete driving episodes. At sample time, for each selected
(state, action) pair at timestep t, draws a future timestep t' > t
(using geometric-γ weighting) and uses the ego vehicle's position at t'
as the HER-relabelled goal for that training example.

Episode storage layout per step:
    state    : (state_dim,) float32  - pure state, no goal features
    action   : (action_dim,) float32 - tanh-squashed action in [-1, 1]
    ego_info : (4,) float32          - [ego_x, ego_y, cos_heading, sin_heading]

The goal for a sample (t → t') is computed in the ego frame at time t:
    dx        = ego_pos_x[t'] - ego_pos_x[t]
    dy        = ego_pos_y[t'] - ego_pos_y[t]
    goal_long = (dx * cos_h[t] + dy * sin_h[t]) / DIST_NORM
    goal_lat  = (-dx * sin_h[t] + dy * cos_h[t]) / DIST_NORM
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

DIST_NORM = 100.0  # normalisation constant matching CollisionAvoidanceEnv


class _Episode:
    """Accumulates transitions for one episode, then finalises to numpy arrays."""

    __slots__ = ("states", "actions", "ego_infos", "_finalised")

    def __init__(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.ego_infos: List[np.ndarray] = []
        self._finalised: bool = False

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        ego_info: np.ndarray,
    ) -> None:
        """Append a single timestep to this episode.

        Args:
            state:    Pure state vector of shape (state_dim,).
            action:   Tanh-squashed action vector of shape (action_dim,).
            ego_info: Ego vehicle info [ego_x, ego_y, cos_heading, sin_heading].
        """
        assert not self._finalised, "Cannot add to a finalised episode."
        self.states.append(np.array(state, dtype=np.float32, copy=True))
        self.actions.append(np.array(action, dtype=np.float32, copy=True))
        # Must copy: callers often pass a row view into a persistent ego buffer.
        # Without a copy, every timestep aliases the same memory and HER goals
        # collapse to 0 (InfoNCE stuck at ln(B)).
        self.ego_infos.append(np.array(ego_info, dtype=np.float32, copy=True))

    def finalise(self) -> None:
        """Convert lists to contiguous arrays for efficient indexing."""
        self.states = np.stack(self.states, axis=0)  # (T, state_dim)
        self.actions = np.stack(self.actions, axis=0)  # (T, action_dim)
        self.ego_infos = np.stack(self.ego_infos, axis=0)  # (T, 4)
        self._finalised = True

    def __len__(self) -> int:
        return len(self.states)


class HERReplayBuffer:
    """Ring-buffer of complete episodes with HER goal relabelling.

    Episodes are stored in full.  At sample time each selected (s, a) pair
    is paired with a HER goal derived from a *future* state drawn from the
    same episode using a geometric-γ probability distribution, so that
    nearer future states are preferred (consistent with temporal discounting).

    Args:
        state_dim:     Pure state dimension (no goal), = 605 for nocturne.
        action_dim:    Continuous action dimension, = 2 for nocturne.
        goal_dim:      Goal encoding dimension, = 2 for nocturne.
        max_episodes:  Maximum episodes to retain (oldest are dropped).
        gamma:         Discount factor used for geometric future sampling.
        num_envs:      Number of parallel environments (one in-progress
                       episode is maintained per env).
        min_ep_len:    Minimum episode length to store (shorter discarded).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        max_episodes: int = 50_000,
        gamma: float = 0.99,
        num_envs: int = 1,
        min_ep_len: int = 2,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.goal_dim = int(goal_dim)
        self.gamma = float(gamma)
        self.min_ep_len = int(min_ep_len)

        # List ring buffer: O(1) random access. A deque would make every
        # sample O(buffer size) because deque[i] is linear in i.
        self._max_episodes = int(max_episodes)
        self._episodes: List[_Episode] = []
        self._write = 0
        # Running total of stored transition steps (approximate after wrap).
        self._total_steps: int = 0

        # One in-progress episode per environment.
        self._num_envs = max(1, int(num_envs))
        self._active: List[_Episode] = [_Episode() for _ in range(self._num_envs)]

    # ------------------------------------------------------------------
    # Insertion API
    # ------------------------------------------------------------------

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        ego_info: np.ndarray,
        done: bool,
        env_id: int = 0,
    ) -> None:
        """Add a single transition; finalise and store the episode when done=True.

        Args:
            state:    Pure state vector (state_dim,).
            action:   Tanh-squashed continuous action (action_dim,).
            ego_info: [ego_x, ego_y, cos_heading, sin_heading] at this step.
            done:     Whether this transition ends the episode.
            env_id:   Which parallel environment this transition belongs to.
        """
        env_id = int(env_id) % self._num_envs
        ep = self._active[env_id]
        ep.add(state, action, ego_info)

        if done:
            ep.finalise()
            if len(ep) >= self.min_ep_len:
                self._total_steps += len(ep)
                if len(self._episodes) < self._max_episodes:
                    self._episodes.append(ep)
                else:
                    self._episodes[self._write] = ep
                    self._write = (self._write + 1) % self._max_episodes
            # Reset active episode for this env regardless of length.
            self._active[env_id] = _Episode()

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        ego_infos: np.ndarray,
        dones: np.ndarray,
        env_ids: Optional[np.ndarray] = None,
    ) -> None:
        """Add a batch of transitions (one per env in a vectorised env step).

        Args:
            states:    (N, state_dim) float32 array.
            actions:   (N, action_dim) float32 array.
            ego_infos: (N, 4) float32 array – [ego_x, ego_y, cos_h, sin_h] per env.
            dones:     (N,) bool/float array.
            env_ids:   (N,) int array of environment indices; defaults to 0…N-1.
        """
        N = len(states)
        for i in range(N):
            eid = int(env_ids[i]) if env_ids is not None else i % self._num_envs
            self.add(
                states[i],
                actions[i],
                ego_infos[i],
                bool(dones[i]),
                eid,
            )

    # ------------------------------------------------------------------
    # Sampling with HER
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """Sample a HER-relabelled batch of (state, action, goal) triples.

        Sampling procedure for each element *i* in the batch:
          1. Pick a random completed episode uniformly.
          2. Pick a random anchor timestep t ∈ [0, ep_len − 2].
          3. Pick a future timestep t' ∈ [t+1, ep_len − 1] with probability
             proportional to γ^(t'−t−1), so closer futures are preferred.
          4. Compute the HER goal from the ego-frame displacement (t → t').

        Returns:
            Dict with keys ``'obs'`` (B, state_dim), ``'action'`` (B, action_dim),
            ``'goal'`` (B, goal_dim), all float32; or ``None`` if the buffer
            contains no completed episodes.
        """
        num_eps = len(self._episodes)
        if num_eps == 0:
            return None

        batch_size = int(batch_size)
        obs_out = np.empty((batch_size, self.state_dim), dtype=np.float32)
        act_out = np.empty((batch_size, self.action_dim), dtype=np.float32)
        goal_out = np.empty((batch_size, self.goal_dim), dtype=np.float32)
        log_gamma = math.log(self.gamma) if 0.0 < self.gamma < 1.0 else None

        ep_indices = np.random.randint(0, num_eps, size=batch_size)
        for i, ep_idx in enumerate(ep_indices):
            ep = self._episodes[int(ep_idx)]
            ep_len = len(ep)
            if ep_len < 2:
                obs_out[i] = ep.states[-1]
                act_out[i] = ep.actions[-1]
                goal_out[i] = 0.0
                continue
            t = int(np.random.randint(0, ep_len - 1))
            t_prime = self._sample_future(t, ep_len, log_gamma)
            obs_out[i] = ep.states[t]
            act_out[i] = ep.actions[t]
            ego_t = ep.ego_infos[t]
            ego_p = ep.ego_infos[t_prime]
            dx = float(ego_p[0]) - float(ego_t[0])
            dy = float(ego_p[1]) - float(ego_t[1])
            cos_h = float(ego_t[2])
            sin_h = float(ego_t[3])
            goal_out[i, 0] = (dx * cos_h + dy * sin_h) / DIST_NORM
            goal_out[i, 1] = (-dx * sin_h + dy * cos_h) / DIST_NORM

        return {"obs": obs_out, "action": act_out, "goal": goal_out}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_future(
        self, t: int, ep_len: int, log_gamma: Optional[float] = None
    ) -> int:
        """Sample a future index t' ∈ [t+1, ep_len−1] with γ^(offset) weights.

        Inverse-CDF of the truncated geometric γ^k / Σ γ^j, same distribution
        as explicitly building the probability vector.
        """
        num_future = ep_len - t - 1
        if num_future <= 1:
            return t + 1
        if log_gamma is None:
            if not (0.0 < self.gamma < 1.0):
                return t + 1 + int(np.random.randint(0, num_future))
            log_gamma = math.log(self.gamma)
        u = float(np.random.random())
        g_n = self.gamma ** num_future
        inner = max(1.0 - u * (1.0 - g_n), 1e-12)
        k = int(math.log(inner) / log_gamma)
        if k < 0:
            k = 0
        elif k >= num_future:
            k = num_future - 1
        return t + 1 + k

    @staticmethod
    def _compute_goal(
        ego_infos: np.ndarray,
        t: int,
        t_prime: int,
    ) -> np.ndarray:
        """Compute the HER goal as an ego-frame displacement, normalised by DIST_NORM.

        The displacement from position at *t* to position at *t'* is rotated into
        the ego heading frame at *t*, then divided by DIST_NORM so the network
        receives a compact, scale-normalised representation.

        Layout of ego_infos rows:
            [ego_x, ego_y, cos_heading, sin_heading]

        Returns:
            (2,) float32 array – [goal_long, goal_lat].
        """
        dx = float(ego_infos[t_prime, 0]) - float(ego_infos[t, 0])
        dy = float(ego_infos[t_prime, 1]) - float(ego_infos[t, 1])
        cos_h = float(ego_infos[t, 2])
        sin_h = float(ego_infos[t, 3])

        goal_long = (dx * cos_h + dy * sin_h) / DIST_NORM
        goal_lat = (-dx * sin_h + dy * cos_h) / DIST_NORM

        return np.array([goal_long, goal_lat], dtype=np.float32)

    # ------------------------------------------------------------------
    # Utilities / properties
    # ------------------------------------------------------------------

    @property
    def num_episodes(self) -> int:
        """Number of completed episodes currently stored."""
        return len(self._episodes)

    def __len__(self) -> int:
        """Total number of stored transition steps across all completed episodes.

        Note: once the deque wraps, this count is *not* decremented for
        evicted episodes — it reflects the cumulative steps ever stored.
        Use ``num_episodes`` for the live episode count.
        """
        return self._total_steps

    def __repr__(self) -> str:
        return (
            f"HERReplayBuffer("
            f"episodes={self.num_episodes}, "
            f"total_steps={self._total_steps}, "
            f"state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, "
            f"goal_dim={self.goal_dim}, "
            f"gamma={self.gamma}"
            f")"
        )
