"""
CRL-adapted collision avoidance environment for nocturne.

Differences from CollisionAvoidanceEnv:
1. Observation = concat([state(605), goal(2)]) instead of 609-dim.
   State: grid(600) + [heading/pi, speed/SPEED_NORM] + ttz(3)
   Goal:  ego-frame displacement to target / DIST_NORM
2. Action: continuous [throttle_tanh, steer_tanh] in [-1, 1]^2
   (scaled to throttle in [-3, 2], steer in [-0.7, 0.7] internally).
3. Reward: always 0.0 (self-supervised CRL; no hand-crafted reward).
4. info dict includes: ego_x, ego_y, ego_cos_h, ego_sin_h
   (needed by the HER buffer to compute goal encodings).
5. Exposes state_dim, goal_dim, get_ego_info(), get_state_goal().
"""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gym.spaces import Box

from examples.drl_collision_avoidance.collision_avoidance_env import (
    DIST_NORM,
    MAX_RESET_ATTEMPTS,
    SPEED_NORM,
    CollisionAvoidanceEnv,
)

# Throttle range: [-3.0, 2.0]  =>  center = -0.5, half-range = 2.5
_THROTTLE_SCALE = 2.5
_THROTTLE_BIAS = -0.5
# Steer range: [-0.7, 0.7]
_STEER_SCALE = 0.7


class CRLCollisionAvoidanceEnv(CollisionAvoidanceEnv):
    """Collision avoidance env adapted for Contrastive RL training.

    The parent class (CollisionAvoidanceEnv) expects cfg keys that differ from
    the CRL YAML layout (``rew_cfg`` vs ``reward``, ``crl`` vs ``drl``, no
    ``action_map``).  This class injects the required parent-config keys from
    CRL equivalents before delegating to ``super().__init__``, so neither the
    parent class nor the CRL config files need to change.

    Args:
        cfg: Same config dict as CollisionAvoidanceEnv; the CRL env uses
             only the base scenario / rew_cfg settings and overrides the
             observation and action spaces.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        # ------------------------------------------------------------------
        # Adapt CRL config keys to what CollisionAvoidanceEnv.__init__ needs.
        # We work on a shallow copy so the caller's dict is never mutated.
        # ------------------------------------------------------------------
        cfg = dict(cfg)

        # Parent reads cfg["reward"] with these exact keys.
        if "reward" not in cfg:
            _rew = cfg.get("rew_cfg", {})
            cfg["reward"] = {
                "goal_bonus": float(_rew.get("goal_achieved_bonus", 0.0)),
                "collision_penalty": float(_rew.get("collision_penalty", 0.0)),
                "step_penalty": 0.0,
                "progress_scale": 0.0,
                "ttz_safe_threshold": 3.0,
                "ttz_reward_scale": 0.0,
                "offroad_penalty": 0.0,
                "heading_scale": 0.0,
            }

        # Parent reads cfg["action_map"] to build a discrete action table.
        # For CRL the table is never used, but the attribute must exist.
        if "action_map" not in cfg:
            cfg["action_map"] = {
                "throttle_levels": [
                    float(cfg.get("accel_lower_bound", -3.0)),
                    0.0,
                    float(cfg.get("accel_upper_bound", 2.0)),
                ],
                "steer_levels": [
                    float(cfg.get("steering_lower_bound", -0.7)),
                    0.0,
                    float(cfg.get("steering_upper_bound", 0.7)),
                ],
            }

        # Parent reads cfg["drl"]["max_episode_steps"].
        if "drl" not in cfg:
            _crl = cfg.get("crl", {})
            max_ep = _crl.get("max_episode_steps") or cfg.get("episode_length", 80)
            cfg["drl"] = {"max_episode_steps": int(max_ep)}

        super().__init__(cfg)

        # ------------------------------------------------------------------
        # CRL-specific dimensions and spaces (override parent's).
        # ------------------------------------------------------------------
        grid_size = self.grid_channels * self.grid_rows * self.grid_cols

        self.state_dim: int = grid_size + 2 + 3  # grid + (heading, speed) + ttz(3)
        self.goal_dim: int = 2

        obs_dim = self.state_dim + self.goal_dim  # default: 600 + 2 + 3 + 2 = 607

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = Box(
            low=np.full(2, -1.0, dtype=np.float32),
            high=np.full(2, 1.0, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the underlying scenario and return a CRL observation."""
        obs_dict: Optional[Dict] = None
        for _ in range(MAX_RESET_ATTEMPTS):
            obs_dict = self.base_env.reset()
            if not obs_dict:
                continue
            ego_id = next(iter(obs_dict))
            vehicle_ids = {v.getID() for v in self.base_env.scenario.getVehicles()}
            if ego_id in vehicle_ids:
                self._ego_id = ego_id
                self._step_count = 0
                self._prev_goal_dist = self._get_goal_dist()
                return self._build_crl_obs()

        # Fallback: use any reachable vehicle id.
        ego_id = self._find_any_ego_id(obs_dict or {})
        if ego_id is None:
            raise RuntimeError(
                f"CRLCollisionAvoidanceEnv: failed to reset after "
                f"{MAX_RESET_ATTEMPTS} attempts."
            )
        self._ego_id = ego_id
        self._step_count = 0
        self._prev_goal_dist = self._get_goal_dist()
        return self._build_crl_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Accept a tanh-squashed continuous action in [-1, 1]^2.

        Internally scales to throttle in [-3, 2] and steer in [-0.7, 0.7].

        Returns:
            obs    : CRL observation (state + goal concatenated).
            reward : Always 0.0 — CRL is fully self-supervised.
            done   : Episode termination flag.
            info   : dict with goal_achieved, collided, veh_veh_collision,
                     veh_edge_collision, ego_x, ego_y, ego_cos_h, ego_sin_h.
        """
        throttle = float(
            np.clip(float(action[0]) * _THROTTLE_SCALE + _THROTTLE_BIAS, -3.0, 2.0)
        )
        steer = float(np.clip(float(action[1]) * _STEER_SCALE, -0.7, 0.7))

        ego_id: int = self._ego_id  # type: ignore[assignment]  # always set by reset()
        action_dict = {ego_id: [throttle, steer, 0.0]}
        _, _rew_dict, done_dict, info_dict = self.base_env.step(action_dict)
        self._step_count += 1

        obs = self._build_crl_obs()
        ego_done = bool(done_dict.get(ego_id, False))
        all_done = bool(done_dict.get("__all__", False))
        done = ego_done or all_done or self._step_count >= self._max_steps

        info = dict(info_dict.get(ego_id, {}))

        # Append ego pose so the HER buffer can compute hindsight goals.
        ego_veh = self._get_ego_vehicle()
        if ego_veh is not None:
            info["ego_x"] = float(ego_veh.position.x)
            info["ego_y"] = float(ego_veh.position.y)
            info["ego_cos_h"] = float(math.cos(ego_veh.heading))
            info["ego_sin_h"] = float(math.sin(ego_veh.heading))
        else:
            info["ego_x"] = 0.0
            info["ego_y"] = 0.0
            info["ego_cos_h"] = 1.0
            info["ego_sin_h"] = 0.0

        # CRL is self-supervised: reward is always 0 regardless of outcome.
        return obs, 0.0, done, info

    # ------------------------------------------------------------------
    # Helpers exposed to the training loop
    # ------------------------------------------------------------------

    def get_state_goal(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split a CRL observation vector into (state, goal) arrays.

        Args:
            obs: Flat observation array of length ``state_dim + goal_dim``.

        Returns:
            state: numpy array of shape ``(state_dim,)``.
            goal:  numpy array of shape ``(goal_dim,)``.
        """
        return obs[: self.state_dim], obs[self.state_dim :]

    def get_ego_info(self) -> Tuple[float, float, float, float]:
        """Return ``(ego_x, ego_y, cos_heading, sin_heading)`` for HER storage.

        Safe to call at any point; returns identity pose if the ego vehicle
        is not currently present in the scene.
        """
        ego_veh = self._get_ego_vehicle()
        if ego_veh is None:
            return 0.0, 0.0, 1.0, 0.0
        return (
            float(ego_veh.position.x),
            float(ego_veh.position.y),
            float(math.cos(ego_veh.heading)),
            float(math.sin(ego_veh.heading)),
        )

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _build_crl_obs(self) -> np.ndarray:
        """Build a ``(state_dim + goal_dim,)`` CRL observation.

        Layout:
            [0 : grid_size]               occupancy grid (flattened, C*H*W)
            [grid_size : grid_size+2]     [heading/pi, speed/SPEED_NORM]
            [grid_size+2 : state_dim]     TTZ features (3 values)
            [state_dim : state_dim+2]     ego-frame goal displacement / DIST_NORM
        """
        ego_veh = self._get_ego_vehicle()
        if ego_veh is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # --- state components ---
        grid = self._build_occupancy_grid(ego_veh).flatten()  # (C*H*W,)
        heading = ego_veh.heading / math.pi  # scalar
        speed = ego_veh.speed / SPEED_NORM  # scalar
        ttz = self._get_ttz_info(ego_veh)  # (3,)

        state = np.concatenate([grid, [heading, speed], ttz])  # (state_dim,)

        # --- goal component ---
        goal = self._compute_goal_encoding(ego_veh)  # (2,)

        return np.concatenate([state, goal]).astype(np.float32)

    def _compute_goal_encoding(self, ego_veh) -> np.ndarray:
        """Compute ego-frame displacement to the Waymo target position.

        Projects the world-frame offset (dx, dy) into the ego vehicle's
        longitudinal / lateral axes and normalises by DIST_NORM.

        Returns:
            np.ndarray of shape (2,): [goal_long / DIST_NORM,
                                       goal_lat  / DIST_NORM]
        """
        dx = ego_veh.target_position.x - ego_veh.position.x
        dy = ego_veh.target_position.y - ego_veh.position.y
        cos_h = math.cos(ego_veh.heading)
        sin_h = math.sin(ego_veh.heading)
        goal_long = (dx * cos_h + dy * sin_h) / DIST_NORM
        goal_lat = (-dx * sin_h + dy * cos_h) / DIST_NORM
        return np.array([goal_long, goal_lat], dtype=np.float32)
