import math
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np
from gym.spaces import Box, Discrete

import nocturne
from nocturne.envs.base_env import BaseEnv

# Sentinel for "no imminent conflict" — keeps reward logic simple while
# still exposing finite values when interaction exists.
NO_CONFLICT_TTZ = 999.0

# Normalization divisors for observation features.
SPEED_NORM = 30.0
DIST_NORM = 100.0

# Minimum distance below which TTZ is considered zero (contact).
TTZ_CONTACT_DIST = 0.5

# Maximum TTZ value exposed to the network (clips the sentinel).
TTZ_OBS_CLIP = 20.0

MAX_RESET_ATTEMPTS = 50


class CollisionAvoidanceEnv(gym.Env):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        cfg.setdefault("single_agent_mode", True)
        cfg.setdefault("max_num_vehicles", 1)

        grid_cfg = cfg["occupancy_grid"]
        self.grid_rows = grid_cfg["rows"]
        self.grid_cols = grid_cfg["cols"]
        self.forward_dist = grid_cfg["forward_dist"]
        self.backward_dist = grid_cfg["backward_dist"]
        self.lateral_dist = grid_cfg["lateral_dist"]
        self.vru_weight = grid_cfg["vru_weight"]
        self.vehicle_weight = grid_cfg["vehicle_weight"]
        self.road_edge_weight = grid_cfg["road_edge_weight"]

        rew_cfg = cfg["reward"]
        self.goal_bonus = rew_cfg["goal_bonus"]
        self.collision_penalty = rew_cfg["collision_penalty"]
        self.step_penalty = rew_cfg["step_penalty"]
        self.progress_scale = rew_cfg["progress_scale"]
        self.ttz_safe_threshold = rew_cfg["ttz_safe_threshold"]
        self.ttz_reward_scale = rew_cfg["ttz_reward_scale"]
        self.offroad_penalty = rew_cfg["offroad_penalty"]
        self.heading_scale = rew_cfg["heading_scale"]

        act_cfg = cfg["action_map"]
        throttle_levels = act_cfg["throttle_levels"]
        steer_levels = act_cfg["steer_levels"]
        self.action_table = [
            (float(t), float(s)) for t in throttle_levels for s in steer_levels
        ]

        self.base_env = BaseEnv(cfg)
        self.cfg = cfg

        obs_dim = self.grid_rows * self.grid_cols + 4 + 2 + 3
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = Discrete(len(self.action_table))

        self._ego_id: Optional[int] = None
        self._prev_goal_dist: float = 0.0
        self._step_count: int = 0
        self._max_steps = cfg["drl"]["max_episode_steps"]
        self._ttz_vehicle: float = NO_CONFLICT_TTZ
        self._ttz_pedestrian: float = NO_CONFLICT_TTZ

    def reset(self) -> np.ndarray:
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
                return self._build_observation()

        ego_id = self._find_any_ego_id(obs_dict)
        if ego_id is None:
            raise RuntimeError(
                f"Failed to initialize environment after {MAX_RESET_ATTEMPTS} reset attempts."
            )
        self._ego_id = ego_id
        self._step_count = 0
        self._prev_goal_dist = self._get_goal_dist()
        return self._build_observation()

    def _find_any_ego_id(self, obs_dict):
        if obs_dict:
            return next(iter(obs_dict))
        controlled = getattr(self.base_env, "controlled_vehicles", []) or []
        if controlled:
            return controlled[0].getID()
        scenario = getattr(self.base_env, "scenario", None)
        vehicles = scenario.getVehicles() if scenario is not None else []
        if vehicles:
            return vehicles[0].getID()
        return None

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        throttle, steer = self.action_table[action]
        action_dict = {self._ego_id: [throttle, steer, 0.0]}
        _, rew_dict, done_dict, info_dict = self.base_env.step(action_dict)
        self._step_count += 1

        obs = self._build_observation()
        reward = self._compute_reward(rew_dict, info_dict, done_dict)

        ego_done = done_dict.get(self._ego_id, False)
        all_done = done_dict.get("__all__", False)
        done = ego_done or all_done or self._step_count >= self._max_steps

        info = dict(info_dict.get(self._ego_id, {}))
        info["ttz_vehicle"] = self._ttz_vehicle
        info["ttz_pedestrian"] = self._ttz_pedestrian
        info["step_count"] = self._step_count
        info["goal_dist"] = self._get_goal_dist()

        return obs, reward, done, info

    def render(self, mode="human"):
        return self.base_env.render(mode)

    def seed(self, seed=None):
        self.base_env.seed(seed)

    def close(self):
        pass

    # --- Observation Building ---

    def _build_observation(self) -> np.ndarray:
        ego_veh = self._get_ego_vehicle()
        if ego_veh is None:
            self._ttz_vehicle = NO_CONFLICT_TTZ
            self._ttz_pedestrian = NO_CONFLICT_TTZ
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        grid = self._build_occupancy_grid(ego_veh)
        ego_state = self._get_ego_state(ego_veh)
        target_info = self._get_target_info(ego_veh)
        ttz_info = self._get_ttz_info(ego_veh)

        return np.concatenate(
            [grid.flatten(), ego_state, target_info, ttz_info]
        ).astype(np.float32)

    def _build_occupancy_grid(self, ego_veh) -> np.ndarray:
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        ego_pos = ego_veh.position
        cos_h, sin_h = math.cos(ego_veh.heading), math.sin(ego_veh.heading)
        cell_long = (self.forward_dist + self.backward_dist) / self.grid_rows
        cell_lat = (2.0 * self.lateral_dist) / self.grid_cols

        for obj in self.base_env.scenario.getVehicles():
            if obj.getID() == self._ego_id:
                continue
            self._project_to_grid(
                obj.position, ego_pos, cos_h, sin_h, cell_long, cell_lat, grid, self.vehicle_weight,
            )

        for obj in self.base_env.scenario.getPedestrians():
            self._project_to_grid(
                obj.position, ego_pos, cos_h, sin_h, cell_long, cell_lat, grid, self.vru_weight,
            )

        for obj in self.base_env.scenario.getCyclists():
            self._project_to_grid(
                obj.position, ego_pos, cos_h, sin_h, cell_long, cell_lat, grid, self.vru_weight,
            )

        for road_line in self.base_env.scenario.getRoadLines():
            if road_line.road_type == nocturne.RoadType.ROAD_EDGE:
                for pt in road_line.geometry_points():
                    self._project_to_grid(
                        pt, ego_pos, cos_h, sin_h, cell_long, cell_lat, grid, self.road_edge_weight,
                    )

        return grid

    def _project_to_grid(self, point, ego_pos, cos_h, sin_h, cell_long, cell_lat, grid, weight):
        """Project a point (any object with .x/.y) into the ego-centric grid."""
        dx = point.x - ego_pos.x
        dy = point.y - ego_pos.y
        local_x = dx * cos_h + dy * sin_h
        local_y = -dx * sin_h + dy * cos_h

        if local_x < -self.backward_dist or local_x > self.forward_dist:
            return
        if abs(local_y) > self.lateral_dist:
            return

        row = int((self.forward_dist - local_x) / cell_long)
        col = int((local_y + self.lateral_dist) / cell_lat)
        row = max(0, min(row, self.grid_rows - 1))
        col = max(0, min(col, self.grid_cols - 1))
        grid[row, col] = max(grid[row, col], weight)

    def _get_ego_state(self, ego_veh) -> np.ndarray:
        heading_err = self._get_heading_error(ego_veh)
        speed_err = (ego_veh.speed - ego_veh.target_speed) / SPEED_NORM
        return np.array(
            [
                ego_veh.heading / math.pi,
                ego_veh.speed / SPEED_NORM,
                heading_err / math.pi,
                speed_err,
            ],
            dtype=np.float32,
        )

    def _get_target_info(self, ego_veh) -> np.ndarray:
        goal = ego_veh.target_position
        ego_pos = ego_veh.position
        dx = goal.x - ego_pos.x
        dy = goal.y - ego_pos.y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-8
        angle_to_goal = math.atan2(dy, dx)
        rel_heading = angle_to_goal - ego_veh.heading
        rel_heading = (rel_heading + math.pi) % (2 * math.pi) - math.pi
        return np.array([dist / DIST_NORM, rel_heading / math.pi], dtype=np.float32)

    def _get_ttz_info(self, ego_veh) -> np.ndarray:
        ego_pos = ego_veh.position
        ego_speed = max(ego_veh.speed, 0.1)

        min_ttz_veh = NO_CONFLICT_TTZ
        min_ttz_ped = NO_CONFLICT_TTZ

        for obj in self.base_env.scenario.getVehicles():
            if obj.getID() == self._ego_id:
                continue
            ttz = self._compute_ttz(ego_pos, ego_speed, ego_veh.heading, obj)
            min_ttz_veh = min(min_ttz_veh, ttz)

        for obj in self.base_env.scenario.getPedestrians():
            ttz = self._compute_ttz(ego_pos, ego_speed, ego_veh.heading, obj)
            min_ttz_ped = min(min_ttz_ped, ttz)

        for obj in self.base_env.scenario.getCyclists():
            ttz = self._compute_ttz(ego_pos, ego_speed, ego_veh.heading, obj)
            min_ttz_ped = min(min_ttz_ped, ttz)

        self._ttz_vehicle = min_ttz_veh
        self._ttz_pedestrian = min_ttz_ped

        return np.array(
            [
                min(min_ttz_veh, TTZ_OBS_CLIP),
                min(min_ttz_ped, TTZ_OBS_CLIP),
                np.clip(min_ttz_veh - min_ttz_ped, -TTZ_OBS_CLIP, TTZ_OBS_CLIP),
            ],
            dtype=np.float32,
        )

    def _compute_ttz(self, ego_pos, ego_speed, ego_heading, obj) -> float:
        dx = obj.position.x - ego_pos.x
        dy = obj.position.y - ego_pos.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < TTZ_CONTACT_DIST:
            return 0.0
        closing_speed = ego_speed * (
            math.cos(ego_heading) * dx / dist + math.sin(ego_heading) * dy / dist
        )
        obj_speed = getattr(obj, "speed", 0.0)
        if obj_speed > 0.01:
            obj_heading = getattr(obj, "heading", 0.0)
            closing_speed -= obj_speed * (
                math.cos(obj_heading) * dx / dist + math.sin(obj_heading) * dy / dist
            )
        if closing_speed <= 0.01:
            return NO_CONFLICT_TTZ
        return dist / closing_speed

    def _get_heading_error(self, ego_veh) -> float:
        goal = ego_veh.target_position
        pos = ego_veh.position
        angle_to_goal = math.atan2(goal.y - pos.y, goal.x - pos.x)
        err = angle_to_goal - ego_veh.heading
        return (err + math.pi) % (2 * math.pi) - math.pi

    # --- Reward ---

    def _compute_reward(self, rew_dict, info_dict, done_dict) -> float:
        reward = 0.0
        info = info_dict.get(self._ego_id, {})

        goal_achieved = info.get("goal_achieved", False)
        collided = info.get("collided", False)
        veh_edge = info.get("veh_edge_collision", False)
        is_terminal = goal_achieved or collided

        if goal_achieved:
            reward += self.goal_bonus

        if collided:
            reward -= self.collision_penalty
            if veh_edge:
                reward -= self.offroad_penalty

        if not is_terminal:
            reward -= self.step_penalty

            curr_dist = self._get_goal_dist()
            progress = self._prev_goal_dist - curr_dist
            reward += self.progress_scale * progress
            self._prev_goal_dist = curr_dist

            ego_veh = self._get_ego_vehicle()
            if ego_veh is not None:
                heading_err = abs(self._get_heading_error(ego_veh))
                reward += self.heading_scale * (1.0 - heading_err / math.pi)

            min_ttz = min(self._ttz_vehicle, self._ttz_pedestrian)
            if min_ttz < self.ttz_safe_threshold and min_ttz > 0:
                reward -= self.ttz_reward_scale * (self.ttz_safe_threshold - min_ttz)

        return reward

    def _get_goal_dist(self) -> float:
        ego_veh = self._get_ego_vehicle()
        if ego_veh is None:
            return 0.0
        goal = ego_veh.target_position
        pos = ego_veh.position
        return math.sqrt((goal.x - pos.x) ** 2 + (goal.y - pos.y) ** 2)

    def _get_ego_vehicle(self):
        for v in self.base_env.controlled_vehicles:
            if v.getID() == self._ego_id:
                return v
        vehs = self.base_env.scenario.getVehicles()
        return vehs[0] if vehs else None
