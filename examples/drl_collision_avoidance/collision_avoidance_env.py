import math
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np
from gym.spaces import Box, Discrete

import nocturne
from nocturne.envs.base_env import BaseEnv


class CollisionAvoidanceEnv(gym.Env):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        cfg.setdefault("single_agent_mode", True)
        cfg.setdefault("max_num_vehicles", 1)

        grid_cfg = cfg.get("occupancy_grid", {})
        self.grid_rows = grid_cfg.get("rows", 25)
        self.grid_cols = grid_cfg.get("cols", 14)
        self.forward_dist = grid_cfg.get("forward_dist", 20.0)
        self.backward_dist = grid_cfg.get("backward_dist", 5.0)
        self.lateral_dist = grid_cfg.get("lateral_dist", 7.0)
        self.vru_weight = grid_cfg.get("vru_weight", 2.0)
        self.vehicle_weight = grid_cfg.get("vehicle_weight", 1.0)
        self.road_edge_weight = grid_cfg.get("road_edge_weight", 3.0)

        rew_cfg = cfg.get("reward", {})
        self.goal_bonus = rew_cfg.get("goal_bonus", 100.0)
        self.collision_penalty = rew_cfg.get("collision_penalty", 0.0)
        self.step_penalty = rew_cfg.get("step_penalty", 0.0)
        self.progress_scale = rew_cfg.get("progress_scale", 0.5)
        self.ttz_safe_threshold = rew_cfg.get("ttz_safe_threshold", 4.0)
        self.ttz_reward_scale = rew_cfg.get("ttz_reward_scale", 0.5)
        self.offroad_penalty = rew_cfg.get("offroad_penalty", 5.0)
        self.heading_scale = rew_cfg.get("heading_scale", 0.3)

        act_cfg = cfg.get("action_map", {})
        throttle_levels = act_cfg.get("throttle_levels", [-4.0, 0.0, 2.0])
        steer_levels = act_cfg.get("steer_levels", [-0.3, 0.0, 0.3])
        self.action_table = []
        for t in throttle_levels:
            for s in steer_levels:
                self.action_table.append((float(t), float(s)))

        self.base_env = BaseEnv(cfg)
        self.cfg = cfg

        # grid(rows*cols) + ego(heading, speed, heading_err, speed_err) + goal(dist, rel_heading) + ttz(3)
        obs_dim = self.grid_rows * self.grid_cols + 4 + 2 + 3
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = Discrete(len(self.action_table))

        self._ego_id: Optional[int] = None
        self._prev_goal_dist: float = 0.0
        self._step_count: int = 0
        self._max_steps = cfg.get("drl", {}).get("max_episode_steps", 400)
        self._ttz_vehicle: float = 999.0
        self._ttz_pedestrian: float = 999.0

    def reset(self) -> np.ndarray:
        # BaseEnv reset can occasionally pick a controlled id that gets removed
        # during startup filtering. Retry to avoid starting from an invalid ego.
        obs_dict = {}
        for _ in range(50):
            obs_dict = self.base_env.reset()
            if not obs_dict:
                continue
            ego_id = list(obs_dict.keys())[0]
            vehicle_ids = {v.getID() for v in self.base_env.scenario.getVehicles()}
            if ego_id in vehicle_ids:
                self._ego_id = ego_id
                self._step_count = 0
                self._prev_goal_dist = self._get_goal_dist()
                return self._build_observation()

        if obs_dict:
            self._ego_id = list(obs_dict.keys())[0]
        else:
            controlled = getattr(self.base_env, "controlled_vehicles", []) or []
            if controlled:
                self._ego_id = controlled[0].getID()
            else:
                scenario = getattr(self.base_env, "scenario", None)
                scenario_vehicles = (
                    scenario.getVehicles() if scenario is not None else []
                )
                if scenario_vehicles:
                    self._ego_id = scenario_vehicles[0].getID()
                else:
                    raise RuntimeError(
                        "Failed to initialize environment after 50 reset attempts."
                    )

        if self._ego_id is None:
            raise RuntimeError(
                "Failed to initialize environment after 50 reset attempts."
            )

        self._step_count = 0
        self._prev_goal_dist = self._get_goal_dist()
        return self._build_observation()

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

    def _build_observation(self) -> np.ndarray:
        ego_veh = self._get_ego_vehicle()
        if ego_veh is None:
            self._ttz_vehicle = 999.0
            self._ttz_pedestrian = 999.0
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        grid = self._build_occupancy_grid(ego_veh)
        ego_state = self._get_ego_state(ego_veh)
        target_info = self._get_target_info(ego_veh)
        ttz_info = self._get_ttz_info(ego_veh)

        obs = np.concatenate([grid.flatten(), ego_state, target_info, ttz_info])
        return obs.astype(np.float32)

    def _build_occupancy_grid(self, ego_veh) -> np.ndarray:
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        ego_pos = ego_veh.position
        ego_heading = ego_veh.heading
        cos_h, sin_h = math.cos(ego_heading), math.sin(ego_heading)

        total_long = self.forward_dist + self.backward_dist
        total_lat = 2.0 * self.lateral_dist
        cell_long = total_long / self.grid_rows
        cell_lat = total_lat / self.grid_cols

        for obj in self.base_env.scenario.getVehicles():
            if obj.getID() == self._ego_id:
                continue
            self._project_to_grid(
                obj,
                ego_pos,
                cos_h,
                sin_h,
                cell_long,
                cell_lat,
                grid,
                self.vehicle_weight,
            )

        for obj in self.base_env.scenario.getPedestrians():
            self._project_to_grid(
                obj, ego_pos, cos_h, sin_h, cell_long, cell_lat, grid, self.vru_weight
            )

        for obj in self.base_env.scenario.getCyclists():
            self._project_to_grid(
                obj, ego_pos, cos_h, sin_h, cell_long, cell_lat, grid, self.vru_weight
            )

        for road_line in self.base_env.scenario.getRoadLines():
            if road_line.road_type == nocturne.RoadType.ROAD_EDGE:
                for pt in road_line.geometry_points():
                    self._project_point_to_grid(
                        pt, ego_pos, cos_h, sin_h,
                        cell_long, cell_lat, grid, self.road_edge_weight,
                    )

        return grid

    def _project_to_grid(
        self, obj, ego_pos, cos_h, sin_h, cell_long, cell_lat, grid, weight
    ):
        dx = obj.position.x - ego_pos.x
        dy = obj.position.y - ego_pos.y
        local_x = dx * cos_h + dy * sin_h
        local_y = -dx * sin_h + dy * cos_h

        if local_x < -self.backward_dist or local_x > self.forward_dist:
            return
        if abs(local_y) > self.lateral_dist:
            return

        row = int((self.forward_dist - local_x) / cell_long)
        col = int((local_y + self.lateral_dist) / cell_lat)
        # Row index is inverted so smaller row numbers correspond to space ahead
        # of the ego vehicle (top of the grid in downstream visualizations).
        row = max(0, min(row, self.grid_rows - 1))
        col = max(0, min(col, self.grid_cols - 1))
        grid[row, col] = max(grid[row, col], weight)

    def _project_point_to_grid(
        self, pt, ego_pos, cos_h, sin_h, cell_long, cell_lat, grid, weight
    ):
        dx = pt.x - ego_pos.x
        dy = pt.y - ego_pos.y
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
        speed_err = (ego_veh.speed - ego_veh.target_speed) / 30.0
        return np.array(
            [
                ego_veh.heading / math.pi,
                ego_veh.speed / 30.0,
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
        return np.array([dist / 100.0, rel_heading / math.pi], dtype=np.float32)

    def _get_ttz_info(self, ego_veh) -> np.ndarray:
        ego_pos = ego_veh.position
        ego_speed = max(ego_veh.speed, 0.1)

        # 999.0 is a sentinel for "no imminent conflict" that keeps reward logic
        # simple while still exposing finite values when interaction exists.
        min_ttz_veh = 999.0
        min_ttz_ped = 999.0

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

        ttz_diff = min_ttz_veh - min_ttz_ped
        # Clip observation features so the network input scale is bounded even
        # when there are no nearby actors (sentinel stays in info/reward attrs).
        return np.array(
            [
                min(min_ttz_veh, 20.0),
                min(min_ttz_ped, 20.0),
                np.clip(ttz_diff, -20.0, 20.0),
            ],
            dtype=np.float32,
        )

    def _compute_ttz(self, ego_pos, ego_speed, ego_heading, obj) -> float:
        dx = obj.position.x - ego_pos.x
        dy = obj.position.y - ego_pos.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.5:
            return 0.0
        # Positive closing speed means the relative motion is reducing distance.
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
            return 999.0
        return dist / closing_speed

    def _get_heading_error(self, ego_veh) -> float:
        goal = ego_veh.target_position
        pos = ego_veh.position
        angle_to_goal = math.atan2(goal.y - pos.y, goal.x - pos.x)
        err = angle_to_goal - ego_veh.heading
        return (err + math.pi) % (2 * math.pi) - math.pi

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

    def _get_ego_vehicle(
        self,
    ):
        for v in self.base_env.controlled_vehicles:
            if v.getID() == self._ego_id:
                return v
        vehs = self.base_env.scenario.getVehicles()
        return vehs[0] if vehs else None
