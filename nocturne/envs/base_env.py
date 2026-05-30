# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Default environment for Nocturne."""
from typing import Any, Dict, Sequence, Union

from collections import defaultdict, deque
from itertools import islice
import json
import os

from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import numpy as np
import torch

from cfgs.config import ERR_VAL as INVALID_POSITION, get_scenario_dict
from nocturne import Action, Simulation
from nocturne.utils.occlusion_features import compute_occlusion_features


class BaseEnv(Env):
    """Default environment for Nocturne."""

    def __init__(self, cfg: Dict[str, Any], rank: int = 0) -> None:
        """Initialize the environment.

        Args
        ----
            cfg (dict): configuration file describing the experiment
            rank (int, optional): [description]. Defaults to 0.
        """
        super().__init__()
        self.cfg = cfg
        with open(os.path.join(cfg['scenario_path'],
                               'valid_files.json')) as file:
            self.valid_veh_dict = json.load(file)
            self.files = list(self.valid_veh_dict.keys())
            # sort the files so that we have a consistent order
            self.files = sorted(self.files)
        if cfg['num_files'] != -1:
            self.files = self.files[0:cfg['num_files']]
        self.file = self.files[np.random.randint(len(self.files))]
        self.simulation = Simulation(os.path.join(cfg['scenario_path'],
                                                  self.file),
                                     config=get_scenario_dict(cfg))

        self.scenario = self.simulation.getScenario()
        self.controlled_vehicles = self.scenario.getObjectsThatMoved()
        self.cfg = cfg
        subscriber_cfg = self.cfg.get('subscriber', {})
        self.n_frames_stacked = subscriber_cfg.get('n_frames_stacked', 1)
        self.render_img_width = subscriber_cfg.get('render_img_width', 1600)
        self.render_img_height = subscriber_cfg.get('render_img_height', 1600)
        self.render_padding = subscriber_cfg.get('render_padding', 50.0)
        if self.n_frames_stacked > 1:
            print(
                'WARNING: you are frame stacking and may want to turn off recurrence if it is enabled\
                  in your agent as frame-stacking may not be needed when using recurrent policies.'
            )
        self.single_agent_mode = cfg['single_agent_mode']
        self.seed(cfg['seed'])
        self.episode_length = cfg['episode_length']
        self.t = 0
        self.step_num = 0
        self.rank = rank
        self.seed(cfg['seed'])

        # Scenario pool: limit working set so OS page cache keeps files hot
        self._pool_size = cfg.get('scenario_pool_size', 0)
        self._resample_interval = cfg.get('resample_pool_interval', 5000)
        self._pool_files = []
        self._pool_reset_count = 0
        if self._pool_size > 0:
            self._pool_files = [self.files[np.random.randint(len(self.files))]
                                for _ in range(self._pool_size)]

        obs_dict, _ = self.reset()
        self.observation_space = Box(low=-np.infty,
                                     high=np.infty,
                                     shape=(obs_dict[list(
                                         obs_dict.keys())[0]].shape[0], ))
        if self.cfg['discretize_actions']:
            self.accel_discretization = self.cfg['accel_discretization']
            self.steering_discretization = self.cfg['steering_discretization']
            self.head_angle_discretization = self.cfg[
                'head_angle_discretization']
            self.action_space = Discrete(self.accel_discretization *
                                         self.steering_discretization *
                                         self.head_angle_discretization)
            self.accel_grid = np.linspace(
                -np.abs(self.cfg['accel_lower_bound']),
                self.cfg['accel_upper_bound'], self.accel_discretization)
            self.steering_grid = np.linspace(
                -np.abs(self.cfg['steering_lower_bound']),
                self.cfg['steering_upper_bound'], self.steering_discretization)
            self.head_angle_grid = np.linspace(
                -np.abs(self.cfg['head_angle_lower_bound']),
                self.cfg['head_angle_upper_bound'],
                self.head_angle_discretization)
            # compute the indexing only once
            self.idx_to_actions = {}
            i = 0
            for accel in self.accel_grid:
                for steer in self.steering_grid:
                    for head_angle in self.head_angle_grid:
                        self.idx_to_actions[i] = [accel, steer, head_angle]
                        i += 1
        else:
            self.action_space = Box(
                low=np.array([
                    self.cfg['accel_lower_bound'],
                    self.cfg['steering_lower_bound'],
                    self.cfg['head_angle_lower_bound']
                ],
                             dtype=np.float32),
                high=np.array([
                    self.cfg['accel_upper_bound'],
                    self.cfg['steering_upper_bound'],
                    self.cfg['head_angle_upper_bound']
                ],
                              dtype=np.float32),
            )

    def apply_actions(
        self, action_dict: Dict[int, Union[Action, np.ndarray, Sequence[float],
                                           int]]
    ) -> None:
        """Apply a dict of actions to the vehicle objects."""
        for veh_obj in self.scenario.getObjectsThatMoved():
            action = action_dict.get(veh_obj.id, None)
            if action is None:
                continue

            # TODO: Make this a util function.
            if isinstance(action, Action):
                veh_obj.apply_action(action)
            elif isinstance(action, np.ndarray):
                veh_obj.apply_action(Action.from_numpy(action))
            elif isinstance(action, (tuple, list)):
                veh_obj.acceleration = action[0]
                veh_obj.steering = action[1]
                veh_obj.head_angle = action[2]
            else:
                accel, steer, head_angle = self.idx_to_actions[action]
                veh_obj.acceleration = accel
                veh_obj.steering = steer
                veh_obj.head_angle = head_angle

    def step(
        self, action_dict: Dict[int, Union[Action, np.ndarray, Sequence[float],
                                           int]]
    ) -> None:
        """See superclass."""
        obs_dict = {}
        rew_dict = {}
        done_dict = {}
        info_dict = defaultdict(dict)
        rew_cfg = self.cfg['rew_cfg']
        self.apply_actions(action_dict)
        self.simulation.step(self.cfg['dt'])
        self.t += self.cfg['dt']
        self.step_num += 1
        objs_to_remove = []
        for veh_obj in self.controlled_vehicles:
            veh_id = veh_obj.getID()
            if veh_id in self.done_ids:
                continue
            self.context_dict[veh_id].append(self.get_observation(veh_obj))
            if self.n_frames_stacked > 1:
                veh_deque = self.context_dict[veh_id]
                context_list = list(
                    islice(veh_deque,
                           len(veh_deque) - self.n_frames_stacked,
                           len(veh_deque)))
                obs_dict[veh_id] = np.concatenate(context_list)
            else:
                obs_dict[veh_id] = self.context_dict[veh_id][-1]
            rew_dict[veh_id] = 0
            done_dict[veh_id] = False
            info_dict[veh_id]['goal_achieved'] = False
            info_dict[veh_id]['collided'] = False
            info_dict[veh_id]['veh_veh_collision'] = False
            info_dict[veh_id]['veh_edge_collision'] = False
            info_dict[veh_id]['cost'] = 0.0
            info_dict[veh_id]['near_miss'] = 0.0
            obj_pos = veh_obj.position
            goal_pos = veh_obj.target_position
            '''############################################
                            Compute rewards
               ############################################'''
            position_target_achieved = True
            speed_target_achieved = True
            heading_target_achieved = True
            if rew_cfg['position_target']:
                position_target_achieved = (
                    goal_pos -
                    obj_pos).norm() < rew_cfg['position_target_tolerance']
            if rew_cfg['speed_target']:
                speed_target_achieved = np.abs(
                    veh_obj.speed -
                    veh_obj.target_speed) < rew_cfg['speed_target_tolerance']
            if rew_cfg['heading_target']:
                heading_target_achieved = np.abs(
                    self.angle_sub(veh_obj.heading, veh_obj.target_heading)
                ) < rew_cfg['heading_target_tolerance']
            if position_target_achieved and speed_target_achieved and heading_target_achieved:
                info_dict[veh_id]['goal_achieved'] = True
                rew_dict[veh_id] += rew_cfg['goal_achieved_bonus'] / rew_cfg[
                    'reward_scaling']
            if rew_cfg['shaped_goal_distance'] and rew_cfg['position_target']:
                # penalize the agent for its distance from goal
                # we scale by goal_dist_normalizers to ensure that this value is always less than the penalty for
                # collision
                if rew_cfg['goal_distance_penalty']:
                    rew_dict[veh_id] -= rew_cfg.get(
                        'shaped_goal_distance_scaling', 1.0) * (
                            (goal_pos - obj_pos).norm() /
                            self.goal_dist_normalizers[veh_id]
                        ) / rew_cfg['reward_scaling']
                else:
                    # the minus one is to ensure that it's not beneficial to collide
                    # we divide by goal_achieved_bonus / episode_length to ensure that
                    # acquiring the maximum "get-close-to-goal" reward at every time-step is
                    # always less than just acquiring the goal reward once
                    # we also assume that vehicles are never more than 400 meters from their goal
                    # which makes sense as the episodes are 9 seconds long i.e. we'd have to go more than
                    # 40 m/s to get there
                    rew_dict[veh_id] += rew_cfg.get(
                        'shaped_goal_distance_scaling',
                        1.0) * (1 - (goal_pos - obj_pos).norm() /
                                self.goal_dist_normalizers[veh_id]
                                ) / rew_cfg['reward_scaling']
                # repeat the same thing for speed and heading
                if rew_cfg['shaped_goal_distance'] and rew_cfg['speed_target']:
                    if rew_cfg['goal_distance_penalty']:
                        rew_dict[veh_id] -= rew_cfg.get(
                            'shaped_goal_distance_scaling', 1.0) * (
                                np.abs(veh_obj.speed - veh_obj.target_speed) /
                                40.0) / rew_cfg['reward_scaling']
                    else:
                        rew_dict[veh_id] += rew_cfg.get(
                            'shaped_goal_distance_scaling', 1.0
                        ) * (1 - np.abs(veh_obj.speed - veh_obj.target_speed) /
                             40.0) / rew_cfg['reward_scaling']
                if rew_cfg['shaped_goal_distance'] and rew_cfg[
                        'heading_target']:
                    if rew_cfg['goal_distance_penalty']:
                        rew_dict[veh_id] -= rew_cfg.get(
                            'shaped_goal_distance_scaling',
                            1.0) * (np.abs(
                                self.angle_sub(veh_obj.heading,
                                               veh_obj.target_heading)) /
                                    (2 * np.pi)) / rew_cfg['reward_scaling']
                    else:
                        rew_dict[veh_id] += rew_cfg.get(
                            'shaped_goal_distance_scaling',
                            1.0) * (1 - np.abs(
                                self.angle_sub(veh_obj.heading,
                                               veh_obj.target_heading)) /
                                    (2 * np.pi)) / rew_cfg['reward_scaling']
            '''############################################
                    Handle potential done conditions
            ############################################'''
            # achieved our goal
            if info_dict[veh_id]['goal_achieved'] and self.cfg.get(
                    'remove_at_goal', True):
                done_dict[veh_id] = True
            if veh_obj.getCollided():
                info_dict[veh_id]['collided'] = True
                info_dict[veh_id]['cost'] = 1.0
                if int(veh_obj.collision_type) == 1:
                    info_dict[veh_id]['veh_veh_collision'] = True
                if int(veh_obj.collision_type) == 2:
                    info_dict[veh_id]['veh_edge_collision'] = True
                rew_dict[veh_id] -= np.abs(
                    rew_cfg['collision_penalty']) / rew_cfg['reward_scaling']
                if self.cfg.get('remove_at_collide', True):
                    done_dict[veh_id] = True
            near_miss_threshold = rew_cfg.get('near_miss_threshold', None)
            if near_miss_threshold is not None and not info_dict[veh_id][
                    'collided']:
                min_distance = np.inf
                for other_obj in self.scenario.getVehicles():
                    if other_obj.getID() == veh_id:
                        continue
                    min_distance = min(
                        min_distance, (other_obj.position - obj_pos).norm())
                info_dict[veh_id]['near_miss'] = float(
                    min_distance < near_miss_threshold)
            # remove the vehicle so that its trajectory doesn't continue. This is important
            # in the multi-agent setting.
            if done_dict[veh_id]:
                self.done_ids.append(veh_id)
                if (info_dict[veh_id]['goal_achieved']
                        and self.cfg.get('remove_at_goal', True)) or (
                            info_dict[veh_id]['collided']
                            and self.cfg.get('remove_at_collide', True)):
                    objs_to_remove.append(veh_obj)

        for veh_obj in objs_to_remove:
            self.scenario.removeVehicle(veh_obj)

        if self.cfg['rew_cfg']['shared_reward']:
            total_reward = np.sum([rew_dict[key] for key in rew_dict.keys()])
            rew_dict = {key: total_reward for key in rew_dict.keys()}

        # fill in the missing observations if we should be doing so
        if self.cfg['subscriber']['keep_inactive_agents']:
            # force all vehicles done to be false since they should persist through the episode
            done_dict = {key: False for key in self.all_vehicle_ids}
            for key in self.all_vehicle_ids:
                if key not in obs_dict.keys():
                    obs_dict[key] = self.dead_feat
                    rew_dict[key] = 0.0
                    info_dict[key]['goal_achieved'] = False
                    info_dict[key]['collided'] = False
                    info_dict[key]['veh_veh_collision'] = False
                    info_dict[key]['veh_edge_collision'] = False
                    info_dict[key]['cost'] = 0.0
                    info_dict[key]['near_miss'] = 0.0

        truncated_dict = {key: False for key in done_dict.keys()}
        if self.step_num >= self.episode_length:
            truncated_dict = {key: True for key in done_dict.keys()}

        all_done = True
        for value in done_dict.values():
            all_done *= value
        done_dict['__all__'] = all_done
        truncated_dict['__all__'] = any(truncated_dict.values())

        return obs_dict, rew_dict, done_dict, truncated_dict, info_dict

    def _get_simulation(self):
        """Return (file, Simulation) — from pool working set if enabled."""
        if self._pool_size > 0:
            self._pool_reset_count += 1
            if self._pool_reset_count % self._resample_interval == 0:
                # Replace one random pool entry to maintain diversity
                idx = np.random.randint(self._pool_size)
                self._pool_files[idx] = self.files[np.random.randint(len(self.files))]
            f = self._pool_files[np.random.randint(self._pool_size)]
        else:
            f = self.files[np.random.randint(len(self.files))]
        path = os.path.join(self.cfg['scenario_path'], f)
        return f, Simulation(path, config=get_scenario_dict(self.cfg))

    def reset(self):
        """See superclass."""
        self.t = 0
        self.step_num = 0

        enough_vehicles = False
        # we don't want to initialize scenes with 0 actors after satisfying
        # all the conditions on a scene that we have
        while not enough_vehicles:
            self.file, self.simulation = self._get_simulation()
            self.scenario = self.simulation.getScenario()
            '''##################################################################
                Construct context dictionary of observations that can be used to
                warm up policies by stepping all vehicles as experts.
            #####################################################################'''
            dead_obs = self.get_observation(self.scenario.getVehicles()[0])
            self.dead_feat = -np.ones(
                dead_obs.shape[0] * self.n_frames_stacked)
            # step all the vehicles forward by one second and record their observations as context
            context_len = max(10, self.n_frames_stacked)
            self.context_dict = {
                veh.getID():
                deque([self.dead_feat for _ in range(context_len)],
                      maxlen=context_len)
                for veh in self.scenario.getObjectsThatMoved()
            }
            for veh in self.scenario.getObjectsThatMoved():
                veh.expert_control = True
            for _ in range(10):
                for veh in self.scenario.getObjectsThatMoved():
                    self.context_dict[veh.getID()].append(
                        self.get_observation(veh))
                self.simulation.step(self.cfg['dt'])
            # now hand back control to our actual controllers
            for veh in self.scenario.getObjectsThatMoved():
                veh.expert_control = False

            # remove all the objects that are in collision or are already in goal dist
            # additionally set the objects that have infeasible goals to be experts
            for veh_obj in self.simulation.getScenario().getObjectsThatMoved():
                obj_pos = veh_obj.getPosition()
                obj_pos = np.array([obj_pos.x, obj_pos.y])
                goal_pos = veh_obj.getGoalPosition()
                goal_pos = np.array([goal_pos.x, goal_pos.y])
                '''############################################
                    Remove vehicles at goal
                ############################################'''
                norm = np.linalg.norm(goal_pos - obj_pos)
                if norm < self.cfg['rew_cfg'][
                        'goal_tolerance'] or veh_obj.getCollided():
                    self.scenario.removeVehicle(veh_obj)
                '''############################################
                    Set all vehicles with unachievable goals to be experts
                ############################################'''
                if self.file in self.valid_veh_dict and veh_obj.getID(
                ) in self.valid_veh_dict[self.file]:
                    veh_obj.expert_control = True
            '''############################################
                Pick out the vehicles that we are controlling
            ############################################'''
            # ensure that we have no more than max_num_vehicles are controlled
            temp_vehicles = self.scenario.getObjectsThatMoved()
            np.random.shuffle(temp_vehicles)
            curr_index = 0
            self.controlled_vehicles = []
            self.expert_controlled_vehicles = []
            self.vehicles_to_delete = []
            for vehicle in temp_vehicles:
                # this vehicle was invalid at the end of the 1 second context
                # step so we need to remove it.
                if np.isclose(vehicle.position.x, INVALID_POSITION):
                    self.vehicles_to_delete.append(vehicle)
                # we don't want to include vehicles that had unachievable goals
                # as controlled vehicles
                elif not vehicle.expert_control and curr_index < self.cfg[
                        'max_num_vehicles']:
                    self.controlled_vehicles.append(vehicle)
                    curr_index += 1
                else:
                    self.expert_controlled_vehicles.append(vehicle)
            self.all_vehicle_ids = [
                veh.getID() for veh in self.controlled_vehicles
            ]
            # make all the vehicles that are in excess of max_num_vehicles controlled by an expert
            for veh in self.expert_controlled_vehicles:
                veh.expert_control = True
            # remove vehicles that are currently at an invalid position
            for veh in self.vehicles_to_delete:
                self.scenario.removeVehicle(veh)

            # check that we have at least one vehicle or if we have just one file, exit anyways
            # or else we might be stuck in an infinite loop
            if len(self.all_vehicle_ids) > 0 or len(self.files) == 1:
                enough_vehicles = True

        # for one reason or another (probably we had a file where all the agents achieved their goals)
        # we have no controlled vehicles
        # just grab a vehicle even if it hasn't moved so that we have something
        # to return obs for even if it's not controlled
        # NOTE: this case only occurs during our eval procedure where we set the
        # self.files list to be length 1. Otherwise, the while loop above will repeat
        # until a file is found.
        if len(self.all_vehicle_ids) == 0:
            self.controlled_vehicles = [self.scenario.getVehicles()[0]]
            self.all_vehicle_ids = [
                veh.getID() for veh in self.controlled_vehicles
            ]

        # construct the observations and goal normalizers
        obs_dict = {}
        self.goal_dist_normalizers = {}
        max_goal_dist = -100
        for veh_obj in self.controlled_vehicles:
            veh_id = veh_obj.getID()
            # store normalizers for each vehicle
            obj_pos = veh_obj.getPosition()
            obj_pos = np.array([obj_pos.x, obj_pos.y])
            goal_pos = veh_obj.getGoalPosition()
            goal_pos = np.array([goal_pos.x, goal_pos.y])
            dist = np.linalg.norm(obj_pos - goal_pos)
            self.goal_dist_normalizers[veh_id] = dist
            # compute the obs
            self.context_dict[veh_id].append(self.get_observation(veh_obj))
            if self.n_frames_stacked > 1:
                veh_deque = self.context_dict[veh_id]
                context_list = list(
                    islice(veh_deque,
                           len(veh_deque) - self.n_frames_stacked,
                           len(veh_deque)))
                obs_dict[veh_id] = np.concatenate(context_list)
            else:
                obs_dict[veh_id] = self.context_dict[veh_id][-1]
            # pick the vehicle that has to travel the furthest distance and use it for rendering
            if dist > max_goal_dist:
                # this attribute is just used for rendering of the view
                # from the ego frame
                self.render_vehicle = veh_obj
                max_goal_dist = dist

        self.done_ids = []
        # we should return obs for the missing agents
        if self.cfg['subscriber']['keep_inactive_agents']:
            max_id = max([int(key) for key in obs_dict.keys()])
            num_missing_agents = max(
                0, self.cfg['max_num_vehicles'] - len(obs_dict))
            for i in range(num_missing_agents):
                obs_dict[max_id + i + 1] = self.dead_feat
            self.dead_agent_ids = [
                max_id + i + 1 for i in range(num_missing_agents)
            ]
            self.all_vehicle_ids = list(obs_dict.keys())
        else:
            self.dead_agent_ids = []
        return obs_dict, {}

    def get_observation(self, veh_obj):
        """Return the observation for a particular vehicle."""
        ego_obs = self.scenario.ego_state(veh_obj)
        subscriber_cfg = self.cfg['subscriber']
        use_occlusion_features = subscriber_cfg.get('use_occlusion_features',
                                                    False)
        if use_occlusion_features:
            occ_obs = compute_occlusion_features(
                self.scenario,
                veh_obj,
                subscriber_cfg['view_dist'],
                subscriber_cfg['view_angle'],
                head_angle=veh_obj.head_angle,
            )
        if self.cfg['subscriber']['use_ego_state'] and self.cfg['subscriber'][
                'use_observations']:
            obs_parts = [
                ego_obs,
                self.scenario.flattened_visible_state(
                    veh_obj,
                    view_dist=self.cfg['subscriber']['view_dist'],
                    view_angle=self.cfg['subscriber']['view_angle'],
                    head_angle=veh_obj.head_angle)
            ]
        elif self.cfg['subscriber']['use_ego_state'] and not self.cfg[
                'subscriber']['use_observations']:
            obs_parts = [ego_obs]
        else:
            obs_parts = [
                self.scenario.flattened_visible_state(
                    veh_obj,
                    view_dist=self.cfg['subscriber']['view_dist'],
                    view_angle=self.cfg['subscriber']['view_angle'],
                    head_angle=veh_obj.head_angle)
            ]
        if use_occlusion_features:
            obs_parts.append(occ_obs)
        obs = np.concatenate(obs_parts)
        return obs

    def make_all_vehicles_experts(self):
        """Force all vehicles to be experts."""
        for veh in self.scenario.getVehicles():
            veh.expert_control = True

    def get_vehicles(self):
        """Return the vehicles."""
        return self.scenario.getVehicles()

    def get_objects_that_moved(self):
        """Return the objects that moved."""
        return self.scenario.getObjectsThatMoved()

    def render(self, mode=None):
        """See superclass."""
        return self.scenario.getImage(
            img_width=self.render_img_width,
            img_height=self.render_img_height,
            draw_target_positions=True,
            padding=self.render_padding,
        )

    def render_ego(self, mode=None):
        """See superclass."""
        if self.render_vehicle.getID() in self.done_ids:
            return None
        else:
            return self.scenario.getConeImage(
                source=self.render_vehicle,
                view_dist=self.cfg['subscriber']['view_dist'],
                view_angle=self.cfg['subscriber']['view_angle'],
                head_angle=self.render_vehicle.head_angle,
                img_width=self.render_img_width,
                img_height=self.render_img_height,
                padding=self.render_padding,
                draw_target_position=True,
            )

    def render_features(self, mode=None):
        """See superclass."""
        if self.render_vehicle.getID() in self.done_ids:
            return None
        else:
            return self.scenario.getFeaturesImage(
                source=self.render_vehicle,
                view_dist=self.cfg['subscriber']['view_dist'],
                view_angle=self.cfg['subscriber']['view_angle'],
                head_angle=self.render_vehicle.head_angle,
                img_width=self.render_img_width,
                img_height=self.render_img_height,
                padding=self.render_padding,
                draw_target_position=True,
            )

    def seed(self, seed=None):
        """Ensure determinism."""
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def angle_sub(self, current_angle, target_angle) -> int:
        """Subtract two angles to find the minimum angle between them."""
        # Subtract the angles, constraining the value to [0, 2 * np.pi)
        diff = (target_angle - current_angle) % (2 * np.pi)

        # If we are more than np.pi we're taking the long way around.
        # Let's instead go in the shorter, negative direction
        if diff > np.pi:
            diff = -(2 * np.pi - diff)
        return diff
