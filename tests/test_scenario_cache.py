# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for bounded Simulation reuse and scenario-pool refill."""
from collections import OrderedDict
from pathlib import Path

import numpy as np

from nocturne import Simulation
from nocturne.envs.base_env import BaseEnv

# Minimal scenario config for loading the fixture via the raw bindings.
_FIXTURE = str(Path(__file__).resolve().parent / "large_file_tfrecord.json")
_SCENARIO_CFG = {
    "start_time": 0,
    "allow_non_vehicles": True,
    "moving_threshold": 0.2,
    "speed_threshold": 0.05,
    "max_visible_objects": 16,
    "max_visible_road_points": 1000,
    "max_visible_traffic_lights": 20,
    "max_visible_stop_signs": 4,
    "sample_every_n": 1,
    "road_edge_first": False,
}


def _tests_dir_cfg():
    """Full BaseEnv config rooted at the tests/ fixture directory."""
    from examples.drl_collision_avoidance.scenario_utils import load_config

    cfg = load_config(
        scenario_path=str(Path(__file__).resolve().parent),
        scenario_split="valid",
    )
    cfg["scenario_pool_size"] = 1
    cfg["scenario_cache_size"] = 2
    cfg["resample_pool_interval"] = 0
    return cfg


def _stub_env(cache_size=2, pool_size=2, files=None):
    env = object.__new__(BaseEnv)
    env.files = list(files or ["a.json", "b.json", "c.json", "d.json"])
    env._pool_size = pool_size
    env._pool_files = list(env.files[:pool_size]) if pool_size > 0 else []
    env._pool_reset_count = 0
    env._resample_interval = 0
    env._cache_size = cache_size
    env._sim_cache = OrderedDict()
    env._bad_files = set()
    env._eval_mode = False
    env._eval_file_override = None
    env._eval_scene_status = "ok"
    env.valid_veh_dict = {name: [] for name in env.files}
    return env


def test_store_simulation_evicts_oldest():
    env = _stub_env(cache_size=2)
    env._store_simulation("a.json", object())
    env._store_simulation("b.json", object())
    env._store_simulation("c.json", object())
    assert list(env._sim_cache.keys()) == ["b.json", "c.json"]
    assert len(env._sim_cache) == 2


def test_cache_hit_moves_entry_to_end():
    env = _stub_env(cache_size=2)
    env._store_simulation("a.json", "sim-a")
    env._store_simulation("b.json", "sim-b")
    env._sim_cache.move_to_end("a.json")
    env._store_simulation("c.json", "sim-c")
    assert list(env._sim_cache.keys()) == ["a.json", "c.json"]


def test_zero_cache_size_does_not_store():
    env = _stub_env(cache_size=0)
    env._store_simulation("a.json", object())
    assert env._sim_cache == OrderedDict()


def test_blacklist_refills_pool_and_drops_cache():
    env = _stub_env(cache_size=4, pool_size=2, files=["a.json", "b.json", "c.json"])
    env._store_simulation("a.json", object())
    env._pool_files = ["a.json", "a.json"]
    env._blacklist_file("a.json")
    assert "a.json" not in env.files
    assert "a.json" not in env._pool_files
    assert "a.json" not in env._sim_cache
    assert len(env._pool_files) == 2
    assert set(env._pool_files) <= set(env.files)


def test_pick_scenario_file_uses_current_pool_length():
    env = _stub_env(pool_size=3, files=["a.json", "b.json"])
    env._pool_files = ["a.json"]
    chosen = {env._pick_scenario_file() for _ in range(20)}
    assert chosen <= set(env.files)
    assert len(env._pool_files) == 3


def test_close_clears_cache():
    env = _stub_env()
    env._store_simulation("a.json", object())
    env.simulation = object()
    env.scenario = object()
    env.close()
    assert env._sim_cache == OrderedDict()
    assert env.simulation is None
    assert env.scenario is None


def test_restore_snapshot_reinserts_removed_objects():
    """Regression test for cached-scenario depopulation.

    ``removeVehicle`` erases objects from the scenario's object containers, so
    a cached Simulation reused across episodes must get the removed objects
    back from the snapshot — otherwise scenarios slowly lose their vehicles
    and every reset becomes a degenerate episode.
    """
    sim = Simulation(scenario_path=_FIXTURE, config=dict(_SCENARIO_CFG))
    scenario = sim.getScenario()
    sim.step(0.1)
    scenario.save_snapshot()

    ids_before = sorted(obj.getID() for obj in scenario.getObjects())
    moving_before = sorted(obj.getID()
                           for obj in scenario.getObjectsThatMoved())
    vehicles_before = len(scenario.getVehicles())
    positions_before = {
        obj.getID(): (obj.getPosition().x, obj.getPosition().y)
        for obj in scenario.getObjects()
    }
    assert len(ids_before) > 1
    assert moving_before

    # Two remove/restore cycles, as a cached Simulation would see across
    # consecutive resets of the same scenario file.
    for idx in range(2):
        victim = scenario.getObjects()[idx]
        victim_id = victim.getID()
        victim.setPosition(9999.0, 9999.0)
        assert scenario.removeVehicle(victim)
        assert victim_id not in {obj.getID() for obj in scenario.getObjects()}

        scenario.restore_snapshot()
        after = {obj.getID(): obj for obj in scenario.getObjects()}
        assert sorted(after) == ids_before
        assert len(scenario.getVehicles()) == vehicles_before
        assert sorted(obj.getID()
                      for obj in scenario.getObjectsThatMoved()) == moving_before
        x_before, y_before = positions_before[victim_id]
        pos = after[victim_id].getPosition()
        assert np.isclose(pos.x, x_before)
        assert np.isclose(pos.y, y_before)


def test_cached_env_resets_preserve_and_restore_objects():
    """Env-level regression for cached-scenario depopulation.

    Resetting the same scenario through the Simulation cache must yield the
    same object pool every time, and a vehicle removed the way the env
    removes it at goal/collision must be back on the next reset.
    """
    cfg = _tests_dir_cfg()
    env = BaseEnv(cfg)
    try:
        counts = []
        for _ in range(3):
            env.reset(build_obs=False)
            counts.append(len(env.scenario.getObjects()))
            assert len(env.all_vehicle_ids) > 0
        assert len(set(counts)) == 1, counts

        victim = None
        for obj in env.scenario.getObjects():
            pos = obj.getPosition()
            goal = obj.getGoalPosition()
            dist = ((goal.x - pos.x) ** 2 + (goal.y - pos.y) ** 2) ** 0.5
            if dist > 15.0:
                victim = obj
                break
        assert victim is not None, "fixture should have a vehicle far from goal"
        victim_id = victim.getID()
        assert env.scenario.removeVehicle(victim)
        assert victim_id not in {o.getID() for o in env.scenario.getObjects()}

        env.reset(build_obs=False)
        after = {o.getID() for o in env.scenario.getObjects()}
        assert victim_id in after
        assert len(after) == counts[-1]
    finally:
        env.close()


def test_reset_removes_vehicles_inside_success_radius():
    """No controlled vehicle may start inside the goal-success radius.

    Reset-time removal must use the same tolerance as the success check in
    step() (position_target_tolerance), otherwise episodes that start 2-5 m
    from the goal are solved on the first step — the "trivial goals" bug.
    """
    cfg = _tests_dir_cfg()
    env = BaseEnv(cfg)
    tol = float(cfg["rew_cfg"]["position_target_tolerance"])
    try:
        for _ in range(2):
            env.reset(build_obs=False)
            for veh in env.controlled_vehicles:
                pos = veh.getPosition()
                goal = veh.getGoalPosition()
                dist = ((goal.x - pos.x) ** 2 + (goal.y - pos.y) ** 2) ** 0.5
                assert dist >= tol, (veh.getID(), dist)
    finally:
        env.close()


def test_deterministic_eval_pins_scene_and_ego():
    """Deterministic eval mode selects the same ego for the same scene."""
    cfg = _tests_dir_cfg()
    cfg["deterministic_eval"] = True
    cfg["scenario_pool_size"] = 0
    cfg["scenario_cache_size"] = 0
    env = BaseEnv(cfg)
    try:
        selections = []
        for _ in range(2):
            env._eval_file_override = "large_file_tfrecord.json"
            env.reset(build_obs=False)
            assert env.file == "large_file_tfrecord.json"
            assert env._eval_scene_status == "ok"
            assert len(env.all_vehicle_ids) > 0
            selections.append(tuple(env.all_vehicle_ids))
        assert selections[0] == selections[1]
    finally:
        env.close()
