# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for bounded Simulation reuse and scenario-pool refill."""
from collections import OrderedDict

from nocturne.envs.base_env import BaseEnv


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
