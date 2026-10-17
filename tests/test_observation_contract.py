"""Regression tests for the Phase 1b observation/reward contract.

These pin the arm matrix in
``paper/workspace/phase2_arms_and_compute.md``: visibility filtering reaches
the grid, the reward TTZ source is independent of the observation source,
grid history is grid-only, and the evaluator never guesses observation
switches.
"""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.drl_collision_avoidance.collision_avoidance_env import (
    CollisionAvoidanceEnv,
)
from examples.drl_collision_avoidance.evaluate import (
    _apply_checkpoint_observation_settings,
    _checkpoint_observation_metadata,
)
from examples.drl_collision_avoidance.observation_config import (
    observation_metadata,
)
from examples.drl_collision_avoidance.scenario_utils import (
    load_config,
    load_valid_files,
)

TESTS_DIR = Path(__file__).resolve().parent
FIXTURE = "large_file_tfrecord.json"
SCENARIO_CFG = {
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

# name -> (visibility_filter, use_occlusion, occlusion_mask, history_frames,
#          history_stride, reward_ttz_source, expected obs_dim)
ARMS = {
    "A": (False, False, False, 1, 1, "privileged", 609),
    "B": (True, False, False, 1, 1, "privileged", 609),
    "C": (True, True, False, 1, 1, "privileged", 613),
    "C_vis": (True, True, False, 1, 1, "visible", 613),
    "D": (True, False, True, 1, 1, "privileged", 809),
    "D_vis": (True, False, True, 1, 1, "visible", 809),
    "O": (True, True, True, 1, 1, "privileged", 813),
    "C_hist": (True, False, False, 3, 5, "privileged", 1809),
}


def _cfg(arm):
    vis, stats, mask, hf, hs, rew, _ = ARMS[arm]
    cfg = load_config(scenario_path=str(TESTS_DIR),
                      scenario_split="valid")
    cfg["scenario_pool_size"] = 0
    cfg["scenario_cache_size"] = 0
    cfg["deterministic_eval"] = True
    cfg["seed"] = 42
    cfg["occupancy_grid"] = dict(cfg["occupancy_grid"])
    cfg["reward"] = dict(cfg["reward"])
    cfg["occupancy_grid"].update(
        visibility_filter=vis,
        use_occlusion=stats,
        occlusion_mask=mask,
        history_frames=hf,
        history_stride=hs,
    )
    cfg["reward"]["reward_ttz_source"] = rew
    return cfg


@pytest.mark.parametrize("arm", list(ARMS))
def test_arm_observation_dimensions(arm):
    cfg = _cfg(arm)
    meta = observation_metadata(cfg)
    assert meta["obs_dim"] == ARMS[arm][-1]
    assert meta["grid_channels"] == 3 * ARMS[arm][3]
    env = CollisionAvoidanceEnv(cfg)
    assert env.observation_space.shape[0] == ARMS[arm][-1]
    assert env.obs_dim == ARMS[arm][-1]


def test_visibility_filtered_grid_is_subset_of_privileged():
    env_a = CollisionAvoidanceEnv(_cfg("A"))
    env_b = CollisionAvoidanceEnv(_cfg("B"))
    for env in (env_a, env_b):
        env.set_eval_scene(FIXTURE)
    obs_a, info_a = env_a.reset()
    obs_b, info_b = env_b.reset()
    assert info_a["ego_id"] == info_b["ego_id"]

    grid_a = obs_a[:600].reshape(3, 20, 10)
    grid_b = obs_b[:600].reshape(3, 20, 10)
    assert np.all(grid_a[0] >= grid_b[0])

    # The env must pass its ray-cast visible ids to the C++ filter.
    scenario = env_b.base_env.scenario
    ego = env_b._get_ego_vehicle()
    visible_ids = []
    for key in ("vehicles", "peds", "cyclists"):
        visible_ids.extend(env_b._visible_set.get(key, ()))
    direct = scenario.occupancy_grid_visible(
        ego=ego, rows=20, cols=10, forward_dist=40.0, backward_dist=5.0,
        lateral_dist=10.0, vehicle_weight=1.0, vru_weight=2.0,
        road_edge_weight=3.0, visible_ids=visible_ids,
        ego_id=env_b._ego_id,
    )
    assert np.array_equal(direct, grid_b)


def test_cpp_grid_visible_removes_actor_cells():
    from nocturne import Scenario

    scenario = Scenario(str(TESTS_DIR / FIXTURE), SCENARIO_CFG)
    ego = scenario.getVehicles()[0]
    kwargs = dict(
        ego=ego, rows=20, cols=10, forward_dist=40.0, backward_dist=5.0,
        lateral_dist=10.0, vehicle_weight=1.0, vru_weight=2.0,
        road_edge_weight=3.0, ego_id=ego.getID(),
    )
    full = scenario.occupancy_grid(**kwargs)
    empty = scenario.occupancy_grid_visible(visible_ids=[], **kwargs)
    assert np.all(full[0] >= empty[0])
    assert full.sum() > empty.sum(), (
        "empty visible set should remove at least one actor cell")


def test_reward_source_is_decoupled_from_observation():
    env_c = CollisionAvoidanceEnv(_cfg("C"))
    env_cv = CollisionAvoidanceEnv(_cfg("C_vis"))
    for env in (env_c, env_cv):
        env.set_eval_scene(FIXTURE)
    obs_c, _ = env_c.reset()
    obs_cv, _ = env_cv.reset()
    assert np.array_equal(obs_c, obs_cv)

    for _ in range(20):
        obs_c, _, done_c, trunc_c, info_c = env_c.step(10)
        obs_cv, _, done_cv, trunc_cv, info_cv = env_cv.step(10)
        # Same observation switches => identical observations.
        assert np.array_equal(obs_c, obs_cv)
        # C uses the privileged reward; C_vis uses the observed (visible) TTZ.
        assert info_c["ttz_vehicle_reward"] == pytest.approx(
            info_c["ttz_vehicle"])
        assert info_cv["ttz_vehicle_reward"] == pytest.approx(
            info_cv["ttz_vehicle_observed"])
        if done_c or trunc_c or done_cv or trunc_cv:
            break


def test_occlusion_mask_geometry():
    env = object.__new__(CollisionAvoidanceEnv)
    env.grid_rows, env.grid_cols = 20, 10
    env.forward_dist, env.backward_dist, env.lateral_dist = 40.0, 5.0, 10.0
    env.occlusion_view_angle = math.pi
    env._grid_cell_azimuth, env._grid_cell_range = \
        env._build_grid_cell_geometry()

    # Inverse of the C++ cell mapping.
    cell_long = (env.forward_dist + env.backward_dist) / env.grid_rows
    cell_lat = (2.0 * env.lateral_dist) / env.grid_cols
    for r, c in ((0, 0), (0, 9), (19, 0), (19, 9), (10, 5)):
        lx = env.forward_dist - (r + 0.5) * cell_long
        ly = -env.lateral_dist + (c + 0.5) * cell_lat
        assert int((env.forward_dist - lx) // cell_long) == r
        assert int((ly + env.lateral_dist) // cell_lat) == c
        assert env._grid_cell_azimuth[r, c] == pytest.approx(
            math.atan2(ly, lx), abs=1e-5)
        assert env._grid_cell_range[r, c] == pytest.approx(
            math.hypot(lx, ly), abs=1e-4)

    # One 1 m-radius blocker 10 m straight ahead.
    obj = np.array([[1.0, 10.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0,
                     1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    mask = env._compute_occlusion_mask(obj).reshape(20, 10)
    half_width = math.atan2(math.hypot(1.0, 1.0), 10.0)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert mask.sum() > 0
    assert np.all(mask[env._grid_cell_range <= 10.0] == 0.0)
    wedge_beyond = (
        (np.abs(env._grid_cell_azimuth) <= half_width) &
        (env._grid_cell_range > 10.0)
    )
    assert np.all(mask[wedge_beyond] == 1.0)
    assert np.all(mask[~wedge_beyond] == 0.0)
    assert env._compute_occlusion_mask(
        np.zeros((0, 13), dtype=np.float32)).sum() == 0.0


def test_grid_history_padding_and_stride():
    env = object.__new__(CollisionAvoidanceEnv)
    env.history_frames = 3
    env.history_stride = 2
    env._grid_history = []
    env._step_count = 0

    def grid(value):
        return np.full((3, 2, 2), float(value), dtype=np.float32)

    out0 = env._stack_grid_history(grid(0))
    assert out0.shape == (9, 2, 2)
    assert np.array_equal(out0[:3], grid(0))
    assert np.all(out0[3:] == 0.0)

    env._step_count = 2
    out2 = env._stack_grid_history(grid(2))
    assert np.array_equal(out2[3:6], grid(0))
    assert np.all(out2[6:] == 0.0)

    env._step_count = 4
    out4 = env._stack_grid_history(grid(4))
    assert np.array_equal(out4[3:6], grid(2))
    assert np.array_equal(out4[6:], grid(0))

    # Not a snapshot step: the deque must not change.
    env._step_count = 5
    out5 = env._stack_grid_history(grid(5))
    assert np.array_equal(out5[3:6], grid(4))
    assert np.array_equal(out5[6:], grid(2))


def _minimal_obs_cfg():
    return {
        "occupancy_grid": {
            "rows": 20, "cols": 10,
            "forward_dist": 40.0, "backward_dist": 5.0,
            "lateral_dist": 10.0,
            "visibility_filter": False, "use_occlusion": False,
            "occlusion_mask": False, "history_frames": 1,
            "history_stride": 1,
        },
        "reward": {"reward_ttz_source": "privileged"},
    }


def test_checkpoint_metadata_roundtrip_and_refusal(tmp_path):
    meta = observation_metadata(_cfg("O"))
    good = tmp_path / "good.pth"
    torch.save({"obs_dim": meta["obs_dim"],
                "observation_metadata": meta}, good)
    assert _checkpoint_observation_metadata(str(good)) == meta

    cfg = _minimal_obs_cfg()
    _apply_checkpoint_observation_settings(cfg, str(good), None)
    assert cfg["occupancy_grid"]["visibility_filter"] is True
    assert cfg["occupancy_grid"]["use_occlusion"] is True
    assert cfg["occupancy_grid"]["occlusion_mask"] is True
    assert cfg["reward"]["reward_ttz_source"] == "privileged"

    bad = tmp_path / "bad.pth"
    torch.save({"obs_dim": meta["obs_dim"]}, bad)
    with pytest.raises(RuntimeError, match="observation_metadata"):
        _checkpoint_observation_metadata(str(bad))


def test_agent_checkpoint_contains_observation_metadata(tmp_path):
    from examples.drl_collision_avoidance.train import _build_agent

    cfg = _cfg("O")
    cfg["device"] = "cpu"
    cfg["drl"] = dict(cfg["drl"])
    cfg["drl"]["hidden_layers"] = [8, 8, 8, 8]
    meta = observation_metadata(cfg)
    agent = _build_agent(cfg, cfg["drl"], meta["obs_dim"], 20)
    path = tmp_path / "agent.pth"
    agent.save(str(path))

    checkpoint = torch.load(path, map_location="cpu")
    assert checkpoint["observation_metadata"] == meta
    assert checkpoint["grid_channels"] == meta["grid_channels"]
