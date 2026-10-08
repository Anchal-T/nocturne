"""Pure checks for the observation split, episode labels, and parameter delta."""

from examples.drl_collision_avoidance.episode_metrics import (
    classify_outcome,
    maneuver_and_lead,
    value_before,
    wilson_interval,
)
from examples.drl_collision_avoidance.observation_config import (
    head_parameter_delta,
    heldout_bucket,
    resolve_observation_flags,
    split_filenames,
)


def test_legacy_occlusion_flag_stays_coupled():
    flags = resolve_observation_flags({"use_occlusion": True, "visibility_filter": None})
    assert flags.visibility_filter is True
    assert flags.occlusion_features is True
    assert flags.legacy_unfiltered_grid is True


def test_explicit_flags_filter_the_grid():
    flags = resolve_observation_flags({
        "use_occlusion": True,
        "visibility_filter": True,
        "occlusion_features": False,
    })
    assert flags.visibility_filter is True
    assert flags.occlusion_features is False
    assert flags.legacy_unfiltered_grid is False


def test_timeout_is_not_success():
    name, success, timeout = classify_outcome(False, False, 80, 80)
    assert name == "timeout"
    assert success is False
    assert timeout is True
    _, success, timeout = classify_outcome(True, False, 20, 80)
    assert success is True and timeout is False
    _, success, _ = classify_outcome(True, True, 20, 80)
    assert success is False


def test_brake_lead_is_positive_when_slowdown_precedes_min_ttz():
    speeds = [10, 10, 8, 6, 6]
    steers = [0, 0, 0.7, 0, 0]
    ttz = [5, 4, 2, 1, 3]
    label, lead = maneuver_and_lead(speeds, steers, ttz, dt=0.1)
    assert label == "both"
    assert lead == 0.1
    assert value_before(ttz, 2) == 2


def test_split_is_stable_and_covers_every_file():
    names = [f"scenario_{i}.json" for i in range(40)]
    first, _ = split_filenames(names, val_fraction=0.5, seed=42)
    again, _ = split_filenames(names, val_fraction=0.5, seed=42)
    assert first == again
    assert heldout_bucket(names[0], 0.5, 42) in ("val", "test")
    val, test = split_filenames(names, 0.5, 42)
    assert set(val).isdisjoint(test)
    assert set(val) | set(test) == set(names)


def test_occlusion_features_add_4096_parameters():
    assert head_parameter_delta(4, head_dim=256, dueling=True, noisy=True) == 4096


def test_wilson_interval_contains_the_rate():
    low, high = wilson_interval(77, 1500)
    assert low < 77 / 1500 < high
