"""Sensor-feasible occlusion features for Nocturne observations."""

import math

import numpy as np


OCCLUSION_FEATURE_SIZE = 4


def _interval_union_length(intervals):
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def compute_occlusion_features(scenario, ego_veh, view_dist, view_angle,
                               head_angle=0.0):
    """Return non-privileged occlusion geometry features.

    The features are computed only from objects that are already visible to the
    ego vehicle. They summarize how much of the sensor cone is blocked by those
    visible objects and avoid exposing labels for truly hidden vehicles.
    """
    visible = scenario.visible_objects_state(
        ego_veh,
        view_dist=view_dist,
        view_angle=view_angle,
        head_angle=head_angle,
        padding=False,
    )
    return compute_occlusion_features_from_objects(
        visible, view_dist, view_angle)


def compute_occlusion_features_from_objects(
    objects: np.ndarray,
    view_dist: float,
    view_angle: float,
) -> np.ndarray:
    """Compute occlusion features from a pre-computed visible-objects array.

    This avoids a redundant C++ visibility pass when the caller has already
    obtained the visible objects (e.g. from a ``visible_state`` call that was
    also used to build the flattened observation).
    """
    if objects is None or len(objects) == 0:
        return np.zeros(OCCLUSION_FEATURE_SIZE, dtype=np.float32)

    intervals = []
    nearest_blocker = view_dist
    blocker_count = 0
    half_view = view_angle * 0.5
    for obj in objects:
        if obj[0] <= 0.0:
            continue
        distance = max(float(obj[1]), 1e-3)
        azimuth = float(obj[2])
        length = max(float(obj[3]), 0.0)
        width = max(float(obj[4]), 0.0)
        radius = 0.5 * math.hypot(length, width)
        angular_half_width = min(math.atan2(radius, distance), half_view)
        start = max(-half_view, azimuth - angular_half_width)
        end = min(half_view, azimuth + angular_half_width)
        if end <= start:
            continue
        intervals.append((start, end))
        nearest_blocker = min(nearest_blocker, distance)
        blocker_count += 1

    occluded_fraction = _interval_union_length(intervals) / max(view_angle, 1e-6)
    nearest_blocker_norm = nearest_blocker / max(view_dist, 1e-6)
    blocker_density = blocker_count / max(len(objects), 1)
    occlusion_pressure = occluded_fraction * (1.0 - nearest_blocker_norm)

    return np.array([
        occluded_fraction,
        nearest_blocker_norm,
        blocker_density,
        occlusion_pressure,
    ],
                    dtype=np.float32)
