"""Episode labels computed from traces the evaluator already records.

Success is reaching the goal without a collision. A low collision rate with a
high timeout rate means the policy stopped instead of driving.
"""

import math
from typing import Optional, Sequence

DT = 0.1
BRAKE_DROP_MPS = 1.0
LATERAL_STEER_RAD = 0.2
TTZ_CONFLICT_S = 3.0
TTZ_FINITE_S = 100.0
CONTACT_PARTNER_DIST_M = 5.0
CONFLICT_WINDOW_STEPS = 20


def classify_outcome(goal: bool, collided: bool, length: int, horizon: int):
    timeout = (not goal) and (not collided) and length >= horizon
    if goal and collided:
        name = "goal_and_collision"
    elif collided:
        name = "collision"
    elif goal:
        name = "goal"
    elif timeout:
        name = "timeout"
    else:
        name = "incomplete"
    return name, bool(goal) and not bool(collided), timeout


def collision_type(collided: bool, veh_veh: bool, veh_edge: bool) -> str:
    if not collided:
        return "none"
    if veh_veh and veh_edge:
        return "vehicle_and_road_edge"
    if veh_edge:
        return "road_edge"
    if veh_veh:
        return "vehicle"
    return "unspecified"


def partner_label(
    kind: str,
    nearest_type: Optional[str],
    nearest_dist: Optional[float],
) -> str:
    """Road-edge comes from the simulator. Object partners use the nearest body."""
    if kind == "road_edge":
        return "road_edge"
    if kind == "none":
        return "none"
    if (
        nearest_type
        and nearest_dist is not None
        and nearest_dist <= CONTACT_PARTNER_DIST_M
    ):
        return nearest_type
    return kind


def value_before(series: Sequence[Optional[float]], steps_back: int) -> Optional[float]:
    if series is None or len(series) <= steps_back:
        return None
    value = series[-1 - steps_back]
    if value is None or value >= TTZ_FINITE_S:
        return None
    return float(value)


def _closest_approach(ttz: Sequence[Optional[float]]):
    finite = [
        (index, float(value))
        for index, value in enumerate(ttz)
        if value is not None and value < TTZ_FINITE_S
    ]
    if not finite:
        return None
    return min(finite, key=lambda item: item[1])[0]


def maneuver_and_lead(
    speeds: Sequence[float],
    steers: Sequence[float],
    ttz: Sequence[Optional[float]],
    dt: float = DT,
):
    """Brake and lateral labels in the two seconds before the closest approach.

    The closest approach is the minimum finite ground-truth time-to-zero.
    Lead time is how long before that step the speed has already fallen by
    1 m/s from its value at the start of the window. A positive lead is an
    anticipatory slowdown.
    """
    n_steps = len(speeds)
    if n_steps == 0:
        return "none", None
    closest = _closest_approach(ttz)
    if closest is None:
        start = max(0, n_steps - CONFLICT_WINDOW_STEPS)
        end = n_steps - 1
    else:
        start = max(0, closest - CONFLICT_WINDOW_STEPS)
        end = closest
    base_speed = float(speeds[start])
    first_brake = None
    for index in range(start, end + 1):
        if base_speed - float(speeds[index]) >= BRAKE_DROP_MPS:
            first_brake = index
            break
    steer_end = min(len(steers), end + 1)
    lateral = any(
        abs(float(steer)) >= LATERAL_STEER_RAD for steer in steers[start:steer_end]
    )
    braking = first_brake is not None
    if braking and lateral:
        label = "both"
    elif braking:
        label = "brake"
    elif lateral:
        label = "lateral"
    else:
        label = "none"
    lead = None
    if first_brake is not None and closest is not None:
        lead = (closest - first_brake) * dt
    return label, lead


def wilson_interval(successes: int, total: int, z: float = 1.96):
    if total <= 0:
        return None, None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
    margin /= denominator
    return centre - margin, centre + margin
