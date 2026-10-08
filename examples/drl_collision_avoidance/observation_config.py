"""Observation-flag resolution and the held-out file split.

The split functions stay here so tests can import them without OmegaConf.

`use_occlusion: true` on older configs couples three behaviors: a visibility
filter on time-to-zero features, four occlusion summaries, and a C++ occupancy
grid that does not receive the visibility set. Explicit `visibility_filter`
and `occlusion_features` keys turn that coupling off so an ablation can change
one input at a time. The C++ grid is then skipped and the Python stamper
receives only visible objects.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ObservationFlags:
    visibility_filter: bool
    occlusion_features: bool
    legacy_unfiltered_grid: bool


def _optional_bool(cfg: Dict[str, Any], key: str) -> Optional[bool]:
    if key not in cfg or cfg[key] is None:
        return None
    return bool(cfg[key])


def resolve_observation_flags(grid_cfg: Dict[str, Any]) -> ObservationFlags:
    legacy = bool(grid_cfg.get("use_occlusion", False))
    visibility = _optional_bool(grid_cfg, "visibility_filter")
    features = _optional_bool(grid_cfg, "occlusion_features")
    if visibility is None and features is None:
        return ObservationFlags(legacy, legacy, legacy)
    return ObservationFlags(
        legacy if visibility is None else visibility,
        legacy if features is None else features,
        False,
    )


def observation_metadata(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Scalars stored in a checkpoint so evaluation can rebuild the observation."""
    flags = resolve_observation_flags(cfg.get("occupancy_grid", {}))
    reward = cfg.get("reward", {})
    source = reward.get("reward_ttz_source") or "visible"
    return {
        "visibility_filter": flags.visibility_filter,
        "occlusion_features": flags.occlusion_features,
        "legacy_unfiltered_grid": flags.legacy_unfiltered_grid,
        "reward_ttz_source": str(source),
        "ttz_reward_scale": float(reward.get("ttz_reward_scale", 0.0)),
        "ttz_safe_threshold": float(reward.get("ttz_safe_threshold", 3.0)),
    }


def heldout_bucket(filename: str, val_fraction: float = 0.5, seed: int = 42) -> str:
    """Map a scenario filename to ``val`` or ``test`` with a stable hash.

    The Waymo training directory stays the training set. This split only
    partitions the held-out directory so checkpoints can be chosen on ``val``
    and reported on ``test``.
    """
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    digest = hashlib.sha256(f"{int(seed)}\n{filename}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "val" if value < float(val_fraction) else "test"


def split_filenames(filenames, val_fraction: float = 0.5, seed: int = 42):
    val, test = [], []
    for name in filenames:
        if heldout_bucket(name, val_fraction, seed) == "val":
            val.append(name)
        else:
            test.append(name)
    return val, test


def head_parameter_delta(
    extra_features: int,
    head_dim: int = 256,
    dueling: bool = True,
    noisy: bool = True,
) -> int:
    """Trainable parameters added when the observation grows by `extra_features`.

    Those features bypass the grid encoder and enter the dueling heads. A
    NoisyLinear stores a mean and a standard-deviation matrix, so each head
    contributes ``2 * head_dim * extra_features`` weights. Biases do not depend
    on the input width.
    """
    heads = 2 if dueling else 1
    matrices = 2 if noisy else 1
    return heads * matrices * head_dim * extra_features
