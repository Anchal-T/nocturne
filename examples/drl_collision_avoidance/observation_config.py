"""Observation/reward switch contract shared by training and evaluation.

The arm matrix in ``paper/workspace/phase2_arms_and_compute.md`` is defined by
these switches. Every checkpoint stores the resulting metadata so evaluation
never has to infer the arm from ``obs_dim`` (A and B are both 609-D).
"""

from typing import Any, Dict

# Keys that must be present in checkpoint observation_metadata.
OBSERVATION_METADATA_KEYS = (
    'obs_dim',
    'visibility_filter',
    'use_occlusion',
    'occlusion_mask',
    'history_frames',
    'history_stride',
    'reward_ttz_source',
    'grid_rows',
    'grid_cols',
    'grid_channels',
)


def observation_metadata(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Derive the observation/reward metadata from a DRL config dict."""
    grid_cfg = cfg['occupancy_grid']
    rows = int(grid_cfg['rows'])
    cols = int(grid_cfg['cols'])
    history_frames = max(1, int(grid_cfg.get('history_frames', 1)))
    grid_channels = 3 * history_frames
    use_occlusion = bool(grid_cfg.get('use_occlusion', False))
    occlusion_mask = bool(grid_cfg.get('occlusion_mask', False))
    obs_dim = grid_channels * rows * cols + 4 + 2 + 3
    if use_occlusion:
        obs_dim += 4
    if occlusion_mask:
        obs_dim += rows * cols

    rew_cfg = cfg.get('reward', {})
    reward_ttz_source = str(
        rew_cfg.get('reward_ttz_source', 'privileged')).lower()
    if reward_ttz_source not in ('privileged', 'visible'):
        raise ValueError(
            "reward.reward_ttz_source must be 'privileged' or 'visible', "
            f"got {reward_ttz_source!r}")

    return {
        'obs_dim': obs_dim,
        'visibility_filter': bool(grid_cfg.get('visibility_filter', False)),
        'use_occlusion': use_occlusion,
        'occlusion_mask': occlusion_mask,
        'history_frames': history_frames,
        'history_stride': max(1, int(grid_cfg.get('history_stride', 1))),
        'reward_ttz_source': reward_ttz_source,
        'grid_rows': rows,
        'grid_cols': cols,
        'grid_channels': grid_channels,
    }
