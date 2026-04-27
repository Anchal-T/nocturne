import hashlib
import os
import subprocess
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from cfgs.config import PROCESSED_TRAIN_NO_TL, PROCESSED_VALID_NO_TL

_SUPPORTED_SPLITS = ('train', 'valid')
_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'cfgs', 'drl_collision_avoidance', 'config.yaml'
)

# Per-process cache: S3 URI → local path. Prevents redundant syncs when
# multiple envs in the same worker process share an S3 scenario_path.
_s3_sync_cache: Dict[str, str] = {}


def _default_path_for_split(split: str) -> str:
    if split == 'train':
        return os.environ.get('PROCESSED_TRAIN_NO_TL', PROCESSED_TRAIN_NO_TL)
    if split == 'valid':
        return os.environ.get('PROCESSED_VALID_NO_TL', PROCESSED_VALID_NO_TL)
    raise ValueError(
        f'Unsupported scenario_split={split!r}. '
        f'Expected one of {_SUPPORTED_SPLITS}.'
    )


def _is_s3_path(path: str) -> bool:
    return path.startswith('s3://')


def _sync_s3_to_local(s3_uri: str) -> str:
    """Sync an S3 prefix to a local temp directory and return the local path.

    Safe to call concurrently from multiple Ray actors — aws s3 sync is
    idempotent and each URI maps to a unique cache directory via SHA-256 hash.
    Each actor process syncs at most once per URI thanks to _s3_sync_cache.
    """
    if s3_uri in _s3_sync_cache:
        return _s3_sync_cache[s3_uri]
    cache_key = hashlib.sha256(s3_uri.encode()).hexdigest()[:16]
    local_dir = os.path.join('/tmp', 'nocturne_s3_cache', cache_key)
    os.makedirs(local_dir, exist_ok=True)
    try:
        subprocess.run(
            ['aws', 's3', 'sync', s3_uri.rstrip('/') + '/', local_dir + '/', '--no-progress'],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ''
        stdout = exc.stdout.strip() if exc.stdout else ''
        details = stderr or stdout or 'No subprocess output captured.'
        raise RuntimeError(
            f'Failed to sync S3 scenario path {s3_uri!r} to local cache {local_dir!r}: '
            f'{details}'
        ) from exc
    _s3_sync_cache[s3_uri] = local_dir
    return local_dir


def resolve_scenario_path(
    scenario_path: Optional[str],
    scenario_split: str = 'train',
) -> str:
    split = str(scenario_split).lower()
    resolved_path = scenario_path or _default_path_for_split(split)

    # S3 URIs are synced to local per-worker at env creation time (in _make_env_fn).
    # Pass them through here without filesystem checks.
    if _is_s3_path(resolved_path):
        return resolved_path

    resolved_path = os.path.abspath(os.path.expanduser(str(resolved_path)))

    if not os.path.isdir(resolved_path):
        raise FileNotFoundError(
            f'Scenario directory does not exist: {resolved_path}. '
            f'Pass --scenario_path explicitly or set scenario_split to one of '
            f'{_SUPPORTED_SPLITS}.'
        )

    valid_files_path = os.path.join(resolved_path, 'valid_files.json')
    if not os.path.isfile(valid_files_path):
        raise FileNotFoundError(
            f'Missing valid_files.json in scenario directory: {resolved_path}. '
            'Nocturne BaseEnv requires this file.'
        )

    return resolved_path


def apply_scenario_path_defaults(
    cfg: Dict[str, Any],
    default_split: str = 'train',
) -> Dict[str, Any]:
    split = str(cfg.get('scenario_split', default_split)).lower()
    cfg['scenario_split'] = split
    cfg['scenario_path'] = resolve_scenario_path(cfg.get('scenario_path'), split)
    return cfg


def load_config(
    scenario_path: Optional[str] = None,
    scenario_split: str = 'valid',
    num_files: int = -1,
    **overrides,
) -> Dict[str, Any]:
    cfg = OmegaConf.load(_CONFIG_PATH)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if scenario_split:
        cfg_dict['scenario_split'] = scenario_split
    if scenario_path:
        cfg_dict['scenario_path'] = scenario_path
    if num_files != -1:
        cfg_dict['num_files'] = num_files

    for key, value in overrides.items():
        cfg_dict[key] = value

    return apply_scenario_path_defaults(cfg_dict, default_split=scenario_split)

