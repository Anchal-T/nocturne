import os
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from cfgs.config import PROCESSED_TRAIN_NO_TL, PROCESSED_VALID_NO_TL

_SUPPORTED_SPLITS = ('train', 'valid')
_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'cfgs', 'drl_collision_avoidance', 'config.yaml'
)


def _default_path_for_split(split: str) -> str:
    if split == 'train':
        return os.environ.get('PROCESSED_TRAIN_NO_TL', PROCESSED_TRAIN_NO_TL)
    if split == 'valid':
        return os.environ.get('PROCESSED_VALID_NO_TL', PROCESSED_VALID_NO_TL)
    raise ValueError(
        f'Unsupported scenario_split={split!r}. '
        f'Expected one of {_SUPPORTED_SPLITS}.'
    )


def resolve_scenario_path(
    scenario_path: Optional[str],
    scenario_split: str = 'train',
) -> str:
    split = str(scenario_split).lower()
    resolved_path = scenario_path or _default_path_for_split(split)
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

