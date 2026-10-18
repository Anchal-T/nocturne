"""Deterministic evaluation for collision-avoidance policies.

Every episode is an explicit (scene_file, ego_id) pair drawn from a
hash-split of the scenario directory's ``valid_files.json``. The environment
runs in deterministic eval mode (fixed scene order, ego chosen by a pure
function of the scene) and the policy runs with noise disabled
(``DDQNAgent.set_eval_mode``), so the exact same episodes can be replayed for
every model arm. Each run writes a JSON file with full provenance
(checkpoint hash, scene-list hash, config, code state), which is the only
accepted input for downstream tables and figures.

Usage:
    python -m examples.drl_collision_avoidance.evaluate \
        --checkpoint checkpoints/.../model_final.pth \
        --scenario_path dataset/full_valid/formatted_json_v2_no_tl_valid \
        --split test --split_seed 0 --num_files 200 \
        --method ddqn_occlusion \
        --output_json "paper/runs data/eval/dqn_occlusion_test.json" \
        --log_steps
"""
import argparse
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cfgs.config import set_display_window
from examples.drl_collision_avoidance.agent_utils import build_cpu_agent
from examples.drl_collision_avoidance.scenario_utils import (
    hash_split_files,
    load_config,
    load_valid_files,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# BaseEnv.reset() advances the simulation by this many expert-controlled
# steps before handing control to the policy, so an action at loop index t is
# taken at scenario time start_time + WARMUP_STEPS + t (same convention as
# train_bc.py).
WARMUP_STEPS = 10

# Checkpoint metadata copied verbatim into the provenance block.
_CHECKPOINT_META_KEYS = (
    'train_steps', 'epsilon', 'obs_dim', 'n_actions', 'dueling', 'noisy',
    'bc_model', 'hidden_layers', 'grid_size', 'grid_rows', 'grid_cols',
    'grid_channels', 'mlp_depth', 'world_size', 'use_muon',
)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_list(items: List[str]) -> str:
    return hashlib.sha256('\n'.join(items).encode()).hexdigest()


def _git_state() -> Dict[str, Any]:
    try:
        head = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
        lines = status.splitlines() if status else []
        return {
            'git_head': head,
            'git_dirty': bool(lines),
            # Record the paths so an auditor can distinguish stale untracked
            # artifacts from a modified evaluator or environment.
            'git_status': lines[:100],
            'git_status_truncated': len(lines) > 100,
        }
    except Exception as exc:  # pragma: no cover - best effort only
        return {'git_head': None, 'git_dirty': None, 'git_error': str(exc)}


def _torch_load_cpu(path: str):
    """torch.load that tolerates torch versions without ``weights_only``."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _jsonable(value):
    """Convert numpy scalars/arrays so results always serialize as JSON."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _checkpoint_metadata(checkpoint_path: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    try:
        checkpoint = _torch_load_cpu(checkpoint_path)
    except Exception as exc:  # pragma: no cover - best effort only
        return {'load_error': str(exc)}
    if isinstance(checkpoint, dict):
        for key in _CHECKPOINT_META_KEYS:
            value = checkpoint.get(key)
            if key in checkpoint and not isinstance(value, dict):
                if key == 'hidden_layers' and value is not None:
                    value = [int(v) for v in value]
                meta[key] = _jsonable(value)
    return meta


def _checkpoint_obs_dim(checkpoint_path: str) -> Optional[int]:
    checkpoint = _torch_load_cpu(checkpoint_path)
    if not isinstance(checkpoint, dict):
        return None
    if checkpoint.get('obs_dim') is not None:
        return int(checkpoint['obs_dim'])

    state_dict = checkpoint.get('online_net') or checkpoint.get('model')
    hidden_layers = checkpoint.get('hidden_layers')
    grid_size = checkpoint.get('grid_size')
    if not state_dict or not hidden_layers or grid_size is None:
        return None

    encoder_dim = int(hidden_layers[1])
    head_weight_names = {
        'advantage_head.initial_layer.0.weight',
        'advantage_head.initial_layer.0.weight_mu',
        'head.initial_layer.0.weight',
        'head.initial_layer.0.weight_mu',
    }
    for key, value in state_dict.items():
        normalized_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ('module.', '_orig_mod.'):
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix):]
                    changed = True
        if normalized_key in head_weight_names:
            head_input_dim = int(value.shape[1])
            return int(grid_size) + head_input_dim - encoder_dim
    return None


def _checkpoint_observation_metadata(checkpoint_path: str) -> Dict[str, Any]:
    checkpoint = _torch_load_cpu(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'Checkpoint {checkpoint_path} is not a dictionary; cannot read '
            'observation metadata.')
    metadata = checkpoint.get('observation_metadata')
    if not isinstance(metadata, dict):
        raise RuntimeError(
            f'Checkpoint {checkpoint_path} has no observation_metadata. '
            'Refusing to infer the observation switches from obs_dim: A and '
            'B are both 609-D and the guess would silently change the arm.')
    required = (
        'obs_dim', 'visibility_filter', 'use_occlusion', 'occlusion_mask',
        'history_frames', 'history_stride', 'reward_ttz_source',
        'grid_rows', 'grid_cols', 'grid_channels',
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise RuntimeError(
            f'Checkpoint {checkpoint_path} observation_metadata is missing '
            f'{", ".join(missing)}.')
    ckpt_obs_dim = checkpoint.get('obs_dim')
    if ckpt_obs_dim is not None and int(ckpt_obs_dim) != int(
            metadata['obs_dim']):
        raise RuntimeError(
            f'Checkpoint {checkpoint_path} stores obs_dim={ckpt_obs_dim} but '
            f'observation_metadata says {metadata["obs_dim"]}.')
    return metadata


def _apply_checkpoint_observation_settings(cfg, checkpoint_path,
                                           use_occlusion):
    grid_cfg = cfg['occupancy_grid']
    metadata = _checkpoint_observation_metadata(checkpoint_path)

    grid_cfg['visibility_filter'] = bool(metadata['visibility_filter'])
    grid_cfg['use_occlusion'] = bool(metadata['use_occlusion'])
    grid_cfg['occlusion_mask'] = bool(metadata['occlusion_mask'])
    grid_cfg['history_frames'] = int(metadata['history_frames'])
    grid_cfg['history_stride'] = int(metadata['history_stride'])
    cfg.setdefault('reward', {})['reward_ttz_source'] = str(
        metadata['reward_ttz_source'])

    if use_occlusion is not None:
        # Sanity-check override only. It changes obs_dim when the stats flag
        # differs from the checkpoint, so the network will normally fail to
        # load; that failure is intentional.
        print(
            'WARNING: overriding checkpoint use_occlusion='
            f'{bool(use_occlusion)} (checkpoint: '
            f'{bool(metadata["use_occlusion"])}); this is a sanity check, not '
            'accepted evidence.')
        grid_cfg['use_occlusion'] = bool(use_occlusion)

    return metadata


def _finite_ttz_min(values):
    finite = [v for v in values if v < 100]
    return min(finite) if finite else None


def _valid_numbers(values):
    return [v for v in values if v is not None]


def _mean_or_none(values):
    values = [float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _env_metadata(env, cfg) -> Dict[str, Any]:
    grid_cfg = cfg['occupancy_grid']
    drl_cfg = cfg['drl']
    return {
        'dt': float(cfg.get('dt', 0.1)),
        'max_episode_steps': int(drl_cfg['max_episode_steps']),
        'warmup_steps': WARMUP_STEPS,
        'single_agent_mode': bool(cfg.get('single_agent_mode', True)),
        'max_num_vehicles': int(cfg.get('max_num_vehicles', 1)),
        'deterministic_eval': bool(cfg.get('deterministic_eval', False)),
        'action_table': [[float(t), float(s)]
                         for t, s in env.action_table],
        'throttle_levels': [float(v) for v in
                            cfg['action_map']['throttle_levels']],
        'steer_levels': [float(v) for v in cfg['action_map']['steer_levels']],
        'occupancy_grid': {
            'rows': int(grid_cfg['rows']),
            'cols': int(grid_cfg['cols']),
            'forward_dist': float(grid_cfg['forward_dist']),
            'backward_dist': float(grid_cfg['backward_dist']),
            'lateral_dist': float(grid_cfg['lateral_dist']),
            'visibility_filter': bool(
                grid_cfg.get('visibility_filter', False)),
            'use_occlusion': bool(grid_cfg.get('use_occlusion', False)),
            'occlusion_mask': bool(grid_cfg.get('occlusion_mask', False)),
            'history_frames': int(grid_cfg.get('history_frames', 1)),
            'history_stride': int(grid_cfg.get('history_stride', 1)),
            'grid_channels': int(getattr(env, 'grid_channels', 0)),
            'obs_dim': int(getattr(env, 'obs_dim', 0)),
        },
        'reward_ttz_source': str(
            cfg.get('reward', {}).get('reward_ttz_source', 'privileged')),
        'subscriber': {
            'view_dist': float(cfg['subscriber']['view_dist']),
            'view_angle': float(cfg['subscriber']['view_angle']),
            'n_frames_stacked': int(cfg['subscriber'].get('n_frames_stacked',
                                                          1)),
        },
        'rew_cfg': {
            'goal_tolerance': float(cfg['rew_cfg']['goal_tolerance']),
            'position_target': bool(cfg['rew_cfg']['position_target']),
            'position_target_tolerance': float(
                cfg['rew_cfg']['position_target_tolerance']),
        },
    }


def _versions_metadata() -> Dict[str, Any]:
    info = {
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'torch': torch.__version__,
    }
    try:
        info['torch_cuda'] = torch.version.cuda
        info['cuda_available'] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info['gpu'] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - best effort only
        info['cuda_error'] = str(exc)
    return info


def evaluate(checkpoint_path: str, scenario_path: Optional[str] = None,
             scenario_split: str = 'valid', split: str = 'test',
             split_seed: int = 0, num_files: int = -1, seed: int = 42,
             render: bool = False, use_occlusion: Optional[bool] = None,
             method: str = 'ddqn', log_steps: bool = False,
             trace_path: Optional[str] = None) -> Dict[str, Any]:
    """Run the paired deterministic evaluation and return the results dict."""
    set_display_window()
    # Evaluation is single-process CPU; keep four concurrent evaluations from
    # each grabbing all 48 intra-op threads while training runs.
    torch.set_num_threads(1)

    from examples.drl_collision_avoidance.collision_avoidance_env import (
        CollisionAvoidanceEnv,
    )

    cfg = load_config(scenario_path=scenario_path,
                      scenario_split=scenario_split, num_files=-1)
    checkpoint_obs_metadata = _apply_checkpoint_observation_settings(
        cfg, checkpoint_path, use_occlusion)
    cfg['deterministic_eval'] = True
    cfg['seed'] = int(seed)
    # One scene is visited exactly once per run; a long-lived Simulation
    # cache only wastes memory, so keep the evaluator's footprint bounded.
    cfg['scenario_pool_size'] = 0
    cfg['scenario_cache_size'] = 0

    scenario_dir = cfg['scenario_path']
    all_files = load_valid_files(scenario_dir)
    eval_files = hash_split_files(all_files, split_seed=split_seed,
                                  half=split)
    if num_files is not None and num_files != -1:
        eval_files = eval_files[:int(num_files)]
    if not eval_files:
        raise RuntimeError(
            f'No scenario files selected (split={split!r}, '
            f'seed={split_seed}, num_files={num_files}).')

    print(f'Scenario dir : {scenario_dir}')
    print(f'Split        : {split} (seed={split_seed}) -> '
          f'{len(eval_files)} of {len(all_files)} files')
    print(f'Occlusion    : {cfg["occupancy_grid"].get("use_occlusion", False)}')
    print(
        'Obs contract : '
        f'visibility_filter={cfg["occupancy_grid"].get("visibility_filter", False)} '
        f'occlusion_mask={cfg["occupancy_grid"].get("occlusion_mask", False)} '
        f'history={cfg["occupancy_grid"].get("history_frames", 1)}/'
        f'{cfg["occupancy_grid"].get("history_stride", 1)} '
        f'reward_ttz={cfg.get("reward", {}).get("reward_ttz_source", "privileged")}')

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    start_time = cfg.get('scenario', {}).get('start_time', 0)

    env = CollisionAvoidanceEnv(cfg)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = build_cpu_agent(cfg, obs_dim, n_actions)
    agent.load(checkpoint_path)
    # Must come after load(): rebuilding networks can flip noisy layers back
    # into sampling mode.
    agent.set_eval_mode()

    if log_steps and trace_path is None:
        trace_path = 'eval_steps.jsonl'
    trace_handle = open(trace_path, 'w') if log_steps else None

    started_at = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    episodes: List[Dict[str, Any]] = []
    ttz_vehicles: List[float] = []
    ttz_pedestrians: List[float] = []

    try:
        for ep_idx, scene_file in enumerate(eval_files):
            env.set_eval_scene(scene_file)
            state, reset_info = env.reset()
            episode: Dict[str, Any] = {
                'episode': ep_idx,
                'scene_file': scene_file,
                'ego_id': reset_info.get('ego_id'),
                'valid': bool(reset_info.get('valid', False)),
                'reason': reset_info.get('reason'),
                'n_vehicles': int(reset_info.get('n_vehicles', 0)),
                'initial_goal_dist': reset_info.get('initial_goal_dist'),
            }
            if not episode['valid']:
                episodes.append(episode)
                print(f'Episode {ep_idx + 1:4d} | {scene_file} | INVALID '
                      f'({episode["reason"]})')
                continue

            total_reward = 0.0
            collided = False
            goal = False
            edge_collision = False
            timed_out = False
            min_goal_dist = float('inf')
            trajectory_pairs = []
            speeds = []
            steerings = []
            episode_ttz_vehicle = []
            episode_ttz_pedestrian = []
            t = -1
            ego_veh = env._get_ego_vehicle()

            for t in range(cfg['drl']['max_episode_steps']):
                ego_veh = env._get_ego_vehicle()
                if ego_veh is not None:
                    speeds.append(float(ego_veh.speed))
                    try:
                        exp_pos = env.base_env.scenario.expert_position(
                            ego_veh, start_time + WARMUP_STEPS + t)
                        trajectory_pairs.append((
                            (float(ego_veh.position.x),
                             float(ego_veh.position.y)),
                            (float(exp_pos.x), float(exp_pos.y)),
                        ))
                    except (IndexError, KeyError, RuntimeError):
                        pass

                action = agent.select_action(state)
                _, steer = env.action_table[action]
                steerings.append(float(steer))

                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += float(reward)
                collided = collided or bool(info.get('collided', False))
                goal = goal or bool(info.get('goal_achieved', False))
                edge_collision = edge_collision or bool(
                    info.get('veh_edge_collision', False))
                gd = float(info.get('goal_dist', float('inf')))
                min_goal_dist = min(min_goal_dist, gd)

                ttz_vehicle = float(info.get('ttz_vehicle', 999.0))
                ttz_pedestrian = float(info.get('ttz_pedestrian', 999.0))
                ttz_vehicles.append(ttz_vehicle)
                ttz_pedestrians.append(ttz_pedestrian)
                episode_ttz_vehicle.append(ttz_vehicle)
                episode_ttz_pedestrian.append(ttz_pedestrian)

                if trace_handle is not None and ego_veh is not None:
                    trace_handle.write(json.dumps({
                        'episode': ep_idx,
                        'scene_file': scene_file,
                        'ego_id': episode['ego_id'],
                        't': t,
                        'action': int(action),
                        'x': float(ego_veh.position.x),
                        'y': float(ego_veh.position.y),
                        'heading': float(ego_veh.heading),
                        'speed': float(ego_veh.speed),
                        'goal_dist': gd,
                        'ttz_vehicle': ttz_vehicle,
                        'ttz_pedestrian': ttz_pedestrian,
                        'reward': float(reward),
                        'collided': bool(info.get('collided', False)),
                        'goal_achieved': bool(
                            info.get('goal_achieved', False)),
                    }) + '\n')

                if render:
                    env.render()
                if done:
                    break

            timed_out = bool(truncated) and not bool(terminated)
            ade = None
            fde = None
            jerk = None
            smoothness = None
            if len(trajectory_pairs) > 1:
                pos = np.array([p for p, _ in trajectory_pairs])
                exp = np.array([p for _, p in trajectory_pairs])
                displacements = np.linalg.norm(pos - exp, axis=1)
                ade = float(np.mean(displacements))
                fde = float(displacements[-1])
            dt = float(cfg.get('dt', 0.1))
            if len(speeds) > 2 and dt > 0:
                accel = np.diff(np.array(speeds)) / dt
                jerk_signal = np.diff(accel) / dt
                jerk = float(np.sqrt(np.mean(jerk_signal ** 2)))
            if len(steerings) > 1 and dt > 0:
                steer_rate = np.diff(np.array(steerings)) / dt
                smoothness = float(np.std(steer_rate))

            episode.update({
                'steps': t + 1,
                'reward': total_reward,
                'collided': collided,
                'goal': goal,
                'edge_collision': edge_collision,
                'timed_out': timed_out,
                'min_goal_dist': min_goal_dist if np.isfinite(min_goal_dist)
                else None,
                'ade': ade,
                'fde': fde,
                'jerk': jerk,
                'smoothness': smoothness,
                'min_ttz_vehicle': _finite_ttz_min(episode_ttz_vehicle),
                'min_ttz_pedestrian': _finite_ttz_min(
                    episode_ttz_pedestrian),
            })
            episodes.append(episode)
            print(f'Episode {ep_idx + 1:4d} | {scene_file} | ego='
                  f'{episode["ego_id"]} | reward={total_reward:8.2f} | '
                  f'len={t + 1:4d} | collided={collided} | goal={goal} | '
                  f'min_goal_dist={min_goal_dist:.2f}')
    finally:
        if trace_handle is not None:
            trace_handle.close()

    valid = [e for e in episodes if e['valid']]
    invalid = [e for e in episodes if not e['valid']]
    aggregate = {
        'num_scenes_requested': len(eval_files),
        'num_valid_episodes': len(valid),
        'num_invalid_episodes': len(invalid),
        'invalid_reasons': sorted({e['reason'] for e in invalid
                                   if e['reason']}),
        'collision_rate': _mean_or_none([e['collided'] for e in valid]),
        'goal_rate': _mean_or_none([e['goal'] for e in valid]),
        'timeout_rate': _mean_or_none([e['timed_out'] for e in valid]),
        'edge_collision_rate': _mean_or_none(
            [e['edge_collision'] for e in valid]),
        'avg_reward': _mean_or_none([e['reward'] for e in valid]),
        'avg_length': _mean_or_none([e['steps'] for e in valid]),
        'avg_initial_goal_dist': _mean_or_none(
            [e['initial_goal_dist'] for e in valid]),
        'ade': _mean_or_none([e['ade'] for e in valid]),
        'fde': _mean_or_none([e['fde'] for e in valid]),
        'jerk': _mean_or_none([e['jerk'] for e in valid]),
        'smoothness': _mean_or_none([e['smoothness'] for e in valid]),
        'ttz_vehicle_min_mean': _mean_or_none(
            [e['min_ttz_vehicle'] for e in valid]),
        'ttz_pedestrian_min_mean': _mean_or_none(
            [e['min_ttz_pedestrian'] for e in valid]),
    }

    results = {
        'method': method,
        'provenance': {
            'checkpoint': {
                'path': os.path.abspath(checkpoint_path),
                'sha256': _sha256_file(checkpoint_path),
                'size_bytes': os.path.getsize(checkpoint_path),
                'mtime': os.path.getmtime(checkpoint_path),
                'metadata': _checkpoint_metadata(checkpoint_path),
                'observation_metadata': checkpoint_obs_metadata,
            },
            'scenario': {
                'path': os.path.abspath(scenario_dir),
                'split': split,
                'split_seed': int(split_seed),
                'valid_files_json_sha256': _sha256_file(
                    os.path.join(scenario_dir, 'valid_files.json')),
                'num_files_total': len(all_files),
                'num_files_selected': len(eval_files),
                'selected_files_sha256': _sha256_list(eval_files),
            },
            'env': _env_metadata(env, cfg),
            'code': _git_state(),
            'versions': _versions_metadata(),
            'run': {
                'seed': int(seed),
                'host': socket.gethostname(),
                'started_at': started_at,
                'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'log_steps': bool(log_steps),
                'trace_path': os.path.abspath(trace_path)
                if trace_path else None,
            },
        },
        'aggregate': aggregate,
        'episodes': episodes,
    }

    print('\n' + '=' * 60)
    print('EVALUATION SUMMARY')
    print('=' * 60)
    print(f'  Valid episodes: {len(valid)}/{len(episodes)}')
    if valid:
        print(f'  Avg reward:     {aggregate["avg_reward"]:.2f}')
        print(f'  Avg length:     {aggregate["avg_length"]:.1f}')
        print(f'  Collision rate: {aggregate["collision_rate"]:.1%}')
        print(f'  Goal rate:      {aggregate["goal_rate"]:.1%}')
        ttz_v = [t for t in ttz_vehicles if t < 100]
        ttz_p = [t for t in ttz_pedestrians if t < 100]
        if ttz_v:
            print(f'  Avg TTZ (veh):  {np.mean(ttz_v):.2f}s')
        if ttz_p:
            print(f'  Avg TTZ (ped):  {np.mean(ttz_p):.2f}s')
        if aggregate['ade'] is not None:
            print(f'  ADE:            {aggregate["ade"]:.2f} m')
            print(f'  FDE:            {aggregate["fde"]:.2f} m')
        if aggregate['jerk'] is not None:
            print(f'  Jerk:           {aggregate["jerk"]:.2f} m/s^3')
        if aggregate['smoothness'] is not None:
            print(f'  Smoothness:     {aggregate["smoothness"]:.2f} rad/s')
    print('=' * 60)
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Deterministic paired evaluation of a trained policy')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint (.pth)')
    parser.add_argument('--scenario_path', type=str, default=None,
                        help='Scenario directory containing valid_files.json')
    parser.add_argument('--scenario_split', type=str, default='valid',
                        choices=['train', 'valid'],
                        help='Dataset split used when --scenario_path is '
                             'omitted')
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'train', 'all'],
                        help='Hash-split half of valid_files.json to evaluate')
    parser.add_argument('--split_seed', type=int, default=0,
                        help='Salt for the hash split (same salt = same '
                             'episodes for every arm)')
    parser.add_argument('--num_files', type=int, default=-1,
                        help='Cap on the number of evaluated scenes')
    # Deprecated alias kept so older command lines keep working.
    parser.add_argument('--num_episodes', type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--render', action='store_true')
    # Python 3.8 has no BooleanOptionalAction; emulate --use_occlusion/--no_use_occlusion
    parser.add_argument('--use_occlusion', action='store_true',
                        help='Override occupancy_grid.use_occlusion to True')
    parser.add_argument('--no_use_occlusion', dest='use_occlusion',
                        action='store_false',
                        help='Override occupancy_grid.use_occlusion to False')
    parser.set_defaults(use_occlusion=None)
    parser.add_argument('--output_json', type=str, default=None,
                        help='Save results to JSON file for downstream '
                             'analysis')
    parser.add_argument('--log_steps', action='store_true',
                        help='Write per-step traces as JSON lines')
    parser.add_argument('--trace_path', type=str, default=None,
                        help='Trace file path (default: <output_json> '
                             '.steps.jsonl)')
    parser.add_argument('--method', type=str, default='ddqn',
                        help='Method name label for JSON output')
    args = parser.parse_args()

    num_files = args.num_files
    if num_files == -1 and args.num_episodes is not None:
        num_files = args.num_episodes

    trace_path = args.trace_path
    if trace_path is None and args.log_steps and args.output_json:
        trace_path = args.output_json + '.steps.jsonl'

    results = evaluate(
        args.checkpoint,
        scenario_path=args.scenario_path,
        scenario_split=args.scenario_split,
        split=args.split,
        split_seed=args.split_seed,
        num_files=num_files,
        seed=args.seed,
        render=args.render,
        use_occlusion=args.use_occlusion,
        method=args.method,
        log_steps=args.log_steps,
        trace_path=trace_path,
    )

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults saved to {args.output_json}')


if __name__ == '__main__':
    main()
