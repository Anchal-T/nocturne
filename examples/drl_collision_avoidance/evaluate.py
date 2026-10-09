import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cfgs.config import set_display_window
from examples.drl_collision_avoidance.agent_utils import build_cpu_agent
from examples.drl_collision_avoidance.episode_metrics import (
    classify_outcome,
    collision_type,
    maneuver_and_lead,
    partner_label,
    value_before,
)
from examples.drl_collision_avoidance.scenario_utils import load_config


def _checkpoint_obs_dim(checkpoint_path: str) -> Optional[int]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
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


def _apply_checkpoint_observation_settings(cfg, checkpoint_path, use_occlusion):
    """Match the observation the checkpoint was trained with.

    Checkpoints that stored ``visibility_filter`` and ``occlusion_features``
    use those independent switches. Older checkpoints only stored ``obs_dim``,
    which cannot separate the two switches, so they keep the coupled
    ``use_occlusion`` path.
    """
    grid_cfg = cfg['occupancy_grid']
    if use_occlusion is not None:
        grid_cfg['use_occlusion'] = bool(use_occlusion)
        grid_cfg['visibility_filter'] = None
        grid_cfg['occlusion_features'] = None
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and (
        'visibility_filter' in checkpoint or 'occlusion_features' in checkpoint
    ):
        if checkpoint.get('legacy_unfiltered_grid'):
            grid_cfg['use_occlusion'] = True
            grid_cfg['visibility_filter'] = None
            grid_cfg['occlusion_features'] = None
        else:
            grid_cfg['use_occlusion'] = False
            grid_cfg['visibility_filter'] = bool(checkpoint.get('visibility_filter', False))
            grid_cfg['occlusion_features'] = bool(checkpoint.get('occlusion_features', False))
        if checkpoint.get('reward_ttz_source'):
            cfg.setdefault('reward', {})['reward_ttz_source'] = checkpoint['reward_ttz_source']
        return

    ckpt_obs_dim = _checkpoint_obs_dim(checkpoint_path)
    if ckpt_obs_dim is None:
        return

    grid_size = 3 * int(grid_cfg['rows']) * int(grid_cfg['cols'])
    no_occ_dim = grid_size + 4 + 2 + 3
    occ_dim = no_occ_dim + 4
    grid_cfg['visibility_filter'] = None
    grid_cfg['occlusion_features'] = None
    if ckpt_obs_dim == occ_dim:
        grid_cfg['use_occlusion'] = True
    elif ckpt_obs_dim == no_occ_dim:
        grid_cfg['use_occlusion'] = False
    else:
        print(
            f'WARNING: checkpoint obs_dim={ckpt_obs_dim} does not match '
            f'expected no-occlusion ({no_occ_dim}) or occlusion ({occ_dim}) dimensions.'
        )


def _apply_eval_overrides(cfg, args):
    grid_cfg = cfg['occupancy_grid']
    if getattr(args, 'visibility_filter', None) is not None:
        grid_cfg['visibility_filter'] = bool(args.visibility_filter)
        grid_cfg['use_occlusion'] = False
    if getattr(args, 'occlusion_features', None) is not None:
        grid_cfg['occlusion_features'] = bool(args.occlusion_features)
        grid_cfg['use_occlusion'] = False
    if getattr(args, 'reward_ttz_source', None):
        cfg['reward']['reward_ttz_source'] = args.reward_ttz_source
    if getattr(args, 'occlusion_view_angle', None) is not None:
        grid_cfg['occlusion_view_angle'] = float(args.occlusion_view_angle)
    cfg.setdefault('robustness', {})
    cfg['robustness']['occlusion_dropout_p'] = float(getattr(args, 'occlusion_dropout_p', 0.0))
    cfg['robustness']['obs_noise_std'] = float(getattr(args, 'obs_noise_std', 0.0))
    cfg['log_privileged'] = True
    heldout = getattr(args, 'heldout_split', 'all')
    if heldout in ('val', 'test'):
        cfg['heldout_split'] = heldout


def _finite_ttz_min(values):
    finite = [v for v in values if v < 100]
    return min(finite) if finite else None


def _valid_numbers(values):
    return [v for v in values if v is not None]


def evaluate(checkpoint_path: str, scenario_path: Optional[str] = None,
             scenario_split: str = 'valid', num_episodes: int = 10,
             num_files: int = 1, render: bool = False,
             use_occlusion: Optional[bool] = None,
             overrides=None):
    set_display_window()

    from examples.drl_collision_avoidance.collision_avoidance_env import CollisionAvoidanceEnv

    cfg = load_config(scenario_path=scenario_path, scenario_split=scenario_split, num_files=num_files)
    _apply_checkpoint_observation_settings(cfg, checkpoint_path, use_occlusion)
    if overrides is not None:
        _apply_eval_overrides(cfg, overrides)
    print(f'Using scenario split={scenario_split} path={cfg["scenario_path"]}')
    print(
        f'visibility_filter={cfg["occupancy_grid"].get("visibility_filter")} '
        f'occlusion_features={cfg["occupancy_grid"].get("occlusion_features")} '
        f'use_occlusion={cfg["occupancy_grid"].get("use_occlusion", False)} '
        f'heldout_split={cfg.get("heldout_split")}'
    )

    env = CollisionAvoidanceEnv(cfg)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = build_cpu_agent(cfg, obs_dim, n_actions)
    agent.load(checkpoint_path)
    agent.epsilon = 0.0
    grid_size = env.grid_channels * env.grid_rows * env.grid_cols
    trace_dir = getattr(overrides, 'trace_dir', None) if overrides is not None else None
    trace_max = int(getattr(overrides, 'trace_max', 0) or 0) if overrides is not None else 0
    trace_counts = {}

    dt = cfg.get('dt', 0.1)
    start_time = cfg.get('scenario', {}).get('start_time', 0)
    # BaseEnv.reset() runs 10 expert-control context steps before handing
    # control to the agent, advancing the simulation clock by 10 timesteps.
    context_steps = 10

    results = {
        'rewards': [], 'lengths': [],
        'collisions': 0, 'goals': 0,
        'collided_list': [], 'goal_list': [],
        'ttz_vehicles': [], 'ttz_pedestrians': [],
        'episode_ttz_vehicle': [], 'episode_ttz_pedestrian': [],
        'ade': [], 'fde': [], 'jerk': [], 'smoothness': [],
        'num_vehicles': [],
        'episodes': [],
    }

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        collided = False
        goal = False
        min_goal_dist = float('inf')

        trajectory_pairs = []
        speeds = []
        steerings = []
        episode_ttz_vehicle = []
        episode_ttz_pedestrian = []
        episode_ttz_gt = []
        episode_speeds = []
        episode_expert_speeds = []
        episode_steers = []
        episode_occ = []
        episode_pressure = []
        episode_density = []
        episode_visible_density = []
        hidden_flags = []
        nearest_types = []
        nearest_dists = []
        veh_veh_flags = []
        veh_edge_flags = []
        best_ttz = 1e9
        best_grid = None
        ade = None
        fde = None
        jerk = None
        smoothness = None

        ego_veh = env._get_ego_vehicle()
        n_vehicles = len(env.base_env.scenario.getVehicles())

        for t in range(cfg['drl']['max_episode_steps']):
            ego_veh = env._get_ego_vehicle()
            if ego_veh is not None:
                speeds.append(float(ego_veh.speed))
                try:
                    exp_pos = env.base_env.scenario.expert_position(
                        ego_veh, start_time + context_steps + t)
                    trajectory_pairs.append((
                        (ego_veh.position.x, ego_veh.position.y),
                        (exp_pos.x, exp_pos.y),
                    ))
                except (IndexError, KeyError, RuntimeError):
                    pass

            action = agent.select_action(state)
            _, steer = env.action_table[action]
            steerings.append(steer)
            episode_steers.append(float(steer))

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            collided = collided or info.get('collided', False)
            goal = goal or info.get('goal_achieved', False)
            gd = info.get('goal_dist', float('inf'))
            min_goal_dist = min(min_goal_dist, gd)

            ttz_vehicle = info.get('ttz_vehicle', 999.0)
            ttz_pedestrian = info.get('ttz_pedestrian', 999.0)
            ttz_gt = min(
                float(info.get('ttz_vehicle_gt', 999.0)),
                float(info.get('ttz_pedestrian_gt', 999.0)),
            )
            results['ttz_vehicles'].append(ttz_vehicle)
            results['ttz_pedestrians'].append(ttz_pedestrian)
            episode_ttz_vehicle.append(ttz_vehicle)
            episode_ttz_pedestrian.append(ttz_pedestrian)
            episode_ttz_gt.append(ttz_gt)
            episode_speeds.append(float(info.get('ego_speed', 0.0)))
            episode_occ.append(float(info.get('occ_frac', 0.0)))
            episode_pressure.append(float(info.get('occ_pressure', 0.0)))
            episode_density.append(int(info.get('vehicles_in_range', 0)))
            visible_density = info.get('vehicles_visible_in_range')
            episode_visible_density.append(
                None if visible_density is None else int(visible_density)
            )
            hidden_flags.append(bool(info.get('hidden_contact', False)))
            nearest_types.append(info.get('nearest_type'))
            nearest_dists.append(info.get('nearest_dist'))
            veh_veh_flags.append(bool(info.get('veh_veh_collision', False)))
            veh_edge_flags.append(bool(info.get('veh_edge_collision', False)))
            if ttz_gt < best_ttz:
                best_ttz = ttz_gt
                best_grid = np.asarray(state[:grid_size], dtype=np.float32).copy()
            ego_now = env._get_ego_vehicle()
            if ego_now is not None:
                try:
                    expert_speed = env.base_env.scenario.expert_speed(
                        ego_now, start_time + context_steps + t)
                    episode_expert_speeds.append(float(expert_speed))
                except (IndexError, KeyError, RuntimeError, TypeError):
                    pass

            if render:
                env.render()
            if done:
                break

        if len(trajectory_pairs) > 1:
            pos = np.array([p for p, _ in trajectory_pairs])
            exp = np.array([p for _, p in trajectory_pairs])
            displacements = np.linalg.norm(pos - exp, axis=1)
            ade = float(np.mean(displacements))
            fde = float(displacements[-1])
        if len(speeds) > 2 and dt > 0:
            accel = np.diff(np.array(speeds)) / dt
            jerk_signal = np.diff(accel) / dt
            jerk = float(np.sqrt(np.mean(jerk_signal ** 2)))
        if len(steerings) > 1 and dt > 0:
            steer_arr = np.array(steerings)
            steer_rate = np.diff(steer_arr) / dt
            smoothness = float(np.std(steer_rate))

        results['rewards'].append(total_reward)
        results['lengths'].append(t + 1)
        results['ade'].append(ade)
        results['fde'].append(fde)
        results['jerk'].append(jerk)
        results['smoothness'].append(smoothness)
        results['collisions'] += int(collided)
        results['goals'] += int(goal)
        results['collided_list'].append(collided)
        results['goal_list'].append(goal)
        length = t + 1
        horizon = cfg['drl']['max_episode_steps']
        outcome, success, timeout = classify_outcome(goal, collided, length, horizon)
        kind = collision_type(collided, any(veh_veh_flags), any(veh_edge_flags))
        partner = partner_label(
            kind,
            nearest_types[-1] if nearest_types else None,
            nearest_dists[-1] if nearest_dists else None,
        )
        maneuver, brake_lead_s = maneuver_and_lead(
            episode_speeds, episode_steers, episode_ttz_gt, dt=dt,
        )
        occ_values = episode_occ
        pressure_values = episode_pressure
        avg_speed = float(np.mean(episode_speeds)) if episode_speeds else None
        expert_avg = float(np.mean(episode_expert_speeds)) if episode_expert_speeds else None
        episode_record = {
            'reward': total_reward,
            'length': length,
            'collided': collided,
            'goal': goal,
            'timeout': timeout,
            'success': success,
            'outcome': outcome,
            'num_vehicles': n_vehicles,
            'vehicles_in_range': float(np.mean(episode_density)) if episode_density else 0.0,
            'vehicles_visible_in_range': (
                float(np.mean([v for v in episode_visible_density if v is not None]))
                if any(v is not None for v in episode_visible_density) else None
            ),
            'ttz_vehicle': _finite_ttz_min(episode_ttz_vehicle),
            'ttz_pedestrian': _finite_ttz_min(episode_ttz_pedestrian),
            'min_ttz_gt': _finite_ttz_min(episode_ttz_gt),
            'ttz_1s_before_collision': value_before(episode_ttz_gt, 10) if collided else None,
            'ttz_2s_before_collision': value_before(episode_ttz_gt, 20) if collided else None,
            'collision_type': kind,
            'partner': partner,
            'hidden_at_collision': bool(hidden_flags[-1]) if collided and hidden_flags else False,
            'maneuver': maneuver,
            'brake_lead_s': brake_lead_s,
            'avg_speed': avg_speed,
            'expert_avg_speed': expert_avg,
            'speed_ratio': (
                avg_speed / expert_avg if avg_speed is not None and expert_avg not in (None, 0.0) else None
            ),
            'mean_occ_frac': float(np.mean(occ_values)) if occ_values else None,
            'max_occ_frac': float(np.max(occ_values)) if occ_values else None,
            'mean_occ_pressure': float(np.mean(pressure_values)) if pressure_values else None,
            'max_occ_pressure': float(np.max(pressure_values)) if pressure_values else None,
            'scenario_file': info.get('scenario_file'),
            'ade': ade,
            'fde': fde,
            'jerk': jerk,
            'smoothness': smoothness,
            'min_goal_dist': min_goal_dist,
        }
        _write_trace(
            trace_dir, trace_counts, trace_max, ep, outcome, episode_record,
            trajectory_pairs, episode_speeds, episode_steers, episode_ttz_gt,
            episode_occ, episode_pressure, best_grid, env,
        )
        results['episodes'].append(episode_record)
        results['num_vehicles'].append(n_vehicles)
        results['episode_ttz_vehicle'].append(episode_record['ttz_vehicle'])
        results['episode_ttz_pedestrian'].append(episode_record['ttz_pedestrian'])
        print(
            f'Episode {ep:3d} | reward={total_reward:8.2f} | len={length:4d} | '
            f'outcome={outcome} | maneuver={maneuver} | min_goal_dist={min_goal_dist:.2f}'
        )

    print('\n' + '=' * 60)
    print('EVALUATION SUMMARY')
    print('=' * 60)
    print(f'  Episodes:       {num_episodes}')
    print(f'  Avg reward:     {np.mean(results["rewards"]):.2f} +/- {np.std(results["rewards"]):.2f}')
    print(f'  Avg length:     {np.mean(results["lengths"]):.1f}')
    n_eps = max(num_episodes, 1)
    n_timeout = sum(1 for ep in results['episodes'] if ep['timeout'])
    n_success = sum(1 for ep in results['episodes'] if ep['success'])
    print(f'  Collision rate:  {results["collisions"] / n_eps:.1%}')
    print(f'  Goal rate:       {results["goals"] / n_eps:.1%}')
    print(f'  Success rate:    {n_success / n_eps:.1%}')
    print(f'  Timeout rate:    {n_timeout / n_eps:.1%}')
    ttz_v = [t for t in results['ttz_vehicles'] if t < 100]
    ttz_p = [t for t in results['ttz_pedestrians'] if t < 100]
    if ttz_v:
        print(f'  Avg TTZ (veh):  {np.mean(ttz_v):.2f}s')
    if ttz_p:
        print(f'  Avg TTZ (ped):  {np.mean(ttz_p):.2f}s')
    ade_values = _valid_numbers(results['ade'])
    fde_values = _valid_numbers(results['fde'])
    jerk_values = _valid_numbers(results['jerk'])
    smoothness_values = _valid_numbers(results['smoothness'])
    if ade_values:
        print(f'  ADE:            {np.mean(ade_values):.2f} +/- {np.std(ade_values):.2f} m')
        print(f'  FDE:            {np.mean(fde_values):.2f} +/- {np.std(fde_values):.2f} m')
    if jerk_values:
        print(f'  Jerk:           {np.mean(jerk_values):.2f} +/- {np.std(jerk_values):.2f} m/s^3')
    if smoothness_values:
        print(f'  Smoothness:     {np.mean(smoothness_values):.2f} +/- {np.std(smoothness_values):.2f} rad/s')
    print('=' * 60)
    return results


def _write_trace(
    trace_dir, trace_counts, trace_max, episode_index, outcome, episode_record,
    trajectory_pairs, speeds, steers, ttz_gt, occ_frac, pressure, best_grid, env,
):
    if not trace_dir or trace_max <= 0:
        return
    if trace_counts.get(outcome, 0) >= trace_max:
        return
    trace_counts[outcome] = trace_counts.get(outcome, 0) + 1
    os.makedirs(trace_dir, exist_ok=True)
    grid = None
    if best_grid is not None:
        grid = best_grid.reshape(env.grid_channels, env.grid_rows, env.grid_cols)[0].tolist()
    payload = {
        'episode': episode_index,
        'outcome': outcome,
        'scenario_file': episode_record.get('scenario_file'),
        'maneuver': episode_record.get('maneuver'),
        'hidden_at_collision': episode_record.get('hidden_at_collision'),
        'ego_xy': [[float(p[0]), float(p[1])] for p, _ in trajectory_pairs],
        'expert_xy': [[float(p[0]), float(p[1])] for _, p in trajectory_pairs],
        'speed': [float(v) for v in speeds],
        'steer': [float(v) for v in steers],
        'ttz_gt': [None if v >= 100 else float(v) for v in ttz_gt],
        'occ_frac': [float(v) for v in occ_frac],
        'occ_pressure': [float(v) for v in pressure],
        'occupancy_at_min_ttz': grid,
        'grid_rows': env.grid_rows,
        'grid_cols': env.grid_cols,
    }
    path = os.path.join(trace_dir, f'ep{episode_index:04d}_{outcome}.json')
    with open(path, 'w') as handle:
        json.dump(payload, handle)


def _add_optional_bool(parser, name, help_text):
    """Add --name / --no_name with a None default, meaning no override.

    argparse.BooleanOptionalAction needs Python 3.9, and this repo supports 3.8.
    """
    parser.add_argument(f'--{name}', dest=name, action='store_true', help=help_text)
    parser.add_argument(f'--no_{name}', dest=name, action='store_false',
                        help=f'Override {name} to False')
    parser.set_defaults(**{name: None})


def _mean_or_none(values):
    usable = _valid_numbers(values)
    if not usable:
        return None
    return float(np.mean(usable))


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained DDQN agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint (.pth)')
    parser.add_argument('--scenario_path', type=str,
                        default=None,
                        help='Override scenario directory with valid_files.json')
    parser.add_argument('--scenario_split', type=str,
                        default='valid', choices=['train', 'valid'],
                        help='Dataset split used when --scenario_path is omitted')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--num_files', type=int, default=-1)
    parser.add_argument('--render', action='store_true')
    _add_optional_bool(parser, 'use_occlusion',
                       'Legacy coupled occlusion switch. Prefer the two flags below.')
    _add_optional_bool(parser, 'visibility_filter',
                       'Override occupancy_grid.visibility_filter')
    _add_optional_bool(parser, 'occlusion_features',
                       'Override occupancy_grid.occlusion_features')
    parser.add_argument('--reward_ttz_source', choices=['visible', 'ground_truth'], default=None)
    parser.add_argument('--occlusion_view_angle', type=float, default=None,
                        help='View-cone angle in radians. pi is 180 degrees; 2*pi/3 is 120 degrees.')
    parser.add_argument('--occlusion_dropout_p', type=float, default=0.0)
    parser.add_argument('--obs_noise_std', type=float, default=0.0)
    parser.add_argument('--heldout_split', choices=['all', 'val', 'test'], default='all')
    parser.add_argument('--trace_dir', type=str, default=None)
    parser.add_argument('--trace_max', type=int, default=0,
                        help='Traces to keep per outcome. 0 disables trace dumps.')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Save results to JSON file for downstream analysis')
    parser.add_argument('--method', type=str, default='ddqn',
                        help='Method name label for JSON output')
    args = parser.parse_args()
    results = evaluate(
        args.checkpoint,
        args.scenario_path,
        args.scenario_split,
        args.num_episodes,
        args.num_files,
        args.render,
        args.use_occlusion,
        overrides=args,
    )
    if args.output_json:
        n = len(results['episodes'])
        successes = sum(1 for ep in results['episodes'] if ep['success'])
        timeouts = sum(1 for ep in results['episodes'] if ep['timeout'])
        output = {
            'method': args.method,
            'checkpoint': args.checkpoint,
            'num_episodes': n,
            'goal_rate': results['goals'] / max(n, 1),
            'collision_rate': results['collisions'] / max(n, 1),
            'success_rate': successes / max(n, 1),
            'timeout_rate': timeouts / max(n, 1),
            'avg_reward': float(np.mean(results['rewards'])) if results['rewards'] else None,
            'avg_speed': _mean_or_none([ep.get('avg_speed') for ep in results['episodes']]),
            'ade': _mean_or_none(results['ade']),
            'fde': _mean_or_none(results['fde']),
            'jerk': _mean_or_none(results['jerk']),
            'smoothness': _mean_or_none(results['smoothness']),
            'occlusion_dropout_p': args.occlusion_dropout_p,
            'obs_noise_std': args.obs_noise_std,
            'occlusion_view_angle': args.occlusion_view_angle,
            'heldout_split': args.heldout_split,
            'episodes': results['episodes'],
        }
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f'\nResults saved to {args.output_json}')


if __name__ == '__main__':
    main()
