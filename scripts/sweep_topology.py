#!/usr/bin/env python3
"""Phase 0 topology sweep: find the best worker/env/ready-fraction settings.

Runs a collection-only loop (no learning) for a fixed number of env steps
across a grid of `num_workers`, `num_envs_per_worker`, and
`min_ready_fraction` settings, and reports collection FPS for each.

This isolates actor-side throughput from the learner so the sweep measures
exactly what the plan asks: how fast can the env fleet produce transitions
for a given topology.

Usage:
    python scripts/sweep_topology.py --scenario-path /path/to/scenarios
    python scripts/sweep_topology.py --scenario-file examples/example_scenario.json \
        --workers 4,8,16 --envs-per-worker 1,2,4
"""
import argparse
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def _build_cfg(scenario_path: str) -> Dict[str, Any]:
    from omegaconf import OmegaConf
    cfg_path = os.path.join(
        os.path.dirname(__file__), '..', 'cfgs',
        'drl_collision_avoidance', 'config.yaml',
    )
    cfg = OmegaConf.load(cfg_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict['scenario_path'] = os.path.abspath(scenario_path)
    cfg_dict['scenario_split'] = 'train'
    cfg_dict['num_files'] = -1
    cfg_dict['seed'] = 42
    cfg_dict['single_agent_mode'] = True
    cfg_dict['max_num_vehicles'] = 1
    return cfg_dict


def _make_scenario_dir(scenario_file, scenario_path):
    import json
    import shutil
    import tempfile
    if scenario_path:
        if not os.path.isfile(os.path.join(scenario_path, 'valid_files.json')):
            raise FileNotFoundError(
                f'No valid_files.json in --scenario-path {scenario_path}')
        return os.path.abspath(scenario_path)
    if not scenario_file:
        raise ValueError('Provide --scenario-file or --scenario-path.')
    scenario_file = os.path.abspath(scenario_file)
    if not os.path.isfile(scenario_file):
        raise FileNotFoundError(f'Scenario file not found: {scenario_file}')
    tmp = tempfile.mkdtemp(prefix='nocturne_sweep_')
    shutil.copy(scenario_file, tmp)
    fname = os.path.basename(scenario_file)
    with open(os.path.join(tmp, 'valid_files.json'), 'w') as fp:
        json.dump({fname: []}, fp)
    return tmp


def _run_topology(
    cfg_dict: Dict[str, Any],
    num_workers: int,
    num_envs_per_worker: int,
    min_ready_fraction: float,
    vec_env_mode: str,
    num_steps: int,
    warmup_steps: int,
) -> Tuple[float, int]:
    """Run a collection-only loop and return (fps, actual_steps)."""
    from examples.drl_collision_avoidance.train import (
        _build_vec_env, _make_env_fn,
    )
    from examples.drl_collision_avoidance.vec_env import (
        AsyncSubprocVecEnv, DummyVecEnv, SubprocVecEnv,
    )

    num_envs = num_workers * num_envs_per_worker
    env_fns = [_make_env_fn(cfg_dict, i) for i in range(num_envs)]

    vec_env_classes = {
        'async': AsyncSubprocVecEnv,
        'sync': SubprocVecEnv,
        'dummy': DummyVecEnv,
    }
    vec_env_cls = vec_env_classes[vec_env_mode]
    kwargs = (
        {"num_envs_per_worker": num_envs_per_worker}
        if vec_env_mode in ("async", "ray") else {}
    )
    vec_env = vec_env_cls(env_fns, **kwargs)

    n_actions = vec_env.action_space.n
    obs = vec_env.reset()

    # Warmup
    actions = np.random.randint(0, n_actions, size=num_envs)
    if vec_env_mode == 'async':
        vec_env.step_async(actions)
        for _ in range(warmup_steps):
            ids, obs, _, _, _ = vec_env.step_wait(
                min_ready=max(1, int(num_envs * min_ready_fraction)))
            actions = np.random.randint(0, n_actions, size=len(ids))
            vec_env.step_async(actions, env_ids=ids)
    else:
        for _ in range(warmup_steps):
            obs, _, _, _ = vec_env.step(actions)
            actions = np.random.randint(0, n_actions, size=num_envs)

    # Timed run
    actual_steps = 0
    t0 = time.perf_counter()
    if vec_env_mode == 'async':
        min_ready = max(1, int(num_envs * min_ready_fraction))
        while actual_steps < num_steps:
            ids, obs, _, _, _ = vec_env.step_wait(min_ready=min_ready)
            if len(ids) == 0:
                continue
            actions = np.random.randint(0, n_actions, size=len(ids))
            vec_env.step_async(actions, env_ids=ids)
            actual_steps += len(ids)
    else:
        while actual_steps < num_steps:
            obs, _, _, _ = vec_env.step(actions)
            actions = np.random.randint(0, n_actions, size=num_envs)
            actual_steps += num_envs
    elapsed = time.perf_counter() - t0

    vec_env.close()
    return actual_steps / max(elapsed, 1e-6), actual_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--scenario-file')
    g.add_argument('--scenario-path')
    parser.add_argument('--workers', default='4,8,16,24',
                        help='Comma-separated num_workers values.')
    parser.add_argument('--envs-per-worker', default='1,2,4',
                        help='Comma-separated num_envs_per_worker values.')
    parser.add_argument('--ready-fractions', default='0.25,0.5,1.0',
                        help='Comma-separated min_ready_fraction values.')
    parser.add_argument('--mode', default='async', choices=['async', 'sync'])
    parser.add_argument('--steps', type=int, default=2000,
                        help='Env steps to collect per config.')
    parser.add_argument('--warmup', type=int, default=100)
    args = parser.parse_args()

    scenario_path = _make_scenario_dir(args.scenario_file, args.scenario_path)
    cfg_dict = _build_cfg(scenario_path)

    workers = _parse_int_list(args.workers)
    envs_per_worker = _parse_int_list(args.envs_per_worker)
    ready_fracs = [float(x) for x in args.ready_fractions.split(',')]

    print(f'\n{"=" * 72}')
    print(f'  Topology sweep: mode={args.mode}, steps={args.steps}, warmup={args.warmup}')
    print(f'{"=" * 72}')
    header = (
        f'{"workers":>8s} {"envs/wk":>7s} {"total":>6s} '
        f'{"ready%":>7s} {"fps":>10s} {"steps":>8s}'
    )
    print(header)
    print('-' * len(header))

    results = []
    for nw in workers:
        for ne in envs_per_worker:
            if args.mode == 'sync' and ne > 1 and nw * ne > 1:
                # sync mode: one su            bprocess per env, so num_envs_per_worker is
                # ignored; just set total = num_workers
                pass
            for rf in ready_fracs:
                if args.mode == 'sync' and rf != 1.0:
                    continue  # sync mode waits for all envs; rf irrelevant
                total = nw * ne
                try:
                    fps, actual = _run_topology(
                        cfg_dict, nw, ne, rf, args.mode,
                        args.steps, args.warmup,
                    )
                    print(f'{nw:8d} {ne:7d} {total:6d} {rf:7.2f} {fps:10.1f} {actual:8d}')
                    results.append((nw, ne, rf, fps))
                except Exception as exc:
                    print(f'{nw:8d} {ne:7d} {total:6d} {rf:7.2f} {"FAILED":>10s} - {exc}')

    print('-' * len(header))
    if results:
        best = max(results, key=lambda r: r[3])
        print(f'\n  Best: workers={best[0]}, envs/worker={best[1]}, '
              f'ready_frac={best[2]:.2f} -> {best[3]:.1f} fps')
    print()


if __name__ == '__main__':
    main()
