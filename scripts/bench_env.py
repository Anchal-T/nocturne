#!/usr/bin/env python3
"""Phase 0 benchmark: measure where env time goes.

Times `reset()` versus `step()` in isolation and breaks `step()` into its
internal components (C++ simulation.step, occupancy grid, TTZ, occlusion
query) by monkeypatching the env's internal methods with timing wrappers.

Runs standalone against a single scenario file or a full scenario directory -
no dataset download and no training loop required.

Usage:
    python scripts/bench_env.py --scenario-file examples/example_scenario.json
    python scripts/bench_env.py --scenario-path /path/to/formatted_json_v2_no_tl_train
    python scripts/bench_env.py --scenario-file examples/example_scenario.json --occlusion both
"""
import argparse
import contextlib
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _build_cfg(scenario_path: str, use_occlusion: bool) -> Dict[str, Any]:
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
    cfg_dict['occupancy_grid']['use_occlusion'] = use_occlusion
    cfg_dict['single_agent_mode'] = True
    cfg_dict['max_num_vehicles'] = 1
    return cfg_dict


def _make_scenario_dir(scenario_file: Optional[str],
                       scenario_path: Optional[str]) -> str:
    """Return a directory containing the scenario + valid_files.json."""
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

    tmp = tempfile.mkdtemp(prefix='nocturne_bench_')
    shutil.copy(scenario_file, tmp)
    fname = os.path.basename(scenario_file)
    with open(os.path.join(tmp, 'valid_files.json'), 'w') as fp:
        json.dump({fname: []}, fp)
    return tmp


class Timer:
    """Accumulates call count and wall time for a wrapped callable."""

    def __init__(self, name: str, store: Dict[str, Tuple[int, float]]):
        self.name = name
        self.store = store
        self._depth = 0

    def wrap(self, fn: Callable) -> Callable:
        store = self.store
        name = self.name

        def timed(*args, **kwargs):
            store.setdefault(name, (0, 0.0))
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                cnt, acc = store[name]
                store[name] = (cnt + 1, acc + dt)

        return timed


def _instrument_env(env, store: Dict[str, Tuple[int, float]]):
    """Monkeypatch env internals with timing wrappers.

    Patches:
      - env._build_occupancy_grid  (Python grid stamping)
      - env._get_ttz_info          (TTZ vectorized compute)
      - env._query_visible_objects (occlusion C++ ray-cast, only if occlusion on)
      - env.base_env.step          (wraps the whole BaseEnv.step incl. simulation.step)
      - env.base_env.simulation.step (the C++ Scenario::Step + collision)
      - env.base_env.get_observation (the full Nocturne obs, used during reset warmup)
    """
    env.base_env.simulation.step = Timer(
        'step.simulation_step', store).wrap(env.base_env.simulation.step)
    env.base_env.step = Timer('step.base_env_step', store).wrap(env.base_env.step)
    env._build_occupancy_grid = Timer(
        'step.build_occupancy_grid', store).wrap(env._build_occupancy_grid)
    env._get_ttz_info = Timer(
        'step.get_ttz_info', store).wrap(env._get_ttz_info)
    if hasattr(env, '_query_visible_objects'):
        env._query_visible_objects = Timer(
            'step.query_visible_objects', store).wrap(env._query_visible_objects)
    env.base_env.get_observation = Timer(
        'reset.get_observation', store).wrap(env.base_env.get_observation)


def _bench_reset(env, n: int) -> Tuple[List[float], Dict[str, float]]:
    times: List[float] = []
    component_totals: Dict[str, float] = defaultdict(float)
    component_counts: Dict[str, int] = defaultdict(int)

    for i in range(n):
        store: Dict[str, Tuple[int, float]] = {}
        _instrument_env(env, store)
        t0 = time.perf_counter()
        env.reset()
        dt = time.perf_counter() - t0
        times.append(dt)
        for k, (cnt, acc) in store.items():
            component_totals[k] += acc
            component_counts[k] += cnt

    return times, {k: v / n for k, v in component_totals.items()}


def _bench_step(env, n: int) -> Tuple[List[float], Dict[str, float]]:
    env.reset()
    times: List[float] = []
    component_totals: Dict[str, float] = defaultdict(float)

    action = 0
    for i in range(n):
        store: Dict[str, Tuple[int, float]] = {}
        _instrument_env(env, store)
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        dt = time.perf_counter() - t0
        times.append(dt)
        for k, (cnt, acc) in store.items():
            component_totals[k] += acc
        if terminated or truncated:
            env.reset()

    return times, {k: v / n for k, v in component_totals.items()}


def _fmt_us(seconds: float) -> str:
    us = seconds * 1e6
    if us < 1000:
        return f'{us:7.1f} us'
    return f'{us / 1000:7.2f} ms'


def _fmt_pct(frac: float) -> str:
    return f'{frac * 100:5.1f}%'


def _print_section(title: str, times: List[float],
                   components: Dict[str, float]):
    total_mean = statistics.mean(times)
    total_std = statistics.stdev(times) if len(times) > 1 else 0.0
    total_p50 = statistics.median(times)
    total_p95 = sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0]

    print(f'\n  {title}')
    print(f'    total: mean={_fmt_us(total_mean)}  std={_fmt_us(total_std)}  '
          f'p50={_fmt_us(total_p50)}  p95={_fmt_us(total_p95)}  n={len(times)}')
    if components:
        comp_total = sum(components.values())
        print(f'    breakdown (mean per call):')
        for name in sorted(components, key=lambda k: -components[k]):
            acc = components[name]
            frac = acc / comp_total if comp_total > 0 else 0.0
            share = acc / total_mean if total_mean > 0 else 0.0
            print(f'      {name:32s}  {_fmt_us(acc):>12s}  '
                  f'{_fmt_pct(share):>6s} of total, '
                  f'{_fmt_pct(frac):>6s} of components')


def _print_env_stats(env):
    try:
        scenario = env.base_env.scenario
        moved = scenario.getObjectsThatMoved()
        vehicles = scenario.getVehicles()
        print(f'  scenario: {len(vehicles)} vehicles, '
              f'{len(moved)} moved, '
              f'{len(scenario.getRoadLines())} road lines')
    except Exception as exc:
        print(f'  (could not read scenario stats: {exc})')


def run_bench(scenario_path: str, use_occlusion: bool,
              num_resets: int, num_steps: int, warmup: int):
    from examples.drl_collision_avoidance.collision_avoidance_env import (
        CollisionAvoidanceEnv,
    )

    cfg = _build_cfg(scenario_path, use_occlusion)
    env = CollisionAvoidanceEnv(cfg)
    _print_env_stats(env)

    for _ in range(warmup):
        env.reset()
        env.step(0)

    reset_times, reset_components = _bench_reset(env, num_resets)
    step_times, step_components = _bench_step(env, num_steps)

    occ_label = 'ON ' if use_occlusion else 'OFF'
    print(f'\n{"=" * 64}')
    print(f'  occlusion: {occ_label}')
    _print_section('RESET', reset_times, reset_components)
    _print_section('STEP', step_times, step_components)
    print(f'{"=" * 64}')

    env.close()
    return {
        'occlusion': use_occlusion,
        'reset_mean_us': statistics.mean(reset_times) * 1e6,
        'step_mean_us': statistics.mean(step_times) * 1e6,
        'reset_components': reset_components,
        'step_components': step_components,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--scenario-file', help='Single scenario JSON file.')
    g.add_argument('--scenario-path',
                   help='Directory with valid_files.json + scenario files.')
    parser.add_argument('--occlusion', choices=['off', 'on', 'both'],
                        default='both',
                        help='Which occlusion settings to benchmark.')
    parser.add_argument('--num-resets', type=int, default=50)
    parser.add_argument('--num-steps', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--json-out', default=None,
                        help='Write results as JSON to this path.')
    args = parser.parse_args()

    scenario_path = _make_scenario_dir(args.scenario_file, args.scenario_path)
    is_temp = args.scenario_file is not None

    occ_settings = [False, True] if args.occlusion == 'both' else [
        args.occlusion == 'on']

    results = []
    for use_occlusion in occ_settings:
        try:
            res = run_bench(scenario_path, use_occlusion,
                            args.num_resets, args.num_steps, args.warmup)
            results.append(res)
        except Exception as exc:
            print(f'\n  FAILED (occlusion={use_occlusion}): {exc}',
                  file=sys.stderr)
            import traceback
            traceback.print_exc()

    if args.json_out:
        with open(args.json_out, 'w') as fp:
            json.dump(results, fp, indent=2)
        print(f'\nWrote results to {args.json_out}')

    if is_temp and scenario_path.startswith(tempfile.gettempdir()):
        with contextlib.suppress(Exception):
            shutil.rmtree(scenario_path)


if __name__ == '__main__':
    main()
