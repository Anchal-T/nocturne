"""Density scaling analysis: group evaluation results by vehicle count.

Usage:
    python scripts/density_analysis.py results.json

Input JSON format (output of unified eval):
    {"episodes": [{"num_vehicles": 12, "collided": false, "goal": true, ...}, ...]}
"""
import json
import sys
from collections import defaultdict

import numpy as np

BINS = [(2, 5), (6, 10), (11, 15), (16, 100)]
BIN_LABELS = ["2-5", "6-10", "11-15", "16+"]


def bin_episodes(episodes):
    binned = defaultdict(list)
    for ep in episodes:
        n = ep.get('num_vehicles', 0)
        for (lo, hi), label in zip(BINS, BIN_LABELS):
            if lo <= n <= hi:
                binned[label].append(ep)
                break
    return binned


def main():
    if len(sys.argv) < 2:
        print("Usage: python density_analysis.py results.json [results2.json ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        with open(path) as f:
            data = json.load(f)

        method = data.get('method', path)
        episodes = data['episodes']
        binned = bin_episodes(episodes)

        print(f"\n{'='*60}")
        print(f"Density Analysis: {method}")
        print(f"{'='*60}")
        print(f"{'Density':<10} {'N':>5} {'Goal%':>8} {'Coll%':>8} {'AvgTTZ':>8}")
        print("-" * 45)

        for label in BIN_LABELS:
            eps = binned.get(label, [])
            if not eps:
                print(f"{label:<10} {'0':>5} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
                continue
            n = len(eps)
            goal_rate = np.mean([e.get('goal', False) for e in eps])
            coll_rate = np.mean([e.get('collided', False) for e in eps])
            ttz_vals = [e['ttz_vehicle'] for e in eps if e.get('ttz_vehicle', 999) < 100]
            avg_ttz = np.mean(ttz_vals) if ttz_vals else float('nan')
            print(f"{label:<10} {n:>5} {goal_rate:>7.1%} {coll_rate:>7.1%} {avg_ttz:>8.2f}")


if __name__ == '__main__':
    main()
