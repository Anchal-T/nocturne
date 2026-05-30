"""Compute statistics across seeds: mean ± std, 95% CI, Welch's t-test.

Usage:
    python scripts/compute_statistics.py results_seed1.json results_seed2.json ...

Each JSON file is the output of evaluate.py --output_json (one file per seed).
Group files by method name (from the 'method' field in each JSON).
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    data = np.array(data)
    means = [np.mean(np.random.choice(data, size=len(data), replace=True))
             for _ in range(n_bootstrap)]
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_statistics.py result1.json result2.json ...")
        sys.exit(1)

    # Group files by method
    by_method = defaultdict(list)
    for path in sys.argv[1:]:
        with open(path) as f:
            data = json.load(f)
        by_method[data['method']].append(data)

    metrics = ['goal_rate', 'collision_rate', 'ade', 'fde', 'jerk', 'smoothness']

    print(f"{'Method':<15} {'Seeds':>5}", end="")
    for m in metrics:
        print(f"  {m:<22}", end="")
    print()
    print("-" * 160)

    method_values = {}
    for method, seeds in by_method.items():
        row = f"{method:<15} {len(seeds):>5}"
        method_values[method] = {}
        for metric in metrics:
            values = [s[metric] for s in seeds if s.get(metric) is not None]
            if not values:
                row += f"  {'N/A':<22}"
                continue
            mean = np.mean(values)
            std = np.std(values)
            lo, hi = bootstrap_ci(values) if len(values) >= 3 else (mean, mean)
            row += f"  {mean:.3f}±{std:.3f} [{lo:.3f},{hi:.3f}]"
            method_values[method][metric] = values
        print(row)

    # Pairwise t-tests
    methods = list(by_method.keys())
    if len(methods) > 1:
        baseline = methods[0]
        print(f"\nWelch's t-test vs '{baseline}':")
        for method in methods[1:]:
            for metric in metrics:
                a = method_values.get(baseline, {}).get(metric)
                b = method_values.get(method, {}).get(metric)
                if a and b and len(a) >= 2 and len(b) >= 2:
                    t, p = stats.ttest_ind(a, b, equal_var=False)
                    sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
                    print(f"  {method} [{metric}]: t={t:.3f}, p={p:.4f} {sig}")


if __name__ == '__main__':
    main()
