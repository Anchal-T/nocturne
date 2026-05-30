"""Compute statistics across seeds: mean ± std, 95% CI, Mann-Whitney U-test.

Follows Agarwal et al., NeurIPS 2021 (arXiv:2108.13264): non-parametric tests for
small N, stratified bootstrap CIs.

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
    """Vectorized bootstrap confidence interval."""
    data = np.asarray(data)
    n = len(data)
    # Generate all bootstrap samples at once: (n_bootstrap, n) index matrix
    idx = np.random.randint(0, n, size=(n_bootstrap, n))
    means = data[idx].mean(axis=1)
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lower), float(upper)


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_statistics.py result1.json result2.json ...")
        sys.exit(1)

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

    methods = list(by_method.keys())
    if len(methods) > 1:
        baseline = methods[0]
        print(f"\nMann-Whitney U-test vs '{baseline}':")
        for method in methods[1:]:
            for metric in metrics:
                a = method_values.get(baseline, {}).get(metric)
                b = method_values.get(method, {}).get(metric)
                if a and b and len(a) >= 2 and len(b) >= 2:
                    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
                    sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
                    print(f"  {method} [{metric}]: U={u:.1f}, p={p:.4f} {sig}")


if __name__ == '__main__':
    main()
