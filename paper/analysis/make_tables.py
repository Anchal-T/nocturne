"""Build the reviewer-response tables from evaluation JSON.

Reads ``paper/runs data/ablation/<arm>/seed_<k>/eval_test.json`` and writes
``paper/analysis/generated_results.tex``. The archived single-checkpoint JSON
is summarized only for fields it actually contains. This script does not
train or step the simulator.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from examples.drl_collision_avoidance.episode_metrics import wilson_interval
ABLATION = ROOT / "runs data" / "ablation"
ARCHIVE = ROOT / "runs data" / "eval_baseline_fixed.json"
OUT = Path(__file__).resolve().parent / "generated_results.tex"

DENSITY_BINS = ((0, 1, "0"), (1, 5, "1--4"), (5, 10, "5--9"), (10, 15, "10--14"), (15, 10**9, "15+"))
OCC_BINS = ((0.0, 0.1, "0--0.1"), (0.1, 0.3, "0.1--0.3"), (0.3, 0.6, "0.3--0.6"), (0.6, 1.01, "0.6--1"))

ARM_ORDER = (
    "A_no_occ_no_risk",
    "B_occ_no_risk",
    "C_no_occ_risk",
    "D_occ_risk",
    "O_full_obs_risk",
)
ARM_LABEL = {
    "A_no_occ_no_risk": "No occlusion features, no TTZ reward",
    "B_occ_no_risk": "Occlusion features, no TTZ reward",
    "C_no_occ_risk": "No occlusion features, TTZ reward",
    "D_occ_risk": "Occlusion features + TTZ reward",
    "O_full_obs_risk": "Full observability + TTZ reward",
}


def _load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def _episodes(payload):
    return payload.get("episodes") or []


def _rate(episodes, key):
    if not episodes:
        return None
    return sum(1 for ep in episodes if ep.get(key)) / len(episodes)


def _mean(episodes, key):
    values = [ep[key] for ep in episodes if ep.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _fmt_pct(rate):
    if rate is None:
        return "--"
    return f"{100.0 * rate:.1f}"


def _fmt_num(value, digits=2):
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def _seed_dirs(arm_dir: Path):
    return sorted(path for path in arm_dir.glob("seed_*") if path.is_dir())


def _arm_seed_payloads():
    found = {}
    if not ABLATION.is_dir():
        return found
    for arm_dir in sorted(path for path in ABLATION.iterdir() if path.is_dir()):
        payloads = []
        for seed_dir in _seed_dirs(arm_dir):
            path = seed_dir / "eval_test.json"
            if path.is_file():
                payloads.append(_load_json(path))
        if payloads:
            found[arm_dir.name] = payloads
    return found


def _mean_std(values):
    usable = [value for value in values if value is not None]
    if not usable:
        return None, None
    mean = sum(usable) / len(usable)
    if len(usable) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in usable) / (len(usable) - 1)
    return mean, math.sqrt(var)


def _seed_metric(payloads, fn):
    return _mean_std([fn(_episodes(payload)) for payload in payloads])


def _tex_rate(mean, std):
    if mean is None:
        return "--"
    return f"{100.0 * mean:.1f} $\\pm$ {100.0 * std:.1f}"


def _comparison_table(arms):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Held-out test rates. Each cell is the mean $\pm$ standard deviation across training seeds. Success is a goal with no collision.}",
        r"\label{tab:ablation}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Arm & Success (\%) & Collision (\%) & Timeout (\%) & Speed (m/s) & Min TTZ (s) \\",
        r"\midrule",
    ]
    for arm in ARM_ORDER:
        payloads = arms.get(arm)
        if not payloads:
            continue
        success = _seed_metric(payloads, lambda eps: _rate(eps, "success"))
        collision = _seed_metric(payloads, lambda eps: _rate(eps, "collided"))
        timeout = _seed_metric(payloads, lambda eps: _rate(eps, "timeout"))
        speed = _seed_metric(payloads, lambda eps: _mean(eps, "avg_speed"))
        ttz = _seed_metric(payloads, lambda eps: _mean(eps, "min_ttz_gt"))
        lines.append(
            f"{ARM_LABEL[arm]} & {_tex_rate(*success)} & {_tex_rate(*collision)} & "
            f"{_tex_rate(*timeout)} & {_fmt_num(speed[0])} $\\pm$ {_fmt_num(speed[1])} & "
            f"{_fmt_num(ttz[0])} $\\pm$ {_fmt_num(ttz[1])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def _in_bin(value, low, high):
    return value is not None and low <= value < high


def _bin_table(arms, key, bins, caption, label):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Arm & Bin & $n$ & Collision (\%) & Timeout (\%) \\",
        r"\midrule",
    ]
    any_row = False
    for arm in ("A_no_occ_no_risk", "D_occ_risk"):
        payloads = arms.get(arm)
        if not payloads:
            continue
        episodes = [ep for payload in payloads for ep in _episodes(payload)]
        for low, high, name in bins:
            group = [ep for ep in episodes if _in_bin(ep.get(key), low, high)]
            if not group:
                continue
            any_row = True
            lines.append(
                f"{ARM_LABEL[arm]} & {name} & {len(group)} & "
                f"{_fmt_pct(_rate(group, 'collided'))} & {_fmt_pct(_rate(group, 'timeout'))} \\\\"
            )
    if not any_row:
        return ""
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _paired_reduction(arms):
    base = arms.get("A_no_occ_no_risk")
    proposed = arms.get("D_occ_risk")
    if not base or not proposed:
        return ""
    by_seed = []
    for left, right in zip(base, proposed):
        right_by_file = {
            ep.get("scenario_file"): ep
            for ep in _episodes(right)
            if ep.get("scenario_file")
        }
        removed = []
        for ep in _episodes(left):
            match = right_by_file.get(ep.get("scenario_file"))
            if match is None or not ep.get("collided") or match.get("collided"):
                continue
            removed.append(ep)
        by_seed.append(removed)
    if not any(by_seed):
        return (
            "Paired scenario files were not available, so the collision-type "
            "reduction is not reported."
        )
    counts = defaultdict(int)
    hidden = 0
    total = 0
    for removed in by_seed:
        for ep in removed:
            total += 1
            counts[ep.get("collision_type") or "unspecified"] += 1
            if ep.get("hidden_at_collision"):
                hidden += 1
    parts = ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))
    return (
        f"Across paired scenarios, arm A collided and arm D did not in {total} "
        f"seed-scenario cases ({parts}). {hidden} of those removed collisions "
        "were labeled hidden-at-collision, meaning the nearest object within "
        "5\\,m was outside the ray-cast set one second earlier."
    )


def _failure_rows(arms):
    payloads = arms.get("D_occ_risk")
    if not payloads:
        return ""
    episodes = [ep for payload in payloads for ep in _episodes(payload) if not ep.get("success")]
    if not episodes:
        return ""
    groups = defaultdict(int)
    for ep in episodes:
        groups[(ep.get("outcome"), ep.get("maneuver"), ep.get("partner"))] += 1
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Failure modes of the proposed arm. A row is an outcome, the maneuver inside the conflict window, and the collision partner.}",
        r"\label{tab:failures}",
        r"\begin{tabular}{lllr}",
        r"\toprule",
        r"Outcome & Maneuver & Partner & Count \\",
        r"\midrule",
    ]
    for (outcome, maneuver, partner), count in sorted(groups.items(), key=lambda item: -item[1]):
        lines.append(f"{outcome} & {maneuver} & {partner} & {count} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _robustness(arm_dir_name="D_occ_risk"):
    seed_dir = ABLATION / arm_dir_name / "seed_42"
    if not seed_dir.is_dir():
        return ""
    rows = []
    for path in sorted(seed_dir.glob("eval_test*.json")):
        payload = _load_json(path)
        episodes = _episodes(payload)
        rows.append((
            path.stem,
            len(episodes),
            _rate(episodes, "success"),
            _rate(episodes, "collided"),
            _rate(episodes, "timeout"),
        ))
    if len(rows) <= 1:
        return ""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Eval-time robustness on the proposed seed-42 checkpoint. Dropout zeros the four occlusion features. Noise is added to the full observation. The 120-degree run changes only the view cone.}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Probe & $n$ & Success (\%) & Collision (\%) & Timeout (\%) \\",
        r"\midrule",
    ]
    for name, n, success, collision, timeout in rows:
        lines.append(
            f"{name} & {n} & {_fmt_pct(success)} & {_fmt_pct(collision)} & {_fmt_pct(timeout)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _archive_note():
    if not ARCHIVE.is_file():
        return ""
    payload = _load_json(ARCHIVE)
    episodes = _episodes(payload)
    n = len(episodes)
    if n == 0:
        return ""
    goals = sum(1 for ep in episodes if ep.get("goal"))
    collisions = sum(1 for ep in episodes if ep.get("collided"))
    timeouts = sum(
        1 for ep in episodes
        if (not ep.get("goal")) and (not ep.get("collided")) and ep.get("length", 0) >= 80
    )
    goal_low, goal_high = wilson_interval(goals, n)
    col_low, col_high = wilson_interval(collisions, n)
    return (
        "The archived checkpoint \\texttt{ddqn\\_ep400000.pth} remains the "
        f"fully trained DDQN reported in the main held-out table: {goals} goals "
        f"and {collisions} collisions in {n} episodes "
        f"(goal Wilson interval {100 * goal_low:.2f}--{100 * goal_high:.2f}\\%, "
        f"collision Wilson interval {100 * col_low:.2f}--{100 * col_high:.2f}\\%). "
        f"Episodes that are neither a goal nor a collision and last 80 steps: {timeouts} "
        f"({100 * timeouts / n:.2f}\\%). That file records \\texttt{{num\\_vehicles}}=50 "
        "on every episode, so it cannot support a traffic-density curve, and it "
        "does not record occlusion fraction, hidden contact, or a maneuver label. "
        "Those cuts come from the revised evaluator."
    )


def main():
    arms = _arm_seed_payloads()
    chunks = [_archive_note()]
    if not arms:
        chunks.append(
            "The 2$\\times$2 evaluation JSONs are not in this draft yet. "
            "After the training machine finishes \\texttt{scripts/launch\\_reviewer\\_ablation.sh}, "
            "re-run \\texttt{python paper/analysis/make\\_tables.py} to fill the tables below."
        )
    else:
        chunks.append(_comparison_table(arms))
        chunks.append(_paired_reduction(arms))
        chunks.append(_bin_table(
            arms, "vehicles_in_range", DENSITY_BINS,
            "Collision and timeout rates by the mean number of vehicles inside 40\\,m.",
            "tab:density",
        ))
        chunks.append(_bin_table(
            arms, "mean_occ_frac", OCC_BINS,
            "Collision and timeout rates by mean occluded view fraction.",
            "tab:occlusion-bins",
        ))
        chunks.append(_failure_rows(arms))
        chunks.append(_robustness())
    text = "\n\n".join(chunk for chunk in chunks if chunk) + "\n"
    OUT.write_text(text)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
