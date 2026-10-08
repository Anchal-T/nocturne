"""Draw one figure per saved evaluation trace.

Each figure shows the ego and logged trajectories, the occupancy channel at
the closest approach, and the speed, time-to-zero, and occlusion traces.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "figures" / "episodes"


def _plot(path: Path, dest: Path):
    with path.open() as handle:
        trace = json.load(handle)
    ego = np.asarray(trace.get("ego_xy") or [], dtype=float)
    expert = np.asarray(trace.get("expert_xy") or [], dtype=float)
    speed = np.asarray(trace.get("speed") or [], dtype=float)
    ttz = np.asarray([np.nan if v is None else v for v in trace.get("ttz_gt") or []], dtype=float)
    occ = np.asarray(trace.get("occ_frac") or [], dtype=float)
    grid = trace.get("occupancy_at_min_ttz")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    ax = axes[0, 0]
    if len(expert):
        ax.plot(expert[:, 0], expert[:, 1], color="#888888", label="logged")
    if len(ego):
        ax.plot(ego[:, 0], ego[:, 1], color="#245b8a", label="agent")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Trajectory")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    if grid:
        ax.imshow(np.asarray(grid), origin="upper", cmap="magma")
    ax.set_title("Occupancy at minimum TTZ")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1, 0]
    steps = np.arange(len(speed)) * 0.1
    if len(speed):
        ax.plot(steps, speed, color="#245b8a", label="speed")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    twin = ax.twinx()
    if len(ttz):
        twin.plot(np.arange(len(ttz)) * 0.1, ttz, color="#a63d40", label="TTZ")
    twin.set_ylabel("Ground-truth TTZ (s)")
    ax.set_title("Speed and time-to-zero")

    ax = axes[1, 1]
    if len(occ):
        ax.plot(np.arange(len(occ)) * 0.1, occ, color="#b06b32")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Occluded fraction")
    ax.set_ylim(0, 1)
    title = f"ep {trace.get('episode')}  {trace.get('outcome')}  {trace.get('maneuver')}"
    ax.set_title("Occlusion")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_dir", type=str, required=True)
    args = parser.parse_args()
    trace_dir = Path(args.trace_dir)
    written = 0
    for path in sorted(trace_dir.glob("*.json")):
        _plot(path, OUT / f"{path.stem}.png")
        written += 1
    print(f"Wrote {written} figures to {OUT}")


if __name__ == "__main__":
    main()
