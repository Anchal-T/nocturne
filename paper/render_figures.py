import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
BLUE, RED, GREEN, ORANGE, GRAY, LIGHT = "#245b8a", "#a63d40", "#3b7a57", "#b06b32", "#555555", "#e9eef2"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8, "axes.linewidth": 0.6,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06, "grid.alpha": 0.18, "grid.linewidth": 0.5,
})

# Figure 1: system overview (conceptual; no numerical claims).
fig, ax = plt.subplots(figsize=(7.0, 3.0)); ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")
def box(x, y, w, h, text, color=LIGHT):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12", fc=color, ec=BLUE, lw=0.8)
    ax.add_patch(p); ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=7.2)
def arrow(x1, y1, x2, y2, label=None):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="->",mutation_scale=9,color=GRAY,lw=0.8))
    if label: ax.text((x1+x2)/2,(y1+y2)/2+0.15,label,ha="center",fontsize=6.3,color=GRAY)
box(.2,4.25,2.0,.85,"Ray-cast visible\nroad users", "#dce8f2")
box(.2,2.75,2.0,.85,"Map geometry +\nego state", "#dce8f2")
box(2.8,4.35,2.3,.75,"3-channel\noccupancy grid", "#e8efe8")
box(2.8,3.15,2.3,.75,"TTZ conflict\nfeatures", "#e8efe8")
box(2.8,1.95,2.3,.75,"Occlusion-risk\nsummaries", "#e8efe8")
box(5.8,3.0,2.1,1.25,"613-D\nobservation", "#f1e7db")
box(8.6,3.0,2.1,1.25,"Dueling noisy\nDDQN", "#dce8f2")
box(11.4,3.0,2.2,1.25,"15 discrete\nthrottle-steering\nactions", "#e8efe8")
box(8.5,.55,2.3,1.0,"Prioritized replay\n+ n-step targets", "#f1e7db")
box(11.4,.55,2.2,1.0,"Simulator actors\n48 environments", "#f1e7db")
for y in (4.68,3.18): arrow(2.2,y,2.8,y)
arrow(5.1,4.72,5.8,3.95); arrow(5.1,3.52,5.8,3.62); arrow(5.1,2.32,5.8,3.3)
arrow(7.9,3.62,8.6,3.62); arrow(10.7,3.62,11.4,3.62)
arrow(12.5,3.0,12.5,1.55,"transition"); arrow(11.4,1.05,10.8,1.05); arrow(9.65,1.55,9.65,3.0,"update")
fig.savefig(OUT/"system_overview.png"); plt.close(fig)

# Figure 2: parse every verifiable training window from the supplied transcript.
text = (ROOT / "runs data" / "agent_results_transcript.txt").read_text(errors="ignore")
pat = re.compile(r"Episode\s+([\d,]+)\s*\|\s*avg_reward=\s*([-+]?\d+(?:\.\d+)?)\s*\|\s*eps=\s*([\d.]+).*?\|\s*env_steps=\s*([\d,]+)")
rows = {}
for m in pat.finditer(text):
    ep, rew, eps, steps = int(m.group(1).replace(",","")), float(m.group(2)), float(m.group(3)), int(m.group(4).replace(",",""))
    rows[ep] = (rew, eps, steps)
points = sorted((ep, *v) for ep,v in rows.items())
if not points:
    raise RuntimeError("No training records found in transcript")
episodes, rewards, epsilon, steps = map(list, zip(*points))
steps_m = [s/1e6 for s in steps]
fig, ax = plt.subplots(figsize=(5.5,3.09)); ax2 = ax.twinx()
# Do not connect intervals missing from the supplied transcript.
segments, start = [], 0
for i in range(1, len(steps_m)):
    if steps_m[i] - steps_m[i-1] > 1.0:
        segments.append((start, i)); start = i
segments.append((start, len(steps_m)))
for j, (a, b) in enumerate(segments):
    ax.plot(steps_m[a:b], rewards[a:b], color=BLUE, lw=1.15,
            label="1,000-episode mean reward" if j == 0 else "_nolegend_")
    ax2.plot(steps_m[a:b], epsilon[a:b], color=RED, lw=1.0, ls="--",
             label=r"Exploration $\epsilon$" if j == 0 else "_nolegend_")
ax.axhline(0,color=GRAY,lw=.6,ls="--"); ax.axvline(1.0,color=ORANGE,lw=.8,ls=":",label="Replay capacity reached")
ax.set_xlabel("Environment steps (millions)"); ax.set_ylabel("Mean episode reward",color=BLUE); ax2.set_ylabel(r"Exploration $\epsilon$",color=RED)
ax.grid(True); ax.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
lines = [l for l in ax.get_lines()+ax2.get_lines() if not l.get_label().startswith("_")]
labels=[l.get_label() for l in lines]
ax.legend(lines,labels,loc="lower right",frameon=True)
ax.set_title("DDQN training dynamics")
fig.tight_layout(); fig.savefig(OUT/"training_curves.png"); plt.close(fig)

# Figure 3: evaluation outcomes and continuous-metric distributions.
data=json.loads((ROOT/"runs data"/"eval_baseline_fixed.json").read_text()); es=data["episodes"]; n=len(es)
fig, axes=plt.subplots(1,3,figsize=(7.0,2.65))
counts=[sum(bool(e.get("goal")) for e in es),sum(bool(e.get("collided")) for e in es)]
axes[0].bar(["Goal","Collision"],[100*c/n for c in counts],color=[GREEN,RED],width=.6)
for i,c in enumerate(counts): axes[0].text(i,100*c/n+2,f"{100*c/n:.2f}%\n({c}/{n})",ha="center",fontsize=7)
axes[0].set_ylabel("Episode rate (%)"); axes[0].set_ylim(0,108); axes[0].set_title("Held-out outcomes")
for ax,k,title,color in [(axes[1],"ade","ADE (m)",BLUE),(axes[2],"jerk","Mean |jerk| (m/s³)",ORANGE)]:
    vals=[float(e[k]) for e in es if e.get(k) is not None]
    ax.hist(vals,bins=28,color=color,alpha=.82,edgecolor="white",linewidth=.3)
    mean=sum(vals)/len(vals); sd=math.sqrt(sum((v-mean)**2 for v in vals)/(len(vals)-1))
    ax.axvline(mean,color=RED,lw=1,ls="--"); ax.set_title(f"{title}\n{mean:.2f} ± {sd:.2f}, n={len(vals):,}")
    ax.set_xlabel(title); ax.set_ylabel("Episodes")
for ax in axes:
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(axis="y")
fig.tight_layout(); fig.savefig(OUT/"evaluation_summary.png"); plt.close(fig)

print(f"Parsed {len(points)} unique training windows; rendered 3 figures to {OUT}")
