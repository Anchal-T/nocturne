# Phase 2 — arm matrix and compute plan (proposal)

Status: proposal for review · Created 2026-10-17
Companion record: `archive/phase1/README.md` (pipeline fixes this plan builds on)

Every number in the paper will be regenerated from these runs and nothing here
is a result yet. The arm names A/B/C/D/O/C_vis/D_vis/C_hist are kept from the
roadmap; their definitions below are mine, because no definition survives
anywhere in the repo or history. Renaming is a one-string change per arm.

## 1. Observation contract (must be fixed before any arm trains)

The paper's method text says Nocturne's ray-cast visibility model "removes every
object outside the view cone or occluded behind a nearer object" and the
surviving objects are encoded into the observation. The code does not implement
that contract yet:

- `occupancy_grid.use_occlusion` toggles visibility filtering of the TTZ/object
  data and appends the 4 occlusion summary features.
- The C++ fast grid (`Scenario::OccupancyGrid`) never receives the visibility
  set; it stamps every vehicle/pedestrian/cyclist in the scenario. The grid is
  therefore privileged — it contains occluded actors — in *both* current modes.
- Consequence: the current "no-occlusion" arm is not the paper's ablation, and
  the current "occlusion" arm still cheats through the grid.

Phase 1b items (blocking, to implement before training):

1. **Visibility-consistent grid.** Extend `Scenario::OccupancyGrid` with an
   optional visible-id filter (sets already computed by
   `_match_visible_to_vehicles`); the same ray-cast result feeds TTZ. When the
   filter is on, no actor information in the observation comes from occluded
   objects; when off, the arm is explicitly privileged.
2. **Config switches** (disjoint from arm labels):
   - `occupancy_grid.visibility_filter` (bool) — filter grid + TTZ to ray-cast
     visible actors.
   - `occupancy_grid.use_occlusion` (bool) — append the 4 summary stats
     (existing switch).
   - `occupancy_grid.occlusion_mask` (bool) — append a spatial occlusion mask
     channel (0/1 grid cell: azimuth inside a merged blocked interval with the
     cell range beyond that interval's blocker distance — same interval sweep
     `_compute_occlusion_features` already performs; ~200 cells of numpy).
   - `subscriber.frame_stack` (int, default 1) — wrapper-level stacking of the
     last k raw observations, zero-padded at episode start.
     `subscriber.n_frames_stacked` currently only affects the legacy BaseEnv
     path, not the DRL wrapper.
3. **Checkpoint observation metadata.** `DDQNAgent.save` currently records
   `obs_dim` but not the observation switches; `evaluate.py` infers
   `use_occlusion` from dim 609 vs 613. Under 8 arms that is ambiguous (A and B
   are both 609; C and C_vis both 613). Checkpoints must record
   `visibility_filter`, `use_occlusion`, `occlusion_mask`, `frame_stack`, and
   the evaluator must take them from the checkpoint (CLI override for sanity
   checks only). Silent fallback would invalidate evidence.

Dimension arithmetic: grid 3×20×10 = 600, ego 4, target 2, TTZ 3 (= 609);
+4 stats = 613; +200-cell mask channel = 809; +stats = 813; history ×k (k=3 ⇒
1839 for C_hist).

## 2. Arm matrix (8 arms × 3 seeds)

All arms: identical dueling-noisy DDQN, identical reward, identical 20-action
table, identical budget (10M env steps). They differ only in the observation,
through the four switches of §1.

| Arm | Context (grid + TTZ) | Occlusion encoding | History | obs_dim | Question it answers |
|---|---|---|---|---|---|
| A | privileged (all actors) | — | 1 | 609 | Upper bound: what perfect actor knowledge buys. |
| B | ray-cast visible | — | 1 | 609 | Primary control: cost of realistic perception (paper's no-occlusion ablation). |
| C | privileged | 4-D summary stats | 1 | 613 | Do stats help when actors are already fully visible? (leak/sanity check) |
| C_vis | ray-cast visible | 4-D summary stats | 1 | 613 | Compact model = draft's 613-D "ours". |
| C_hist | ray-cast visible | 4-D summary stats | 3 | 1839 | Is explicit occlusion better than generic memory? |
| D | privileged | spatial mask | 1 | 809 | Spatial encoding under perfect perception. |
| D_vis | ray-cast visible | spatial mask | 1 | 809 | Spatial encoding under realistic perception. |
| O | ray-cast visible | 4-D stats + mask | 1 | 813 | Ours: combined encoding. |

Core contrasts:

- **O vs B** — main claim: occlusion encoding under realistic perception.
- **A vs B** — perception gap: cost of only seeing what is visible.
- **C_vis vs C, D_vis vs D** — does each encoding matter more/only when
  perception is realistic?
- **C_hist vs C_vis** — generic temporal memory vs explicit occlusion.
- **O vs C_vis, O vs D_vis** — combined vs single encoding.

Out of scope (per roadmap decisions): IQN/CVaR, any SOTA goal-rate claim, and
the draft's depth/optimizer ablation rows (they need their own budget decision;
they are not part of the 24 runs).

Seeds: 42, 123, 456 — fixed a priori for all arms. No per-arm tuning; the val
half of `full_valid` is used for sanity checks only, never for checkpoint
selection. Evaluation uses the final checkpoint.

## 3. Compute plan

### 3.1 Hardware (verified 2026-10-17)

- 4 × Tesla K80 (11 GB, sm_37), all idle; torch 1.12.1+cu102 sees all four,
  matmul verified on `cuda:0`.
- 48 vCPU, 251 GB RAM.
- No X11/Xvfb; training runs headless (already the case).

### 3.2 Per-run layout (pinned)

- `drl.vec_env_mode=sync`, `drl.num_workers=4`, `drl.num_envs_per_worker=12`
  (48 envs), `drl.train_freq=128`, `drl.batch_size=512`.
- `drl.max_env_steps=10000000`, `drl.min_runtime_seconds=0`.
- Everything else stays at the shipped defaults (dueling + noisy DDQN, PER,
  n-step 3, Muon, SPR) — inherited pipeline, not claimed contributions.

### 3.3 Parallelism

- 4 concurrent jobs, one per GPU (`CUDA_VISIBLE_DEVICES`), `OMP_NUM_THREADS=1`.
- Footprint per job: 4 env worker processes + 1 learner + 1 main ≈ 6 processes;
  4 jobs ≈ 24 of 48 cores (headroom for evaluation and other work).
- 24 runs = 6 waves of 4. Resume is not needed (fixed budget), but checkpoints
  are retained.

### 3.4 Throughput

- Calibration run (200k steps, exact layout above) started 2026-10-17:
  measured throughput **TBD** env-steps/s per job from the run logs after
  warmup.
- Per-run time = 10M / (R × 3600) h; total wall clock ≈ 6 waves × per-run time
  (worst case, with 4-way parallelism). Fill in once R is measured.

### 3.5 Launch template

```
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 \
python -m examples.drl_collision_avoidance.train \
  seed=$SEED \
  drl.checkpoint_dir=checkpoints/$ARM/seed_$SEED \
  drl.max_env_steps=10000000 drl.min_runtime_seconds=0 \
  drl.num_workers=4 drl.num_envs_per_worker=12 drl.num_envs=48 \
  drl.vec_env_mode=sync drl.train_freq=128 drl.batch_size=512 \
  occupancy_grid.visibility_filter=$VIS occupancy_grid.use_occlusion=$OCC \
  occupancy_grid.occlusion_mask=$MASK subscriber.frame_stack=$STACK \
  hydra.run.dir=logs/$ARM/seed_$SEED
```

Arm → flags (transcription table for the launcher once approved):

| Arm | visibility_filter | use_occlusion | occlusion_mask | frame_stack |
|---|---|---|---|---|
| A | false | false | false | 1 |
| B | true | false | false | 1 |
| C | false | true | false | 1 |
| C_vis | true | true | false | 1 |
| C_hist | true | true | false | 3 |
| D | false | false | true | 1 |
| D_vis | true | false | true | 1 |
| O | true | true | true | 1 |

Planned launcher: `scripts/launch_controlled_study.sh <arm> <seed> <gpu>`
(created after review), one line per job appended to
`paper/workspace/launches.txt` with the exact command + git sha.

### 3.6 Evaluation (accepted evidence only)

```
python examples/drl_collision_avoidance/evaluate.py \
  --checkpoint checkpoints/$ARM/seed_$SEED/ddqn_final.pth \
  --scenario_path dataset/full_valid/formatted_json_v2_no_tl_valid \
  --split test --split_seed 0 --num_files -1 \
  --output_json paper/workspace/eval/${ARM}_seed${SEED}.json \
  --log_steps
```

- Full test half = 6,102 scenes, one paired ego per scene, no subsampling.
- Eval is single-process CPU; 4–6 evals in parallel after a wave completes,
  overlapped with the next wave where cores allow.
- Provenance JSONs in `paper/workspace/eval/` are the only inputs to Phase 4
  tables/figures.

### 3.7 Storage layout

- `checkpoints/<arm>/seed_<seed>/` — `ddqn_final.pth` (+ periodic saves).
- `logs/<arm>/seed_<seed>/` — hydra + tensorboard.
- `paper/workspace/eval/<arm>_seed<seed>.json` (+ `.steps.jsonl`).
- `paper/workspace/launches.txt` — exact command per run.

## 4. Order of operations

1. Implement Phase 1b (§1) + tests: arm configs round-trip through
   checkpoint/load; obs dims match the table; mask deterministic; evaluator
   refuses checkpoints without observation metadata.
2. Smoke-run every arm config for ~1k steps; verify obs dim, provenance.
3. Freeze configs, record git sha, launch wave 1 (4 arms × 4 GPUs).
4. Six waves; evaluate each finished run on the test half.
5. Phase 3/4: regenerate every table/figure from `paper/workspace/eval/*.json`.

## 5. Needs sign-off before freezing

1. The arm definitions above (esp. `_vis` = ray-cast-visible context vs
   privileged; `_hist` = 3-frame stack).
2. "Ours" = O (stats + mask, 813-D) vs the draft's 613-D stats-only model. If
   O must stay the 613-D model, then O := C_vis, and the mask family (D/O)
   becomes a secondary experiment — but the 8-arm matrix then needs one more
   distinct arm.
3. Seeds 42/123/456 and "final checkpoint is the evaluated one".
