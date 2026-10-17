# Phase 2 — arm matrix and compute plan (frozen for implementation)

Status: frozen 2026-10-17 after sign-off; Phase 1b implemented and smoke-tested
2026-10-17 · Created 2026-10-17
Companion record: `archive/phase1/README.md` (pipeline fixes this plan builds on)

Every number in the paper will be regenerated from these runs and nothing here
is a result yet.

## 0. Where the arm definitions come from

The roadmap pasted on 2026-10-17 is the only surviving source for the arm
names. It does **not** define A/B/C/D/O; it says only "A, B, C, D, and O, as
they exist today", and those variants do not exist in the code. A search of
the repo (including untracked files), every git ref and the pre-purge paper,
the Cursor plan directory, and the OpenCode session database found no other
definition. The roadmap's only explicit constraints are:

- `C_vis` / `D_vis` = C and D with `reward.reward_ttz_source=visible`;
  `{C, C_vis, D, D_vis}` form a 2x2 of occlusion summaries x reward source.
- `C_hist` = memory baseline, occlusion summaries **off**, grid history **on**
  (`history_frames=3`, `history_stride=5`, i.e. 1 s).
- H1 = cost of partial observability, O vs A.
- H2 = summaries cut hidden-partner collisions.
- H5 = summaries help beyond memory, D vs `C_hist`.

The matrix below is the minimal reconstruction that satisfies all of these:
A is the privileged-observation upper bound; B is the realistic-perception
baseline; C (features) and D (spatial mask) are the two occlusion-summary
encodings; O is their union (813-D, confirmed); `C_vis`/`D_vis` change only the
reward source; `C_hist` swaps explicit summaries for grid history. Letters are
kept for continuity with the roadmap. This is a documented design decision,
not a recovered original.

## 1. Observation and reward contract (must be fixed before any arm trains)

The paper's method text says the ray-cast visibility model removes every
object outside the view cone or occluded behind a nearer object before the
observation is built. Before Phase 1b the code did not implement that contract:

- `occupancy_grid.use_occlusion` simultaneously (a) enabled visibility
  filtering of the TTZ/object data and (b) appended the 4 occlusion summary
  features. Visibility and encoding were not independent switches.
- The C++ fast grid (`Scenario::OccupancyGrid`) never received the visibility
  set; it stamped every vehicle/pedestrian/cyclist in the scenario. The grid
  was therefore privileged — it contained occluded actors — in *both* modes.
- The reward reused the observation TTZ, so turning visibility filtering on
  silently changed the reward source too.
- `train.py` hardcoded `grid_channels = 3`; there was no grid history.

Phase 1b is now implemented (commit `phase1-pipeline-fixes`, 2026-10-17):

1. **Visibility-consistent grid.** Extend `Scenario::OccupancyGrid` with an
   optional visible-id filter (the set already computed by
   `_match_visible_to_vehicles`); the same ray-cast result feeds TTZ. When the
   filter is on, no actor information in the observation comes from occluded
   objects; when off, the arm is explicitly privileged.
2. **Config switches** (orthogonal; arm labels are not switches):
   - `occupancy_grid.visibility_filter` (bool) — filter grid + observation
     TTZ/object features to ray-cast visible actors.
   - `occupancy_grid.use_occlusion` (bool) — append the 4 summary stats.
   - `occupancy_grid.occlusion_mask` (bool) — append a 200-cell spatial
     occlusion mask (0/1 per grid cell: azimuth inside a merged blocked
     interval and range beyond that interval's blocker distance).
   - `occupancy_grid.history_frames` (int, default 1) and
     `occupancy_grid.history_stride` (int, default 1) — grid-only history.
     Keep a deque of past grids sampled every `history_stride` sim steps,
     current grid first, zero-padded at episode start. `grid_channels =
     3 * history_frames`. This replaces the earlier full-observation
     `subscriber.frame_stack` idea and matches the roadmap (3 frames,
     stride 5 = 1 s).
   - `reward.reward_ttz_source` in `{privileged, visible}`, default
     `privileged` — the reward TTZ is computed independently of the
     observation source. `privileged` uses all actors; `visible` uses only
     ray-cast-visible actors. This removes today's coupling.
3. **Checkpoint observation metadata.** `DDQNAgent.save` now records
   `observation_metadata` from
   `examples/drl_collision_avoidance/observation_config.py` (the single
   contract shared with BC); `evaluate.py` reads it and refuses checkpoints
   without it (CLI override only for sanity checks). Keys: `obs_dim`,
   `visibility_filter`, `use_occlusion`, `occlusion_mask`, `history_frames`,
   `history_stride`, `reward_ttz_source`, `grid_rows`, `grid_cols`,
   `grid_channels`. A and B are both 609-D, so inference would be silent
   evidence corruption.

Implementation evidence (2026-10-17):

- C++ `Scenario::OccupancyGridVisible` filters actors by the ray-cast id set;
  road edges stay. Exposed as `occupancy_grid_visible` and wired through
  `_build_occupancy_grid`.
- `_build_observation` computes observation TTZ from the visible set when
  `visibility_filter` is on, and reward TTZ independently from
  `reward_ttz_source`. `info` logs `ttz_vehicle` / `ttz_pedestrian` as the
  privileged/physical TTZ, plus `*_observed` and `*_reward` variants, so the
  reward ablation cannot be mistaken for a scene change.
- Tests: `tests/test_observation_contract.py` (15 tests: dimensions, grid
  subset, C++ filter, reward decoupling, mask geometry, history padding,
  metadata round-trip/refusal, agent save). Full suite: 58 passed (excluding
  the pre-existing headless `getConeImage` and missing-`matplotlib` tests).
- Smokes: all 8 arm configs at 300 steps (609/613/809/813/1809 as expected);
  O and C_hist at the production 4x12 layout (48 envs, 1000 steps, 17 s each)
  with final checkpoints; evaluator ran O on 2 test-half scenes with metadata
  enforcement.

Dimension arithmetic: grid 3x20x10 = 600, ego 4, target 2, TTZ 3 (= 609);
+4 stats = 613; +200-cell mask = 809; +stats+mask = 813; grid history k=3
(stats off) = 3x600 + 4 + 2 + 3 = **1809**.

## 2. Arm matrix (8 arms x 3 seeds)

All arms: identical dueling-noisy DDQN, identical reward (except the explicit
reward-source switch in `C_vis`/`D_vis`), identical 20-action table, identical
budget (10M env steps). They differ only in the observation and reward-source
switches above. All arms use the privileged reward except `C_vis`/`D_vis`.

| Arm | Observation context | Occlusion encoding | Grid history | obs_dim | Question it answers |
|---|---|---|---|---|---|
| A | privileged (all actors) | — | 1 | 609 | Upper bound; H1 anchor. |
| B | ray-cast visible | — | 1 | 609 | Realistic-perception baseline. |
| C | ray-cast visible | 4-D summary stats | 1 | 613 | Compact summaries. |
| C_vis | ray-cast visible | 4-D summary stats | 1 | 613 | C + visible reward; H4. |
| D | ray-cast visible | spatial mask | 1 | 809 | Spatial encoding; H5 anchor. |
| D_vis | ray-cast visible | spatial mask | 1 | 809 | D + visible reward; H4. |
| O | ray-cast visible | 4-D stats + mask | 1 | 813 | Ours: combined encoding. |
| C_hist | ray-cast visible | none (summaries off) | 3 / stride 5 | 1809 | Memory baseline; H5. |

Core contrasts:

- **O vs B** — main claim: occlusion encoding under realistic perception.
- **A vs B** — perception gap; with O vs A it measures the cost of partial
  observability (H1).
- **C vs B, D vs B, O vs B** — do summaries cut hidden-partner collisions
  (H2)?
- **C_vis vs C, D_vis vs D** — does the reward source change the value of the
  summaries (H4; the roadmap's 2x2)?
- **D vs C_hist** — spatial summaries vs generic grid memory (H5).
- **O vs C, O vs D** — combined vs single encoding.

Out of scope (per roadmap decisions): IQN/CVaR, any SOTA goal-rate claim, and
the draft's depth/optimizer ablation rows (they need their own budget decision;
they are not part of the 24 runs).

Seeds: 42, 123, 456 — fixed a priori for all arms. No per-arm tuning; the val
half of `full_valid` is used for sanity checks only, never for checkpoint
selection. Evaluation uses the final checkpoint.

## 3. Compute plan

### 3.1 Hardware (verified 2026-10-17)

- 4 x Tesla K80 (11 GB, sm_37), all idle; torch 1.12.1+cu102 sees all four,
  matmul verified on `cuda:0`.
- 48 vCPU, 251 GB RAM. The K80s run near 0% utilization: training is
  CPU/simulation-bound.
- No X11/Xvfb; training runs headless (already the case).

### 3.2 Per-run layout (pinned)

- `drl.vec_env_mode=sync`, `drl.num_workers=4`, `drl.num_envs_per_worker=12`
  (48 envs), `drl.train_freq=128`, `drl.batch_size=512`.
- `drl.max_env_steps=10000000`, `drl.min_runtime_seconds=0`.
- Everything else stays at the shipped defaults (dueling + noisy DDQN, PER,
  n-step 3, Muon, SPR) — inherited pipeline, not claimed contributions.

### 3.3 Parallelism

- **2 concurrent training jobs**, one per GPU (`CUDA_VISIBLE_DEVICES`),
  `OMP_NUM_THREADS=1`. Aggregate throughput is flat above 2 jobs (see 3.4), so
  4-way concurrency only increases per-run wall time and starves evaluation.
- Footprint per job: 4 env worker processes + 1 learner + 1 main ~= 6
  processes; 2 jobs ~= 12 of 48 cores, leaving room for 4-6 parallel
  evaluations on the remaining cores.
- 12 waves of 2 runs. Re-probe if the external load (other users) changes.

### 3.4 Throughput (measured 2026-10-17)

| Mode | Per-job | Aggregate | Notes |
|---|---|---|---|
| Solo 200k-step calibration | 436/s incl. startup; 515-570/s steady | 515-570/s | 200,016 steps in 457.8 s |
| 2-way probe, 60k steps/job | 248-303/s (last 150 s) | **~536/s** (sum of last-150 s); 551/s sum of per-job averages | Wave wall 238 s |
| 4-way probe, 60k steps/job | 74-203/s (last 150 s) | ~509/s (last 150 s); 511/s sum of per-job averages | One job degraded by external load |

Aggregate throughput is essentially flat from 1 to 4 jobs: the host CPU, not
the GPUs, is the bottleneck. Pick 2-way.

- Per-run time at ~265 steps/s per job = 10M / 265 ~= 37,700 s ~= 10.5 h.
- 24 runs / 2 concurrent = 12 waves x 10.5 h ~= **126 h (5.3 days)** of
  training, worst case with today's external load. Add ~10% buffer for
  evaluation overlap and restarts: **~5.8 days**.
- Evaluations (6,102 test scenes per run) run on the spare cores while the
  next wave trains; they are not on the critical path unless a wave is
  delayed.

### 3.5 Launch template

```
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 \
python -m examples.drl_collision_avoidance.train \
  seed=$SEED \
  drl.checkpoint_dir=checkpoints/$ARM/seed_$SEED \
  drl.max_env_steps=10000000 drl.min_runtime_seconds=0 \
  drl.num_workers=4 drl.num_envs_per_worker=12 drl.num_envs=48 \
  drl.vec_env_mode=sync drl.train_freq=128 drl.batch_size=512 \
  occupancy_grid.visibility_filter=$VIS occupancy_grid.use_occlusion=$STATS \
  occupancy_grid.occlusion_mask=$MASK occupancy_grid.history_frames=$HF \
  occupancy_grid.history_stride=$HS reward.reward_ttz_source=$REW \
  hydra.run.dir=logs/$ARM/seed_$SEED
```

Arm -> flags (transcription table for the launcher):

| Arm | visibility_filter | use_occlusion (stats) | occlusion_mask | history_frames | history_stride | reward_ttz_source |
|---|---|---|---|---|---|---|
| A | false | false | false | 1 | 1 | privileged |
| B | true | false | false | 1 | 1 | privileged |
| C | true | true | false | 1 | 1 | privileged |
| C_vis | true | true | false | 1 | 1 | visible |
| D | true | false | true | 1 | 1 | privileged |
| D_vis | true | false | true | 1 | 1 | visible |
| O | true | true | true | 1 | 1 | privileged |
| C_hist | true | false | false | 3 | 5 | privileged |

Planned launcher: `scripts/launch_controlled_study.sh <arm> <seed> <gpu>`
(created after Phase 1b lands), one line per job appended to
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
- Eval is single-process CPU; 4-6 evals in parallel after a wave completes,
  overlapped with the next wave where cores allow.
- Provenance JSONs in `paper/workspace/eval/` are the only inputs to Phase 4
  tables/figures.

### 3.7 Storage layout

- `checkpoints/<arm>/seed_<seed>/` — `ddqn_final.pth` (+ periodic saves).
- `logs/<arm>/seed_<seed>/` — hydra + tensorboard.
- `paper/workspace/eval/<arm>_seed<seed>.json` (+ `.steps.jsonl`).
- `paper/workspace/launches.txt` — exact command per run.

## 4. Order of operations

1. ~~Implement Phase 1b (§1) + tests.~~ **Done 2026-10-17.** C++ visible-id
   grid filter; orthogonal config switches; reward-source decoupling;
   occlusion mask; grid-only history; `observation_config.py` metadata in
   every DDQN/BC checkpoint; evaluator refuses checkpoints without it.
   `tests/test_observation_contract.py` = 15 tests; full suite 58 passed.
2. ~~Smoke-run every arm config.~~ **Done 2026-10-17.** All 8 arms at 300
   steps with the expected dims; O and C_hist at 4x12/48 envs, 1000 steps,
   17 s each; O evaluated on 2 test-half scenes.
3. Freeze configs, record git sha, launch wave 1 (2 runs, 2 GPUs).
4. Twelve waves; evaluate each finished run on the test half.
5. Phase 3/4: regenerate every table/figure from `paper/workspace/eval/*.json`.

## 5. Sign-off (resolved 2026-10-17)

1. **Arm definitions.** No original definition exists (search of repo, all git
   history, pre-purge paper, Cursor plans, and session database). The matrix
   above is the minimal reconstruction of the roadmap's explicit constraints
   (§0) and is frozen. Letters are kept for continuity.
2. **"Ours" = O (stats + mask, 813-D).** Confirmed. The 613-D stats-only model
   is arm C.
3. **Seeds 42/123/456 and final-checkpoint evaluation.** Confirmed.

Changes from the first draft, all forced by the recovered roadmap text:

- `_vis` now changes the **reward source** (`reward_ttz_source=visible`), not
  the observation visibility. C/D already use ray-cast-visible observations.
- `C_hist` is **grid-only history with summaries off** (1809-D), not a
  3x full-observation stack (1839-D).
- A is the only privileged-observation arm; B/C/D/O/C_hist all use the
  visibility filter.

Phase 1b implementation starts only after this frozen matrix; any later change
requires a new commit and a new git sha in `launches.txt`.
