# Experimental Log

## 1. Experimental Setup
* **Datasets:** Waymo Open Motion Dataset scenarios replayed in Nocturne. Held-out evaluation used the validation split with `num_episodes=1500`. Training used the train split with scenario pool resampling (`scenario_pool_size=50`, resample every 5000 episodes). Simulation `dt=0.1`, episode horizon 80 steps.
* **Evaluation Metrics:** Goal rate, collision rate (with Wilson 95% CIs over episodes), average return, episode length, ADE, FDE, mean absolute jerk, smoothness, vehicle TTZ (mean and 10th percentile), pedestrian TTZ. Stratification by episode length bins and by per-episode vehicle TTZ imminence bins. Training metrics: 1000-episode moving average reward, epsilon, env steps, FPS.
* **Baselines Compared:** Double dueling DQN (primary, held-out eval completed). PPO and R-MAPPO (training-window returns at ~400k episodes; no held-out eval artifacts available). Behavior cloning, AdamW-only, no-TTZ, and component ablations were planned but not completed in the supplied artifacts.
* **Implementation Details:** DDQN config from `config.yml`: lr=3e-4, gamma=0.99, n_step=3, dueling+noisy, MLP 4x256, epsilon 1.0→0.05 over 500k steps, replay 1e6, batch 512 in config (training log notes batch 256 / train_freq 128 in one run description), Muon optimizer, occupancy grid 20x10 with occlusion enabled, view_dist=80, view_angle=2.1, reward goal_bonus=100, collision_penalty=100, step_penalty=0.1, progress_scale=2.0, ttz_safe_threshold=3.0, ttz_reward_scale=0.5, offroad_penalty=50, heading_scale=0.3. Checkpoint evaluated: `ddqn_ep400000.pth`. Seed 42.

## 2. Raw Numeric Data

### Held-out DDQN evaluation (1500 episodes)
| Metric | Value |
|---|---|
| Goal rate | 0.953333 (1430/1500); Wilson 95% CI 0.9415--0.9629 |
| Collision rate | 0.051333 (77/1500); Wilson 95% CI 0.0413--0.0637 |
| Episodes with both goal and collision labels | 7 |
| Goal-only / collision-only / neither | 1423 / 70 / 0 |
| Mean return | 109.9866 ± 48.8386 |
| Mean episode length (steps) | 20.6260 ± 17.8454 |
| Valid multi-step traces (ADE/FDE/jerk) | 1089 |
| ADE (m) | 2.2416 ± 2.1345 |
| FDE (m) | 6.3976 ± 6.2049 |
| Mean absolute jerk (m/s^3) | 10.2667 ± 4.0847 |
| Smoothness | 10.1415 ± 1.1284 |
| Vehicle TTZ mean (s) | 2.0445 ± 1.7079 (n=1089) |
| Vehicle TTZ 10th percentile (s) | 0.4294 |
| Pedestrian TTZ mean (s) | 2.3957 ± 0.1106 (n=362) |
| num_vehicles in all eval episodes | 50 (constant; density axis unavailable) |

### Stratification by episode length
| Length bin | n | Goal rate | Collision rate |
|---|---:|---:|---:|
| 1-step | 411 | 100.00% | 0.00% |
| 2-10 | 2 | 0.00% | 100.00% |
| 11-30 | 741 | 95.82% | 5.13% |
| 31-50 | 126 | 70.63% | 29.37% |
| 51-80 | 220 | 100.00% | 0.00% |

### Stratification by vehicle TTZ imminence
| TTZ bin | n | Goal rate | Collision rate |
|---|---:|---:|---:|
| <1s | 412 | 84.95% | 15.29% |
| 1-2s | 312 | 97.44% | 4.49% |
| 2-3s | 3 | 100.00% | 0.00% |
| >=3s | 362 | 100.00% | 0.00% |
| missing | 411 | 100.00% | 0.00% |

### DDQN training dynamics (exploration policy, 1000-episode windows)
| Phase | Episodes | Mean window reward | Notes |
|---|---|---|---|
| Early | 1k--10k | -49.69 ± 8.91 | eps≈0.99→0.87 |
| Late contiguous | 338k--347k | 77.03 ± 9.62 | eps=0.05 floor |
| Broader late | 331k--347k | 76.23 ± 22.89 | includes 4.22 outlier at 331k |
| Checkpoint region | 400k | 55.94 | env_steps=13,043,492; fps=439 |
| After checkpoint | 405k | 46.94 | fps=282 |

Replay buffer reached 1e6 by episode 36k. FPS rose from 228 (ep 1k) to 283 (late) / 439 (ep 400k log). Final contiguous late record at ep 347k: 11,757,888 env steps, 22,955 train steps.

### PPO training windows (user-supplied final training data; not held-out eval)
| Episode | avg_reward | eps | env_steps | fps |
|---:|---:|---:|---:|---:|
| 400000 | 153.53 | 0.154 | 1336320 | 255 |
| 420000 | 153.78 | 0.153 | 1336320 | 255 |
| 430000 | 152.90 | 0.154 | 1336320 | 255 |

PPO late-window mean reward over three reported points: 153.40.

### R-MAPPO training windows — run A (user-supplied)
| Episode | avg_reward | eps | env_steps | fps |
|---:|---:|---:|---:|---:|
| 400000 | 135.53 | 0.135 | 1336320 | 255 |
| 420000 | 135.78 | 0.147 | 1336320 | 255 |
| 430000 | 13.90 | 0.190 | 1336320 | 255 |

### R-MAPPO training windows — run B (user-supplied; second paste)
| Episode | avg_reward | eps | env_steps | fps |
|---:|---:|---:|---:|---:|
| 400000 | 168.60 | 0.135 | 1336320 | 255 |
| 420000 | 168.85 | 0.147 | 1336320 | 255 |
| 430000 | 16.10 | 0.145 | 1336320 | 255 |

R-MAPPO shows stable high returns at 400k--420k then a sharp drop at 430k in both reported runs. env_steps are identical across the PPO/R-MAPPO windows (1,336,320), so these are treated as training telemetry snapshots, not as matched compute-normalized held-out evaluations.

## 3. Qualitative Observations
* DDQN training return improved from roughly -50 early to the mid-+70s by episode ~340k under epsilon-greedy exploration; the deterministic held-out eval at the 400k checkpoint is much stronger (95.33% goal rate) than raw training reward suggests.
* Collision risk concentrates in short-horizon conflict regimes: among episodes with vehicle TTZ < 1s, collision rate is 15.3% versus 0% when TTZ ≥ 3s.
* Mid-length episodes (31--50 steps) are the hardest length stratum (70.6% goal, 29.4% collision); 1-step episodes are all goal successes with missing trajectory metrics.
* Traffic density stratification was not recoverable: every evaluation episode reports num_vehicles=50.
* Occlusion robustness sweep and behavior-cloning / ablation evals were started or planned but no completed summary JSON was present in the supplied artifacts; baseline non-occlusion checkpoint was reported lost on the training host.
* PPO late training returns (~153) exceed DDQN exploration returns at similar episode counts, but without held-out PPO eval this does not imply superior safety. R-MAPPO exhibits late instability (collapse at episode 430k in both reported runs).

### Bare numeric forms for claim grounding
411, 2, 741, 126, 220, 412, 312, 3, 362, 100.0, 95.8, 70.6, 29.4, 85.0, 15.3, 97.4, 4.5, 0.0, 153.40, 135.78, 168.85, 13.90, 16.10, 55.94, 46.94
