## Problem Statement
Autonomous vehicles must avoid collisions under partial observability: other road users may be occluded behind blockers or outside a limited sensor field of view. Most RL driving results assume full observability in synthetic traffic, which removes the hardest part of the problem. We study collision avoidance on real Waymo Open Motion Dataset scenarios replayed in the Nocturne simulator, whose ray-cast visibility model hides occluded agents from the ego policy.

## Core Hypothesis
Making occlusion risk explicit in the observation (ray-cast filtered occupancy, TTZ conflict features, and low-dimensional occlusion-risk summaries), combined with a TTZ-barrier shaped reward, enables a value-based deep RL agent to reach goals while keeping collisions low under physically grounded partial observability. Comparing asynchronous double-dueling DQN against on-policy PPO and risk-aware R-MAPPO training runs clarifies which learning paradigm yields usable policies in this log-replay setting.

## Proposed Methodology (Detailed Technical Approach)
- Environment: Nocturne + Waymo scenarios, $\Delta t=0.1$s, horizon $T_{\max}=80$, single ego controlled; others replay logged trajectories.
- Observation ($o \in \mathbb{R}^{613}$ with occlusion features): ego-centric $3\times20\times10$ occupancy grid (class-weighted), analytic TTZ features over visible agents, four occlusion-risk summaries (occluded view fraction, nearest-blocker proximity, blocker density, occlusion pressure), ego/goal features; 3-frame stack.
- Action space: discrete $3\times5$ throttle$\times$steering ($|\mathcal{A}|=15$).
- Reward: goal bonus, collision/off-road penalties, progress, heading, and TTZ-barrier term that activates when min TTZ falls below a safety threshold.
- Agents: (i) double dueling DQN with noisy nets, $n$-step returns, prioritized replay, Muon+AdamW; asynchronous actor--learner; (ii) PPO; (iii) R-MAPPO with Lagrangian safety constraint.
- Evaluation: held-out deterministic rollouts measuring goal rate, collision rate, ADE/FDE, jerk/smoothness, and TTZ statistics; stratification by episode length and conflict imminence (vehicle TTZ).

## Expected Contribution
An occlusion-aware, risk-shaped RL formulation for log-replay driving under ray-cast partial observability; a scalable DDQN training system; statistically reported held-out operating point; training-dynamics comparison against PPO and R-MAPPO; and stratified failure analysis by conflict imminence rather than aggregate rates alone.
