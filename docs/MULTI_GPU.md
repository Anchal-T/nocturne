# Multi-GPU training

Single-GPU remains the default. Repo-owned trainers (R-MAPPO/MAPPO/Lagrangian,
Waymo BC, collision-avoidance BC, DDQN, CRL) use one process per GPU with NCCL
when launched through `torchrun`. RLlib and Sample Factory keep their own
distribution APIs.

Do not use GPUs 0 or 3 while they are owned by another job. Validation
commands below pin `CUDA_VISIBLE_DEVICES=1,2`. Switching to four GPUs later is
the same launcher with `CUDA_VISIBLE_DEVICES=0,1,2,3` and `NPROC=4`.

K80s have no NVLink. Expect sublinear scaling; CPU simulation and IPC remain
the bottleneck.

## Launchers

Two-GPU (GPUs 1 and 2):

```bash
CUDA_VISIBLE_DEVICES=1,2 ./scripts/run_ppo_ddp.sh
CUDA_VISIBLE_DEVICES=1 NPROC=1 ./scripts/run_ppo_smoke.sh
CUDA_VISIBLE_DEVICES=1,2 ./scripts/run_imitation_ddp.sh
CUDA_VISIBLE_DEVICES=1,2 ./scripts/run_bc_ddp.sh
CUDA_VISIBLE_DEVICES=1,2 ./scripts/run_ddqn_ddp.sh
CUDA_VISIBLE_DEVICES=1,2 ./scripts/run_crl_ddp.sh
CUDA_VISIBLE_DEVICES=1,2 ./scripts/run_rllib_multi_gpu.sh
CUDA_VISIBLE_DEVICES=1,2 ./scripts/run_sample_factory_multi_gpu.sh
```

Four-GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 ./scripts/run_ppo_ddp.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 ./scripts/run_imitation_ddp.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 ./scripts/run_bc_ddp.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 ./scripts/run_ddqn_ddp.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 ./scripts/run_crl_ddp.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/run_rllib_multi_gpu.sh rllib.num_gpus=4
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/run_sample_factory_multi_gpu.sh algorithm.actor_worker_gpus=[1,2,3]
```

Hydra/algorithm overrides pass through after the script name, for example
`CUDA_VISIBLE_DEVICES=1,2 ./scripts/run_ppo_ddp.sh algorithm.use_lagrangian=True`.

`scripts/run_ppo_smoke.sh` is a 400k-step PPO profile with a bounded scenario
cache and fewer PPO updates. It does not change `cfgs/algorithm/ppo.yaml`.
Launchers default temporary files to `$HOME/tmp`. Evaluate a saved actor and
critic without optimizer or ValueNorm state:

```bash
python examples/on_policy_files/evaluate_ppo.py \
  --checkpoint /path/to/models \
  --scenario-path dataset/formatted_json_v2_no_tl_valid
```

## Stack notes

- **R-MAPPO family**: DDP wraps actor, reward critic, and cost critic. Empty
  active minibatches still backward a dummy loss so ranks cannot deadlock.
  Advantage mean/std and ValueNorm buffers are reduced/broadcast. Rank 0 writes
  checkpoints without a `module.` prefix (`actor.pt`, `critic.pt`, optional
  `trainer_state.pt`).
- **Waymo BC**: scenario files are round-robin sharded by rank; loss/accuracy
  are reduced; rank 0 logs and saves.
- **Collision-avoidance BC**: each rank generates its own expert data, then
  trains with a synced batch count.
- **RLlib**: Ray 1.11 `num_gpus` is the multi-GPU learner API. `num_learners`
  is not used unless Ray is upgraded and retested. Cluster launch path is
  `examples/rllib_files/run_rllib.py`.
- **Sample Factory 1.123**: one learner. Extra GPUs can run actor-side
  inference via `algorithm.actor_worker_gpus`; that is not multi-GPU SGD.
- **DDQN**: `ProcessLearner` is disabled under torchrun. Each rank owns local
  rollouts and a PER shard; online-net gradients sync through DDP. CUDA graphs
  and `torch.compile` are gated off when distributed. A dead learner process
  raises instead of hanging.
- **CRL**: rank-local parallel env workers and HER buffers. Encoders and actor
  are DDP-wrapped. Negatives are local; gradients average. Dummy train steps
  keep ranks aligned while buffers fill.

## Tests

```bash
python -m pytest tests/test_distributed.py tests/test_ddp_sharding.py -q
```

The NCCL two-GPU smoke test in `tests/test_distributed.py` is opt-in:

```bash
CUDA_VISIBLE_DEVICES=1,2 RUN_NCCL_SMOKE=1 python -m pytest tests/test_distributed.py::test_nccl_two_gpu_parameter_sync -q
```

## Validation snapshot

The short smoke runs below were completed on GPUs 1 and 2 on 2026-09-01.
They include checkpoint writes; the Sample Factory run was also resumed from
its saved checkpoint. The NCCL test compared parameter checksums across ranks.

- R-MAPPO: 21 FPS on one GPU and 21 FPS across two GPUs for an 8-step run.
- Collision-avoidance BC: 70 local samples in 11.6 s on one GPU versus 171
  global samples in 11.6 s across two ranks.
- Waymo BC: 8 local samples in 11.6 s on one GPU versus 16 global samples in
  11.5 s across two ranks.
- DDQN: 5 environment steps/s on one GPU versus 10 environment steps/s across
  two ranks.
- CRL: 2.61 episodes/s on one GPU and approximately 2.75 episodes/s during
  the two-rank collection smoke.
- RLlib: approximately 151 sampled agent steps/s with one GPU and 149 with
  two GPUs; this is the legacy Ray multi-GPU learner path.
- Sample Factory: 231.5 FPS with one learner GPU and 150.8 FPS with one
  learner plus one actor GPU. SF 1.123 has one learner, so extra GPUs do not
  provide same-policy multi-GPU SGD.

These are startup-heavy correctness smokes, not production scaling
benchmarks. Repeat with longer runs and the same rollout/data settings before
making performance decisions.
