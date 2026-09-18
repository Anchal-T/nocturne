# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CPU/Gloo tests for the shared distributed helpers."""
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from nocturne.utils.distributed import (
    DistInfo,
    all_reduce_mean_std,
    all_reduce_scalar,
    control_barrier,
    dummy_loss,
    init_distributed,
    is_launched_distributed,
    parameter_checksum,
    rank_offset_seed,
    reduce_metrics,
    shard_sequence,
    unwrap_module,
    wrap_ddp,
)


def test_single_process_init_is_noop():
    os.environ.pop("WORLD_SIZE", None)
    os.environ.pop("RANK", None)
    os.environ.pop("LOCAL_RANK", None)
    info = init_distributed(requested_device="cpu")
    assert info.world_size == 1
    assert info.rank == 0
    assert not info.is_distributed
    assert not is_launched_distributed()


def test_shard_and_seed_are_disjoint():
    items = list(range(10))
    shards = [shard_sequence(items, rank, 4) for rank in range(4)]
    flattened = sorted(item for shard in shards for item in shard)
    assert flattened == items
    seeds = [rank_offset_seed(7, rank) for rank in range(4)]
    assert len(set(seeds)) == 4


def test_unwrap_identity():
    module = torch.nn.Linear(3, 3)
    assert unwrap_module(module) is module


def _gloo_worker(rank, world_size, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    info = init_distributed(requested_device="cpu")
    values = np.array([float(rank + 1), float(rank + 1)])
    mean, std, count = all_reduce_mean_std(values, None, info.device)
    reduced = all_reduce_scalar(float(rank), info.device, op="mean")
    metrics = reduce_metrics({"loss": float(rank + 1)}, info.device)
    module = torch.nn.Linear(2, 2).to(info.device)
    wrapped = wrap_ddp(module, info, find_unused_parameters=True)
    loss = dummy_loss(wrapped)
    loss.backward()
    checksum = parameter_checksum(wrapped)
    result_queue.put({
        "rank": rank,
        "world_size": info.world_size,
        "mean": mean,
        "count": count,
        "std": std,
        "reduced": reduced,
        "metrics_loss": metrics["loss"],
        "checksum": checksum,
    })
    import torch.distributed as dist
    dist.barrier()
    dist.destroy_process_group()


def test_gloo_collectives_two_ranks():
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    world_size = 2
    processes = [
        ctx.Process(target=_gloo_worker, args=(rank, world_size, queue))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    results = [queue.get() for _ in range(world_size)]
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0
    assert {item["world_size"] for item in results} == {2}
    for item in results:
        assert abs(item["mean"] - 1.5) < 1e-6
        assert abs(item["count"] - 4.0) < 1e-6
        assert abs(item["reduced"] - 0.5) < 1e-6
        assert abs(item["metrics_loss"] - 1.5) < 1e-6
    checksums = {item["checksum"] for item in results}
    assert len(checksums) == 1


def test_distinfo_rank0():
    info = DistInfo(rank=1, world_size=2)
    assert info.is_distributed
    assert not info.is_rank0


def test_all_reduce_scalar_min_is_identity_when_single_process():
    os.environ.pop("WORLD_SIZE", None)
    os.environ.pop("RANK", None)
    info = init_distributed(requested_device="cpu")
    assert all_reduce_scalar(3.0, info.device, op="min") == 3.0
    assert all_reduce_scalar(3.0, info.device, op="max") == 3.0


def test_dummy_loss_touches_parameters():
    module = torch.nn.Linear(4, 4)
    loss = dummy_loss(module)
    loss.backward()
    assert all(param.grad is not None for param in module.parameters())


def test_process_learner_dead_process_raises():
    import queue
    from examples.drl_collision_avoidance.train import ProcessLearner

    learner = ProcessLearner.__new__(ProcessLearner)
    learner._process = type("Dead", (), {"is_alive": staticmethod(lambda: False)})()
    learner._pending_responses = []

    class _EmptyQueue:
        def get_nowait(self):
            raise queue.Empty

    learner._response_queue = _EmptyQueue()
    try:
        learner._ensure_process_alive()
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "exited unexpectedly" in str(exc)


def _nccl_worker(rank, world_size, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29521"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    info = init_distributed()
    module = torch.nn.Linear(8, 8).to(info.device)
    wrapped = wrap_ddp(module, info, find_unused_parameters=True)
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.1)
    x = torch.ones(4, 8, device=info.device) * (rank + 1)
    loss = wrapped(x).sum()
    optimizer.zero_grad()
    loss.backward()
    assert all(
        torch.isfinite(param.grad).all()
        for param in wrapped.parameters() if param.grad is not None
    )
    optimizer.step()
    # Model rank 0 doing a relatively long evaluation while its peer waits.
    # The control barrier must not leave a CUDA/NCCL kernel spinning.
    if rank == 0:
        time.sleep(10)
    control_barrier(info)
    result_queue.put(parameter_checksum(wrapped))
    import torch.distributed as dist
    dist.destroy_process_group()


def test_nccl_two_gpu_parameter_sync():
    if os.environ.get("RUN_NCCL_SMOKE") != "1":
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    processes = [
        ctx.Process(target=_nccl_worker, args=(rank, 2, queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    checksums = [queue.get() for _ in range(2)]
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0
    assert abs(checksums[0] - checksums[1]) < 1e-3
