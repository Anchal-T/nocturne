# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Shared torch.distributed helpers for repo-owned trainers.

Single-process callers keep working: ``init_distributed()`` is a no-op unless
``WORLD_SIZE`` is set to a value greater than 1 (as ``torchrun`` does).
"""
from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


_CONTROL_GROUP = None


@dataclass(frozen=True)
class DistInfo:
    """Resolved rank / device assignment for this process."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = torch.device("cpu")
    backend: str = "gloo"

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def is_launched_distributed() -> bool:
    """True when torchrun/launch exported a WORLD_SIZE greater than 1."""
    return _env_int("WORLD_SIZE", 1) > 1


def init_distributed(requested_device: Optional[str] = None,
                     timeout_seconds: float = 1800.0) -> DistInfo:
    """Initialize the process group if launched with torchrun.

    ``requested_device`` is used only for the single-process fallback. Under
    torchrun the local rank maps onto ``cuda:<LOCAL_RANK>`` after
    ``CUDA_VISIBLE_DEVICES`` has already sliced the physical GPUs.
    """
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)

    if world_size <= 1:
        if requested_device is None:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(requested_device)
            if device.type == "cuda" and not torch.cuda.is_available():
                device = torch.device("cpu")
        return DistInfo(device=device)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
    # NCCL barriers launch a CUDA kernel. They are appropriate for short
    # training synchronization, but a rank waiting while another rank runs a
    # lengthy evaluation can trip the display driver's kernel watchdog. Keep
    # a CPU-backed group for those long control-plane waits.
    global _CONTROL_GROUP
    if backend == "nccl" and _CONTROL_GROUP is None:
        _CONTROL_GROUP = dist.new_group(
            backend="gloo",
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return DistInfo(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        backend=backend,
    )


def cleanup_distributed() -> None:
    """Destroy the process group if this process created one."""
    global _CONTROL_GROUP
    if dist.is_available() and dist.is_initialized():
        # Do not add a final barrier here: if a peer failed, waiting for it
        # would mask the original exception and can leave surviving ranks
        # stuck until the process-group timeout.
        dist.destroy_process_group()
    _CONTROL_GROUP = None


def wrap_ddp(module: torch.nn.Module,
             dist_info: DistInfo,
             find_unused_parameters: bool = True,
             broadcast_buffers: bool = True) -> torch.nn.Module:
    """Wrap ``module`` with DDP when running distributed, else return it."""
    if not dist_info.is_distributed:
        return module
    device_ids = None
    output_device = None
    if dist_info.device.type == "cuda":
        device_ids = [dist_info.local_rank]
        output_device = dist_info.local_rank
    return DDP(
        module,
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
    )


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP / DataParallel / torch.compile wrappers for save/load."""
    current = module
    for _ in range(4):
        inner = getattr(current, "module", None)
        orig = getattr(current, "_orig_mod", None)
        if inner is not None:
            current = inner
            continue
        if orig is not None:
            current = orig
            continue
        break
    return current


def rank0_only(dist_info: DistInfo) -> bool:
    return dist_info.is_rank0


def barrier(dist_info: Optional[DistInfo] = None) -> None:
    if dist_info is not None and not dist_info.is_distributed:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def control_barrier(dist_info: Optional[DistInfo] = None) -> None:
    """Synchronize ranks without holding a CUDA kernel while waiting."""
    if dist_info is not None and not dist_info.is_distributed:
        return
    if dist.is_available() and dist.is_initialized():
        if _CONTROL_GROUP is not None:
            dist.barrier(group=_CONTROL_GROUP)
        else:
            dist.barrier()


def _is_distributed() -> bool:
    return (dist.is_available() and dist.is_initialized()
            and dist.get_world_size() > 1)


def _control_reduce(tensor: torch.Tensor, op) -> torch.Tensor:
    """All-reduce tiny stats on the CPU control group when one exists.

    Gradient synchronization stays on NCCL through DDP. Scalar metrics and
    advantage statistics should not launch NCCL kernels that can trip a
    watchdog while a peer is still on CPU work.
    """
    if not _is_distributed():
        return tensor
    group = _CONTROL_GROUP
    if group is None:
        dist.all_reduce(tensor, op=op)
        return tensor
    if tensor.device.type == "cpu":
        dist.all_reduce(tensor, op=op, group=group)
        return tensor
    cpu_tensor = tensor.detach().to(device="cpu")
    dist.all_reduce(cpu_tensor, op=op, group=group)
    tensor.copy_(cpu_tensor.to(device=tensor.device, dtype=tensor.dtype))
    return tensor


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """In-place SUM all-reduce. No-op when not distributed."""
    return _control_reduce(tensor, dist.ReduceOp.SUM)


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if _is_distributed():
        _control_reduce(tensor, dist.ReduceOp.SUM)
        tensor.div_(dist.get_world_size())
    return tensor


def all_reduce_scalar(value: float,
                      device: torch.device,
                      op: str = "mean") -> float:
    """Reduce a Python float across ranks and return the result on every rank."""
    tensor_device = torch.device("cpu") if _CONTROL_GROUP is not None else device
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=tensor_device)
    if op == "sum":
        all_reduce_sum(tensor)
    elif op == "mean":
        all_reduce_mean(tensor)
    elif op == "max":
        _control_reduce(tensor, dist.ReduceOp.MAX)
    elif op == "min":
        _control_reduce(tensor, dist.ReduceOp.MIN)
    else:
        raise ValueError(f"unsupported reduction {op!r}")
    return float(tensor.item())


def all_reduce_mean_std(values: np.ndarray,
                        mask: Optional[np.ndarray],
                        device: torch.device) -> tuple:
    """Global mean/std of ``values`` over ``mask`` (True = keep)."""
    arr = np.asarray(values, dtype=np.float64)
    if mask is not None:
        keep = np.asarray(mask, dtype=bool)
        arr = arr[keep]
    if arr.size == 0:
        local_n = 0.0
        local_sum = 0.0
        local_sumsq = 0.0
    else:
        local_n = float(arr.size)
        local_sum = float(arr.sum())
        local_sumsq = float(np.square(arr).sum())
    stats_device = torch.device("cpu") if _CONTROL_GROUP is not None else device
    stats = torch.tensor([local_n, local_sum, local_sumsq],
                         dtype=torch.float64,
                         device=stats_device)
    all_reduce_sum(stats)
    total_n, total_sum, total_sumsq = [float(x) for x in stats.tolist()]
    if total_n < 1.0:
        return 0.0, 1.0, 0.0
    mean = total_sum / total_n
    var = max(total_sumsq / total_n - mean * mean, 0.0)
    std = float(np.sqrt(var))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return float(mean), std, total_n


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a picklable Python object from ``src`` to every rank."""
    if not (dist.is_available() and dist.is_initialized()
            and dist.get_world_size() > 1):
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def broadcast_module_state(module: torch.nn.Module, src: int = 0) -> None:
    """Overwrite every rank's ``module`` state with rank ``src``."""
    if not (dist.is_available() and dist.is_initialized()
            and dist.get_world_size() > 1):
        return
    state = unwrap_module(module).state_dict()
    for key in sorted(state.keys()):
        tensor = state[key]
        if torch.is_tensor(tensor):
            dist.broadcast(tensor, src=src)


def dummy_loss(module: torch.nn.Module) -> torch.Tensor:
    """Zero loss that still touches parameters so DDP all-reduce can run."""
    total = None
    for param in module.parameters():
        if not param.requires_grad:
            continue
        contrib = param.sum() * 0.0
        total = contrib if total is None else total + contrib
    if total is None:
        device = torch.device("cpu")
        try:
            device = next(module.parameters()).device
        except StopIteration:
            pass
        return torch.zeros((), device=device)
    return total


def parameter_checksum(module: torch.nn.Module) -> float:
    """Scalar checksum used by tests to compare ranks."""
    total = 0.0
    for param in unwrap_module(module).parameters():
        total += float(param.detach().float().sum().cpu())
    return total


def rank_offset_seed(base_seed: int, rank: int, stride: int = 100003) -> int:
    return int(base_seed) + int(rank) * int(stride)


def shard_sequence(items: Sequence, rank: int, world_size: int) -> list:
    """Round-robin shard so ranks see disjoint items and cover the full set."""
    if world_size <= 1:
        return list(items)
    return [item for index, item in enumerate(items) if index % world_size == rank]


def reduce_metrics(metrics: dict, device: torch.device) -> dict:
    """All-reduce numeric metric values; leave non-numeric entries untouched."""
    if not metrics:
        return metrics
    reduced = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating, np.integer)):
            reduced[key] = all_reduce_scalar(float(value), device, op="mean")
        elif torch.is_tensor(value) and value.numel() == 1:
            tensor_device = (torch.device("cpu")
                             if _CONTROL_GROUP is not None else device)
            tensor = value.detach().to(device=tensor_device, dtype=torch.float64)
            all_reduce_mean(tensor)
            reduced[key] = float(tensor.item())
        else:
            reduced[key] = value
    return reduced


def make_checkpoint(model_state: dict,
                    extra: Optional[dict] = None,
                    dist_info: Optional[DistInfo] = None,
                    global_step: int = 0) -> dict:
    """Build a rank-portable checkpoint dict without DDP ``module.`` prefixes."""
    payload = {
        "model": model_state,
        "global_step": int(global_step),
        "world_size": 1 if dist_info is None else dist_info.world_size,
    }
    if extra:
        payload.update(extra)
    return payload
