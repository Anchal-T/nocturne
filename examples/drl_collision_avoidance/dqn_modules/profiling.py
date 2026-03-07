import os
from typing import Optional

import torch
import torch.nn as nn


class CudaTrainProfiler:
    """Manages CUDA profiling for training steps.

    Wraps torch.profiler with a schedule-based lifecycle: start once,
    advance each train step, export a Chrome trace when the schedule
    completes (or when finalize() is called early).
    """

    def __init__(
        self,
        device: torch.device,
        enabled: bool = False,
        wait_steps: int = 8,
        warmup_steps: int = 4,
        active_steps: int = 24,
        trace_path: str = "trace.json",
        memory_log_interval: int = 200,
    ):
        self._device = device
        self._enabled = enabled and device.type == "cuda"
        self._wait_steps = max(0, wait_steps)
        self._warmup_steps = max(0, warmup_steps)
        self._active_steps = max(1, active_steps)
        self._trace_path = str(trace_path)
        self._memory_log_interval = max(1, memory_log_interval)

        self._profiler: Optional[torch.profiler.profile] = None
        self._step_calls = 0
        self._exported = False
        self._last_memory_log_step = 0

    @property
    def is_active(self) -> bool:
        return self._profiler is not None

    def maybe_start(self) -> None:
        if not self._enabled or self._profiler is not None or self._exported:
            return
        if not hasattr(torch, "profiler"):
            return

        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self._wait_steps,
                warmup=self._warmup_steps,
                active=self._active_steps,
                repeat=1,
            ),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )
        self._profiler.start()
        self._step_calls = 0
        print(
            f"[CUDA] train-step profiler started "
            f"(wait={self._wait_steps}, warmup={self._warmup_steps}, "
            f"active={self._active_steps}) -> {self._trace_path}"
        )

    def advance(self) -> None:
        if self._profiler is None:
            return
        self._profiler.step()
        self._step_calls += 1
        total = self._wait_steps + self._warmup_steps + self._active_steps
        if self._step_calls >= total:
            self.finalize(export_trace=True)

    def finalize(self, export_trace: bool = True) -> None:
        if self._profiler is None:
            return
        profiler = self._profiler
        self._profiler = None
        try:
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            profiler.stop()
            if export_trace and not self._exported:
                trace_path = os.path.abspath(self._trace_path)
                trace_dir = os.path.dirname(trace_path)
                if trace_dir:
                    os.makedirs(trace_dir, exist_ok=True)
                profiler.export_chrome_trace(trace_path)
                self._exported = True
                print(f"[CUDA] train-step profiler trace saved: {trace_path}")
        except Exception as exc:
            print(f"[CUDA] failed to finalize train-step profiler: {exc}")

    def maybe_log_memory(self, train_steps: int) -> None:
        if not self._enabled:
            return
        if train_steps - self._last_memory_log_step < self._memory_log_interval:
            return
        self._last_memory_log_step = train_steps
        allocated = torch.cuda.memory_allocated(self._device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(self._device) / (1024 ** 2)
        print(
            f"[CUDA] step={train_steps} mem_alloc={allocated:.1f}MiB "
            f"mem_reserved={reserved:.1f}MiB"
        )
