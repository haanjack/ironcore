"""Wall-clock profiler using torch.cuda.synchronize()."""

import time
from pathlib import Path
from typing import Callable

import torch

from experiments.generation.profilers.base import Profiler, ProfilerResult


class WallClockProfiler(Profiler):
    """Simple wall-clock timing profiler.

    Always available on PyTorch systems. Uses torch.cuda.synchronize()
    to ensure accurate GPU timing.
    """

    def is_available(self) -> bool:
        """Wall-clock profiler is always available."""
        return True

    def profile(
        self,
        fn: Callable,
        inputs: tuple,
        label: str,
    ) -> ProfilerResult:
        """Profile function using wall-clock time.

        Returns median time across `repeat` iterations.
        """
        self._do_warmup(fn, inputs)

        times = []
        for _ in range(self.repeat):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn(*inputs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        times.sort()
        median_ms = times[len(times) // 2]

        return ProfilerResult(
            profiler_name="wallclock",
            time_ms=median_ms,
            metrics={
                "min_ms": min(times),
                "max_ms": max(times),
                "std_ms": (sum((t - median_ms) ** 2 for t in times) / len(times)) ** 0.5,
            },
        )

    def parse_results(self, output_path: Path) -> dict:
        """Wall-clock profiler doesn't produce output files."""
        return {}
