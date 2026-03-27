"""PyTorch built-in profiler integration.

Uses torch.profiler to capture CUDA kernel execution metrics.
This is more reliable than nsys for automated profiling.
"""

import time
from pathlib import Path
from typing import Callable

import torch

from experiments.generation.profilers.base import Profiler, ProfilerResult


class TorchProfiler(Profiler):
    """PyTorch built-in profiler.

    Uses torch.profiler to capture CUDA kernel execution.
    Produces JSON trace files that can be opened in Chrome Trace Viewer (chrome://tracing).
    """

    def is_available(self) -> bool:
        """PyTorch profiler is available when CUDA is available."""
        return torch.cuda.is_available()

    def profile(
        self,
        fn: Callable,
        inputs: tuple,
        label: str,
    ) -> ProfilerResult:
        """Profile function using PyTorch profiler."""
        if not self.is_available():
            return ProfilerResult(
                profiler_name="torch",
                time_ms=0.0,
                success=False,
                error_msg="CUDA not available, cannot use PyTorch profiler",
            )

        output_path = self.output_dir / f"{label}_torch_trace.json"

        # Warmup
        self._do_warmup(fn, inputs)

        # Profile with torch.profiler
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            # Run multiple iterations
            for _ in range(self.repeat):
                fn(*inputs)

        # Export to Chrome trace format
        try:
            prof.export_chrome_trace(str(output_path))

            # Extract metrics from the profiler
            metrics = self._extract_metrics(prof)

            # Get average time from profiler
            # Calculate manually if events aren't available
            times = []
            torch.cuda.synchronize()
            for _ in range(self.repeat):
                torch.cuda.synchronize()
                start = time.perf_counter()
                fn(*inputs)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)

            times.sort()
            median_ms = times[len(times) // 2]

            metrics["wallclock_min_ms"] = min(times)
            metrics["wallclock_max_ms"] = max(times)
            metrics["wallclock_median_ms"] = median_ms
            metrics["wallclock_std_ms"] = (sum((t - median_ms) ** 2 for t in times) / len(times)) ** 0.5

            # Get kernel counts from profiler
            kernel_count = 0
            total_cuda_time = 0.0
            for event in prof.events():
                if event.device_type == torch.profiler.DeviceType.CUDA:
                    kernel_count += 1
                    # Use device_time_total (cuda_time_total is deprecated)
                    if hasattr(event, 'device_time_total'):
                        total_cuda_time += event.device_time_total / 1000  # Convert to ms
                    elif hasattr(event, 'cuda_time_total'):
                        total_cuda_time += event.cuda_time_total / 1000  # Convert to ms

            if kernel_count > 0:
                metrics["kernel_count"] = kernel_count
                metrics["cuda_time_total_ms"] = total_cuda_time

            # Get memory info
            if hasattr(prof, 'key_averages'):
                averages = prof.key_averages()
                if averages:
                    metrics["profiler_events"] = len(averages)

            return ProfilerResult(
                profiler_name="torch",
                time_ms=metrics.get("cuda_time_total_ms", median_ms) or median_ms,
                metrics=metrics,
                output_path=output_path,
                success=True,
            )

        except Exception as e:
            return ProfilerResult(
                profiler_name="torch",
                time_ms=0.0,
                success=False,
                error_msg=f"PyTorch profiler failed: {e}",
            )

    def _extract_metrics(self, prof: torch.profiler.profile) -> dict:
        """Extract metrics from PyTorch profiler."""
        metrics = {}

        try:
            # Get key averages
            averages = prof.key_averages()
            if averages:
                for event in averages:
                    if event.device_type == torch.profiler.DeviceType.CUDA:
                        # Extract per-kernel metrics
                        name = event.key.replace("/", "_").replace(" ", "_")
                        if hasattr(event, 'cuda_time_total'):
                            metrics[f"kernel_{name}_cuda_time_us"] = event.cuda_time_total
                        if hasattr(event, 'self_cuda_time_total'):
                            metrics[f"kernel_{name}_self_cuda_time_us"] = event.self_cuda_time_total

                        # Memory stats
                        if hasattr(event, 'cuda_memory_usage'):
                            mem = event.cuda_memory_usage
                            if mem:
                                metrics[f"kernel_{name}_memory_bytes"] = mem

        except Exception:
            pass

        return metrics

    def parse_results(self, output_path: Path) -> dict:
        """Parse torch profiler JSON output (not used, metrics extracted directly)."""
        return {}
