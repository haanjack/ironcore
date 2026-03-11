# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import re
import threading
import time
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from ironcore.config import MainConfig
from ironcore.global_vars import get_logger


class CommProfiler:
    """Thread-safe singleton for collecting distributed communication timings."""

    _instance: ClassVar["CommProfiler | None"] = None
    _class_lock: ClassVar[threading.Lock] = threading.Lock()

    # Instance attribute declarations for type checkers
    enabled: bool
    _stats_lock: threading.Lock
    _stats: dict[str, list[float]]
    _initialized: bool

    def __new__(cls) -> "CommProfiler":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return
        self.enabled = False
        self._stats_lock = threading.Lock()
        self._stats = {}
        self._initialized = True

    def enable(self):
        self.enabled = True
        self.reset()

    def disable(self):
        self.enabled = False

    def record(self, op_name: str, duration_ms: float):
        if not self.enabled:
            return
        with self._stats_lock:
            if op_name not in self._stats:
                self._stats[op_name] = []
            self._stats[op_name].append(duration_ms)

    def get_and_reset_stats(self) -> dict[str, dict]:
        with self._stats_lock:
            result = {}
            for op, durs in self._stats.items():
                result[op] = {
                    "count": len(durs),
                    "total_ms": sum(durs),
                    "mean_ms": sum(durs) / len(durs),
                    "max_ms": max(durs),
                }
            self._stats = {}
        return result

    def reset(self):
        with self._stats_lock:
            self._stats = {}


@contextlib.contextmanager
def timed_comm(op_name: str):
    """Context manager for timing synchronous distributed communication operations."""
    profiler = CommProfiler()
    if not profiler.enabled:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        profiler.record(op_name, duration_ms)


class LayerTimingCollector:
    """Singleton for collecting per-layer forward/backward GPU timing via CUDA events."""

    _instance: ClassVar["LayerTimingCollector | None"] = None
    _class_lock: ClassVar[threading.Lock] = threading.Lock()

    # Instance attribute declarations for type checkers
    enabled: bool
    _pending: dict[int, tuple]
    _completed: list[tuple]
    logger: Any
    _initialized: bool

    def __new__(cls) -> "LayerTimingCollector":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return
        self.enabled = False
        self._pending = {}
        self._completed = []
        try:
            self.logger = get_logger()
        except AssertionError:
            import logging

            self.logger = logging.getLogger(__name__)
        self._initialized = True

    def enable(self):
        self.enabled = True
        self.reset()

    def disable(self):
        self.enabled = False

    def start(self, module_id: int, layer_name: str, phase: str):
        if not self.enabled or not torch.cuda.is_available():
            return
        if module_id in self._pending:
            # Recompute or reentrant call — discard the overwritten entry
            self.logger.debug(f"LayerTimingCollector: overwriting pending entry for {layer_name}")
        event = torch.cuda.Event(enable_timing=True)  # type: ignore  # torch stub requires stream=, not needed at runtime
        event.record()  # type: ignore  # same as above
        self._pending[module_id] = (layer_name, phase, event)

    def end(self, module_id: int):
        if not self.enabled or not torch.cuda.is_available():
            return
        entry = self._pending.pop(module_id, None)
        if entry is None:
            self.logger.debug(
                f"LayerTimingCollector: no pending entry for module {module_id} in end()"
            )
            return
        layer_name, phase, start_event = entry
        end_event = torch.cuda.Event(enable_timing=True)  # type: ignore  # torch stub requires stream=, not needed at runtime
        end_event.record()  # type: ignore  # same as above
        self._completed.append((layer_name, phase, start_event, end_event))

    def get_summary(self) -> str:
        if not self._completed:
            return "No layer timing data collected."
        if torch.cuda.is_available():
            # Global sync is intentional: we need all CUDA streams to complete before
            # reading elapsed times. Called only once at profiling stop, so impact is acceptable.
            torch.cuda.synchronize()
        stats: dict[str, dict[str, float]] = {}
        for layer_name, phase, start_event, end_event in self._completed:
            elapsed_ms = start_event.elapsed_time(end_event)
            if layer_name not in stats:
                stats[layer_name] = {"forward_ms": 0.0, "backward_ms": 0.0}
            stats[layer_name][f"{phase}_ms"] += elapsed_ms
        sorted_layers = sorted(
            stats.items(),
            key=lambda x: x[1]["forward_ms"] + x[1]["backward_ms"],
            reverse=True,
        )
        lines = [
            "=" * 70,
            f"{'Layer':<35} {'Fwd (ms)':>10} {'Bwd (ms)':>10} {'Total (ms)':>10}",
            "-" * 70,
        ]
        for layer_name, times in sorted_layers:
            fwd = times["forward_ms"]
            bwd = times["backward_ms"]
            total = fwd + bwd
            lines.append(f"{layer_name:<35} {fwd:>10.2f} {bwd:>10.2f} {total:>10.2f}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def reset(self):
        self._pending.clear()
        self._completed.clear()


def get_layer_timing_collector() -> LayerTimingCollector:
    """Get the global LayerTimingCollector singleton."""
    return LayerTimingCollector()


class TimedDataIterator:
    """Wraps a data iterator and measures time spent in __next__ (data loading + transfer)."""

    def __init__(self, iterator) -> None:
        self._iterator = iterator
        self._total_ms: float = 0.0
        self._call_count: int = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._iterator)  # propagates TypeError if unsupported

    def __next__(self):
        start = time.perf_counter()
        batch = next(self._iterator)
        self._total_ms += (time.perf_counter() - start) * 1000.0
        self._call_count += 1
        return batch

    def get_and_reset_stats(self) -> dict[str, float]:
        stats = {"total_ms": self._total_ms, "count": float(self._call_count)}
        self._total_ms = 0.0
        self._call_count = 0
        return stats


class ProfileManager:
    """Manages profiling lifecycles, versioning, and hardware-specific hooks."""

    def __init__(self, config: MainConfig):
        self.config = config.profiler
        self.logger = get_logger()
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # Check if this rank should profile
        self.should_profile = self.rank in self.config.ranks

        self.torch_profiler: profile | None = None
        self.current_version = self._get_next_version()
        self.is_active = False

        self._comm_profiler = CommProfiler()
        self._layer_timing = LayerTimingCollector()
        self._timed_data_iter: TimedDataIterator | None = None

        if self.should_profile:
            # Reset singleton state so re-initializing ProfileManager starts from a known
            # clean state, without clobbering an active profiling session on another rank.
            self._comm_profiler.disable()
            self._comm_profiler.reset()
            self._layer_timing.disable()
            self._layer_timing.reset()
            self._init_profilers()

    def _get_next_version(self) -> str:
        """Finds the next available version number for the given profile name."""
        if not self.should_profile:
            return "v0"

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix = self.config.name
        existing_versions = []

        for f in output_dir.glob(f"{prefix}_v*.json"):
            match = re.search(r"_v(\d+)\.json$", f.name)
            if match:
                existing_versions.append(int(match.group(1)))

        next_ver = max(existing_versions) + 1 if existing_versions else 0
        return f"v{next_ver}"

    def _init_profilers(self):
        """Initializes the PyTorch profiler if enabled."""
        if self.config.torch_profiler:
            trace_path = Path(self.config.output_dir)

            # export_chrome_trace and on_trace_ready are mutually exclusive: when a schedule
            # is active, on_trace_ready flushes and clears the event buffer after each cycle,
            # leaving export_chrome_trace with nothing to write. Use one or the other.
            if self.config.export_chrome_trace:
                on_trace_ready = None
            else:
                on_trace_ready = torch.profiler.tensorboard_trace_handler(
                    str(trace_path),
                    worker_name=f"{self.config.name}_{self.current_version}_rank{self.rank}",
                )

            self.torch_profiler = profile(
                activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                on_trace_ready=on_trace_ready,
                schedule=torch.profiler.schedule(
                    wait=self.config.wait_steps,
                    warmup=self.config.warmup_steps,
                    active=self.config.active_steps,
                    repeat=self.config.repeat,
                ),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                with_flops=True,
            )
            self.logger.info(f"Initialized Torch Profiler (Version: {self.current_version})")

    def step(self, step: int):
        """Advances the profiler and checks for step/memory triggers."""
        if not self.should_profile:
            return

        # 1. Step Trigger
        if step == self.config.start:
            self.start()

        # 2. OOM / Memory Trigger
        if self.config.oom_monitor and not self.is_active:
            self._check_memory_threshold()

        # 3. Advance Torch Profiler
        if self.torch_profiler:
            self.torch_profiler.step()

        # 3b. Log per-step comm stats if enabled
        if self.config.comm_profiler and self.is_active:
            stats = self._comm_profiler.get_and_reset_stats()
            if stats:
                for op, s in stats.items():
                    self.logger.debug(
                        f"[step={step}] comm/{op}: total={s['total_ms']:.2f}ms, mean={s['mean_ms']:.2f}ms"
                    )

        # 4. End Trigger
        if step == self.config.end:
            self.stop()

    def _check_memory_threshold(self):
        """Checks if current GPU memory usage exceeds the threshold."""
        if not torch.cuda.is_available():
            return

        device = torch.cuda.current_device()
        total_mem = torch.cuda.get_device_properties(device).total_memory
        used_mem = torch.cuda.memory_reserved(device)
        usage_percent = (used_mem / total_mem) * 100

        if usage_percent >= self.config.oom_threshold:
            self.logger.warning(
                f"Memory usage ({usage_percent:.1f}%) exceeded threshold ({self.config.oom_threshold}%). "
                "Triggering emergency profiling."
            )
            self.start()

    def start(self):
        """Starts hardware and framework-level captures."""
        if self.is_active or not self.should_profile:
            return

        self.logger.info(
            f"Starting hardware capture (ROCTX/NVTX) for {self.config.name} {self.current_version}"
        )

        # Synchronize CUDA before capture to avoid queued work contaminating the first step
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Hardware profiler trigger
        if self.config.gpu_profiler and hasattr(torch.cuda, "profiler"):
            torch.cuda.profiler.start()

        # Torch profiler start (if not already managed by schedule)
        if self.torch_profiler:
            self.torch_profiler.start()

        # Communication profiling
        if self.config.comm_profiler:
            self._comm_profiler.enable()

        # Memory snapshot recording
        if self.config.memory_snapshot and torch.cuda.is_available():
            torch.cuda.memory._record_memory_history(max_entries=100_000)
            self.logger.info("Started CUDA memory history recording")

        # Layer timing
        if self.config.layer_timing:
            self._layer_timing.enable()
            self.logger.info("Started per-layer timing collection")

        self.is_active = True

    def stop(self):
        """Stops all captures and flushes data."""
        if not self.is_active or not self.should_profile:
            return

        self.logger.info(f"Stopping hardware capture and flushing traces for {self.config.name}")

        if self.config.gpu_profiler and hasattr(torch.cuda, "profiler"):
            torch.cuda.profiler.stop()

        if self.torch_profiler:
            self.torch_profiler.stop()
            self._export_trace_formats()

        # Dump memory snapshot
        if self.config.memory_snapshot and torch.cuda.is_available():
            snapshot_path = (
                Path(self.config.output_dir)
                / f"{self.config.name}_{self.current_version}_rank{self.rank}_memory.pickle"
            )
            try:
                torch.cuda.memory._dump_snapshot(str(snapshot_path))
                self.logger.info(f"Saved memory snapshot to {snapshot_path}")
            except Exception as e:
                self.logger.warning(f"Failed to dump memory snapshot: {e}")
            torch.cuda.memory._record_memory_history(enabled=None)

        # Log layer timing summary
        if self.config.layer_timing:
            summary = self._layer_timing.get_summary()
            self.logger.info(f"Per-Layer Timing Summary:\n{summary}")
            self._layer_timing.disable()

        # Log final comm profiling summary
        if self.config.comm_profiler:
            stats = self._comm_profiler.get_and_reset_stats()
            if stats:
                lines = ["Communication Profiling Summary:"]
                for op, s in sorted(stats.items()):
                    lines.append(
                        f"  {op}: count={s['count']}, total={s['total_ms']:.2f}ms, "
                        f"mean={s['mean_ms']:.2f}ms, max={s['max_ms']:.2f}ms"
                    )
                self.logger.info("\n".join(lines))
            self._comm_profiler.disable()

        self.is_active = False

    def _export_trace_formats(self):
        """Export Chrome Tracing JSON and/or CSV after the torch profiler stops."""
        if self.torch_profiler is None:
            return

        output_dir = Path(self.config.output_dir)
        base = f"{self.config.name}_{self.current_version}_rank{self.rank}"

        if self.config.export_chrome_trace:
            chrome_path = output_dir / f"{base}_chrome.json"
            try:
                self.torch_profiler.export_chrome_trace(str(chrome_path))
                self.logger.info(f"Exported Chrome trace to {chrome_path}")
            except Exception as e:
                self.logger.warning(f"Chrome trace export failed: {e}")

        if self.config.export_csv:
            csv_path = output_dir / f"{base}_key_averages.csv"
            try:
                import csv as csv_mod

                events = self.torch_profiler.key_averages()
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv_mod.writer(f)
                    writer.writerow(
                        [
                            "name",
                            "cpu_time_total_us",
                            "cuda_time_total_us",
                            "count",
                            "cpu_time_avg_us",
                            "cuda_time_avg_us",
                            "flops",
                        ]
                    )
                    for e in events:
                        writer.writerow(
                            [
                                e.key,
                                int(e.cpu_time_total),
                                int(e.cuda_time_total),
                                e.count,
                                int(e.cpu_time_total / max(e.count, 1)),
                                int(e.cuda_time_total / max(e.count, 1)),
                                getattr(e, "flops", 0) or 0,
                            ]
                        )
                self.logger.info(f"Exported key-averages CSV to {csv_path}")
            except Exception as e:
                self.logger.warning(f"CSV export failed: {e}")

    def wrap_data_iterator(self, iterator) -> "TimedDataIterator":
        """Wrap a data iterator for per-step load time measurement."""
        timed = TimedDataIterator(iterator)
        self._timed_data_iter = timed
        return timed

    def get_data_load_stats(self) -> "dict[str, float] | None":
        """Return and reset per-step data loading stats, or None if not enabled."""
        if self._timed_data_iter is None:
            return None
        return self._timed_data_iter.get_and_reset_stats()
