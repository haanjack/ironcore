# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

import re
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from ironcore.config import MainConfig
from ironcore.global_vars import get_logger


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

        if self.should_profile:
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

            self.torch_profiler = profile(
                activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(trace_path),
                    worker_name=f"{self.config.name}_{self.current_version}_rank{self.rank}",
                ),
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

        # 4. End Trigger
        if step == self.config.end:
            self.stop()
            if self.config.stop_at_end:
                self.logger.info("Stopping training as requested by stop_at_end")
                sys.exit(0)

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

        # Hardware profiler trigger
        if self.config.gpu_profiler and hasattr(torch.cuda, "profiler"):
            torch.cuda.profiler.start()

        # Torch profiler start (if not already managed by schedule)
        if self.torch_profiler:
            self.torch_profiler.start()

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

        self.is_active = False
