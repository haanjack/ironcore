# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Export CUDA/CPU traces for one actual context-block benchmark trainer update.

All unrecognized arguments are forwarded to benchmark_context_blocks.py. The
resulting benchmark timings include profiling/export overhead: use separate,
uninstrumented runs for performance comparisons.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

from ironcore.layers import context_parallel_attention
from ironcore.layers.moe import scheduled
from ironcore.trainers import LanguageModelTrainer


def _annotate(module, name, label):
    original = getattr(module, name)

    def wrapped(*args, **kwargs):
        with record_function(label):
            return original(*args, **kwargs)

    setattr(module, name, wrapped)


def main() -> None:
    """Reuse the production trainer and save one trace per CUDA rank."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-directory", type=Path, required=True)
    parser.add_argument("--profile-update", type=int, default=2)
    args, forwarded = parser.parse_known_args()
    if args.profile_update < 1:
        parser.error("profile-update must be positive")
    args.trace_directory.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    trace_path = args.trace_directory / f"rank{rank}.json"
    if trace_path.exists():
        parser.error("Use a fresh trace directory")
    for name in ("_pack", "_project", "_scatter"):
        _annotate(scheduled, name, f"moe{name}")
    _annotate(context_parallel_attention, "_exchange", "cp_post_exchange")
    original_step = LanguageModelTrainer.train_step
    exported = False

    def measured_step(trainer, step):
        nonlocal exported
        if step + 1 != args.profile_update:
            return original_step(trainer, step)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            with record_function(f"PROFILE_UPDATE_{args.profile_update}"):
                result = original_step(trainer, step)
                torch.cuda.synchronize()
        profiler.export_chrome_trace(str(trace_path))
        table_path = args.trace_directory / f"rank{dist.get_rank()}-table.txt"
        table_path.write_text(
            profiler.key_averages().table(sort_by="self_device_time_total", row_limit=45)
        )
        exported = True
        print(f"PROFILE_EXPORTED rank={dist.get_rank()}", flush=True)
        return result

    LanguageModelTrainer.train_step = measured_step
    path = Path(__file__).with_name("benchmark_context_blocks.py")
    spec = importlib.util.spec_from_file_location("context_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(path), *forwarded]
    print(
        "PROFILE ONLY: recorded update timings include instrumentation/export overhead.", flush=True
    )
    module.main()
    if not exported:
        raise ValueError("Requested profile update was not reached")


if __name__ == "__main__":
    main()
