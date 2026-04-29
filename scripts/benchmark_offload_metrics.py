# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Benchmark: iteration speed + memory (VRAM + host RAM) for offload modes."""

import os
import sys
import time
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.fixtures.config_fixtures import create_test_config

from ironcore.global_vars import reset_global_states
from ironcore.trainers import LanguageModelTrainer
from ironcore.utils.memory import get_host_memory_usage, get_memory_usage

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29501")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")


def setup():
    reset_global_states()
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)


def mock_forward_step(model, data_iterator):
    device = next(model.parameters()).device
    input_ids = torch.randint(0, 1000, (2, 16), device=device)
    labels = input_ids.clone()
    logits = model(input_ids, labels=None)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def mock_data_iter():
    return {"train": iter([]), "eval": iter([]), "test": iter([])}


def run_benchmark(label, config):
    setup()
    with (
        patch(
            "ironcore.trainers.base_trainer.get_data_iterator",
            return_value=mock_data_iter(),
        ),
        patch("ironcore.trainers.base_trainer.get_evaluators", return_value=[]),
    ):
        trainer = LanguageModelTrainer(config, mock_forward_step, F.cross_entropy)
        trainer._initialize()

        # Warmup step
        trainer.train_step(step=0)

        # Timed steps
        times = []
        for step in range(1, 6):
            t0 = time.perf_counter()
            loss, _, _ = trainer.train_step(step=step)
            times.append(time.perf_counter() - t0)

        avg_iter = sum(times) / len(times)
        host_mem = get_host_memory_usage()
        gpu_mem = get_memory_usage(in_mib=True)

        print(f"\n=== {label} ===")
        print(f"  Iteration speed:  {avg_iter * 1000:.1f} ms/step  ({1 / avg_iter:.1f} steps/sec)")
        print(f"  Loss (last step): {loss:.4f}")
        print(f"  Host RAM RSS:     {host_mem['rss_mb']:.0f} MB")
        print(f"  Host RAM Peak:    {host_mem['peak_rss_mb']:.0f} MB")
        if gpu_mem:
            print(f"  VRAM Allocated:   {gpu_mem.get('memory_allocated', 0)} MiB")
            print(f"  VRAM Reserved:    {gpu_mem.get('memory_reserved', 0)} MiB")
            print(f"  VRAM Peak Alloc:  {gpu_mem.get('max_memory_allocated', 0)} MiB")

        if trainer._offload_scheduler is not None:
            m = trainer._offload_scheduler.get_metrics()
            print(f"  Step elapsed:     {m['step_elapsed_ms']:.1f} ms")
            print(f"  H2D prefetch:     {m['h2d_ms']:.1f} ms")
            print(f"  D2H snapshot:     {m['d2h_snapshot_ms']:.1f} ms")
            print(f"  Host pool used:   {m['host_pool_used_mb']:.1f} MB")
        else:
            print("  (no offload scheduler)")

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    # GPT-2 Large: d_model=1280, heads=20, d_ffn=5120, layers=36 (~774M params)
    gpt2_large = dict(
        d_model=1280,
        num_attention_heads=20,
        num_attention_groups=20,
        head_dim=64,
        d_ffn=5120,
        num_layers=36,
        max_seq_len=128,
        precision="float32",
        seed=42,
    )

    # Baseline (no offload)
    cfg = create_test_config(**gpt2_large)
    cfg.trainer.gradient_accumulation_steps = 1
    run_benchmark("GPT-2 Large Baseline (~774M)", cfg)

    # M1: Optimizer offload
    cfg = create_test_config(**gpt2_large)
    cfg.offload.enabled = True
    cfg.offload.optimizer_offload = True
    cfg.trainer.gradient_accumulation_steps = 1
    run_benchmark("GPT-2 Large + M1 Optimizer Offload", cfg)

    # M2: Weight streaming
    cfg = create_test_config(**gpt2_large)
    cfg.offload.enabled = True
    cfg.offload.weight_offload = True
    cfg.trainer.gradient_accumulation_steps = 1
    run_benchmark("GPT-2 Large + M2 Weight Streaming", cfg)
