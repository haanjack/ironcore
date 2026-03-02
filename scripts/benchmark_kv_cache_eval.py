#!/usr/bin/env python
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Benchmark script to compare evaluation speed with and without KV cache.
# Usage: python scripts/benchmark_kv_cache_eval.py [--num-samples N] [--no-cache]

import argparse
import time
import sys
from pathlib import Path

import torch
import torch.distributed as dist

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ironcore import get_tokenizer, set_global_states
from ironcore.config import MainConfig, load_trainer_config
from ironcore.eval.tasks.hellaswag import HellaSwag
from ironcore.language_model import LanguageModel
from ironcore.parallel.parallel_states import initialize_model_parallel
from ironcore.tokenizer import build_tokenizer
from ironcore.utils import get_model_dtype


def run_benchmark_simple(num_samples: int = 100, use_cache: bool = False):
    """Run a simple benchmark comparing cache vs no-cache."""

    # Load config from existing yaml
    config_path = Path(__file__).parent.parent / "configs" / "data" / "full_owt_pretrain.yaml"
    if not config_path.exists():
        print(f"Config not found at {config_path}")
        return None

    # Override use_kv_cache setting
    # Note: This requires the config system to support this

    print(f"Running benchmark with KV cache={use_cache}, samples={num_samples}")

    # For now, just test the evaluation task directly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create a simple model for testing
    from ironcore.config.config_model import ModelConfig, KVCacheConfig, PositionalEmbeddingConfig

    kv_cache_config = KVCacheConfig(
        enabled=use_cache,
        max_batch_size=4,
        max_seq_length=512,
    )

    pos_emb = PositionalEmbeddingConfig(type="rope")

    model_config = ModelConfig(
        d_model=256,
        num_attention_heads=8,
        num_attention_groups=4,
        head_dim=32,
        num_layers=2,
        d_ffn=512,
        max_seq_len=512,
        max_position_embeddings=512,
        positional_embedding=pos_emb,
        kv_cache=kv_cache_config,
    )

    # Build a minimal model
    from ironcore.config import (
        DataConfig, InitConfig, OptimConfig, ParallelConfig,
        OperationConfig, TrainerConfig, UtilsConfig, ProfilerConfig,
    )

    config = MainConfig(
        model=model_config,
        trainer=TrainerConfig(
            tensor_model_parallel_size=1,
            use_flash_attn=False,
            use_kv_cache_in_eval=use_cache,
        ),
        init=InitConfig(seed=42),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        operation=OperationConfig(train_steps=1),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
    )

    # Initialize torch distributed FIRST (required for model parallel)
    import os
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    # Initialize parallel states (requires dist to be initialized)
    initialize_model_parallel(1, timeout_in_minutes=10.0)

    # Set global states (required for tokenizer)
    from ironcore.global_vars import GLOBAL_STATES
    if GLOBAL_STATES is not None:
        # Reset global states for second run
        import ironcore.global_vars as gv
        gv.GLOBAL_STATES = None
    set_global_states(config)

    # Build tokenizer
    build_tokenizer(config)
    tokenizer = get_tokenizer()

    # Create model
    model = LanguageModel(config).to(device=device, dtype=torch.float32)
    model.eval()

    # Initialize cache if needed
    if use_cache and hasattr(model, "initialize_cache"):
        model.initialize_cache(batch_size=4, device=device)
        print("KV cache initialized")

    # Create evaluator
    evaluator = HellaSwag(
        tokenizer=tokenizer,
        batch_size=4,
        num_samples=num_samples,
        cache_dir=Path("./cache"),
    )

    # Time the evaluation
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()

    result = evaluator.process(model)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time

    return {
        "accuracy": result["score"],
        "time": elapsed_time,
        "samples_per_sec": num_samples / elapsed_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV cache for HellaSwag evaluation")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of HellaSwag samples")
    parser.add_argument("--no-cache", action="store_true", help="Run without KV cache comparison")
    args = parser.parse_args()

    print(f"Benchmarking on HellaSwag ({args.num_samples} samples)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU (will be slow)")

    results = {}

    for use_cache in [False, True]:
        if args.no_cache and use_cache:
            continue

        print(f"\n--- Running with KV cache: {use_cache} ---")

        result = run_benchmark_simple(num_samples=args.num_samples, use_cache=use_cache)

        if result is None:
            print("Benchmark failed")
            continue

        results[use_cache] = result

        print(f"Accuracy: {result['accuracy']:.2f}%")
        print(f"Time: {result['time']:.2f}s")
        print(f"Throughput: {result['samples_per_sec']:.2f} samples/sec")

        # Cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print comparison
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"{'Metric':<20} {'No Cache':<15} {'With Cache':<15} {'Speedup':<10}")
        print("-" * 60)

        no_cache = results[False]
        with_cache = results[True]

        speedup_time = no_cache['time'] / with_cache['time'] if with_cache['time'] > 0 else 0
        speedup_throughput = with_cache['samples_per_sec'] / no_cache['samples_per_sec'] if no_cache['samples_per_sec'] > 0 else 0

        print(f"{'Time (s)':<20} {no_cache['time']:<15.2f} {with_cache['time']:<15.2f} {speedup_time:<10.2f}x")
        print(f"{'Samples/sec':<20} {no_cache['samples_per_sec']:<15.2f} {with_cache['samples_per_sec']:<15.2f} {speedup_throughput:<10.2f}x")
        print(f"{'Accuracy (%)':<20} {no_cache['accuracy']:<15.2f} {with_cache['accuracy']:<15.2f} {'-':<10}")

        # Verify accuracy is identical
        if abs(no_cache['accuracy'] - with_cache['accuracy']) < 0.01:
            print("\n✓ Accuracy matches - KV cache produces identical results")
        else:
            print(f"\n✗ WARNING: Accuracy differs by {abs(no_cache['accuracy'] - with_cache['accuracy']):.2f}%")


if __name__ == "__main__":
    main()
