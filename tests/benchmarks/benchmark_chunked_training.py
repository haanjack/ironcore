#!/usr/bin/env python
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Training benchmark for chunked tensor parallelism with TP=1 and TP=2.

Measures iteration time and peak memory across:
  - Tensor parallel sizes: 1, 2
  - Sequence lengths: 1024, 4096, 8192
  - Chunk sizes: None (baseline), 2048, 1024, 512, 256
  - Uses flash attention + bfloat16

Metrics collected:
  - Average/min/max iteration time
  - Peak memory allocation and reservation
  - Loss values for correctness validation
  - Throughput (tokens/sec)

Usage:
    # TP=1 (single GPU)
    python tests/benchmark_chunked_training.py --tp 1

    # TP=2 (2 GPUs with async all-reduce)
    torchrun --nproc_per_node=2 tests/benchmark_chunked_training.py --tp 2

    # Run both and compare (recommended)
    python tests/benchmark_chunked_training.py --run-both
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ironcore.config import (
    DataConfig,
    InitConfig,
    MainConfig,
    ModelConfig,
    OperationConfig,
    OptimConfig,
    ParallelConfig,
    ProfilerConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.models.transformer import TransformerModel
from ironcore.parallel import parallel_states

# ── Model ──────────────────────────────────────────────────────────────
D_MODEL = 512
NUM_HEADS = 8
HEAD_DIM = D_MODEL // NUM_HEADS  # 64
NUM_GROUPS = 8
D_FFN = 2048
NUM_LAYERS = 6
BATCH_SIZE = 1

# ── Benchmark ──────────────────────────────────────────────────────────
WARMUP_STEPS = 5
MEASURE_STEPS = 10

SEQUENCE_LENGTHS = [1024, 4096, 8192]
CHUNK_SIZES = [None, 2048, 1024, 512, 256]  # None = baseline (no chunking)


def create_config(seq_len, chunk_size, tp_size, use_flash_attn=False):
    """Create configuration for benchmarking.

    NOTE: A general-purpose config factory is available at
    tests.fixtures.config_fixtures.create_test_config. This local version is
    kept because it uses benchmark-specific constants and parameter ordering.
    """
    return MainConfig(
        model=ModelConfig(
            d_model=D_MODEL,
            num_attention_heads=NUM_HEADS,
            num_attention_groups=NUM_GROUPS,
            head_dim=HEAD_DIM,
            d_ffn=D_FFN,
            num_layers=NUM_LAYERS,
            max_seq_len=seq_len,
            max_position_embeddings=seq_len,
            dropout_attn=0.0,
            dropout_mlp=0.0,
            dropout_embd=0.0,
            attention_bias=True,
            mlp_bias=True,
            layernorm_bias=True,
            precision="bfloat16",
        ),
        trainer=TrainerConfig(
            tensor_model_parallel_size=tp_size,
            use_flash_attn=use_flash_attn,
            sequence_chunk_size=chunk_size,
        ),
        init=InitConfig(seed=42, init_std=0.02),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(timeout_minute=30),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
    )


def benchmark_one(seq_len, chunk_size, tp_size, device, use_flash_attn=False):
    """Benchmark a single (seq_len, chunk_size, tp_size) configuration."""
    config = create_config(seq_len, chunk_size, tp_size, use_flash_attn=use_flash_attn)

    rank = dist.get_rank() if dist.is_initialized() else 0

    # Set seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = TransformerModel(config).to(device=device, dtype=torch.bfloat16)
    model.init_weights()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Input — no explicit mask needed for flash attention (causal=True internally)
    hidden = torch.randn(BATCH_SIZE, seq_len, D_MODEL, device=device, dtype=torch.bfloat16)

    # ── Warmup ─────────────────────────────────────────────────────────
    for _ in range(WARMUP_STEPS):
        optimizer.zero_grad()
        output = model(hidden, attention_mask=None, rotary_pos_emb=None)
        loss = output.pow(2).mean()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    # ── Reset memory stats after warmup ────────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)

    # ── Measure ────────────────────────────────────────────────────────
    times = []
    losses = []

    for _ in range(MEASURE_STEPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        output = model(hidden, attention_mask=None, rotary_pos_emb=None)
        loss = output.pow(2).mean()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        times.append(t1 - t0)
        losses.append(loss.item())

    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**2)
    peak_resv = torch.cuda.max_memory_reserved(device) / (1024**2)

    # Calculate throughput
    avg_time = sum(times) / len(times)
    throughput = (BATCH_SIZE * seq_len) / avg_time  # tokens/sec

    # ── Cleanup ────────────────────────────────────────────────────────
    del model, optimizer, hidden, output, loss
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "rank": rank,
        "tp_size": tp_size,
        "seq_len": seq_len,
        "chunk_size": chunk_size,
        "avg_time": avg_time,
        "min_time": min(times),
        "max_time": max(times),
        "std_time": (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5,
        "peak_alloc_mib": peak_alloc,
        "peak_resv_mib": peak_resv,
        "final_loss": losses[-1],
        "avg_loss": sum(losses) / len(losses),
        "throughput_tokens_per_sec": throughput,
    }


def run_benchmark(tp_size):
    """Run benchmark for a given TP size."""
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401

        use_flash_attn = True
    except ImportError:
        print("WARNING: flash_attn not installed, using standard attention")
        use_flash_attn = False

    # Initialize distributed
    if tp_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Set CUDA device for this rank
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        parallel_states.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            timeout_in_minutes=30,
        )
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0")

        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", rank=0, world_size=1)

        try:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=1.0,
            )
        except Exception:
            pass

    if rank == 0:
        gpu_name = torch.cuda.get_device_properties(0).name
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        print(f"\n{'=' * 90}")
        print(f"BENCHMARK: TP={tp_size} ({world_size} GPU{'s' if world_size > 1 else ''})")
        print(f"{'=' * 90}")
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"Model: d={D_MODEL}, heads={NUM_HEADS}, ffn={D_FFN}, layers={NUM_LAYERS}")
        print(f"Batch: {BATCH_SIZE}, dtype: bfloat16, flash_attn: {use_flash_attn}")
        print(f"Warmup: {WARMUP_STEPS} steps, Measure: {MEASURE_STEPS} steps")
        print()

    all_results = []

    for seq_len in SEQUENCE_LENGTHS:
        for chunk_size in CHUNK_SIZES:
            # Skip incompatible configurations
            if chunk_size is not None and chunk_size >= seq_len:
                continue

            label = f"seq={seq_len:<5} chunk={'none' if chunk_size is None else chunk_size:<5}"

            if rank == 0:
                print(f"  {label} tp={tp_size} ...", end=" ", flush=True)

            try:
                result = benchmark_one(
                    seq_len, chunk_size, tp_size, device, use_flash_attn=use_flash_attn
                )
                all_results.append(result)

                if rank == 0:
                    print(
                        f"avg={result['avg_time'] * 1000:.2f}ms  "
                        f"mem={result['peak_alloc_mib']:.0f}MB  "
                        f"throughput={result['throughput_tokens_per_sec']:.0f} tok/s"
                    )

            except torch.cuda.OutOfMemoryError:
                if rank == 0:
                    print("OOM")
                torch.cuda.empty_cache()
                gc.collect()
                all_results.append(
                    {
                        "rank": rank,
                        "tp_size": tp_size,
                        "seq_len": seq_len,
                        "chunk_size": chunk_size,
                        "error": "OOM",
                    }
                )
            except Exception as e:
                if rank == 0:
                    print(f"ERROR: {e}")
                all_results.append(
                    {
                        "rank": rank,
                        "tp_size": tp_size,
                        "seq_len": seq_len,
                        "chunk_size": chunk_size,
                        "error": str(e),
                    }
                )

    # Gather results from all ranks
    if world_size > 1:
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, all_results)
        if rank == 0:
            all_results = []
            for rank_results in gathered_results:
                if rank_results is not None:
                    all_results.extend(rank_results)

    # Save and print results
    if rank == 0:
        output_dir = Path("logs/chunked_benchmark")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"benchmark_tp{tp_size}.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

        # Print summary table
        print_summary(all_results, tp_size)

    # Cleanup
    if dist.is_initialized():
        dist.barrier()
        if tp_size > 1:
            dist.destroy_process_group()

    return all_results


def print_summary(results, tp_size):
    """Print summary table of results."""
    print(f"\n{'=' * 120}")
    print(f"SUMMARY: TP={tp_size}")
    print(f"{'=' * 120}")

    # Filter rank 0 results and group by seq_len
    rank0_results = [r for r in results if r.get("rank", 0) == 0 and "error" not in r]

    if not rank0_results:
        print("No successful results to display")
        return

    # Find baselines (chunk_size=None) for each seq_len
    baselines = {}
    for r in rank0_results:
        if r["chunk_size"] is None:
            baselines[r["seq_len"]] = r

    print(
        f"{'Seq':>5}  {'Chunk':>6}  {'Chunks':>7}  "
        f"{'Time (ms)':>11}  {'vs Base':>9}  "
        f"{'Mem (MB)':>10}  {'Mem Δ':>9}  "
        f"{'Throughput':>12}"
    )
    print("-" * 120)

    for seq_len in sorted(set(r["seq_len"] for r in rank0_results)):
        seq_results = [r for r in rank0_results if r["seq_len"] == seq_len]
        baseline = baselines.get(seq_len)

        for r in sorted(
            seq_results, key=lambda x: (x["chunk_size"] is None, x["chunk_size"] or 0), reverse=True
        ):
            chunk_label = "none" if r["chunk_size"] is None else str(r["chunk_size"])
            num_chunks = (
                1 if r["chunk_size"] is None else (seq_len + r["chunk_size"] - 1) // r["chunk_size"]
            )

            if r["chunk_size"] is None or baseline is None:
                speedup_str = "baseline"
                mem_delta_str = "—"
            else:
                speedup = baseline["avg_time"] / r["avg_time"]
                speedup_str = f"{speedup:.3f}x"
                mem_delta = r["peak_alloc_mib"] - baseline["peak_alloc_mib"]
                mem_delta_str = f"{mem_delta:+.0f}MB"

            print(
                f"{seq_len:>5}  {chunk_label:>6}  {num_chunks:>7}  "
                f"{r['avg_time'] * 1000:>10.2f}ms  {speedup_str:>9}  "
                f"{r['peak_alloc_mib']:>9.0f}MB  {mem_delta_str:>9}  "
                f"{r['throughput_tokens_per_sec']:>11.0f} t/s"
            )

    print()


def compare_tp1_tp2():
    """Compare TP=1 and TP=2 results."""
    logs_dir = Path("logs/chunked_benchmark")
    tp1_file = logs_dir / "benchmark_tp1.json"
    tp2_file = logs_dir / "benchmark_tp2.json"

    if not tp1_file.exists() or not tp2_file.exists():
        print("ERROR: Missing benchmark files. Run both TP=1 and TP=2 first.")
        return

    with open(tp1_file) as f:
        tp1_results = json.load(f)
    with open(tp2_file) as f:
        tp2_results = json.load(f)

    # Filter rank 0 results
    tp1_results = [r for r in tp1_results if r.get("rank", 0) == 0 and "error" not in r]
    tp2_results = [r for r in tp2_results if r.get("rank", 0) == 0 and "error" not in r]

    print(f"\n{'=' * 120}")
    print("TP=1 vs TP=2 COMPARISON")
    print(f"{'=' * 120}\n")

    print(
        f"{'Seq':>5}  {'Chunk':>6}  "
        f"{'TP=1 Time':>11}  {'TP=2 Time':>11}  {'Speedup':>9}  "
        f"{'TP=1 Mem':>10}  {'TP=2 Mem':>10}  "
        f"{'Loss Δ':>10}"
    )
    print("-" * 120)

    # Create lookup for TP=2
    tp2_lookup = {}
    for r in tp2_results:
        key = (r["seq_len"], r["chunk_size"])
        tp2_lookup[key] = r

    for r1 in tp1_results:
        key = (r1["seq_len"], r1["chunk_size"])
        r2 = tp2_lookup.get(key)

        chunk_label = "none" if r1["chunk_size"] is None else str(r1["chunk_size"])

        if r2 is None:
            print(
                f"{r1['seq_len']:>5}  {chunk_label:>6}  "
                f"{r1['avg_time'] * 1000:>10.2f}ms  {'—':>11}  {'—':>9}  "
                f"{r1['peak_alloc_mib']:>9.0f}MB  {'—':>10}  {'—':>10}"
            )
            continue

        speedup = r1["avg_time"] / r2["avg_time"]
        loss_diff = abs(r1["final_loss"] - r2["final_loss"])

        print(
            f"{r1['seq_len']:>5}  {chunk_label:>6}  "
            f"{r1['avg_time'] * 1000:>10.2f}ms  {r2['avg_time'] * 1000:>10.2f}ms  {speedup:>8.3f}x  "
            f"{r1['peak_alloc_mib']:>9.0f}MB  {r2['peak_alloc_mib']:>9.0f}MB  "
            f"{loss_diff:>10.2e}"
        )

    print()


def run_both():
    """Run benchmarks for both TP=1 and TP=2."""
    print("=" * 90)
    print("COMPREHENSIVE CHUNKED TENSOR PARALLELISM BENCHMARK")
    print("=" * 90)
    print()

    # Run TP=1
    print("Running TP=1 benchmark...")
    result_tp1 = subprocess.run(
        [sys.executable, __file__, "--tp", "1"],
        cwd=Path(__file__).parent.parent,
        check=False,
    )

    if result_tp1.returncode != 0:
        print("ERROR: TP=1 benchmark failed!")
        return

    # Run TP=2
    print("\nRunning TP=2 benchmark...")
    result_tp2 = subprocess.run(
        ["torchrun", "--nproc_per_node=2", __file__, "--tp", "2"],
        cwd=Path(__file__).parent.parent,
        check=False,
    )

    if result_tp2.returncode != 0:
        print("ERROR: TP=2 benchmark failed!")
        return

    # Compare results
    compare_tp1_tp2()


def main():
    parser = argparse.ArgumentParser(description="Benchmark chunked tensor parallelism")
    parser.add_argument("--tp", type=int, choices=[1, 2], help="Tensor parallel size")
    parser.add_argument(
        "--run-both", action="store_true", help="Run both TP=1 and TP=2 and compare"
    )
    args = parser.parse_args()

    if args.run_both:
        run_both()
    elif args.tp is not None:
        run_benchmark(args.tp)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
