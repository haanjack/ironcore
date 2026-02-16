# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""
Benchmark: Python vs Triton Paged Attention

Compares performance of:
1. Python gather + attention (baseline)
2. Python batched gather + SDPA
3. Triton gather + SDPA

Run with: python benchmark_triton_paged_attention.py
"""

import time
from dataclasses import dataclass

import torch

# Check Triton availability
try:
    from ironcore.layers.triton_paged_attention import (
        TRITON_AVAILABLE,
        python_paged_attention,
        python_paged_attention_batched,
        triton_paged_attention,
    )
except ImportError:
    TRITON_AVAILABLE = False


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark."""

    batch_size: int = 4
    num_heads: int = 8
    num_kv_heads: int = 2  # GQA: 8 query heads, 2 KV heads
    head_dim: int = 64
    page_size: int = 16
    max_seq_len: int = 4096  # Increased for longer sequence tests
    num_pages: int = 2048  # Increased to accommodate longer sequences
    warmup_iters: int = 10
    benchmark_iters: int = 100
    triton_compile_iters: int = 3  # Extra warmup for Triton compilation
    device: str = "cuda"


def create_test_data(config: BenchmarkConfig, seq_lengths: list[int]):
    """Create test tensors for benchmarking."""
    batch_size = config.batch_size
    num_heads = config.num_heads
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim
    page_size = config.page_size
    num_pages = config.num_pages
    device = config.device

    # Query: [batch, 1, num_heads, head_dim]
    query = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=torch.float16)

    # Physical K/V cache: [num_pages, num_kv_heads, page_size, head_dim]
    key_cache = torch.randn(
        num_pages, num_kv_heads, page_size, head_dim, device=device, dtype=torch.float16
    )
    value_cache = torch.randn(
        num_pages, num_kv_heads, page_size, head_dim, device=device, dtype=torch.float16
    )

    # Block tables: [batch, max_pages_per_seq]
    max_pages_per_seq = (config.max_seq_len + page_size - 1) // page_size
    block_tables = torch.zeros(batch_size, max_pages_per_seq, device=device, dtype=torch.long)

    # Context lengths
    context_lens = torch.tensor(seq_lengths, device=device, dtype=torch.long)

    # Fill block tables with sequential physical page indices
    page_idx = 0
    for b, seq_len in enumerate(seq_lengths):
        pages_needed = (seq_len + page_size - 1) // page_size
        for p in range(pages_needed):
            block_tables[b, p] = page_idx
            page_idx += 1

    return query, key_cache, value_cache, block_tables, context_lens


def benchmark_function(
    func,
    args,
    warmup_iters: int,
    benchmark_iters: int,
    name: str,
    triton_compile_iters: int = 0,
) -> dict:
    """Benchmark a function with proper warmup."""
    # Warmup (includes Triton compilation)
    total_warmup = warmup_iters + triton_compile_iters
    for i in range(total_warmup):
        _ = func(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    for _ in range(benchmark_iters):
        _ = func(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    end_time = time.perf_counter()
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    avg_time_ms = (end_time - start_time) / benchmark_iters * 1000

    return {
        "name": name,
        "avg_time_ms": avg_time_ms,
        "peak_memory_mb": peak_memory,
    }


def run_benchmark(config: BenchmarkConfig, seq_lengths: list[int], label: str):
    """Run benchmark for a specific configuration."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {label}")
    print(f"Batch size: {config.batch_size}, Heads: {config.num_heads}/{config.num_kv_heads}")
    print(f"Head dim: {config.head_dim}, Page size: {config.page_size}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"{'=' * 60}")

    query, key_cache, value_cache, block_tables, context_lens = create_test_data(
        config, seq_lengths
    )

    num_heads = config.num_heads
    num_kv_heads = config.num_kv_heads
    page_size = config.page_size

    results = []

    # 1. Python baseline (loop per sequence)
    print("\nRunning Python (loop) baseline...")
    result = benchmark_function(
        python_paged_attention,
        (
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            num_heads,
            num_kv_heads,
            page_size,
        ),
        config.warmup_iters,
        config.benchmark_iters,
        "Python (loop)",
    )
    results.append(result)
    print(f"  {result['name']}: {result['avg_time_ms']:.3f} ms, {result['peak_memory_mb']:.1f} MB")

    # 2. Python batched
    print("\nRunning Python (batched + SDPA)...")
    result = benchmark_function(
        python_paged_attention_batched,
        (
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            num_heads,
            num_kv_heads,
            page_size,
        ),
        config.warmup_iters,
        config.benchmark_iters,
        "Python (batched)",
    )
    results.append(result)
    print(f"  {result['name']}: {result['avg_time_ms']:.3f} ms, {result['peak_memory_mb']:.1f} MB")

    # 3. Triton kernel (with extra warmup for compilation)
    if TRITON_AVAILABLE:
        print("\nRunning Triton (gather + SDPA)...")
        result = benchmark_function(
            triton_paged_attention,
            (
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                num_heads,
                num_kv_heads,
                page_size,
            ),
            config.warmup_iters,
            config.benchmark_iters,
            "Triton",
            triton_compile_iters=config.triton_compile_iters,
        )
        results.append(result)
        print(
            f"  {result['name']}: {result['avg_time_ms']:.3f} ms, {result['peak_memory_mb']:.1f} MB"
        )

        # Verify correctness
        py_output = python_paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            num_heads,
            num_kv_heads,
            page_size,
        )
        triton_output = triton_paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            num_heads,
            num_kv_heads,
            page_size,
        )
        max_diff = (py_output - triton_output).abs().max().item()
        print(f"  Max difference vs Python: {max_diff:.6f}")
    else:
        print("\nTriton not available, skipping...")

    return results


def verify_correctness(config: BenchmarkConfig):
    """Verify that implementations produce the same output."""
    print("\n" + "=" * 60)
    print("Correctness Verification")
    print("=" * 60)

    # Use batch_size matching config
    seq_lengths = [32, 64, 128, 256][: config.batch_size]
    query, key_cache, value_cache, block_tables, context_lens = create_test_data(
        config, seq_lengths
    )

    py_output = python_paged_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        config.num_heads,
        config.num_kv_heads,
        config.page_size,
    )

    py_batched_output = python_paged_attention_batched(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        config.num_heads,
        config.num_kv_heads,
        config.page_size,
    )

    loop_vs_batched_diff = (py_output - py_batched_output).abs().max().item()
    print(f"Python loop vs batched max diff: {loop_vs_batched_diff:.6f}")

    if TRITON_AVAILABLE:
        triton_output = triton_paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            config.num_heads,
            config.num_kv_heads,
            config.page_size,
        )

        diff = (py_output - triton_output).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"Python vs Triton max diff: {max_diff:.6f}")
        print(f"Python vs Triton mean diff: {mean_diff:.6f}")

        # Check if outputs are close (allowing for float16 precision)
        if max_diff < 0.05:
            print("✅ Correctness verified (within float16 tolerance)")
        else:
            print("⚠️  Outputs differ more than expected")
            # Debug: print shapes
            print(f"   py_output shape: {py_output.shape}")
            print(f"   triton_output shape: {triton_output.shape}")

    print()


def print_summary(all_results: list[dict]):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(
        f"{'Configuration':<40} {'Python (ms)':<12} {'Batched (ms)':<12} {'Triton (ms)':<12} {'Speedup':<10}"
    )
    print("-" * 80)

    for config_name, results in all_results:
        py_time = next((r["avg_time_ms"] for r in results if "loop" in r["name"]), 0)
        batched_time = next((r["avg_time_ms"] for r in results if "batched" in r["name"]), 0)
        triton_time = next((r["avg_time_ms"] for r in results if r["name"] == "Triton"), 0)

        speedup = f"{py_time / triton_time:.2f}x" if triton_time > 0 else "N/A"

        print(
            f"{config_name:<40} {py_time:<12.3f} {batched_time:<12.3f} {triton_time:<12.3f} {speedup:<10}"
        )

    print("=" * 80)


def main():
    """Run all benchmarks."""
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU (will be slow)")
        device = "cpu"
    else:
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    print(f"Triton available: {TRITON_AVAILABLE}")

    # Standard configuration
    config = BenchmarkConfig(device=device)

    # Verify correctness first
    verify_correctness(config)

    all_results = []

    # Test different sequence length patterns
    test_cases = [
        ("Short sequences (16-64)", [16, 32, 48, 64]),
        ("Medium sequences (64-256)", [64, 128, 192, 256]),
        ("Long sequences (256-512)", [256, 384, 448, 512]),
        ("Mixed lengths (32-512)", [32, 128, 256, 512]),
        ("Uniform 128", [128, 128, 128, 128]),
        ("Uniform 256", [256, 256, 256, 256]),
        # Longer sequence tests
        ("Long sequences (512-1024)", [512, 768, 896, 1024]),
        ("Very long (1024-2048)", [1024, 1536, 1792, 2048]),
        ("Ultra long (2048-4096)", [2048, 3072, 3584, 4096]),
        ("Uniform 1024", [1024, 1024, 1024, 1024]),
        ("Uniform 2048", [2048, 2048, 2048, 2048]),
    ]

    for name, seq_lengths in test_cases:
        results = run_benchmark(config, seq_lengths, name)
        all_results.append((name, results))

    # Print summary
    print_summary(all_results)

    # Additional: Test with larger batch
    print("\n" + "=" * 60)
    print("LARGE BATCH TEST (batch_size=16)")
    print("=" * 60)

    large_config = BenchmarkConfig(batch_size=16, device=device)
    results = run_benchmark(large_config, [128] * 16, "Batch=16, all 128 tokens")
    all_results.append(("Large batch (16x128)", results))

    # Final summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
