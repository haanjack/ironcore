# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark script for MoE performance comparison.

This script benchmarks:
1. Dense MLP baseline
2. MoE with varying expert counts
3. MoE with varying top-k values
4. Memory usage comparison

Usage:
    python tests/performance/benchmark_moe.py
"""

import time
from dataclasses import dataclass

import torch
from torch import nn

from ironcore.config import MainConfig, PEFTConfig
from ironcore.config.config_data import DataConfig
from ironcore.config.config_model import ModelConfig
from ironcore.config.config_moe import MoEConfig
from ironcore.config.config_optim import OptimConfig
from ironcore.config.config_parallel import ParallelConfig
from ironcore.config.config_trainer import InitConfig, OperationConfig, TrainerConfig
from ironcore.config.config_utils import UtilsConfig
from ironcore.layers.mlp import MLP
from ironcore.layers.moe import MoEMLP


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    memory_mb: float
    params_m: float


def create_test_config(
    hidden_size: int = 1024,
    intermediate_size: int = 4096,
    num_shared_experts: int = 2,
    num_routed_experts: int = 64,
    top_k: int = 6,
) -> MainConfig:
    """Create test configuration."""
    return MainConfig(
        model=ModelConfig(
            d_model=hidden_size,
            d_ffn=intermediate_size,
            dropout_mlp=0.0,
            activation_type="gelu",
            moe=MoEConfig(
                use_moe=True,
                num_shared_experts=num_shared_experts,
                num_routed_experts=num_routed_experts,
                num_experts_per_token=top_k,
            ),
        ),
        init=InitConfig(),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        trainer=TrainerConfig(),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        peft=PEFTConfig(),
    )


def get_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def count_parameters(model: nn.Module) -> float:
    """Count model parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def benchmark_forward_backward(
    model: nn.Module,
    x: torch.Tensor,
    num_warmup: int = 5,
    num_iterations: int = 20,
) -> tuple[float, float]:
    """Benchmark forward and backward pass times.

    Returns:
        Tuple of (forward_time_ms, backward_time_ms)
    """

    # Warmup
    for _ in range(num_warmup):
        output = model(x)
        if isinstance(output, tuple):
            output = output[0]
        output.sum().backward()
        model.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark forward
    forward_times = []
    for _ in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(x)
        if isinstance(output, tuple):
            output = output[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        forward_times.append((end - start) * 1000)

    # Benchmark backward
    backward_times = []
    for _ in range(num_iterations):
        output = model(x)
        if isinstance(output, tuple):
            output = output[0]
        loss = output.sum()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        backward_times.append((end - start) * 1000)
        model.zero_grad()

    return (
        sum(forward_times) / len(forward_times),
        sum(backward_times) / len(backward_times),
    )


def run_benchmark(
    name: str,
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    device: torch.device,
) -> BenchmarkResult:
    """Run a single benchmark."""
    # Reset memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)

    # Measure memory before
    mem_before = get_memory_usage()

    # Run benchmark
    forward_time, backward_time = benchmark_forward_backward(model, x)

    # Measure memory after
    mem_after = get_memory_usage()

    return BenchmarkResult(
        name=name,
        forward_time_ms=forward_time,
        backward_time_ms=backward_time,
        total_time_ms=forward_time + backward_time,
        memory_mb=mem_after - mem_before,
        params_m=count_parameters(model),
    )


def print_results(results: list[BenchmarkResult]):
    """Print benchmark results in a table."""
    print("\n" + "=" * 90)
    print(
        f"{'Name':<40} {'Fwd (ms)':<12} {'Bwd (ms)':<12} {'Total (ms)':<12} {'Mem (MB)':<12} {'Params (M)':<10}"
    )
    print("=" * 90)

    for r in results:
        print(
            f"{r.name:<40} {r.forward_time_ms:<12.2f} {r.backward_time_ms:<12.2f} "
            f"{r.total_time_ms:<12.2f} {r.memory_mb:<12.1f} {r.params_m:<10.2f}"
        )
    print("=" * 90)


def main():
    """Run all benchmarks."""
    # Configuration
    batch_size = 4
    seq_len = 512
    hidden_size = 1024
    intermediate_size = 4096

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = []

    # 1. Dense MLP baseline
    print("\nBenchmarking Dense MLP...")
    config = create_test_config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    config.model.moe.use_moe = False
    dense_model = MLP(config).to(device)
    results.append(
        run_benchmark(
            "Dense MLP",
            dense_model,
            batch_size,
            seq_len,
            hidden_size,
            device,
        )
    )

    # 2. MoE with varying expert counts
    for num_experts in [8, 16, 32, 64]:
        print(f"\nBenchmarking MoE (experts={num_experts})...")
        config = create_test_config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_routed_experts=num_experts,
            top_k=min(4, num_experts),
        )
        moe_model = MoEMLP(config).to(device)
        results.append(
            run_benchmark(
                f"MoE (E={num_experts}, K=4)",
                moe_model,
                batch_size,
                seq_len,
                hidden_size,
                device,
            )
        )

    # 3. MoE with varying top-k
    for top_k in [1, 2, 4, 6, 8]:
        print(f"\nBenchmarking MoE (top_k={top_k})...")
        config = create_test_config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_routed_experts=64,
            top_k=top_k,
        )
        moe_model = MoEMLP(config).to(device)
        results.append(
            run_benchmark(
                f"MoE (E=64, K={top_k})",
                moe_model,
                batch_size,
                seq_len,
                hidden_size,
                device,
            )
        )

    # Print results
    print_results(results)

    # Summary
    print("\n### Summary ###")
    dense_result = results[0]
    moe_results = results[1:]

    print(f"\nDense MLP: {dense_result.total_time_ms:.2f}ms, {dense_result.params_m:.2f}M params")
    print("\nMoE speedup vs Dense (higher is slower):")
    for r in moe_results[:4]:  # Expert count variations
        slowdown = r.total_time_ms / dense_result.total_time_ms
        print(f"  {r.name}: {slowdown:.2f}x")


if __name__ == "__main__":
    main()
