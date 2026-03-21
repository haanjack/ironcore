#!/usr/bin/env python
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark comparing All-Reduce vs All-to-All communication modes.

This benchmark measures:
1. Forward pass latency
2. Backward pass latency
3. Memory usage
4. Throughput (tokens/second)

Usage:
    torchrun --nproc_per_node=2 tests/multi_gpu/benchmark_comm_modes.py
"""

import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

from ironcore.config import MainConfig, PEFTConfig
from ironcore.config.config_data import DataConfig
from ironcore.config.config_model import ModelConfig
from ironcore.config.config_moe import MoEConfig
from ironcore.config.config_optim import OptimConfig
from ironcore.config.config_parallel import ParallelConfig
from ironcore.config.config_trainer import InitConfig, OperationConfig, TrainerConfig
from ironcore.config.config_utils import UtilsConfig
from ironcore.layers.moe import CommunicationMode, MoEMLP
from ironcore.parallel.expert_parallel import (
    destroy_expert_parallel,
    initialize_expert_parallel,
)
from ironcore.parallel.parallel_states import (
    destroy_model_parallel,
    initialize_model_parallel,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    mode: str
    batch_size: int
    seq_len: int
    hidden_size: int
    num_experts: int
    top_k: int
    forward_ms: float
    backward_ms: float
    total_ms: float
    memory_mb: float
    tokens_per_sec: float


def setup_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    return rank, world_size


def cleanup_distributed():
    """Cleanup distributed environment."""
    destroy_expert_parallel()
    destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


def create_config(
    hidden_size: int,
    intermediate_size: int,
    num_shared_experts: int,
    num_routed_experts: int,
    top_k: int,
    ep_size: int,
) -> MainConfig:
    """Create MoE configuration.

    NOTE: A general-purpose MoE config factory is available at
    tests.fixtures.config_fixtures.create_moe_test_config. This local version
    is kept because it uses benchmark-specific parameters (ep_size, etc.).
    """
    return MainConfig(
        model=ModelConfig(
            d_model=hidden_size,
            d_ffn=intermediate_size,
            moe=MoEConfig(
                use_moe=True,
                num_shared_experts=num_shared_experts,
                num_routed_experts=num_routed_experts,
                num_experts_per_token=top_k,
                expert_model_parallel_size=ep_size,
                aux_loss_alpha=0.01,
            ),
        ),
        init=InitConfig(),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        trainer=TrainerConfig(tensor_model_parallel_size=1),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        peft=PEFTConfig(),
    )


def warmup(model, x, num_iterations=5):
    """Warmup CUDA kernels."""
    for _ in range(num_iterations):
        output = model(x)
        if isinstance(output, tuple):
            output = output[0]
        output.sum().backward()
    torch.cuda.synchronize()


def benchmark_forward_backward(
    model: MoEMLP,
    x: torch.Tensor,
    num_iterations: int = 20,
) -> tuple[float, float]:
    """Benchmark forward and backward pass.

    Returns:
        Tuple of (forward_ms, backward_ms)
    """

    # Warmup
    warmup(model, x)

    # Benchmark forward
    torch.cuda.synchronize()
    forward_times = []
    outputs = []

    for _ in range(num_iterations):
        start = time.perf_counter()
        output = model(x)
        if isinstance(output, tuple):
            output = output[0]
        torch.cuda.synchronize()
        end = time.perf_counter()
        forward_times.append((end - start) * 1000)
        outputs.append(output)

    forward_ms = sum(forward_times) / len(forward_times)

    # Benchmark backward
    torch.cuda.synchronize()
    backward_times = []

    for output in outputs:
        output.retain_grad()
        start = time.perf_counter()
        output.sum().backward()
        torch.cuda.synchronize()
        end = time.perf_counter()
        backward_times.append((end - start) * 1000)

    backward_ms = sum(backward_times) / len(backward_times)

    return forward_ms, backward_ms


def get_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def run_benchmark(
    mode: CommunicationMode,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    ep_size: int,
    num_iterations: int = 20,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    rank, _ = setup_distributed()

    # Initialize parallel groups
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )
    initialize_expert_parallel(
        expert_model_parallel_size=ep_size,
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )

    device = torch.device(f"cuda:{rank}")

    # Create config and model
    config = create_config(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_shared_experts=2,
        num_routed_experts=num_experts,
        top_k=top_k,
        ep_size=ep_size,
    )

    model = MoEMLP(config, communication_mode=mode).to(device)
    model.init_weights()
    model.train()

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)

    # Reset memory tracking
    torch.cuda.reset_peak_memory_stats()

    # Run benchmark
    forward_ms, backward_ms = benchmark_forward_backward(model, x, num_iterations)

    # Get memory usage
    memory_mb = get_memory_usage()

    # Calculate throughput
    total_tokens = batch_size * seq_len
    total_ms = forward_ms + backward_ms
    tokens_per_sec = (total_tokens / total_ms) * 1000

    # Cleanup
    destroy_expert_parallel()
    destroy_model_parallel()

    return BenchmarkResult(
        mode=mode.value,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        forward_ms=forward_ms,
        backward_ms=backward_ms,
        total_ms=total_ms,
        memory_mb=memory_mb,
        tokens_per_sec=tokens_per_sec,
    )


def print_results(results: list[BenchmarkResult]):
    """Print benchmark results in a table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS: All-Reduce vs All-to-All Communication")
    print("=" * 100)

    # Header
    print(
        f"\n{'Mode':<12} {'Batch':<8} {'Seq':<6} {'Hidden':<8} {'Experts':<8} "
        f"{'Top-K':<6} {'Fwd(ms)':<10} {'Bwd(ms)':<10} {'Total(ms)':<10} "
        f"{'Memory(MB)':<12} {'Tokens/s':<12}"
    )
    print("-" * 100)

    for r in results:
        print(
            f"{r.mode:<12} {r.batch_size:<8} {r.seq_len:<6} {r.hidden_size:<8} "
            f"{r.num_experts:<8} {r.top_k:<6} {r.forward_ms:<10.2f} {r.backward_ms:<10.2f} "
            f"{r.total_ms:<10.2f} {r.memory_mb:<12.1f} {r.tokens_per_sec:<12.0f}"
        )

    # Summary comparison
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)

    # Group by configuration
    configs = {}
    for r in results:
        key = (r.batch_size, r.seq_len, r.hidden_size, r.num_experts, r.top_k)
        if key not in configs:
            configs[key] = {}
        configs[key][r.mode] = r

    print(
        f"\n{'Config':<40} {'All-Reduce(ms)':<15} {'All-to-All(ms)':<15} {'Speedup':<10} {'Winner'}"
    )
    print("-" * 100)

    for key, modes in configs.items():
        if "all_reduce" in modes and "all_to_all" in modes:
            ar = modes["all_reduce"]
            a2a = modes["all_to_all"]
            config_str = f"b={key[0]}, s={key[1]}, h={key[2]}, e={key[3]}, k={key[4]}"
            speedup = ar.total_ms / a2a.total_ms
            winner = "All-to-All" if speedup > 1.0 else "All-Reduce"
            print(
                f"{config_str:<40} {ar.total_ms:<15.2f} {a2a.total_ms:<15.2f} "
                f"{speedup:<10.2f}x {winner}"
            )


def main():
    """Run all benchmarks."""
    print("=" * 100)
    print("MoE Communication Mode Benchmark")
    print("=" * 100)
    print("\nComparing All-Reduce vs All-to-All communication for Expert Parallelism")
    print("All-Reduce: Each rank processes local tokens, then all-reduce combines outputs")
    print("All-to-All: Tokens dispatched to expert ranks via all-to-all, processed, then gathered")

    # Configuration for benchmarks
    ep_size = 2  # Expert parallel size

    # Test configurations: (batch_size, seq_len, hidden_size, num_experts, top_k)
    # Simplified to fewer configs for faster benchmarking
    configs = [
        # Small config
        (2, 128, 256, 8, 2),
        # Medium config
        (4, 256, 512, 16, 4),
        # Larger config
        (8, 256, 512, 32, 4),
    ]

    results = []

    for config in configs:
        batch_size, seq_len, hidden_size, num_experts, top_k = config

        print(
            f"\nBenchmarking: batch={batch_size}, seq={seq_len}, hidden={hidden_size}, "
            f"experts={num_experts}, top_k={top_k}"
        )

        # Test All-Reduce mode
        print("  Running All-Reduce mode...")
        ar_result = run_benchmark(
            mode=CommunicationMode.ALL_REDUCE,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            ep_size=ep_size,
        )
        results.append(ar_result)

        # Test All-to-All mode
        print("  Running All-to-All mode...")
        a2a_result = run_benchmark(
            mode=CommunicationMode.ALL_TO_ALL,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            ep_size=ep_size,
        )
        results.append(a2a_result)

    # Print final results
    if dist.get_rank() == 0:
        print_results(results)

    cleanup_distributed()


if __name__ == "__main__":
    main()
