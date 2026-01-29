#!/usr/bin/env python3
"""
Corrected comprehensive benchmarking with proper warmup.

Key fixes:
1. Warmup both forward AND backward passes
2. Run multiple iterations and take median
3. Time forward+backward together to avoid timing artifacts
"""
import os
import sys
import time
import json
import torch
import torch.distributed as dist
from dataclasses import dataclass, asdict
from typing import Optional, List
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironcore.config import (
    MainConfig, ModelConfig, TrainerConfig, InitConfig, OptimConfig,
    DataConfig, ParallelConfig, OperationConfig, UtilsConfig
)
from ironcore.layers.attention import Attention
from ironcore.layers.positional_embedding.rotary import RotaryPositionalEmbedding
from ironcore.parallel import parallel_states


@dataclass
class BenchmarkResult:
    """Store benchmark results."""
    tp_size: int
    attention_type: str
    position_embedding: str
    algorithm: str
    num_heads: int
    num_groups: int
    seq_len: int
    d_model: int

    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    peak_memory_mb: float
    parameters: int
    output_norm: float


def create_config(d_model=512, num_heads=8, num_groups=8, head_dim=64,
                  seq_len=128, tp_size=1, use_flash_attn=False, dropout=0.0):
    """Create configuration."""
    model_config = ModelConfig(
        d_model=d_model, num_attention_heads=num_heads, num_attention_groups=num_groups,
        head_dim=head_dim, max_seq_len=seq_len, max_position_embeddings=seq_len,
        dropout_attn=dropout, no_bias=False,
    )
    trainer_config = TrainerConfig(tensor_model_parallel_size=tp_size, use_flash_attn=use_flash_attn)
    init_config = InitConfig(seed=42, init_std=0.02)
    optim_config = OptimConfig(max_lr=1e-3, weight_decay=0.01)
    data_config = DataConfig()
    parallel_config = ParallelConfig()
    operation_config = OperationConfig(train_steps=100)
    utils_config = UtilsConfig()

    return MainConfig(
        model=model_config, trainer=trainer_config, init=init_config,
        optim=optim_config, data=data_config, parallel=parallel_config,
        operation=operation_config, utils=utils_config,
    )


def run_benchmark(config, batch_size=2, seq_len=128, use_rope=False,
                  device=torch.device('cuda:0'), num_iterations=10):
    """Run benchmark with proper warmup and multiple iterations."""

    attention_type = "MHA" if config.model.num_attention_heads == config.model.num_attention_groups else (
        "MQA" if config.model.num_attention_groups == 1 else "GQA"
    )
    position_embedding = "RoPE" if use_rope else "None"
    algorithm = "Flash" if config.trainer.use_flash_attn else "Standard"
    tp_size = config.trainer.tensor_model_parallel_size

    # Initialize model parallel if needed
    try:
        parallel_states.get_tensor_model_parallel_world_size()
    except AssertionError:
        parallel_states.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, timeout_in_minutes=10.0
        )

    # Set seed
    import random
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Create attention layer
    attention = Attention(config).to(device)
    attention.init_weights()

    # Create rotary embedding if needed
    rotary_pos_emb = None
    if use_rope:
        head_dim = config.model.d_model // config.model.num_attention_heads
        rotary_pos_emb = RotaryPositionalEmbedding(
            head_dim=head_dim, max_seq_len=seq_len, base=10000,
        ).to(device)

    # Create input
    def create_input():
        hidden_states = torch.randn(
            batch_size, seq_len, config.model.d_model,
            device=device, requires_grad=True
        )
        attention_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device)
        ).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        return hidden_states, attention_mask

    # Warmup - run BOTH forward and backward multiple times
    print(f"  Warming up...", end='', flush=True)
    for _ in range(5):
        hidden_states, attention_mask = create_input()
        if use_rope:
            output = attention(hidden_states, attention_mask, rotary_pos_emb)
        else:
            output = attention(hidden_states, attention_mask)
        output.sum().backward()
        # Clear gradients
        attention.zero_grad()
    torch.cuda.synchronize()
    print(" done", flush=True)

    # Benchmark iterations
    forward_times = []
    backward_times = []
    total_times = []

    for i in range(num_iterations):
        hidden_states, attention_mask = create_input()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Measure forward
        start = time.time()
        if use_rope:
            output = attention(hidden_states, attention_mask, rotary_pos_emb)
        else:
            output = attention(hidden_states, attention_mask)
        torch.cuda.synchronize()
        forward_time = (time.time() - start) * 1000

        # Measure backward
        start = time.time()
        output.sum().backward()
        torch.cuda.synchronize()
        backward_time = (time.time() - start) * 1000

        total_time = forward_time + backward_time
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2

        forward_times.append(forward_time)
        backward_times.append(backward_time)
        total_times.append(total_time)

        # Clear gradients for next iteration
        attention.zero_grad()

        if (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}/{num_iterations}: Fwd={forward_time:.2f}ms, "
                  f"Bwd={backward_time:.2f}ms, Total={total_time:.2f}ms", flush=True)

    # Use median to filter outliers
    forward_time_ms = float(np.median(forward_times))
    backward_time_ms = float(np.median(backward_times))
    total_time_ms = float(np.median(total_times))
    peak_memory_mb = peak_memory

    parameters = sum(p.numel() for p in attention.parameters())
    output_norm = output.norm().item()

    print(f"  Median: Fwd={forward_time_ms:.2f}ms, Bwd={backward_time_ms:.2f}ms, "
          f"Total={total_time_ms:.2f}ms, Mem={peak_memory_mb:.2f}MB", flush=True)

    return BenchmarkResult(
        tp_size=tp_size,
        attention_type=attention_type,
        position_embedding=position_embedding,
        algorithm=algorithm,
        num_heads=config.model.num_attention_heads,
        num_groups=config.model.num_attention_groups,
        seq_len=seq_len,
        d_model=config.model.d_model,
        forward_time_ms=forward_time_ms,
        backward_time_ms=backward_time_ms,
        total_time_ms=total_time_ms,
        peak_memory_mb=peak_memory_mb,
        parameters=parameters,
        output_norm=output_norm,
    )


def main():
    """Run corrected benchmarks."""

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print("CORRECTED ATTENTION BENCHMARK - With Proper Warmup")
    print("="*80)

    results = []

    # Test configurations
    test_configs = [
        # (d_model, num_heads, num_groups, seq_len, use_flash, use_rope, name)
        (512, 8, 8, 128, False, False, "MHA-Standard"),
        (512, 8, 8, 128, True, False, "MHA-Flash"),
        (512, 8, 4, 128, False, False, "GQA-Standard"),
        (512, 8, 4, 128, True, False, "GQA-Flash"),
        (512, 8, 1, 128, False, False, "MQA-Standard"),
        (512, 8, 1, 128, True, False, "MQA-Flash"),
        (512, 8, 8, 256, False, False, "MHA-Standard-seq256"),
        (512, 8, 8, 256, True, False, "MHA-Flash-seq256"),
    ]

    for i, (d_model, num_heads, num_groups, seq_len, use_flash, use_rope, name) in enumerate(test_configs):
        print(f"\n[{i+1}/{len(test_configs)}] Testing: {name}")
        print(f"  d_model={d_model}, heads={num_heads}, groups={num_groups}, seq_len={seq_len}")

        config = create_config(
            d_model=d_model, num_heads=num_heads, num_groups=num_groups,
            seq_len=seq_len, use_flash_attn=use_flash,
        )

        try:
            result = run_benchmark(
                config=config, seq_len=seq_len, use_rope=use_rope,
                device=device, num_iterations=10,
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Print results table
    print("\n" + "="*120)
    print("CORRECTED RESULTS")
    print("="*120)
    print(f"{'Config':<25} {'Params':<12} {'Fwd(ms)':<10} {'Bwd(ms)':<10} {'Total(ms)':<11} {'Mem(MB)':<10}")
    print("-"*120)

    for r in results:
        print(f"{r.attention_type + '-' + r.algorithm:<25} {r.parameters:<12,} "
              f"{r.forward_time_ms:<10.2f} {r.backward_time_ms:<10.2f} "
              f"{r.total_time_ms:<11.2f} {r.peak_memory_mb:<10.2f}")

    print("="*120)

    # Save results
    os.makedirs("logs", exist_ok=True)
    output_file = "logs/attention_benchmark_corrected.json"
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print comparison
    print("\n" + "="*120)
    print("COMPARISON: Standard vs Flash Attention")
    print("="*120)

    for attn_type in ["MHA", "GQA", "MQA"]:
        type_results = [r for r in results if r.attention_type == attn_type and r.seq_len == 128]
        if len(type_results) >= 2:
            std = next(r for r in type_results if r.algorithm == "Standard")
            flash = next(r for r in type_results if r.algorithm == "Flash")

            fwd_speedup = std.forward_time_ms / flash.forward_time_ms
            bwd_speedup = std.backward_time_ms / flash.backward_time_ms
            total_speedup = std.total_time_ms / flash.total_time_ms

            print(f"\n{attn_type}:")
            print(f"  Forward:  {std.forward_time_ms:.2f}ms -> {flash.forward_time_ms:.2f}ms "
                  f"({fwd_speedup:.2f}x)")
            print(f"  Backward: {std.backward_time_ms:.2f}ms -> {flash.backward_time_ms:.2f}ms "
                  f"({bwd_speedup:.2f}x)")
            print(f"  Total:    {std.total_time_ms:.2f}ms -> {flash.total_time_ms:.2f}ms "
                  f"({total_speedup:.2f}x)")


if __name__ == "__main__":
    main()
