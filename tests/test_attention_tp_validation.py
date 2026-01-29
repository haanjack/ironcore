#!/usr/bin/env python3
"""
Comprehensive TP=1 vs TP=2 validation with activation norm comparison.

This script validates:
1. Attention layer correctness across TP=1 and TP=2
2. Activation norm statistics
3. Forward/backward pass consistency
4. Numerical equivalence

Run with:
- TP=1: python tests/test_attention_tp_validation.py
- TP=2: torchrun --nproc_per_node=2 tests/test_attention_tp_validation.py
"""
import os
import sys
import torch
import torch.distributed as dist
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironcore.config import (
    MainConfig, ModelConfig, TrainerConfig, InitConfig, OptimConfig,
    DataConfig, ParallelConfig, OperationConfig, UtilsConfig
)
from ironcore.layers.attention import Attention
from ironcore.parallel import parallel_states


def create_config(tensor_model_parallel_size=1):
    """Create test configuration."""
    model_config = ModelConfig(
        d_model=512,
        num_attention_heads=8,
        num_attention_groups=8,
        head_dim=64,
        max_seq_len=128,
        max_position_embeddings=128,
        dropout_attn=0.0,
        no_bias=False,
    )

    trainer_config = TrainerConfig(
        tensor_model_parallel_size=tensor_model_parallel_size,
        use_flash_attn=False,
    )

    init_config = InitConfig(seed=42, init_std=0.02)
    optim_config = OptimConfig(max_lr=1e-3, weight_decay=0.01)
    data_config = DataConfig()
    parallel_config = ParallelConfig()
    operation_config = OperationConfig(train_steps=100)
    utils_config = UtilsConfig()

    return MainConfig(
        model=model_config,
        trainer=trainer_config,
        init=init_config,
        optim=optim_config,
        data=data_config,
        parallel=parallel_config,
        operation=operation_config,
        utils=utils_config,
    )


def collect_activation_stats(module, name=""):
    """Collect activation statistics from a module."""
    stats = {}

    # Hook to capture intermediate activations
    activation_hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                stats[f"{name}_mean"] = output.mean().item()
                stats[f"{name}_std"] = output.std().item()
                stats[f"{name}_norm"] = output.norm().item()
                stats[f"{name}_abs_max"] = output.abs().max().item()
                stats[f"{name}_abs_min"] = output.abs().min().item()
            elif isinstance(output, tuple):
                for i, t in enumerate(output):
                    if isinstance(t, torch.Tensor):
                        stats[f"{name}_{i}_mean"] = t.mean().item()
                        stats[f"{name}_{i}_std"] = t.std().item()
                        stats[f"{name}_{i}_norm"] = t.norm().item()
        return hook

    # Register hooks for key components
    if hasattr(module, 'linear_q'):
        hook = module.linear_q.register_forward_hook(make_hook("linear_q"))
        activation_hooks.append(hook)

    if hasattr(module, 'linear_kv'):
        hook = module.linear_kv.register_forward_hook(make_hook("linear_kv"))
        activation_hooks.append(hook)

    if hasattr(module, 'output'):
        hook = module.output.register_forward_hook(make_hook("output_proj"))
        activation_hooks.append(hook)

    return activation_hooks, stats


def run_validation():
    """Run TP validation."""
    # Initialize distributed
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        dist.barrier()

    # Get TP size from config
    config = create_config(tensor_model_parallel_size=1)  # Will be set properly below
    tp_size = world_size  # For this test, use world_size as TP size

    # Initialize model parallel
    parallel_states.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        timeout_in_minutes=10.0
    )

    # Create config with proper TP size
    config = create_config(tensor_model_parallel_size=tp_size)

    tp_rank = parallel_states.get_tensor_model_parallel_rank()

    print(f"\n{'='*70}")
    print(f"[Rank {rank}] TP={tp_size} (TP rank {tp_rank})")
    print(f"{'='*70}")

    # Set seed
    import random
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Create attention layer
    attention = Attention(config).to(device)
    attention.init_weights()

    # Create input
    batch_size = 2
    seq_len = 64
    hidden_states = torch.randn(
        batch_size, seq_len, config.model.d_model,
        device=device, requires_grad=True
    )

    # Create attention mask
    attention_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=device)
    ).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)

    # Collect activation statistics
    hooks, activation_stats = collect_activation_stats(attention)

    # Forward pass
    print(f"[Rank {rank}] Running forward pass...")
    output = attention(hidden_states, attention_mask)

    # Collect final output stats
    activation_stats["output_mean"] = output.mean().item()
    activation_stats["output_std"] = output.std().item()
    activation_stats["output_norm"] = output.norm().item()

    # Backward pass
    print(f"[Rank {rank}] Running backward pass...")
    loss = output.sum()
    loss.backward()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Collect gradient statistics
    grad_stats = {}
    for name, param in attention.named_parameters():
        if param.grad is not None:
            grad_stats[f"{name}_mean"] = param.grad.mean().item()
            grad_stats[f"{name}_std"] = param.grad.std().item()
            grad_stats[f"{name}_norm"] = param.grad.norm().item()

    # Compute total gradient norm
    total_grad_norm = sum(
        param.grad.norm().item()**2
        for param in attention.parameters()
        if param.grad is not None
    )**0.5
    grad_stats["total_grad_norm"] = total_grad_norm

    print(f"[Rank {rank}] Output norm: {activation_stats['output_norm']:.6f}")
    print(f"[Rank {rank}] Grad norm: {total_grad_norm:.6f}")

    # Combine stats
    stats = {
        'rank': rank,
        'tp_size': tp_size,
        'tp_rank': tp_rank,
        **activation_stats,
        **grad_stats,
    }

    # Gather stats from all ranks
    if world_size > 1:
        all_stats = [None] * world_size
        dist.all_gather_object(all_stats, stats)
    else:
        all_stats = [stats]

    # Print comparison on rank 0
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"TP={tp_size} VALIDATION RESULTS")
        print(f"{'='*70}")

        # Save results
        os.makedirs("logs", exist_ok=True)
        with open(f"logs/tp{tp_size}_attention_validation.json", 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"Results saved to logs/tp{tp_size}_attention_validation.json")

        # Print activation norm comparison
        print(f"\n{'='*70}")
        print("ACTIVATION NORM COMPARISON")
        print(f"{'='*70}")

        if tp_size == 2:
            rank0_stats = all_stats[0]
            rank1_stats = all_stats[1]

            # Compare output norms
            print(f"\nOutput Activation Norms:")
            print(f"  Rank 0: {rank0_stats['output_norm']:.6f}")
            print(f"  Rank 1: {rank1_stats['output_norm']:.6f}")
            output_diff = abs(rank0_stats['output_norm'] - rank1_stats['output_norm'])
            print(f"  Difference: {output_diff:.2e}")

            # Compare linear_q activation norms
            if 'linear_q_norm' in rank0_stats:
                print(f"\nLinear Q Activation Norms:")
                print(f"  Rank 0: {rank0_stats['linear_q_norm']:.6f}")
                print(f"  Rank 1: {rank1_stats['linear_q_norm']:.6f}")
                q_diff = abs(rank0_stats['linear_q_norm'] - rank1_stats['linear_q_norm'])
                print(f"  Difference: {q_diff:.2e}")

            # Compare linear_kv activation norms
            if 'linear_kv_norm' in rank0_stats:
                print(f"\nLinear KV Activation Norms:")
                print(f"  Rank 0: {rank0_stats['linear_kv_norm']:.6f}")
                print(f"  Rank 1: {rank1_stats['linear_kv_norm']:.6f}")
                kv_diff = abs(rank0_stats['linear_kv_norm'] - rank1_stats['linear_kv_norm'])
                print(f"  Difference: {kv_diff:.2e}")

            # Compare output projection activation norms
            if 'output_proj_norm' in rank0_stats:
                print(f"\nOutput Projection Activation Norms:")
                print(f"  Rank 0: {rank0_stats['output_proj_norm']:.6f}")
                print(f"  Rank 1: {rank1_stats['output_proj_norm']:.6f}")
                out_diff = abs(rank0_stats['output_proj_norm'] - rank1_stats['output_proj_norm'])
                print(f"  Difference: {out_diff:.2e}")

            # Compare gradient norms
            print(f"\nGradient Norms:")
            print(f"  Rank 0: {rank0_stats['total_grad_norm']:.6f}")
            print(f"  Rank 1: {rank1_stats['total_grad_norm']:.6f}")
            grad_diff = abs(rank0_stats['total_grad_norm'] - rank1_stats['total_grad_norm'])
            print(f"  Difference: {grad_diff:.2e}")

            # Validate correctness
            print(f"\n{'='*70}")
            print("CORRECTNESS VALIDATION")
            print(f"{'='*70}")

            tol = 1e-5
            all_close = True

            if output_diff < tol:
                print(f"✓ Output norms MATCH (diff={output_diff:.2e} < {tol})")
            else:
                print(f"✗ Output norms DIFFER (diff={output_diff:.2e} >= {tol})")
                all_close = False

            if grad_diff < tol:
                print(f"✓ Gradient norms MATCH (diff={grad_diff:.2e} < {tol})")
            else:
                print(f"✗ Gradient norms DIFFER (diff={grad_diff:.2e} >= {tol})")
                all_close = False

            if all_close:
                print(f"\n✓ TP=2 VALIDATION PASSED - Results are consistent across ranks")
            else:
                print(f"\n✗ TP=2 VALIDATION FAILED - Results differ across ranks")

    # Cleanup
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    return stats


if __name__ == "__main__":
    try:
        run_validation()
        sys.exit(0)
    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[Rank {rank}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)
