#!/usr/bin/env python3
"""
Simplified TP=1 vs TP=2 validation with activation norm comparison.
Run with:
- TP=1: python tests/test_attention_tp_simple.py
- TP=2: torchrun --nproc_per_node=2 tests/test_attention_tp_simple.py
"""
import os
import sys
import torch
import torch.distributed as dist

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


def run_validation():
    """Run TP validation."""
    # Get rank info
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Initialize distributed for multi-GPU
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    tp_size = world_size

    # Initialize model parallel
    parallel_states.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        timeout_in_minutes=10.0
    )

    config = create_config(tensor_model_parallel_size=tp_size)
    tp_rank = parallel_states.get_tensor_model_parallel_rank()

    print(f"\n[Rank {rank}] TP={tp_size} (TP rank {tp_rank})")

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

    # Forward pass
    output = attention(hidden_states, attention_mask)
    output_norm = output.norm().item()
    output_mean = output.mean().item()
    output_std = output.std().item()

    print(f"[Rank {rank}] Output norm: {output_norm:.6f}")

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Compute total gradient norm
    total_grad_norm = sum(
        param.grad.norm().item()**2
        for param in attention.parameters()
        if param.grad is not None
    )**0.5

    print(f"[Rank {rank}] Grad norm: {total_grad_norm:.6f}")

    # For TP=2, gather key metrics for comparison
    if world_size > 1:
        # Create tensors for all_gather
        output_norm_tensor = torch.tensor([output_norm], device=device)
        grad_norm_tensor = torch.tensor([total_grad_norm], device=device)

        gathered_output_norms = [torch.zeros_like(output_norm_tensor) for _ in range(world_size)]
        gathered_grad_norms = [torch.zeros_like(grad_norm_tensor) for _ in range(world_size)]

        dist.all_gather(gathered_output_norms, output_norm_tensor)
        dist.all_gather(gathered_grad_norms, grad_norm_tensor)

        if rank == 0:
            print(f"\n{'='*70}")
            print(f"TP={tp_size} VALIDATION RESULTS")
            print(f"{'='*70}")

            print(f"\nOutput Norms:")
            for i in range(world_size):
                print(f"  Rank {i}: {gathered_output_norms[i].item():.6f}")

            output_diff = abs(gathered_output_norms[0].item() - gathered_output_norms[1].item())
            print(f"  Difference: {output_diff:.2e}")

            print(f"\nGradient Norms:")
            for i in range(world_size):
                print(f"  Rank {i}: {gathered_grad_norms[i].item():.6f}")

            grad_diff = abs(gathered_grad_norms[0].item() - gathered_grad_norms[1].item())
            print(f"  Difference: {grad_diff:.2e}")

            # Validate
            tol = 1e-4
            if output_diff < tol and grad_diff < tol:
                print(f"\n✓ TP={tp_size} VALIDATION PASSED (differences < {tol})")
            else:
                print(f"\n✗ TP={tp_size} VALIDATION FAILED (differences >= {tol})")

        dist.barrier()
        dist.destroy_process_group()

    return 0


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
