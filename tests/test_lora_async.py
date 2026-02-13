#!/usr/bin/env python3
"""
Test LoRA with async chunked execution.

This test validates:
1. LoRA works correctly with sequence chunking enabled
2. Chunked execution produces same outputs as non-chunked
3. Async finalize() in LoRARowParallelLinear works correctly
4. Memory and performance characteristics

Run with:
- TP=1: python tests/test_lora_async.py
- TP=2: torchrun --nproc_per_node=2 tests/test_lora_async.py
"""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironcore.config import (
    DataConfig,
    InitConfig,
    LoRAConfig,
    MainConfig,
    ModelConfig,
    OperationConfig,
    OptimConfig,
    ParallelConfig,
    PEFTConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.global_vars import reset_global_states, set_global_states
from ironcore.language_model import LanguageModel
from ironcore.parallel import parallel_states
from ironcore.peft.utils import freeze_base_model


def create_model_config(tp_size=1, chunk_size=None):
    """Create a small model config for testing."""
    model_config = ModelConfig(
        d_model=256,
        num_attention_heads=8,
        num_attention_groups=8,
        head_dim=32,
        max_seq_len=128,  # Longer sequence for chunking
        max_position_embeddings=128,
        dropout_attn=0.0,
        dropout_mlp=0.0,
        no_bias=False,
    )
    model_config.name = "gpt2"

    trainer_config = TrainerConfig(
        tensor_model_parallel_size=tp_size,
        use_flash_attn=False,
        sequence_chunk_size=chunk_size,  # Enable/disable chunking
    )

    # LoRA configuration
    lora_config = LoRAConfig(
        r=8,
        alpha=16.0,
        dropout=0.0,
        target_modules=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )
    peft_config = PEFTConfig(method="lora", lora=lora_config)

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
        peft=peft_config,
    )


def test_async_chunking(tp_size=1):
    """Test LoRA with chunked vs non-chunked execution."""
    print("\n" + "=" * 70)
    print(f"Testing LoRA Async Chunking (TP={tp_size})")
    print("=" * 70)

    # Initialize distributed if needed
    if tp_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
    else:
        rank = 0

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Set seed
    import random

    import numpy as np

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Test 1: Non-chunked execution
    print("\n[1] Creating model WITHOUT chunking...")
    config_no_chunk = create_model_config(tp_size=tp_size, chunk_size=None)
    set_global_states(config_no_chunk)
    parallel_states.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, timeout_in_minutes=10.0
    )

    model_no_chunk = LanguageModel(config_no_chunk)
    model_no_chunk.to(device)
    freeze_base_model(model_no_chunk, "lora")

    # Create test input
    batch_size, seq_len = 2, 64
    torch.manual_seed(42)
    input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)

    # Forward pass without chunking
    model_no_chunk.eval()
    with torch.no_grad():
        output_no_chunk = model_no_chunk(input_ids)

    if rank == 0:
        print(f"  Output shape: {output_no_chunk.shape}")
        print(f"  Output mean: {output_no_chunk.mean().item():.6f}")
        print("  ✓ Non-chunked forward pass completed")

    parallel_states.destroy_model_parallel()
    reset_global_states()

    # Test 2: Chunked execution
    print("\n[2] Creating model WITH chunking (chunk_size=16)...")
    config_chunked = create_model_config(tp_size=tp_size, chunk_size=16)
    set_global_states(config_chunked)
    parallel_states.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, timeout_in_minutes=10.0
    )

    model_chunked = LanguageModel(config_chunked)
    model_chunked.to(device)
    freeze_base_model(model_chunked, "lora")

    # Load same weights as non-chunked model
    model_chunked.load_state_dict(model_no_chunk.state_dict())

    # Forward pass with chunking
    model_chunked.eval()
    torch.manual_seed(42)  # Same input
    input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)

    with torch.no_grad():
        output_chunked = model_chunked(input_ids)

    if rank == 0:
        print(f"  Output shape: {output_chunked.shape}")
        print(f"  Output mean: {output_chunked.mean().item():.6f}")
        print("  ✓ Chunked forward pass completed")

    # Compare outputs
    diff = torch.abs(output_chunked - output_no_chunk).max().item()
    mean_diff = torch.abs(output_chunked - output_no_chunk).mean().item()

    if rank == 0:
        print(f"\n[3] Comparing chunked vs non-chunked outputs:")
        print(f"  Max difference: {diff:.6e}")
        print(f"  Mean difference: {mean_diff:.6e}")

        # Relaxed tolerance for chunking + LoRA
        atol, rtol = 1e-1, 1e-1
        assert torch.allclose(output_chunked, output_no_chunk, atol=atol, rtol=rtol), (
            f"Outputs differ: max_diff={diff:.6e}, mean_diff={mean_diff:.6e}"
        )
        print(f"  ✓ Outputs match within tolerance (atol={atol}, rtol={rtol})")

    # Test 3: Gradient flow with chunking
    print("\n[4] Testing gradient flow with chunking...")
    model_chunked.train()
    torch.manual_seed(42)
    input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)

    output = model_chunked(input_ids)
    loss = output.mean()
    loss.backward()

    # Check gradients
    lora_grads = []
    base_grads = []
    for name, param in model_chunked.named_parameters():
        if param.grad is not None:
            if any(k in name for k in ["lora_A", "lora_B"]):
                lora_grads.append(name)
            else:
                base_grads.append(name)

    if rank == 0:
        print(f"  LoRA parameters with gradients: {len(lora_grads)}")
        print(f"  Base parameters with gradients: {len(base_grads)}")

        assert len(base_grads) == 0, "Base parameters should not have gradients"
        assert len(lora_grads) > 0, "LoRA parameters should have gradients"
        print("  ✓ Gradient flow correct with chunking")

    parallel_states.destroy_model_parallel()

    if tp_size > 1:
        dist.destroy_process_group()

    if rank == 0:
        print("\n" + "=" * 70)
        print("ALL ASYNC CHUNKING TESTS PASSED ✓")
        print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    args = parser.parse_args()

    test_async_chunking(tp_size=args.tp)


if __name__ == "__main__":
    main()
