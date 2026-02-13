# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""Shared utilities for LoRA tests to reduce code duplication."""

import os
import random
import sys

import numpy as np
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
from ironcore.parallel import parallel_states


def create_lora_test_config(
    tp_size: int = 1,
    enable_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_target_modules: list[str] | None = None,
    max_seq_len: int = 64,
    d_model: int = 256,
    num_attention_heads: int = 8,
    chunk_size: int | None = None,
    model_path: str | None = None,
    untie_embed: bool = False,
    xavier_init: bool = True,
) -> MainConfig:
    """
    Create a standard test configuration for LoRA tests.

    Args:
        tp_size: Tensor parallel size
        enable_lora: Whether to enable LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling parameter
        lora_target_modules: List of target modules for LoRA
        max_seq_len: Maximum sequence length
        d_model: Model hidden dimension
        num_attention_heads: Number of attention heads
        chunk_size: Sequence chunk size (None to disable chunking)
        model_path: Path for model checkpoints
        untie_embed: Whether to untie embedding weights
        xavier_init: Whether to use Xavier initialization

    Returns:
        MainConfig instance
    """
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"]

    model_config = ModelConfig(
        d_model=d_model,
        num_attention_heads=num_attention_heads,
        num_attention_groups=num_attention_heads,
        head_dim=d_model // num_attention_heads,
        max_seq_len=max_seq_len,
        max_position_embeddings=max_seq_len,
        dropout_attn=0.0,
        dropout_mlp=0.0,
        dropout_embd=0.0,
        no_bias=False,
        untie_embed=untie_embed,
    )
    trainer_config = TrainerConfig(
        tensor_model_parallel_size=tp_size,
        use_flash_attn=False,
        vocab_padding_unit=256,
        sequence_chunk_size=chunk_size,
    )
    if model_path:
        trainer_config.model_path = model_path

    # LoRA configuration
    if enable_lora:
        lora_config = LoRAConfig(
            r=lora_r,
            alpha=lora_alpha,
            dropout=0.0,
            target_modules=lora_target_modules,
        )
        peft_config = PEFTConfig(method="lora", lora=lora_config)
    else:
        peft_config = PEFTConfig(method="none")

    init_config = InitConfig(seed=42, init_std=0.02, xavier_init=xavier_init)
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


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> tuple[int, int, int]:
    """
    Count trainable, total, and LoRA parameters.

    Args:
        model: The model to analyze

    Returns:
        Tuple of (trainable_params, total_params, lora_params)
    """
    trainable = 0
    total = 0
    lora = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total += num_params

        if param.requires_grad:
            trainable += num_params

            if any(k in name for k in ["lora_A", "lora_B"]):
                lora += num_params

    return trainable, total, lora


def check_gradient_flow(model) -> tuple[list[str], list[str]]:
    """
    Check which parameters have gradients.

    Args:
        model: The model to check

    Returns:
        Tuple of (base_params_with_grads, lora_params_with_grads)
    """
    base_grads = []
    lora_grads = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            if any(k in name for k in ["lora_A", "lora_B"]):
                lora_grads.append(name)
            else:
                base_grads.append(name)

    return base_grads, lora_grads


def print_model_structure(model, layer_idx: int = 0):
    """Print the structure of a specific layer for debugging."""
    layer = model.model.layers[layer_idx]
    print(f"\nModel structure (layers[{layer_idx}]):")
    print(f"  linear_q: {type(layer.linear_q)}")
    print(f"  linear_kv: {type(layer.linear_kv)}")
    print(f"  attn_output: {type(layer.attn_output)}")
    print(f"  mlp.up_proj: {type(layer.mlp.up_proj)}")
    print(f"  mlp.down_proj: {type(layer.mlp.down_proj)}")


def print_parameter_stats(model):
    """Print parameter counts and ratios."""
    trainable, total, lora = count_parameters(model)
    print("\nParameter counts:")
    print(f"  Trainable: {trainable:,}")
    print(f"  Total: {total:,}")
    print(f"  LoRA: {lora:,}")
    print(f"  Trainable ratio: {100.0 * trainable / total:.2f}%")
    return trainable, total, lora


def init_parallel(tp_size: int = 1, timeout_minutes: float = 10.0):
    """Initialize parallel states."""
    parallel_states.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        timeout_in_minutes=timeout_minutes,
    )


def cleanup_parallel(tp_size: int = 1):
    """Clean up parallel states and process groups."""
    parallel_states.destroy_model_parallel()
    if tp_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def get_device(rank: int = 0) -> torch.device:
    """Get the appropriate device for the given rank."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def create_test_input(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    seed: int = 42,
) -> torch.Tensor:
    """Create a test input tensor with the specified seed."""
    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


def assert_gradient_correctness(
    model,
    expect_lora_grads: bool = True,
    expect_base_grads: bool = False,
):
    """
    Assert gradient flow is correct.

    Args:
        model: The model to check
        expect_lora_grads: Whether LoRA parameters should have gradients
        expect_base_grads: Whether base parameters should have gradients

    Raises:
        AssertionError: If gradient flow doesn't match expectations
    """
    base_grads, lora_grads = check_gradient_flow(model)

    if expect_lora_grads and not expect_base_grads:
        assert len(base_grads) == 0, (
            f"Base parameters should not have gradients, but {len(base_grads)} do: {base_grads[:3]}..."
        )
        assert len(lora_grads) > 0, "LoRA parameters should have gradients"
    elif expect_base_grads and not expect_lora_grads:
        assert len(lora_grads) == 0, "LoRA parameters should not have gradients"
        assert len(base_grads) > 0, "Base parameters should have gradients"


def compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    name: str = "tensors",
    atol: float = 1e-2,
    rtol: float = 1e-2,
    verbose: bool = True,
) -> bool:
    """
    Compare two tensors and optionally print statistics.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        name: Name for logging
        atol: Absolute tolerance
        rtol: Relative tolerance
        verbose: Whether to print comparison details

    Returns:
        True if tensors match within tolerance
    """
    diff = torch.abs(tensor1 - tensor2)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    matches = torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol)

    if verbose:
        print(f"\n{name} comparison:")
        print(f"  Max difference: {max_diff:.6e}")
        print(f"  Mean difference: {mean_diff:.6e}")
        print(f"  Matches (atol={atol}, rtol={rtol}): {matches}")

    return matches


def run_training_step(
    model,
    optimizer,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """
    Run a single training step.

    Args:
        model: The model to train
        optimizer: The optimizer
        input_ids: Input token IDs

    Returns:
        Tuple of (output tensor, loss value)
    """
    model.train()
    optimizer.zero_grad()

    output = model(input_ids)
    loss = output.mean()
    loss.backward()

    optimizer.step()

    return output.detach(), loss.item()


def get_lora_parameters(model) -> dict[str, torch.Tensor]:
    """Get all LoRA parameters from a model."""
    return {name: param.clone() for name, param in model.named_parameters() if "lora_" in name}


def get_base_parameters(model) -> dict[str, torch.Tensor]:
    """Get all base (non-LoRA) parameters from a model."""
    return {name: param.clone() for name, param in model.named_parameters() if "lora_" not in name}


def compare_lora_parameters(
    params1: dict[str, torch.Tensor],
    params2: dict[str, torch.Tensor],
    atol: float = 1e-6,
    verbose: bool = True,
) -> tuple[bool, float]:
    """
    Compare LoRA parameters between two models.

    Args:
        params1: First set of LoRA parameters
        params2: Second set of LoRA parameters
        atol: Absolute tolerance
        verbose: Whether to print details

    Returns:
        Tuple of (all_match, max_difference)
    """
    max_diff = 0.0
    all_match = True

    for name, value in params1.items():
        if name not in params2:
            if verbose:
                print(f"  Missing in params2: {name}")
            all_match = False
            continue

        diff = torch.abs(value - params2[name]).max().item()
        max_diff = max(max_diff, diff)

        if diff > atol:
            if verbose:
                print(f"  {name}: diff={diff:.6e}")
            all_match = False

    if verbose:
        print(f"  Max difference: {max_diff:.6e}")

    return all_match, max_diff
