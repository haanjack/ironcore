# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for MoE layers."""

import torch


def flatten_moe_inputs(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Flatten batch and sequence dimensions for MoE processing.

    Converts 3D tensors [batch, seq, hidden] to 2D [num_tokens, hidden]
    for efficient expert processing.

    Args:
        x: [batch, seq, hidden_size] input tensor
        topk_weights: [batch, seq, top_k] routing weights
        topk_indices: [batch, seq, top_k] expert indices

    Returns:
        Tuple containing:
            x_flat: [num_tokens, hidden_size] flattened input
            weights_flat: [num_tokens, top_k] flattened weights
            indices_flat: [num_tokens, top_k] flattened indices
            num_tokens: total number of tokens (batch * seq)
            hidden_size: hidden dimension size
    """
    batch_size, seq_len, hidden_size = x.shape
    num_tokens = batch_size * seq_len
    top_k = topk_indices.shape[-1]

    return (
        x.view(num_tokens, hidden_size),
        topk_weights.view(num_tokens, top_k),
        topk_indices.view(num_tokens, top_k),
        num_tokens,
        hidden_size,
    )


def validate_moe_input(
    x: torch.Tensor,
    expected_hidden_size: int,
    name: str = "MoE",
) -> None:
    """Validate input tensor for MoE forward pass.

    Args:
        x: Input tensor to validate
        expected_hidden_size: Expected hidden dimension size
        name: Name to use in error messages

    Raises:
        ValueError: If input is invalid
    """
    if x.ndim != 3:
        raise ValueError(f"{name} expected 3D input [batch, seq, hidden], got {x.ndim}D tensor")

    if x.shape[2] != expected_hidden_size:
        raise ValueError(f"{name} expected hidden_size={expected_hidden_size}, got {x.shape[2]}")

    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(
            f"{name} batch size and sequence length must be positive, "
            f"got batch={x.shape[0]}, seq={x.shape[1]}"
        )

    if torch.isnan(x).any():
        nan_count = torch.isnan(x).sum().item()
        total_count = x.numel()
        raise ValueError(
            f"{name} input contains NaN values: {nan_count}/{total_count} elements are NaN "
            f"(shape={x.shape}, dtype={x.dtype}). "
            f"This typically indicates numerical instability in attention layer. "
            f"Try: 1) reducing learning rate, 2) using smaller weight init, "
            f"3) gradient clipping."
        )

    if torch.isinf(x).any():
        inf_count = torch.isinf(x).sum().item()
        total_count = x.numel()
        raise ValueError(
            f"{name} input contains Inf values: {inf_count}/{total_count} elements are Inf "
            f"(shape={x.shape}, dtype={x.dtype}). "
            f"This indicates gradient explosion. Try gradient clipping or lower learning rate."
        )
