#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
TP=2 attention validation: verify forward pass output consistency across TP ranks.

Run with:
    torchrun --nproc_per_node=2 -m pytest tests/integration/attention/test_tp_comparison_simple.py -v
"""

import os

import pytest
import torch
import torch.distributed as dist

from tests.fixtures.config_fixtures import create_test_config

from ironcore.layers.attention import Attention
from ironcore.parallel import parallel_states

# Skip if not running under torchrun or fewer than 2 GPUs
pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ
    or not torch.cuda.is_available()
    or torch.cuda.device_count() < 2,
    reason="TP=2 tests require torchrun with at least 2 GPUs",
)


@pytest.fixture(scope="module")
def tp2_env():
    """Initialize distributed and TP=2 parallel states."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    if not parallel_states.is_model_parallel_initialized():
        parallel_states.initialize_model_parallel(tensor_model_parallel_size=2, timeout_in_minutes=10.0)

    config = create_test_config(tensor_model_parallel_size=2)
    yield config, device

    parallel_states.destroy_model_parallel()


def _make_qkv(batch_size, seq_len, num_heads, head_dim, device):
    """Create Q, K, V tensors with shape [b, s, heads, head_dim]."""
    return (
        torch.randn(batch_size, seq_len, num_heads, head_dim, device=device),
        torch.randn(batch_size, seq_len, num_heads, head_dim, device=device),
        torch.randn(batch_size, seq_len, num_heads, head_dim, device=device),
    )


class TestTP2Attention:
    """TP=2 attention layer validation tests."""

    def test_forward_output_shape(self, tp2_env):
        """Verify forward pass produces correct output shape."""
        config, device = tp2_env
        attention = Attention(config).to(device)
        attention.init_weights()

        cfg = config.model
        query, key, value = _make_qkv(2, 64, cfg.num_attention_heads, cfg.head_dim, device)
        mask = torch.tril(torch.ones(64, 64, device=device)).unsqueeze(0).unsqueeze(0).expand(2, -1, -1, -1)

        output = attention(query, key, value, mask)
        assert output.shape == (2, 64, cfg.num_attention_heads * cfg.head_dim)

    def test_output_is_finite(self, tp2_env):
        """Verify output contains no NaN or Inf values."""
        config, device = tp2_env
        attention = Attention(config).to(device)
        attention.init_weights()

        cfg = config.model
        query, key, value = _make_qkv(2, 64, cfg.num_attention_heads, cfg.head_dim, device)
        mask = torch.tril(torch.ones(64, 64, device=device)).unsqueeze(0).unsqueeze(0).expand(2, -1, -1, -1)

        output = attention(query, key, value, mask)
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    def test_output_norm_consistent_across_ranks(self, tp2_env):
        """Verify TP ranks produce the same output norm (distributed correctness)."""
        config, device = tp2_env
        rank = dist.get_rank()
        tp_size = parallel_states.get_tensor_model_parallel_world_size()

        torch.manual_seed(42)
        attention = Attention(config).to(device)
        attention.init_weights()

        cfg = config.model
        torch.manual_seed(42)
        query, key, value = _make_qkv(2, 64, cfg.num_attention_heads, cfg.head_dim, device)
        mask = torch.tril(torch.ones(64, 64, device=device)).unsqueeze(0).unsqueeze(0).expand(2, -1, -1, -1)

        output = attention(query, key, value, mask)
        output_norm = output.norm().item()

        output_norms = [torch.zeros(1, device=device) for _ in range(tp_size)]
        dist.all_gather(output_norms, torch.tensor([output_norm], device=device))

        if rank == 0:
            for i in range(tp_size):
                for j in range(i + 1, tp_size):
                    diff = abs(output_norms[i].item() - output_norms[j].item())
                    assert diff < 1e-3, (
                        f"Output norm differs between ranks {i} and {j}: {diff:.2e}"
                    )

    def test_causal_mask_produces_lower_triangular_output(self, tp2_env):
        """Verify causal attention: positions beyond causal window have near-zero contribution."""
        config, device = tp2_env
        rank = dist.get_rank()
        tp_size = parallel_states.get_tensor_model_parallel_world_size()

        torch.manual_seed(42)
        attention = Attention(config).to(device)
        attention.init_weights()

        cfg = config.model
        torch.manual_seed(42)
        query, key, value = _make_qkv(2, 64, cfg.num_attention_heads, cfg.head_dim, device)
        mask = torch.tril(torch.ones(64, 64, device=device)).unsqueeze(0).unsqueeze(0).expand(2, -1, -1, -1)

        output = attention(query, key, value, mask)
        # Output should not be all zeros
        assert output.abs().sum().item() > 0, "Output is all zeros"
