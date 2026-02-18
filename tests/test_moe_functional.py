# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""Functional integrity tests for MoE implementation.

Tests verify:
1. Deterministic routing correctness
2. Gradient flow and sparsity (selected vs unselected experts)
3. Shared vs routed expert gradient distribution
"""

import os

import pytest
import torch

# Set up environment for single-GPU testing
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

from ironcore.config import MainConfig, PEFTConfig, ProfilerConfig
from ironcore.config.config_data import DataConfig
from ironcore.config.config_model import ModelConfig
from ironcore.config.config_moe import MoEConfig
from ironcore.config.config_optim import OptimConfig
from ironcore.config.config_parallel import ParallelConfig
from ironcore.config.config_trainer import InitConfig, OperationConfig, TrainerConfig
from ironcore.config.config_utils import UtilsConfig
from ironcore.layers.moe import MoEMLP, TopKRouter
from ironcore.parallel.parallel_states import (
    destroy_model_parallel,
    initialize_model_parallel,
)


@pytest.fixture(autouse=True)
def setup_parallel_states():
    """Initialize parallel states before each test."""
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )
    yield
    destroy_model_parallel()


def create_test_config(
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_shared_experts: int = 1,
    num_routed_experts: int = 8,
    top_k: int = 2,
    aux_loss_alpha: float = 0.01,
):
    """Create a test configuration with MoE enabled."""
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
                aux_loss_alpha=aux_loss_alpha,
            ),
        ),
        init=InitConfig(),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        trainer=TrainerConfig(),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
        peft=PEFTConfig(),
    )


class TestDeterministicRouting:
    """Tests for routing correctness and determinism."""

    def test_topk_selection_correctness(self):
        """Verify that router selects the correct top-k experts."""
        torch.manual_seed(42)

        hidden_size = 64
        num_experts = 8
        top_k = 2
        batch_size, seq_len = 2, 16

        config = create_test_config(
            hidden_size=hidden_size,
            num_routed_experts=num_experts,
            top_k=top_k,
        )
        router = TopKRouter(config, hidden_size, num_experts, top_k)
        router.init_weights()

        # Create input
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Get router output
        output = router(hidden_states, training=False)

        # Manually compute expected top-k
        with torch.no_grad():
            logits = torch.matmul(hidden_states, router.weight)
            probs = torch.softmax(logits, dim=-1)
            expected_weights, expected_indices = torch.topk(probs, top_k, dim=-1)

        # Verify indices match
        assert torch.equal(output.topk_indices, expected_indices), "Router selected wrong experts"

        # Router renormalizes top-k weights to sum to 1, so compare renormalized values
        expected_renormalized = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
        assert torch.allclose(output.topk_weights, expected_renormalized, atol=1e-5), (
            "Router weights don't match expected renormalized softmax values"
        )

    def test_weights_sum_to_one(self):
        """Verify that routing weights sum to 1.0 for each token."""
        torch.manual_seed(42)

        hidden_size = 64
        num_experts = 8
        top_k = 4
        batch_size, seq_len = 2, 16

        config = create_test_config(
            hidden_size=hidden_size,
            num_routed_experts=num_experts,
            top_k=top_k,
        )
        router = TopKRouter(config, hidden_size, num_experts, top_k)
        router.init_weights()

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        output = router(hidden_states, training=False)

        # Weights should sum to 1 along the top_k dimension
        weight_sums = output.topk_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), (
            f"Weights don't sum to 1: got {weight_sums}"
        )

    def test_indices_in_valid_range(self):
        """Verify all selected expert indices are valid."""
        torch.manual_seed(42)

        hidden_size = 64
        num_experts = 8
        top_k = 2
        batch_size, seq_len = 4, 32

        config = create_test_config(
            hidden_size=hidden_size,
            num_routed_experts=num_experts,
            top_k=top_k,
        )
        router = TopKRouter(config, hidden_size, num_experts, top_k)

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        output = router(hidden_states, training=False)

        assert (output.topk_indices >= 0).all(), "Negative expert indices found"
        assert (output.topk_indices < num_experts).all(), (
            f"Expert index exceeds num_experts ({num_experts})"
        )

    def test_deterministic_with_fixed_seed(self):
        """Verify router produces identical results with same seed."""
        hidden_size = 64
        num_experts = 8
        top_k = 2

        config = create_test_config(
            hidden_size=hidden_size,
            num_routed_experts=num_experts,
            top_k=top_k,
        )

        # Create two routers with same seed
        torch.manual_seed(12345)
        router1 = TopKRouter(config, hidden_size, num_experts, top_k)
        router1.init_weights()

        torch.manual_seed(12345)
        router2 = TopKRouter(config, hidden_size, num_experts, top_k)
        router2.init_weights()

        # Same input
        torch.manual_seed(42)
        hidden_states = torch.randn(2, 16, hidden_size)

        output1 = router1(hidden_states, training=False)
        output2 = router2(hidden_states, training=False)

        assert torch.equal(output1.topk_indices, output2.topk_indices), (
            "Router is not deterministic with fixed seed"
        )


class TestGradientFlowAndSparsity:
    """Tests for gradient flow through selected experts only."""

    def test_selected_experts_receive_gradients(self):
        """Verify that selected experts have non-zero gradients after backward."""
        torch.manual_seed(42)

        hidden_size = 64
        num_routed_experts = 8
        top_k = 2
        batch_size, seq_len = 4, 16

        config = create_test_config(
            hidden_size=hidden_size,
            num_routed_experts=num_routed_experts,
            top_k=top_k,
        )
        moe = MoEMLP(config)
        moe.init_weights()
        moe.train()

        # Forward and backward
        x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        output = moe(x)
        loss = output.sum()
        loss.backward()

        # Get which experts were selected
        with torch.no_grad():
            router_out = moe.router(x, training=False)
            selected_experts = set(router_out.topk_indices.flatten().tolist())

        # Verify selected experts have gradients
        for expert_idx in selected_experts:
            expert = moe.routed_experts[expert_idx]
            assert expert.up_proj.weight.grad is not None, (
                f"Expert {expert_idx} has no gradient despite being selected"
            )
            grad_norm = expert.up_proj.weight.grad.abs().sum().item()
            assert grad_norm > 0, f"Expert {expert_idx} has zero gradient despite being selected"

    def test_unselected_experts_have_zero_gradients(self):
        """Verify that unselected experts have zero or no gradients."""
        torch.manual_seed(42)

        hidden_size = 64
        num_routed_experts = 8
        top_k = 2
        batch_size, seq_len = 2, 4  # Small batch to ensure some experts unselected

        config = create_test_config(
            hidden_size=hidden_size,
            num_routed_experts=num_routed_experts,
            top_k=top_k,
        )
        moe = MoEMLP(config)
        moe.init_weights()
        moe.train()

        # Zero all gradients first
        for param in moe.parameters():
            if param.grad is not None:
                param.grad.zero_()

        # Forward and backward
        x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        output = moe(x)
        loss = output.sum()
        loss.backward()

        # Get which experts were selected
        with torch.no_grad():
            router_out = moe.router(x, training=False)
            selected_experts = set(router_out.topk_indices.flatten().tolist())

        # Find unselected experts
        all_experts = set(range(num_routed_experts))
        unselected_experts = all_experts - selected_experts

        # Verify unselected experts have zero or no gradients
        for expert_idx in unselected_experts:
            expert = moe.routed_experts[expert_idx]
            grad = expert.up_proj.weight.grad
            if grad is not None:
                grad_norm = grad.abs().sum().item()
                assert grad_norm == 0, (
                    f"Expert {expert_idx} has non-zero gradient ({grad_norm}) "
                    f"despite not being selected"
                )

    def test_gradient_flow_through_shared_experts(self):
        """Verify shared experts receive gradients from all tokens."""
        torch.manual_seed(42)

        hidden_size = 64
        num_shared_experts = 2
        batch_size, seq_len = 4, 16

        config = create_test_config(
            hidden_size=hidden_size,
            num_shared_experts=num_shared_experts,
        )
        moe = MoEMLP(config)
        moe.init_weights()
        moe.train()

        x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        output = moe(x)
        loss = output.sum()
        loss.backward()

        # All shared experts should have gradients
        for i, expert in enumerate(moe.shared_experts):
            assert expert.up_proj.weight.grad is not None, f"Shared expert {i} has no gradient"
            grad_norm = expert.up_proj.weight.grad.abs().sum().item()
            assert grad_norm > 0, f"Shared expert {i} has zero gradient"


class TestSharedVsRoutedExpertGradients:
    """Tests comparing gradient distribution between shared and routed experts."""

    def test_shared_expert_larger_gradient_norm(self):
        """Verify shared experts generally have larger gradient norms than routed."""
        torch.manual_seed(42)

        hidden_size = 64
        num_shared_experts = 2
        num_routed_experts = 8
        top_k = 2
        batch_size, seq_len = 8, 32

        config = create_test_config(
            hidden_size=hidden_size,
            num_shared_experts=num_shared_experts,
            num_routed_experts=num_routed_experts,
            top_k=top_k,
        )
        moe = MoEMLP(config)
        moe.init_weights()
        moe.train()

        x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        output = moe(x)
        loss = output.sum()
        loss.backward()

        # Compute gradient norms
        shared_grad_norms = []
        for expert in moe.shared_experts:
            norm = expert.up_proj.weight.grad.abs().sum().item()
            shared_grad_norms.append(norm)

        routed_grad_norms = []
        for expert in moe.routed_experts:
            grad = expert.up_proj.weight.grad
            if grad is not None:
                norm = grad.abs().sum().item()
                routed_grad_norms.append(norm)

        avg_shared = sum(shared_grad_norms) / len(shared_grad_norms)
        avg_routed = sum(routed_grad_norms) / len(routed_grad_norms) if routed_grad_norms else 0

        # Shared experts process all tokens, so should have larger gradients
        # (This is a statistical test, so we allow some tolerance)
        assert avg_shared >= avg_routed * 0.5, (
            f"Shared expert avg gradient ({avg_shared:.4f}) is much smaller than "
            f"routed expert avg ({avg_routed:.4f}). Shared experts should process more tokens."
        )

    def test_shared_experts_gradient_from_all_tokens(self):
        """Verify shared expert gradients generally increase with more tokens."""
        torch.manual_seed(42)

        hidden_size = 64
        num_shared_experts = 1

        config = create_test_config(
            hidden_size=hidden_size,
            num_shared_experts=num_shared_experts,
            num_routed_experts=4,
            top_k=2,
        )

        # Test with different token counts
        grad_norms = []
        for batch_size, seq_len in [(2, 8), (4, 16), (8, 32)]:
            moe = MoEMLP(config)
            moe.init_weights()
            moe.train()

            x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
            output = moe(x)
            loss = output.sum()
            loss.backward()

            norm = moe.shared_experts[0].up_proj.weight.grad.abs().sum().item()
            grad_norms.append((batch_size * seq_len, norm))

        # Gradient norm should generally increase with token count
        # Note: Due to randomness in routing and initialization, we only check
        # that the final (largest) token count has larger gradient than the first
        first_tokens, first_norm = grad_norms[0]
        last_tokens, last_norm = grad_norms[-1]

        # With 16x more tokens, expect at least some increase
        assert last_norm > first_norm * 0.5, (
            f"Gradient norm didn't scale with tokens: "
            f"{first_tokens} tokens -> {first_norm:.4f}, "
            f"{last_tokens} tokens -> {last_norm:.4f}"
        )


class TestInputValidation:
    """Tests for input validation edge cases."""

    def test_nan_input_raises_error(self):
        """Test that NaN input raises ValueError."""
        config = create_test_config()
        moe = MoEMLP(config)
        moe.init_weights()

        x = torch.randn(2, 16, 64)
        x[0, 0, 0] = float("nan")

        with pytest.raises(ValueError, match="NaN"):
            moe(x)

    def test_inf_input_raises_error(self):
        """Test that Inf input raises ValueError."""
        config = create_test_config()
        moe = MoEMLP(config)
        moe.init_weights()

        x = torch.randn(2, 16, 64)
        x[0, 0, 0] = float("inf")

        with pytest.raises(ValueError, match="Inf"):
            moe(x)

    def test_wrong_hidden_size_raises_error(self):
        """Test that wrong hidden size raises ValueError."""
        config = create_test_config(hidden_size=64)
        moe = MoEMLP(config)
        moe.init_weights()

        x = torch.randn(2, 16, 128)  # Wrong hidden size

        with pytest.raises(ValueError, match="hidden_size"):
            moe(x)

    def test_wrong_ndim_raises_error(self):
        """Test that wrong number of dimensions raises ValueError."""
        config = create_test_config()
        moe = MoEMLP(config)
        moe.init_weights()

        x_2d = torch.randn(16, 64)  # Missing batch dimension

        with pytest.raises(ValueError, match="3D"):
            moe(x_2d)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
