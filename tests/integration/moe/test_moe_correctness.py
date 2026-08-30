# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for MoE implementation.

These tests verify:
1. MoE produces same output as dense when all experts are used
2. Gradient correctness
3. Load balancing behavior
"""

import pytest
import torch
from tests.fixtures.config_fixtures import create_moe_test_config
from tests.fixtures.utils import single_gpu_env

from ironcore.layers.mlp import MLP
from ironcore.layers.moe import MoEMLP
from ironcore.parallel.parallel_states import initialize_model_parallel


@pytest.fixture(autouse=True)
def setup_parallel_states():
    """Initialize parallel states before each test."""
    with single_gpu_env():
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            timeout_in_minutes=10.0,
        )
        yield
        # Cleanup to prevent state leakage
        from ironcore.parallel.parallel_states import destroy_model_parallel

        destroy_model_parallel()


class TestMoECorrectness:
    """Correctness tests for MoE layer."""

    def test_single_expert_matches_dense(self):
        """Test that MoE with 1 expert + 0 shared matches dense MLP output.

        When there's only 1 routed expert and top_k=1, every token goes
        to that single expert, so the output should be equivalent to a
        standard MLP (up to initialization differences).
        """
        hidden_size, intermediate_size = 256, 512
        batch_size, seq_len = 2, 16

        # Create MoE config with single expert
        moe_config = create_moe_test_config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_shared_experts=0,
            num_routed_experts=1,
            num_experts_per_token=1,
            aux_loss_alpha=0.0,
            attention_bias=False,
            mlp_bias=False,
            layernorm_bias=False,
        )

        # Create dense config
        dense_config = create_moe_test_config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            aux_loss_alpha=0.0,
            attention_bias=False,
            mlp_bias=False,
            layernorm_bias=False,
        )
        dense_config.model.moe.use_moe = False

        moe = MoEMLP(moe_config)
        moe.init_weights()  # Initialize weights to avoid NaN
        dense = MLP(dense_config)

        # Manually initialize dense MLP weights (can't use init_weights due to config issue)
        from torch import nn

        with torch.no_grad():
            nn.init.normal_(dense.up_proj.weight, std=0.02)
            nn.init.normal_(dense.down_proj.weight, std=0.02)
            if dense.up_proj.bias is not None:
                nn.init.zeros_(dense.up_proj.bias)
            if dense.down_proj.bias is not None:
                nn.init.zeros_(dense.down_proj.bias)

        # Copy weights: expert.up_proj <- dense.up_proj, expert.down_proj <- dense.down_proj
        with torch.no_grad():
            moe.routed_experts[0].up_proj.weight.copy_(dense.up_proj.weight)
            moe.routed_experts[0].down_proj.weight.copy_(dense.down_proj.weight)

        # Same input
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, hidden_size)

        # Forward
        moe_out = moe(x)
        dense_out = dense(x)

        # Check for NaN first
        assert not torch.isnan(moe_out).any(), "MoE output contains NaN"
        assert not torch.isnan(dense_out).any(), "Dense output contains NaN"

        # Should be very close (not exact due to routing softmax)
        # Note: Even with 1 expert, softmax is applied to weights
        max_diff = (moe_out - dense_out).abs().max().item()
        assert max_diff < 1e-2, f"Max diff: {max_diff}"

    def test_gradient_correctness(self):
        """Test that gradients flow correctly through MoE layer."""
        hidden_size = 256
        batch_size, seq_len = 2, 16

        config = create_moe_test_config(
            hidden_size=hidden_size,
            num_routed_experts=4,
            num_experts_per_token=2,
            aux_loss_alpha=0.0,
            attention_bias=False,
            mlp_bias=False,
            layernorm_bias=False,
        )
        moe = MoEMLP(config)
        moe.init_weights()  # Initialize weights to avoid NaN

        x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        output = moe(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist and are non-zero
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

        # Check router gradient
        assert moe.router.weight.grad is not None
        assert moe.router.weight.grad.abs().sum() > 0

        # Check expert gradients
        for expert in moe.routed_experts:
            assert expert.up_proj.weight.grad is not None
            assert expert.down_proj.weight.grad is not None

    def test_shared_experts_contribution(self):
        """Test that shared experts contribute to output."""
        hidden_size = 256
        batch_size, seq_len = 2, 16

        config = create_moe_test_config(
            hidden_size=hidden_size,
            num_shared_experts=2,
            num_routed_experts=4,
            num_experts_per_token=2,
            aux_loss_alpha=0.0,
            attention_bias=False,
            mlp_bias=False,
            layernorm_bias=False,
        )
        moe = MoEMLP(config)
        moe.init_weights()  # Initialize weights to avoid all zeros

        x = torch.randn(batch_size, seq_len, hidden_size)

        # Get full output with shared experts
        full_output = moe(x)

        # Zero out shared experts to see only routed expert contribution
        for expert in moe.shared_experts:
            with torch.no_grad():
                expert.up_proj.weight.zero_()
                expert.down_proj.weight.zero_()

        # Get output without shared expert contribution
        output_no_shared = moe(x)

        # Outputs should differ (shared experts contribute)
        assert not torch.allclose(full_output, output_no_shared, rtol=1e-5, atol=1e-5), (
            "Shared experts should contribute to output"
        )

    def test_output_shape_preserved(self):
        """Test that output shape is always preserved regardless of routing."""
        hidden_size = 256
        batch_size, seq_len = 2, 16

        # Test various configurations
        configs = [
            {"num_routed_experts": 8, "num_experts_per_token": 1},
            {"num_routed_experts": 8, "num_experts_per_token": 4},
            {"num_routed_experts": 64, "num_experts_per_token": 6},
            {"num_routed_experts": 256, "num_experts_per_token": 8},
        ]

        for config_kwargs in configs:
            config = create_moe_test_config(
                hidden_size=hidden_size,
                aux_loss_alpha=0.0,
                attention_bias=False,
                mlp_bias=False,
                layernorm_bias=False,
                **config_kwargs,
            )
            moe = MoEMLP(config)

            for _ in range(3):  # Multiple random inputs
                x = torch.randn(batch_size, seq_len, hidden_size)
                output = moe(x)
                assert output.shape == (batch_size, seq_len, hidden_size), (
                    f"Shape mismatch for config {config_kwargs}"
                )

    def test_load_balance_improves_with_training(self):
        """Test that load balance loss encourages even expert utilization."""
        # Seeded: model init and all 20 input batches were drawn from the ambient
        # RNG, so the result depended on whatever ran before it in the same pytest
        # process. It passed standalone and failed inside the full suite
        # (first=0.0208, second=0.0249, 9% over the tolerance). The assertion
        # itself is sound — it holds for all of seeds 0-7, 42 and 123 — so this
        # pins the draw rather than loosening the check.
        torch.manual_seed(42)
        hidden_size = 256
        batch_size, seq_len = 4, 32
        num_experts, top_k = 8, 2

        config = create_moe_test_config(
            hidden_size=hidden_size,
            num_routed_experts=num_experts,
            num_experts_per_token=top_k,
            num_shared_experts=0,
            aux_loss_alpha=0.0,
            attention_bias=False,
            mlp_bias=False,
            layernorm_bias=False,
        )
        config.model.moe.aux_loss_alpha = 0.1  # Strong aux loss

        moe = MoEMLP(config)
        optimizer = torch.optim.Adam(moe.parameters(), lr=0.01)

        # Collect utilization over training
        utilizations = []

        for step in range(20):
            x = torch.randn(batch_size, seq_len, hidden_size)

            optimizer.zero_grad()
            output = moe(x)

            # Get aux loss
            aux_loss = moe.get_aux_loss()
            if aux_loss is not None:
                # Compute utilization
                router_out = moe.router(x, training=False)
                util = moe.get_expert_utilization(router_out.topk_indices)
                utilizations.append(util.std().item())  # Std deviation of utilization

                # Backward with aux loss
                (output.sum() + aux_loss).backward()
            else:
                output.sum().backward()

            optimizer.step()
            moe.clear_aux_loss()

        # Utilization should become more even (lower std) over time
        # First half vs second half
        first_half_std = sum(utilizations[:10]) / 10
        second_half_std = sum(utilizations[10:]) / 10

        # Second half should have lower or similar std (more balanced)
        # Allow 10% degradation tolerance
        assert second_half_std <= first_half_std * 1.1, (
            f"Load balance did not improve or stay stable: first={first_half_std:.4f}, second={second_half_std:.4f}"
        )

    def test_no_nan_with_large_inputs(self):
        """Test that MoE handles large inputs without producing NaN."""
        hidden_size = 256
        batch_size, seq_len = 2, 16

        config = create_moe_test_config(
            hidden_size=hidden_size,
            num_routed_experts=8,
            num_experts_per_token=2,
            aux_loss_alpha=0.0,
            attention_bias=False,
            mlp_bias=False,
            layernorm_bias=False,
        )
        moe = MoEMLP(config)
        moe.init_weights()  # Initialize weights to avoid NaN

        # Large inputs
        for scale in [10.0, 100.0, 1000.0]:
            x = torch.randn(batch_size, seq_len, hidden_size) * scale
            output = moe(x)

            assert not torch.isnan(output).any(), f"NaN in output with scale {scale}"
            assert not torch.isinf(output).any(), f"Inf in output with scale {scale}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
