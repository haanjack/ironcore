# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for MoEMLP layer.

Integration tests for the complete MoE layer:
- End-to-end forward/backward
- Auxiliary loss computation
- Different configurations
"""

import os

import pytest
import torch

# Set up environment for single-GPU testing
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

from tests.fixtures.config_fixtures import create_moe_test_config

from ironcore.layers.moe import MoEMLP
from ironcore.parallel.parallel_states import initialize_model_parallel


@pytest.fixture(autouse=True)
def setup_parallel_states():
    """Initialize parallel states before each test."""
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )
    yield
    # Cleanup to prevent state leakage
    from ironcore.parallel.parallel_states import destroy_model_parallel

    destroy_model_parallel()


class TestMoEMLP:
    """Test cases for MoEMLP layer."""

    def test_moe_output_shape(self):
        """Test that MoE layer produces correct output shape."""
        batch_size, seq_len, hidden_size = 2, 16, 256

        config = create_moe_test_config(hidden_size=hidden_size)
        moe = MoEMLP(config)
        moe.init_weights()  # Initialize weights to avoid NaN

        x = torch.randn(batch_size, seq_len, hidden_size)
        output = moe(x)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_moe_forward_backward(self):
        """Test that MoE layer supports forward and backward passes."""
        batch_size, seq_len, hidden_size = 2, 16, 256

        config = create_moe_test_config(hidden_size=hidden_size)
        moe = MoEMLP(config)
        moe.init_weights()  # Initialize weights to avoid NaN

        x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        output = moe(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        # Check router and expert gradients
        assert moe.router.weight.grad is not None

    def test_shared_experts_process_all_tokens(self):
        """Test that shared experts process all tokens."""
        batch_size, seq_len, hidden_size = 2, 16, 256
        num_shared_experts = 2

        # Use minimum routed experts (1) to avoid validation error
        config = create_moe_test_config(
            hidden_size=hidden_size,
            num_shared_experts=num_shared_experts,
            num_routed_experts=1,  # Minimum 1
            num_experts_per_token=1,
        )

        moe = MoEMLP(config)
        moe.init_weights()  # Initialize weights to avoid NaN

        # Check number of shared experts
        assert len(moe.shared_experts) == num_shared_experts

        x = torch.randn(batch_size, seq_len, hidden_size)
        output = moe(x)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_aux_loss_computation(self):
        """Test that auxiliary loss is computed during training."""
        batch_size, seq_len, hidden_size = 2, 16, 256

        config = create_moe_test_config(hidden_size=hidden_size, aux_loss_alpha=0.01)
        moe = MoEMLP(config)
        moe.init_weights()  # Initialize weights to avoid NaN
        moe.train()

        x = torch.randn(batch_size, seq_len, hidden_size)
        _ = moe(x)

        aux_loss = moe.get_aux_loss()
        assert aux_loss is not None
        assert aux_loss.item() >= 0.0

    def test_aux_loss_not_computed_in_eval(self):
        """Test that auxiliary loss is not stored in eval mode."""
        batch_size, seq_len, hidden_size = 2, 16, 256

        config = create_moe_test_config(hidden_size=hidden_size, aux_loss_alpha=0.01)
        moe = MoEMLP(config)
        moe.init_weights()  # Initialize weights to avoid NaN
        moe.eval()

        x = torch.randn(batch_size, seq_len, hidden_size)

        with torch.no_grad():
            _ = moe(x)

        # In eval mode, aux_loss should be None or zero
        aux_loss = moe.get_aux_loss()
        assert aux_loss is None or aux_loss.item() == 0.0, (
            f"Aux loss should be zero or None in eval mode, got {aux_loss}"
        )

    def test_different_top_k_values(self):
        """Test MoE with different top_k values."""
        batch_size, seq_len, hidden_size = 2, 16, 256
        num_routed_experts = 16

        for top_k in [1, 2, 4, 8]:
            config = create_moe_test_config(
                hidden_size=hidden_size,
                num_routed_experts=num_routed_experts,
                num_experts_per_token=top_k,
            )
            moe = MoEMLP(config)
            moe.init_weights()  # Initialize weights to avoid NaN

            x = torch.randn(batch_size, seq_len, hidden_size)
            output = moe(x)

            assert output.shape == (batch_size, seq_len, hidden_size)

    def test_different_expert_counts(self):
        """Test MoE with different expert counts."""
        batch_size, seq_len, hidden_size = 2, 16, 256

        for num_experts in [4, 8, 16, 32, 64]:
            config = create_moe_test_config(
                hidden_size=hidden_size,
                num_routed_experts=num_experts,
            )
            moe = MoEMLP(config)
            moe.init_weights()  # Initialize weights to avoid NaN

            x = torch.randn(batch_size, seq_len, hidden_size)
            output = moe(x)

            assert output.shape == (batch_size, seq_len, hidden_size)

    def test_output_is_not_nan(self):
        """Test that output doesn't contain NaN values."""
        batch_size, seq_len, hidden_size = 2, 16, 256

        config = create_moe_test_config(hidden_size=hidden_size)
        moe = MoEMLP(config)
        moe.init_weights()  # Initialize weights to avoid NaN

        x = torch.randn(batch_size, seq_len, hidden_size)
        output = moe(x)

        assert not torch.isnan(output).any()

    def test_deterministic_output(self):
        """Test deterministic output in eval mode."""
        batch_size, seq_len, hidden_size = 2, 16, 256

        config = create_moe_test_config(hidden_size=hidden_size)
        moe = MoEMLP(config)
        moe.init_weights()  # Initialize weights to avoid NaN
        moe.eval()

        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, hidden_size)

        output1 = moe(x)
        output2 = moe(x)

        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-5)

    def test_expert_utilization(self):
        """Test expert utilization computation."""
        batch_size, seq_len, hidden_size = 2, 16, 256
        num_routed_experts = 16
        top_k = 2

        config = create_moe_test_config(
            hidden_size=hidden_size,
            num_routed_experts=num_routed_experts,
            num_experts_per_token=top_k,
        )
        moe = MoEMLP(config)

        x = torch.randn(batch_size, seq_len, hidden_size)
        router_output = moe.router(x, training=False)

        utilization = moe.get_expert_utilization(router_output.topk_indices)

        assert utilization.shape == (num_routed_experts,)
        assert torch.all(utilization >= 0)
        assert torch.allclose(utilization.sum(), torch.tensor(1.0), atol=1e-5)


class TestMoEMLPAllToAll:
    """Test cases for MoEMLP layer with ALL_TO_ALL communication mode."""

    def test_alltoall_output_shape(self):
        """Test that ALL_TO_ALL mode produces correct output shape."""
        from ironcore.layers.moe import CommunicationMode

        batch_size, seq_len, hidden_size = 2, 16, 256

        config = create_moe_test_config(hidden_size=hidden_size)
        moe = MoEMLP(config, communication_mode=CommunicationMode.ALL_TO_ALL)
        moe.init_weights()

        x = torch.randn(batch_size, seq_len, hidden_size)
        output = moe(x)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_alltoall_forward_backward(self):
        """Test forward and backward pass with ALL_TO_ALL mode."""
        from ironcore.layers.moe import CommunicationMode

        batch_size, seq_len, hidden_size = 2, 16, 256

        config = create_moe_test_config(hidden_size=hidden_size)
        moe = MoEMLP(config, communication_mode=CommunicationMode.ALL_TO_ALL)
        moe.init_weights()

        x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        output = moe(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert moe.router.weight.grad is not None

    def test_alltoall_no_nan(self):
        """Test that ALL_TO_ALL mode doesn't produce NaN."""
        from ironcore.layers.moe import CommunicationMode

        batch_size, seq_len, hidden_size = 2, 16, 256

        config = create_moe_test_config(hidden_size=hidden_size)
        moe = MoEMLP(config, communication_mode=CommunicationMode.ALL_TO_ALL)
        moe.init_weights()

        x = torch.randn(batch_size, seq_len, hidden_size)
        output = moe(x)

        assert not torch.isnan(output).any()

    def test_alltoall_deterministic_in_eval(self):
        """Test deterministic output in eval mode with ALL_TO_ALL."""
        from ironcore.layers.moe import CommunicationMode

        batch_size, seq_len, hidden_size = 2, 16, 256

        config = create_moe_test_config(hidden_size=hidden_size)
        moe = MoEMLP(config, communication_mode=CommunicationMode.ALL_TO_ALL)
        moe.init_weights()
        moe.eval()

        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, hidden_size)

        output1 = moe(x)
        output2 = moe(x)

        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-5)

    def test_alltoall_aux_loss_computation(self):
        """Test auxiliary loss with ALL_TO_ALL mode."""
        from ironcore.layers.moe import CommunicationMode

        batch_size, seq_len, hidden_size = 2, 16, 256

        config = create_moe_test_config(hidden_size=hidden_size, aux_loss_alpha=0.01)
        moe = MoEMLP(config, communication_mode=CommunicationMode.ALL_TO_ALL)
        moe.init_weights()
        moe.train()

        x = torch.randn(batch_size, seq_len, hidden_size)
        _ = moe(x)

        aux_loss = moe.get_aux_loss()
        assert aux_loss is not None
        assert aux_loss.item() >= 0.0


class TestMoEConfigValidation:
    """Test cases for MoE configuration validation."""

    def test_invalid_top_k(self):
        """Test that top_k > num_experts raises error."""
        from ironcore.config.config_moe import MoEConfig

        with pytest.raises(ValueError):
            MoEConfig(
                use_moe=True,
                num_routed_experts=4,
                num_experts_per_token=8,  # Invalid: top_k > num_experts
            )

    def test_zero_experts(self):
        """Test that zero routed experts raises error."""
        from ironcore.config.config_moe import MoEConfig

        with pytest.raises(ValueError):
            MoEConfig(
                use_moe=True,
                num_routed_experts=0,  # Invalid
                num_experts_per_token=1,
            )

    def test_negative_aux_loss(self):
        """Test that negative aux_loss_alpha raises error."""
        from ironcore.config.config_moe import MoEConfig

        with pytest.raises(ValueError):
            MoEConfig(
                use_moe=True,
                num_routed_experts=16,
                num_experts_per_token=2,
                aux_loss_alpha=-0.01,  # Invalid
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
