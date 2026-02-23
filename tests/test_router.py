# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for TopKRouter."""

import pytest
import torch

from ironcore.config import MainConfig, PEFTConfig
from ironcore.config.config_data import DataConfig
from ironcore.config.config_model import ModelConfig
from ironcore.config.config_moe import MoEConfig
from ironcore.config.config_optim import OptimConfig
from ironcore.config.config_parallel import ParallelConfig
from ironcore.config.config_trainer import InitConfig, OperationConfig, TrainerConfig
from ironcore.config.config_utils import UtilsConfig
from ironcore.layers.moe.router import TopKRouter


def create_test_config(
    hidden_size: int = 256,
    num_experts: int = 16,
    top_k: int = 2,
    jitter_noise: float = 0.0,
):
    """Create a test configuration."""
    config = MainConfig(
        model=ModelConfig(
            d_model=hidden_size,
            d_ffn=512,
            moe=MoEConfig(
                use_moe=True,
                num_routed_experts=num_experts,
                num_experts_per_token=top_k,
                router_jitter_noise=jitter_noise,
            ),
        ),
        init=InitConfig(),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        trainer=TrainerConfig(),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        peft=PEFTConfig(),
    )
    return config


class TestTopKRouter:
    """Test cases for TopKRouter."""

    def test_router_output_shapes(self):
        """Test that router produces correct output shapes."""
        batch_size, seq_len, hidden_size = 2, 16, 256
        num_experts, top_k = 16, 2

        config = create_test_config(hidden_size, num_experts, top_k)
        router = TopKRouter(config, hidden_size, num_experts, top_k)

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        output = router(hidden_states, training=False)

        assert output.topk_weights.shape == (batch_size, seq_len, top_k)
        assert output.topk_indices.shape == (batch_size, seq_len, top_k)
        assert output.router_logits.shape == (batch_size, seq_len, num_experts)

    def test_weights_sum_to_one(self):
        """Test that routing weights sum to 1 for each token."""
        batch_size, seq_len, hidden_size = 2, 16, 256
        num_experts, top_k = 16, 4

        config = create_test_config(hidden_size, num_experts, top_k)
        router = TopKRouter(config, hidden_size, num_experts, top_k)
        router.init_weights()  # Initialize weights to avoid NaN

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        output = router(hidden_states, training=False)

        # Weights should sum to 1 along the top_k dimension
        weight_sums = output.topk_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_indices_in_valid_range(self):
        """Test that all indices are valid expert indices."""
        batch_size, seq_len, hidden_size = 2, 16, 256
        num_experts, top_k = 16, 4

        config = create_test_config(hidden_size, num_experts, top_k)
        router = TopKRouter(config, hidden_size, num_experts, top_k)

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        output = router(hidden_states, training=False)

        assert (output.topk_indices >= 0).all()
        assert (output.topk_indices < num_experts).all()

    def test_top_k_selection(self):
        """Test that top-k experts are actually selected."""
        batch_size, seq_len, hidden_size = 2, 4, 256
        num_experts, top_k = 8, 2

        config = create_test_config(hidden_size, num_experts, top_k)
        router = TopKRouter(config, hidden_size, num_experts, top_k)
        router.init_weights()  # Initialize weights to avoid NaN

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        output = router(hidden_states, training=False)

        # Verify top-k by checking indices correspond to highest logits
        sorted_logits, sorted_indices = torch.sort(output.router_logits, dim=-1, descending=True)
        expected_topk = sorted_indices[:, :, :top_k]

        assert torch.equal(output.topk_indices, expected_topk)

    def test_jitter_noise_disabled_in_eval(self):
        """Test that jitter noise is not added during evaluation."""
        batch_size, seq_len, hidden_size = 2, 16, 256
        num_experts, top_k = 16, 2

        config = create_test_config(hidden_size, num_experts, top_k, jitter_noise=0.5)
        router = TopKRouter(config, hidden_size, num_experts, top_k)

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Run twice with eval mode - should get same results
        router.eval()
        output1 = router(hidden_states, training=False)
        output2 = router(hidden_states, training=False)

        assert torch.equal(output1.topk_indices, output2.topk_indices)

    def test_deterministic_without_noise(self):
        """Test deterministic behavior when noise is disabled."""
        batch_size, seq_len, hidden_size = 2, 16, 256
        num_experts, top_k = 16, 2

        config = create_test_config(hidden_size, num_experts, top_k, jitter_noise=0.0)
        router = TopKRouter(config, hidden_size, num_experts, top_k)

        torch.manual_seed(42)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        torch.manual_seed(42)
        output1 = router(hidden_states, training=True)
        output2 = router(hidden_states, training=True)

        assert torch.equal(output1.topk_indices, output2.topk_indices)

    def test_router_logits_stored_during_training(self):
        """Test that router logits are stored during training."""
        batch_size, seq_len, hidden_size = 2, 16, 256
        num_experts, top_k = 16, 2

        config = create_test_config(hidden_size, num_experts, top_k)
        router = TopKRouter(config, hidden_size, num_experts, top_k)

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Clear any stored logits
        router.clear_router_logits()

        # Forward pass in training mode
        router.train()
        _ = router(hidden_states, training=True)

        # Check logits are stored
        stored_logits = router.get_router_logits()
        assert stored_logits is not None
        assert stored_logits.shape == (batch_size, seq_len, num_experts)

    def test_gradient_flow(self):
        """Test that gradients flow through router."""
        batch_size, seq_len, hidden_size = 2, 16, 256
        num_experts, top_k = 16, 2

        config = create_test_config(hidden_size, num_experts, top_k)
        router = TopKRouter(config, hidden_size, num_experts, top_k)

        hidden_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        output = router(hidden_states, training=True)

        # Backward through weights
        loss = output.topk_weights.sum()
        loss.backward()

        # Check gradients exist
        assert hidden_states.grad is not None
        assert router.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
