# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ExpertMLP.

Unit tests for the MoE expert component:
- Output shape validation
- Forward/backward pass
- Different activation functions
- Bias handling
- Dropout behavior
"""

import os

import pytest
import torch

# Set up environment for single-GPU testing
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

from ironcore.config import MainConfig, PEFTConfig
from ironcore.config.config_data import DataConfig
from ironcore.config.config_model import ModelConfig
from ironcore.config.config_optim import OptimConfig
from ironcore.config.config_parallel import ParallelConfig
from ironcore.config.config_trainer import InitConfig, OperationConfig, TrainerConfig
from ironcore.config.config_utils import ProfilerConfig, UtilsConfig
from ironcore.layers.moe.expert import ExpertMLP
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


def create_test_config(
    hidden_size: int = 256,
    intermediate_size: int = 512,
    dropout: float = 0.0,
    attention_bias: bool = True,
    mlp_bias: bool = True,
    layernorm_bias: bool = True,
):
    """Create a test configuration."""
    config = MainConfig(
        model=ModelConfig(
            d_model=hidden_size,
            d_ffn=intermediate_size,
            dropout_mlp=dropout,
            attention_bias=attention_bias,
            mlp_bias=mlp_bias,
            layernorm_bias=layernorm_bias,
            activation_type="gelu",
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
    return config


class TestExpertMLP:
    """Test cases for ExpertMLP."""

    def test_expert_output_shape(self):
        """Test that expert produces correct output shape."""
        num_tokens, hidden_size, intermediate_size = 32, 256, 512

        config = create_test_config(hidden_size, intermediate_size)
        expert = ExpertMLP(config, hidden_size, intermediate_size, expert_id=0)

        x = torch.randn(num_tokens, hidden_size)
        output = expert(x)

        assert output.shape == (num_tokens, hidden_size)

    def test_expert_forward_backward(self):
        """Test that expert supports forward and backward passes."""
        num_tokens, hidden_size, intermediate_size = 32, 256, 512

        config = create_test_config(hidden_size, intermediate_size)
        expert = ExpertMLP(config, hidden_size, intermediate_size, expert_id=0)

        x = torch.randn(num_tokens, hidden_size, requires_grad=True)
        output = expert(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        for param in expert.parameters():
            assert param.grad is not None

    def test_expert_varying_token_count(self):
        """Test that expert handles different token counts."""
        hidden_size, intermediate_size = 256, 512

        config = create_test_config(hidden_size, intermediate_size)
        expert = ExpertMLP(config, hidden_size, intermediate_size, expert_id=0)

        for num_tokens in [1, 16, 64, 128]:
            x = torch.randn(num_tokens, hidden_size)
            output = expert(x)
            assert output.shape == (num_tokens, hidden_size)

    def test_expert_different_activations(self):
        """Test expert with different activation functions."""
        num_tokens, hidden_size, intermediate_size = 32, 256, 512

        # Test standard activations (GLU variants need special handling)
        for activation in ["gelu", "relu", "silu"]:
            config = create_test_config(hidden_size, intermediate_size)
            config.model.activation_type = activation
            expert = ExpertMLP(config, hidden_size, intermediate_size, expert_id=0)

            x = torch.randn(num_tokens, hidden_size)
            output = expert(x)
            assert output.shape == (num_tokens, hidden_size)

    def test_expert_with_bias(self):
        """Test expert with bias enabled."""
        num_tokens, hidden_size, intermediate_size = 32, 256, 512

        config = create_test_config(
            hidden_size, intermediate_size, attention_bias=True, mlp_bias=True, layernorm_bias=True
        )
        expert = ExpertMLP(config, hidden_size, intermediate_size, expert_id=0)

        assert expert.up_proj.bias is not None
        assert expert.down_proj.bias is not None

        x = torch.randn(num_tokens, hidden_size)
        output = expert(x)
        assert output.shape == (num_tokens, hidden_size)

    def test_expert_without_bias(self):
        """Test expert with bias disabled."""
        num_tokens, hidden_size, intermediate_size = 32, 256, 512

        config = create_test_config(
            hidden_size,
            intermediate_size,
            attention_bias=False,
            mlp_bias=False,
            layernorm_bias=False,
        )
        expert = ExpertMLP(config, hidden_size, intermediate_size, expert_id=0)

        assert expert.up_proj.bias is None
        assert expert.down_proj.bias is None

        x = torch.randn(num_tokens, hidden_size)
        output = expert(x)
        assert output.shape == (num_tokens, hidden_size)

    def test_expert_dropout_training_vs_eval(self):
        """Test that dropout behaves differently in training vs eval."""
        num_tokens, hidden_size, intermediate_size = 32, 256, 512

        config = create_test_config(hidden_size, intermediate_size, dropout=0.5)
        expert = ExpertMLP(config, hidden_size, intermediate_size, expert_id=0)
        expert.init_weights()  # Initialize weights to avoid NaN

        x = torch.randn(num_tokens, hidden_size)

        # Training mode: outputs should vary due to dropout
        expert.train()
        outputs = [expert(x) for _ in range(5)]
        # At least some outputs should differ (probabilistic)
        # Use allclose to handle minor numerical differences
        differs_from_first = [
            not torch.allclose(outputs[0], o, rtol=1e-5, atol=1e-5) for o in outputs[1:]
        ]
        assert any(differs_from_first), "Dropout should cause outputs to differ in training mode"

        # Eval mode: dropout should be disabled
        expert.eval()
        outputs_eval = [expert(x) for _ in range(5)]
        # Outputs should be close (may not be exactly equal due to TP all-reduce)
        for o in outputs_eval[1:]:
            assert torch.allclose(outputs_eval[0], o, rtol=1e-5, atol=1e-5)

    def test_expert_id_tracking(self):
        """Test that expert ID is correctly stored."""
        hidden_size, intermediate_size = 256, 512
        config = create_test_config(hidden_size, intermediate_size)

        for expert_id in range(5):
            expert = ExpertMLP(config, hidden_size, intermediate_size, expert_id=expert_id)
            assert expert.expert_id == expert_id

    def test_expert_deterministic(self):
        """Test that expert produces deterministic outputs in eval mode."""
        num_tokens, hidden_size, intermediate_size = 32, 256, 512

        config = create_test_config(hidden_size, intermediate_size, dropout=0.0)
        expert = ExpertMLP(config, hidden_size, intermediate_size, expert_id=0)
        expert.init_weights()  # Initialize weights to avoid NaN
        expert.eval()

        torch.manual_seed(42)
        x = torch.randn(num_tokens, hidden_size)

        output1 = expert(x)
        output2 = expert(x)

        # Use allclose for floating-point tolerance
        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
