# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Expert Parallelism functionality.

These tests verify:
1. EP initialization and state management
2. All-reduce with gradient support
3. Communication primitives (when EP=1, should be no-ops)
"""

import os

import pytest
import torch

# Set up environment for single-GPU testing
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

from ironcore.parallel.expert_parallel import (
    all_reduce_ep,
    all_reduce_ep_with_grad,
    destroy_expert_parallel,
    get_expert_model_parallel_group,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
    initialize_expert_parallel,
)


class TestExpertParallelStates:
    """Test cases for expert parallel state management."""

    def test_initialization_single_gpu(self):
        """Test EP initialization with EP=1 (single GPU)."""
        # Initialize with EP=1
        initialize_expert_parallel(
            expert_model_parallel_size=1,
            tensor_model_parallel_size=1,
            timeout_in_minutes=10.0,
        )

        # Check state
        assert get_expert_model_parallel_world_size() == 1
        assert get_expert_model_parallel_rank() == 0
        # Group may be None when dist is not initialized
        get_expert_model_parallel_group()
        # For single GPU without dist init, group should be None or valid

        # Cleanup
        destroy_expert_parallel()

    def test_rank_calculation(self):
        """Test that rank calculation is correct."""
        initialize_expert_parallel(
            expert_model_parallel_size=1,
            tensor_model_parallel_size=1,
        )

        # Single GPU should always have rank 0
        assert get_expert_model_parallel_rank() == 0

        destroy_expert_parallel()

    def test_destroy_and_reinit(self):
        """Test that destroy and reinitialize works correctly."""
        # First initialization
        initialize_expert_parallel(
            expert_model_parallel_size=1,
            tensor_model_parallel_size=1,
        )
        assert get_expert_model_parallel_world_size() == 1

        # Destroy
        destroy_expert_parallel()

        # Reinitialize
        initialize_expert_parallel(
            expert_model_parallel_size=1,
            tensor_model_parallel_size=1,
        )
        assert get_expert_model_parallel_world_size() == 1

        destroy_expert_parallel()


class TestExpertParallelCommunication:
    """Test cases for EP communication primitives."""

    def setup_method(self):
        """Initialize EP before each test."""
        initialize_expert_parallel(
            expert_model_parallel_size=1,
            tensor_model_parallel_size=1,
        )

    def teardown_method(self):
        """Destroy EP after each test."""
        destroy_expert_parallel()

    def test_all_reduce_ep_single_gpu(self):
        """Test all_reduce_ep with EP=1 (should be no-op)."""
        x = torch.randn(4, 8)
        result = all_reduce_ep(x.clone())

        # With EP=1, all_reduce should be a no-op (or return input unchanged)
        assert torch.allclose(x, result)

    def test_all_reduce_ep_with_grad_single_gpu(self):
        """Test all_reduce_ep_with_grad with EP=1."""
        x = torch.randn(4, 8, requires_grad=True)
        result = all_reduce_ep_with_grad(x)

        # With EP=1, should return same values
        assert torch.allclose(x, result)

        # Test gradient flow
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.all(x.grad == 1.0)

    def test_all_reduce_ep_gradient_correctness(self):
        """Test that gradients flow correctly through all_reduce."""
        x = torch.randn(4, 8, requires_grad=True)

        # Forward
        y = all_reduce_ep_with_grad(x)
        y2 = y * 2  # Scale by 2

        # Backward
        y2.sum().backward()

        # Gradient should be 2.0 for each element (from the * 2)
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x.grad) * 2.0)


class TestMoEWithEP:
    """Test MoE layer with expert parallelism configuration."""

    def setup_method(self):
        """Initialize parallel states before each test."""
        from ironcore.parallel.parallel_states import initialize_model_parallel

        initialize_model_parallel(
            tensor_model_parallel_size=1,
            timeout_in_minutes=10.0,
        )
        initialize_expert_parallel(
            expert_model_parallel_size=1,
            tensor_model_parallel_size=1,
        )

    def teardown_method(self):
        """Destroy parallel states after each test."""
        from ironcore.parallel.parallel_states import destroy_model_parallel

        destroy_expert_parallel()
        destroy_model_parallel()

    def test_moe_with_ep_config(self):
        """Test MoE layer with EP=1 configuration."""
        from ironcore.config import AlignmentConfig, MainConfig, PEFTConfig, ProfilerConfig
        from ironcore.config.config_data import DataConfig
        from ironcore.config.config_model import ModelConfig
        from ironcore.config.config_moe import MoEConfig
        from ironcore.config.config_optim import OptimConfig
        from ironcore.config.config_parallel import ParallelConfig
        from ironcore.config.config_trainer import InitConfig, OperationConfig, TrainerConfig
        from ironcore.config.config_utils import UtilsConfig
        from ironcore.layers.moe import MoEMLP

        config = MainConfig(
            model=ModelConfig(
                d_model=256,
                d_ffn=512,
                moe=MoEConfig(
                    use_moe=True,
                    num_shared_experts=2,
                    num_routed_experts=16,
                    num_experts_per_token=2,
                    expert_model_parallel_size=1,  # EP=1
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
            alignment=AlignmentConfig(),
        )

        moe = MoEMLP(config)
        moe.init_weights()

        x = torch.randn(2, 16, 256)
        output = moe(x)

        assert output.shape == (2, 16, 256)
        assert not torch.isnan(output).any()

    def test_moe_gradient_with_ep(self):
        """Test gradient flow through MoE with EP=1."""
        from ironcore.config import AlignmentConfig, MainConfig, PEFTConfig, ProfilerConfig
        from ironcore.config.config_data import DataConfig
        from ironcore.config.config_model import ModelConfig
        from ironcore.config.config_moe import MoEConfig
        from ironcore.config.config_optim import OptimConfig
        from ironcore.config.config_parallel import ParallelConfig
        from ironcore.config.config_trainer import InitConfig, OperationConfig, TrainerConfig
        from ironcore.config.config_utils import UtilsConfig
        from ironcore.layers.moe import MoEMLP

        config = MainConfig(
            model=ModelConfig(
                d_model=256,
                d_ffn=512,
                moe=MoEConfig(
                    use_moe=True,
                    num_shared_experts=2,
                    num_routed_experts=16,
                    num_experts_per_token=2,
                    expert_model_parallel_size=1,
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
            alignment=AlignmentConfig(),
        )

        moe = MoEMLP(config)
        moe.init_weights()

        x = torch.randn(2, 16, 256, requires_grad=True)
        output = moe(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
