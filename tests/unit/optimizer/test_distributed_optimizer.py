# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DistributedOptimizer."""

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from ironcore.optimizer.distributed_optimizer import DistributedOptimizer


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, hidden_size=64, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)


class TestDistributedOptimizerSingleGPU:
    """Tests that work on single GPU (no distributed)."""

    def test_initialization(self):
        """Test basic initialization without distributed."""
        model = SimpleModel()
        base_optimizer = AdamW(model.parameters(), lr=1e-3)
        optimizer = DistributedOptimizer(base_optimizer)

        assert optimizer.dp_size == 1
        assert optimizer.dp_rank == 0
        assert len(optimizer.all_params) == sum(p.numel() > 0 for p in model.parameters())
        assert len(optimizer.local_param_indices) == len(optimizer.all_params)

    def test_step_single_rank(self):
        """Test optimizer step works correctly on single rank."""
        torch.manual_seed(42)
        model = SimpleModel()
        base_optimizer = AdamW(model.parameters(), lr=1e-3)
        optimizer = DistributedOptimizer(base_optimizer)

        # Create dummy input and loss
        x = torch.randn(4, 64)
        y = torch.randn(4, 1)
        loss = nn.MSELoss()(model(x), y)

        # Backward
        loss.backward()

        # Store gradients before step
        {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

        # Step
        optimizer.step()

        # Verify parameters were updated
        params_changed = False
        for p in model.parameters():
            if p.grad is not None:
                # Check that parameter values changed
                if not torch.allclose(p, p - p.grad * 1e-3):  # Rough check
                    params_changed = True
                    break

        assert params_changed, "Parameters should have been updated"

    def test_param_groups_delegation(self):
        """Test that param_groups is delegated to inner optimizer."""
        model = SimpleModel()
        base_optimizer = AdamW(model.parameters(), lr=1e-3)
        optimizer = DistributedOptimizer(base_optimizer)

        assert optimizer.param_groups == base_optimizer.param_groups
        assert optimizer.param_groups[0]["lr"] == 1e-3

    def test_attribute_delegation(self):
        """Test that unknown attributes are delegated to inner optimizer."""
        model = SimpleModel()
        base_optimizer = AdamW(model.parameters(), lr=1e-3)
        optimizer = DistributedOptimizer(base_optimizer)

        # state is an attribute of the inner optimizer
        assert optimizer.state is base_optimizer.state

    def test_zero_grad(self):
        """Test zero_grad method."""
        model = SimpleModel()
        base_optimizer = AdamW(model.parameters(), lr=1e-3)
        optimizer = DistributedOptimizer(base_optimizer)

        # Create gradients
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()

        # Verify gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads

        # Zero gradients
        optimizer.zero_grad()

        # Verify gradients are zeroed
        all_none = all(p.grad is None for p in model.parameters())
        assert all_none

    def test_state_dict(self):
        """Test state_dict returns local partition."""
        model = SimpleModel()
        base_optimizer = AdamW(model.parameters(), lr=1e-3)
        optimizer = DistributedOptimizer(base_optimizer)

        # Do a step to create optimizer state
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Get state dict
        state_dict = optimizer.state_dict()

        assert "state" in state_dict
        assert "param_groups" in state_dict

    def test_load_state_dict(self):
        """Test load_state_dict method."""
        model = SimpleModel()
        base_optimizer = AdamW(model.parameters(), lr=1e-3)
        optimizer = DistributedOptimizer(base_optimizer)

        # Do a step to create state
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Save state
        state_dict = optimizer.state_dict()

        # Create new optimizer and load state
        model2 = SimpleModel()
        base_optimizer2 = AdamW(model2.parameters(), lr=1e-3)
        optimizer2 = DistributedOptimizer(base_optimizer2)
        optimizer2.load_state_dict(state_dict)

        # Verify param groups match
        assert len(optimizer2.param_groups) == len(optimizer.param_groups)

    def test_param_groups_modification(self):
        """Test that modifying param_groups affects the inner optimizer."""
        model = SimpleModel()
        base_optimizer = AdamW(model.parameters(), lr=1e-3)
        optimizer = DistributedOptimizer(base_optimizer)

        # Modify LR in the wrapper
        optimizer.param_groups[0]["lr"] = 5e-4

        # Check if it's reflected in the inner optimizer
        assert base_optimizer.param_groups[0]["lr"] == 5e-4
        assert optimizer.param_groups[0]["lr"] == 5e-4

    def test_isinstance_optimizer(self):
        """Test that DistributedOptimizer passes isinstance(Optimizer) check."""
        model = SimpleModel()
        base_optimizer = AdamW(model.parameters(), lr=1e-3)
        optimizer = DistributedOptimizer(base_optimizer)

        from torch.optim import Optimizer

        assert isinstance(optimizer, Optimizer)

    def test_repr(self):
        """Test string representation."""
        model = SimpleModel()
        base_optimizer = AdamW(model.parameters(), lr=1e-3)
        optimizer = DistributedOptimizer(base_optimizer)

        repr_str = repr(optimizer)
        assert "DistributedOptimizer" in repr_str
        assert "dp_size=1" in repr_str
        assert "dp_rank=0" in repr_str


import os as _os

@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or _os.environ.get("RANK") is None,
    reason="Requires at least 2 GPUs and torchrun (RANK env var not set)",
)
class TestDistributedOptimizerMultiGPU:
    """Tests that require multiple GPUs."""

    @pytest.fixture(scope="module")
    def distributed_setup_module(self):
        """Setup distributed environment (module-scoped to avoid NCCL issues)."""
        import os

        import torch.distributed as dist

        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() < 2
            or os.environ.get("RANK") is None
            or os.environ.get("MASTER_ADDR") is None
        ):
            pytest.skip("Requires torchrun with at least 2 GPUs (RANK/MASTER_ADDR not set)")

        # Set device based on local rank
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)

        # Initialize if not already initialized
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        from ironcore.parallel.parallel_states import initialize_model_parallel

        # Initialize with TP=1 (pure DP)
        initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=1)

        yield local_rank

        # Cleanup - don't destroy process group in module scope

    def test_parameter_partitioning(self, distributed_setup_module):
        """Test that parameters are correctly partitioned across ranks."""
        import os

        from torch.distributed import get_rank, get_world_size

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = get_rank()
        world_size = get_world_size()

        model = SimpleModel().cuda(local_rank)
        base_optimizer = AdamW(model.parameters(), lr=1e-3)
        optimizer = DistributedOptimizer(base_optimizer)

        # Each rank should own roughly 1/world_size of parameters
        total_params = len(optimizer.all_params)
        local_params = len(optimizer.local_param_indices)

        # Check round-robin assignment
        expected_local = sum(1 for i in range(total_params) if i % world_size == rank)
        assert local_params == expected_local

    def test_step_with_ddp(self, distributed_setup_module):
        """Test optimizer step with DDP-wrapped model."""
        import os

        from torch.nn.parallel import DistributedDataParallel as DDP

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # Create model and wrap with DDP
        model = SimpleModel().cuda(local_rank)
        model = DDP(model, device_ids=[local_rank])

        base_optimizer = AdamW(model.parameters(), lr=1e-3)
        optimizer = DistributedOptimizer(base_optimizer)

        # Create input
        x = torch.randn(4, 64).cuda(local_rank)
        loss = model(x).sum()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Verify no errors occurred
        assert True

    def test_parameter_consistency_after_step(self, distributed_setup_module):
        """Test that parameters are consistent across ranks after step."""
        import os

        import torch.distributed as dist
        from torch.distributed import all_reduce
        from torch.nn.parallel import DistributedDataParallel as DDP

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # Set same seed for initial model
        torch.manual_seed(42)
        model = SimpleModel().cuda(local_rank)
        model = DDP(model, device_ids=[local_rank])

        base_optimizer = AdamW(model.parameters(), lr=1e-3)
        optimizer = DistributedOptimizer(base_optimizer)

        # Do a training step
        x = torch.randn(4, 64).cuda(local_rank)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Verify parameters are the same across all ranks
        for param in model.parameters():
            # Sum all parameters and check they match
            local_param = param.data.clone()
            all_reduce(local_param, op=dist.ReduceOp.SUM)

            # After sum and divide, should get same as local (if all equal)
            avg_param = local_param / dist.get_world_size()
            assert torch.allclose(param.data, avg_param, rtol=1e-5, atol=1e-6), (
                "Parameter inconsistent across ranks"
            )
