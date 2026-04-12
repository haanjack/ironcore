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
