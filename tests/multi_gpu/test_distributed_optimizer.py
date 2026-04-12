# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU tests for DistributedOptimizer.

Requires torchrun --nproc_per_node=2.

Usage:
    torchrun --nproc_per_node=2 -m pytest tests/multi_gpu/test_distributed_optimizer.py -v
"""

import os

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


class TestDistributedOptimizerMultiGPU:
    """Tests that require multiple GPUs."""

    @pytest.fixture(scope="module")
    def distributed_setup_module(self):
        """Setup distributed environment (module-scoped to avoid NCCL issues)."""
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

    def test_parameter_partitioning(self, distributed_setup_module):
        """Test that parameters are correctly partitioned across ranks."""
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
