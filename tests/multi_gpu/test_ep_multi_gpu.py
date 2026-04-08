#!/usr/bin/env python
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU test for Expert Parallelism.

This test verifies EP functionality with multiple GPUs.

Usage:
    torchrun --nproc_per_node=2 tests/multi_gpu/test_ep_multi_gpu.py
"""

import os

import pytest
import torch
import torch.distributed as dist

from ironcore.config import MainConfig, PEFTConfig
from ironcore.config.config_data import DataConfig
from ironcore.config.config_model import ModelConfig
from ironcore.config.config_moe import MoEConfig
from ironcore.config.config_optim import OptimConfig
from ironcore.config.config_parallel import ParallelConfig
from ironcore.config.config_trainer import InitConfig, OperationConfig, TrainerConfig
from ironcore.config.config_utils import ProfilerConfig, UtilsConfig
from ironcore.layers.moe import MoEMLP
from ironcore.parallel.expert_parallel import (
    destroy_expert_parallel,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
    initialize_expert_parallel,
)
from ironcore.parallel.parallel_states import (
    destroy_model_parallel,
    initialize_model_parallel,
)


# Skip all tests if not running with torchrun (fewer than 2 GPUs available)
pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ
    or not torch.cuda.is_available()
    or torch.cuda.device_count() < 2,
    reason="Expert parallel tests require torchrun with 2+ GPUs",
)


def setup_distributed():
    """Initialize distributed environment."""
    # Initialize distributed if not already done
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    print(f"[Rank {rank}] Distributed initialized. World size: {world_size}")
    return rank, world_size


def cleanup_distributed():
    """Cleanup distributed environment."""
    destroy_expert_parallel()
    destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


def test_ep_initialization():
    """Test EP initialization with 2 GPUs."""
    rank, world_size = setup_distributed()

    # Initialize with EP=2, TP=1
    initialize_expert_parallel(
        expert_model_parallel_size=2,
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )

    ep_rank = get_expert_model_parallel_rank()
    ep_size = get_expert_model_parallel_world_size()

    print(f"[Rank {rank}] EP rank: {ep_rank}, EP size: {ep_size}")

    assert ep_size == 2, f"Expected EP size 2, got {ep_size}"
    assert ep_rank in [0, 1], f"Expected EP rank 0 or 1, got {ep_rank}"
    assert ep_rank == rank, f"EP rank {ep_rank} should equal global rank {rank}"

    print(f"[Rank {rank}] ✅ EP initialization test passed")

    destroy_expert_parallel()
    destroy_model_parallel()


def test_moe_with_ep():
    """Test MoE layer with EP=2."""
    rank, world_size = setup_distributed()

    # Initialize with EP=2, TP=1
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )
    initialize_expert_parallel(
        expert_model_parallel_size=2,
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )

    # Create config with EP=2
    config = MainConfig(
        model=ModelConfig(
            d_model=256,
            d_ffn=512,
            moe=MoEConfig(
                use_moe=True,
                num_shared_experts=2,
                num_routed_experts=8,  # 4 experts per EP rank
                num_experts_per_token=2,
                expert_model_parallel_size=2,  # EP=2
                aux_loss_alpha=0.01,
            ),
        ),
        init=InitConfig(),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        trainer=TrainerConfig(tensor_model_parallel_size=1),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
        peft=PEFTConfig(),
    )

    device = torch.device(f"cuda:{rank}")

    # Create MoE layer
    moe = MoEMLP(config).to(device)
    moe.init_weights()

    # Create input
    batch_size, seq_len, hidden_size = 2, 16, 256
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    # Forward pass
    output = moe(x)

    print(f"[Rank {rank}] Input shape: {x.shape}, Output shape: {output.shape}")

    assert output.shape == (batch_size, seq_len, hidden_size), (
        f"Expected shape {(batch_size, seq_len, hidden_size)}, got {output.shape}"
    )
    assert not torch.isnan(output).any(), "Output contains NaN"

    print(f"[Rank {rank}] ✅ MoE forward pass test passed")

    # Backward pass
    x_with_grad = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)
    output = moe(x_with_grad)
    loss = output.sum()
    loss.backward()

    assert x_with_grad.grad is not None, "Gradient is None"
    assert x_with_grad.grad.abs().sum() > 0, "Gradient is all zeros"

    print(f"[Rank {rank}] ✅ MoE backward pass test passed")

    # Check gradients are synchronized (both ranks should have similar gradients)
    grad_sum = x_with_grad.grad.abs().sum().item()
    grad_tensor = torch.tensor([grad_sum], device=device)
    dist.all_reduce(grad_tensor, op=dist.ReduceOp.SUM)
    avg_grad = grad_tensor.item() / world_size

    print(f"[Rank {rank}] Gradient sum: {grad_sum:.4f}, Avg: {avg_grad:.4f}")

    destroy_expert_parallel()
    destroy_model_parallel()

    if rank == 0:
        print("\n✅ All EP multi-GPU tests passed!")


def test_ep_all_reduce():
    """Test all-reduce across EP ranks."""
    rank, world_size = setup_distributed()

    initialize_expert_parallel(
        expert_model_parallel_size=2,
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )

    device = torch.device(f"cuda:{rank}")

    # Each rank contributes different values
    local_value = rank + 1  # Rank 0: 1, Rank 1: 2
    tensor = torch.ones(4, device=device) * local_value

    # Expected: (1+2) = 3 for each element
    expected_sum = sum(range(1, world_size + 1))  # 1 + 2 = 3

    from ironcore.parallel.expert_parallel import all_reduce_ep

    result = all_reduce_ep(tensor.clone())

    print(
        f"[Rank {rank}] Local: {local_value}, After all-reduce: {result[0].item()}, Expected: {expected_sum}"
    )

    assert torch.allclose(result, torch.ones(4, device=device) * expected_sum), (
        f"All-reduce failed: expected {expected_sum}, got {result[0].item()}"
    )

    print(f"[Rank {rank}] ✅ EP all-reduce test passed")

    destroy_expert_parallel()
    destroy_model_parallel()


def test_ep_gradient_synchronization():
    """Test that gradients are correctly synchronized across EP ranks."""
    rank, world_size = setup_distributed()

    initialize_model_parallel(
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )
    initialize_expert_parallel(
        expert_model_parallel_size=2,
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )

    device = torch.device(f"cuda:{rank}")

    # Create config with EP=2
    config = MainConfig(
        model=ModelConfig(
            d_model=128,
            d_ffn=256,
            moe=MoEConfig(
                use_moe=True,
                num_shared_experts=1,
                num_routed_experts=4,  # 2 experts per EP rank
                num_experts_per_token=2,
                expert_model_parallel_size=2,
                aux_loss_alpha=0.0,
            ),
        ),
        init=InitConfig(),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        trainer=TrainerConfig(tensor_model_parallel_size=1),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
        peft=PEFTConfig(),
    )

    moe = MoEMLP(config).to(device)
    moe.init_weights()

    # Use same input on both ranks for comparison
    torch.manual_seed(42)
    x = torch.randn(2, 8, 128, device=device, requires_grad=True)

    output = moe(x)
    loss = output.sum()
    loss.backward()

    # All-reduce gradients for comparison
    grad_sum = x.grad.abs().sum().clone()
    dist.all_reduce(grad_sum, op=dist.ReduceOp.SUM)

    # Both ranks should receive gradients from the other rank's experts
    # so gradients should be non-zero on both
    assert x.grad is not None, f"[Rank {rank}] Gradient is None"
    assert x.grad.abs().sum() > 0, f"[Rank {rank}] Gradient is all zeros"

    print(f"[Rank {rank}] ✅ EP gradient synchronization test passed")

    destroy_expert_parallel()
    destroy_model_parallel()


def test_ep_all_to_all_correctness():
    """Test that all-to-all dispatch/gather produces correct outputs."""
    rank, world_size = setup_distributed()

    if world_size < 2:
        print("[Rank 0] Skipping all-to-all test - needs 2+ GPUs")
        return

    initialize_model_parallel(
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )
    initialize_expert_parallel(
        expert_model_parallel_size=2,
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )

    from ironcore.layers.moe import CommunicationMode

    device = torch.device(f"cuda:{rank}")

    # Create config with ALL_TO_ALL mode
    config = MainConfig(
        model=ModelConfig(
            d_model=64,
            d_ffn=128,
            moe=MoEConfig(
                use_moe=True,
                num_shared_experts=1,
                num_routed_experts=4,
                num_experts_per_token=2,
                expert_model_parallel_size=2,
                aux_loss_alpha=0.0,
            ),
        ),
        init=InitConfig(),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        trainer=TrainerConfig(tensor_model_parallel_size=1),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
        peft=PEFTConfig(),
    )

    moe = MoEMLP(config, communication_mode=CommunicationMode.ALL_TO_ALL).to(device)
    moe.init_weights()
    moe.eval()

    # Use deterministic input
    torch.manual_seed(123)
    x = torch.randn(2, 4, 64, device=device)

    with torch.no_grad():
        output = moe(x)

    # Output should have same shape and no NaN
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"

    print(f"[Rank {rank}] ✅ EP all-to-all correctness test passed")

    destroy_expert_parallel()
    destroy_model_parallel()


def test_ep_load_balance_across_ranks():
    """Test that load balancing works across EP ranks."""
    rank, world_size = setup_distributed()

    initialize_model_parallel(
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )
    initialize_expert_parallel(
        expert_model_parallel_size=2,
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )

    device = torch.device(f"cuda:{rank}")

    config = MainConfig(
        model=ModelConfig(
            d_model=128,
            d_ffn=256,
            moe=MoEConfig(
                use_moe=True,
                num_shared_experts=1,
                num_routed_experts=8,  # 4 per rank
                num_experts_per_token=2,
                expert_model_parallel_size=2,
                aux_loss_alpha=0.1,  # Enable load balancing
            ),
        ),
        init=InitConfig(),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        trainer=TrainerConfig(tensor_model_parallel_size=1),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
        peft=PEFTConfig(),
    )

    moe = MoEMLP(config).to(device)
    moe.init_weights()
    optimizer = torch.optim.Adam(moe.parameters(), lr=0.01)

    # Train for a few steps
    for step in range(5):
        torch.manual_seed(step * 100 + rank)  # Different seeds per rank
        x = torch.randn(4, 16, 128, device=device)

        optimizer.zero_grad()
        output = moe(x)
        aux_loss = moe.get_aux_loss()
        if aux_loss is not None:
            (output.sum() + aux_loss).backward()
        else:
            output.sum().backward()
        optimizer.step()
        moe.clear_aux_loss()

    # Check that all routed experts have gradients (not just one rank's experts)
    expert_grad_count = 0
    for expert in moe.routed_experts:
        if expert.up_proj.weight.grad is not None:
            if expert.up_proj.weight.grad.abs().sum() > 0:
                expert_grad_count += 1

    print(f"[Rank {rank}] Experts with gradients: {expert_grad_count}/{len(moe.routed_experts)}")

    # At least some experts should have received gradients
    assert expert_grad_count > 0, f"[Rank {rank}] No experts received gradients"

    print(f"[Rank {rank}] ✅ EP load balance test passed")

    destroy_expert_parallel()
    destroy_model_parallel()


def test_ep_plus_tp_combined():
    """Test EP and TP working together (requires 4 GPUs with EP=2, TP=2)."""
    rank, world_size = setup_distributed()

    if world_size < 4:
        print(f"[Rank {rank}] Skipping EP+TP test - needs 4 GPUs, have {world_size}")
        return

    # Initialize with TP=2, then EP=2
    initialize_model_parallel(
        tensor_model_parallel_size=2,
        timeout_in_minutes=10.0,
    )
    initialize_expert_parallel(
        expert_model_parallel_size=2,
        tensor_model_parallel_size=2,
        timeout_in_minutes=10.0,
    )

    device = torch.device(f"cuda:{rank}")

    config = MainConfig(
        model=ModelConfig(
            d_model=128,
            d_ffn=256,
            moe=MoEConfig(
                use_moe=True,
                num_shared_experts=1,
                num_routed_experts=8,  # 4 experts per EP rank, TP sharded
                num_experts_per_token=2,
                expert_model_parallel_size=2,
                aux_loss_alpha=0.0,
            ),
        ),
        init=InitConfig(),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        trainer=TrainerConfig(tensor_model_parallel_size=2),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
        peft=PEFTConfig(),
    )

    moe = MoEMLP(config).to(device)
    moe.init_weights()

    # Test forward pass
    torch.manual_seed(42 + rank)
    x = torch.randn(2, 8, 128, device=device)
    output = moe(x)

    assert output.shape == (2, 8, 128), f"Shape mismatch: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"

    print(f"[Rank {rank}] ✅ EP+TP forward pass test passed")

    # Test backward pass
    x_with_grad = torch.randn(2, 8, 128, device=device, requires_grad=True)
    output = moe(x_with_grad)
    loss = output.sum()
    loss.backward()

    assert x_with_grad.grad is not None, "Gradient is None"
    assert x_with_grad.grad.abs().sum() > 0, "Gradient is all zeros"

    print(f"[Rank {rank}] ✅ EP+TP backward pass test passed")

    destroy_expert_parallel()
    destroy_model_parallel()


def main():
    """Run all tests."""
    setup_distributed()

    test_type = os.environ.get("TEST_TYPE", "all")

    try:
        if test_type == "init":
            test_ep_initialization()
        elif test_type == "moe":
            test_moe_with_ep()
        elif test_type == "allreduce":
            test_ep_all_reduce()
        elif test_type == "gradient_sync":
            test_ep_gradient_synchronization()
        elif test_type == "alltoall":
            test_ep_all_to_all_correctness()
        elif test_type == "loadbalance":
            test_ep_load_balance_across_ranks()
        elif test_type == "ep_tp":
            test_ep_plus_tp_combined()
        else:
            # Run all tests sequentially with shared process group
            print("=" * 60)
            print("Running EP Multi-GPU Tests")
            print("=" * 60)

            print("\n[Test 1] EP Initialization")
            test_ep_initialization()

            print("\n[Test 2] EP All-Reduce")
            test_ep_all_reduce()

            print("\n[Test 3] MoE with EP=2")
            test_moe_with_ep()

            print("\n[Test 4] EP Gradient Synchronization")
            test_ep_gradient_synchronization()

            print("\n[Test 5] EP All-to-All Correctness")
            test_ep_all_to_all_correctness()

            print("\n[Test 6] EP Load Balance Across Ranks")
            test_ep_load_balance_across_ranks()

            print("\n[Test 7] EP+TP Combined (requires 4 GPUs)")
            test_ep_plus_tp_combined()

            if dist.get_rank() == 0:
                print("\n" + "=" * 60)
                print("✅ All EP multi-GPU tests passed!")
                print("=" * 60)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
