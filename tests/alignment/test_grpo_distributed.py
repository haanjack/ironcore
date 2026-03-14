#!/usr/bin/env python
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Distributed unit tests for GRPO advantage computation.

Tests the all-gather synchronization of rewards before group normalization.

Usage:
    torchrun --nproc_per_node=2 tests/alignment/test_grpo_distributed.py
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed process group and model parallel states."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

    # Initialize model parallel states for ironcore
    from ironcore.parallel.parallel_states import initialize_model_parallel

    # Use default: 1 tensor parallel group, world_size data parallel groups
    initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=30)


def cleanup_distributed():
    """Cleanup distributed process group."""
    dist.destroy_process_group()


def get_device():
    """Get CUDA device for current rank."""
    return torch.device(f"cuda:{dist.get_rank()}")


@pytest.mark.skip(
    reason="Run with torchrun --nproc_per_node=2 tests/alignment/test_grpo_distributed.py"
)
def test_distributed_advantage_allgather():
    """
    Test that advantage computation correctly all-gathers rewards across ranks.

    Setup:
    - 2 ranks, batch_size=2, group_size=4 (8 samples total)
    - Each rank has 4 samples: two from each group
    - Rewards distributed across ranks

    Expected:
    - After all-gather, each rank should have all 8 rewards
    - Advantages should be computed correctly (sum=0, std=1 within groups)
    """
    from ironcore.alignment.loss.grpo import compute_advantages

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = get_device()

    # Each rank has samples from all groups (sharded by sample, not by group)
    # Rank 0: samples [0, 2, 4, 6] (even indices)
    # Rank 1: samples [1, 3, 5, 7] (odd indices)

    batch_size = 2
    group_size = 4
    total_samples = batch_size * group_size

    # Create local rewards ON CUDA
    # Group 0: [0.0, 1.0, 2.0, 3.0]
    # Group 1: [4.0, 5.0, 6.0, 7.0]
    all_rewards = torch.arange(total_samples, dtype=torch.float32, device=device)
    all_group_ids = (
        torch.arange(batch_size, device=device)
        .unsqueeze(1)
        .expand(batch_size, group_size)
        .reshape(-1)
    )

    # Each rank gets its slice (strided by world_size)
    local_rewards = all_rewards[rank::world_size].clone()
    local_group_ids = all_group_ids[rank::world_size].clone()

    print(f"[Rank {rank}] Local rewards: {local_rewards.tolist()}")
    print(f"[Rank {rank}] Local group_ids: {local_group_ids.tolist()}")

    # Compute advantages (with distributed=True)
    advantages = compute_advantages(local_rewards, local_group_ids, distributed=True)

    print(f"[Rank {rank}] Advantages: {advantages.tolist()}")

    # Expected advantages (manually computed from full rewards)
    # Group 0: [0,1,2,3], mean=1.5, std≈1.29
    #   (0-1.5)/1.29 = -1.16, (1-1.5)/1.29 = -0.39, (2-1.5)/1.29 = 0.39, (3-1.5)/1.29 = 1.16
    # Group 1: [4,5,6,7], mean=5.5, std≈1.29
    #   Same normalized values: -1.16, -0.39, 0.39, 1.16

    # Verify sum of advantages within groups is 0
    # Gather all advantages to rank 0 for verification
    gathered_advantages = [torch.zeros_like(advantages) for _ in range(world_size)]
    dist.all_gather(gathered_advantages, advantages)

    if rank == 0:
        # Reconstruct full advantages in original order
        full_advantages = torch.zeros(total_samples, device=device)
        for r, adv in enumerate(gathered_advantages):
            full_advantages[r::world_size] = adv

        print(f"\n[Rank 0] Full advantages: {full_advantages.tolist()}")

        # Check sum within each group is 0
        for g in range(batch_size):
            group_mask = all_group_ids == g
            group_advantages = full_advantages[group_mask]
            group_sum = group_advantages.sum().item()
            print(f"  Group {g}: sum={group_sum:.6f}")

            assert abs(group_sum) < 1e-5, f"Group {g} advantage sum should be 0, got {group_sum}"

        # Check std within each group is 1
        for g in range(batch_size):
            group_mask = all_group_ids == g
            group_advantages = full_advantages[group_mask]
            group_std = group_advantages.std().item()
            print(f"  Group {g}: std={group_std:.6f}")

            assert abs(group_std - 1.0) < 1e-4, (
                f"Group {g} advantage std should be 1, got {group_std}"
            )

        print("\n✓ PASS: All distributed advantage tests passed!")

    dist.barrier()


@pytest.mark.skip(
    reason="Run with torchrun --nproc_per_node=2 tests/alignment/test_grpo_distributed.py"
)
def test_distributed_identical_rewards():
    """
    Test that identical rewards produce exactly 0 advantage across ranks.
    """
    from ironcore.alignment.loss.grpo import compute_advantages

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = get_device()

    # All rewards are identical within groups - ON CUDA
    all_rewards = torch.tensor([5.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0], device=device)
    all_group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device)

    local_rewards = all_rewards[rank::world_size].clone()
    local_group_ids = all_group_ids[rank::world_size].clone()

    advantages = compute_advantages(local_rewards, local_group_ids, distributed=True)

    # All advantages should be exactly 0
    assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-8), (
        f"[Rank {rank}] Identical rewards should produce 0 advantage, got {advantages}"
    )

    if rank == 0:
        print("✓ PASS: Identical rewards produce 0 advantage across all ranks!")

    dist.barrier()


@pytest.mark.skip(
    reason="Run with torchrun --nproc_per_node=2 tests/alignment/test_grpo_distributed.py"
)
def test_distributed_partial_group():
    """
    Test advantage computation when groups are split across ranks.

    This tests the critical case where not all samples of a group
    are on the same rank.
    """
    from ironcore.alignment.loss.grpo import compute_advantages

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = get_device()

    # 1 group split across 2 ranks
    batch_size = 1
    group_size = 4
    total_samples = batch_size * group_size

    # Single group: rewards [1, 2, 3, 4] - ON CUDA
    all_rewards = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    all_group_ids = torch.tensor([0, 0, 0, 0], device=device)

    local_rewards = all_rewards[rank::world_size].clone()
    local_group_ids = all_group_ids[rank::world_size].clone()

    print(f"[Rank {rank}] Local rewards: {local_rewards.tolist()}")

    advantages = compute_advantages(local_rewards, local_group_ids, distributed=True)

    print(f"[Rank {rank}] Advantages: {advantages.tolist()}")

    # Gather and verify
    gathered_advantages = [torch.zeros_like(advantages) for _ in range(world_size)]
    dist.all_gather(gathered_advantages, advantages)

    if rank == 0:
        full_advantages = torch.zeros(total_samples, device=device)
        for r, adv in enumerate(gathered_advantages):
            full_advantages[r::world_size] = adv

        # Sum should be 0
        assert abs(full_advantages.sum().item()) < 1e-5, (
            f"Sum should be 0, got {full_advantages.sum()}"
        )
        # Std should be 1
        assert abs(full_advantages.std().item() - 1.0) < 1e-4, (
            f"Std should be 1, got {full_advantages.std()}"
        )

        print("✓ PASS: Partial group split across ranks works correctly!")

    dist.barrier()


def main():
    """Run all distributed tests."""
    try:
        setup_distributed()

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if rank == 0:
            print("=" * 60)
            print("GRPO Distributed Advantage Tests")
            print(f"World size: {world_size}")
            print("=" * 60)

        print(f"\n[Rank {rank}] Running test_distributed_advantage_allgather...")
        test_distributed_advantage_allgather()

        print(f"\n[Rank {rank}] Running test_distributed_identical_rewards...")
        test_distributed_identical_rewards()

        print(f"\n[Rank {rank}] Running test_distributed_partial_group...")
        test_distributed_partial_group()

        if rank == 0:
            print("\n" + "=" * 60)
            print("ALL DISTRIBUTED TESTS PASSED ✓")
            print("=" * 60)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
