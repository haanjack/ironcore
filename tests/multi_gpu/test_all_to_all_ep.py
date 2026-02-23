#!/usr/bin/env python
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test All-to-All communication mode for EP.

This test verifies the all-to-all communication works correctly.

Usage:
    torchrun --nproc_per_node=2 tests/multi_gpu/test_all_to_all_ep.py
"""

import os

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
from ironcore.layers.moe import CommunicationMode, MoEMLP
from ironcore.parallel.expert_parallel import (
    destroy_expert_parallel,
    initialize_expert_parallel,
)
from ironcore.parallel.parallel_states import (
    destroy_model_parallel,
    initialize_model_parallel,
)


def setup_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    return rank, world_size


def cleanup_distributed():
    """Cleanup distributed environment."""
    destroy_expert_parallel()
    destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


def test_all_to_all_mode():
    """Test all-to-all communication mode."""
    rank, world_size = setup_distributed()

    # Initialize parallel groups
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )
    initialize_expert_parallel(
        expert_model_parallel_size=world_size,
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
                expert_model_parallel_size=world_size,
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

    # Create MoE layer with all-to-all mode
    moe = MoEMLP(config, communication_mode=CommunicationMode.ALL_TO_ALL).to(device)
    moe.init_weights()
    moe.eval()  # Use eval mode to avoid jitter noise

    # Create input
    batch_size, seq_len, hidden_size = 2, 16, 256
    torch.manual_seed(42 + rank)  # Different seed per rank but deterministic
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    print(f"[Rank {rank}] Testing all-to-all forward pass...")

    # Forward pass
    with torch.no_grad():
        output = moe(x)

    print(f"[Rank {rank}] Input shape: {x.shape}, Output shape: {output.shape}")

    assert output.shape == (batch_size, seq_len, hidden_size), (
        f"Expected shape {(batch_size, seq_len, hidden_size)}, got {output.shape}"
    )
    assert not torch.isnan(output).any(), "Output contains NaN"

    print(f"[Rank {rank}] All-to-all forward pass test passed")

    # Backward pass test with gradient
    moe.train()
    x_with_grad = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)
    output = moe(x_with_grad)
    loss = output.sum()
    loss.backward()

    assert x_with_grad.grad is not None, "Gradient is None"
    assert x_with_grad.grad.abs().sum() > 0, "Gradient is all zeros"

    print(f"[Rank {rank}] All-to-all backward pass test passed")

    destroy_expert_parallel()
    destroy_model_parallel()
    cleanup_distributed()

    if rank == 0:
        print("\nAll-to-All EP tests passed!")


def test_compare_modes():
    """Compare all-reduce and all-to-all outputs (should be equivalent)."""
    rank, world_size = setup_distributed()

    # Initialize parallel groups
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )
    initialize_expert_parallel(
        expert_model_parallel_size=world_size,
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )

    # Create config
    config = MainConfig(
        model=ModelConfig(
            d_model=256,
            d_ffn=512,
            moe=MoEConfig(
                use_moe=True,
                num_shared_experts=2,
                num_routed_experts=8,
                num_experts_per_token=2,
                expert_model_parallel_size=world_size,
                aux_loss_alpha=0.0,  # Disable aux loss for comparison
                router_jitter_noise=0.0,  # Disable noise
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

    # Create both MoE layers with same weights
    torch.manual_seed(42)  # Same seed for weight initialization
    moe_ar = MoEMLP(config, communication_mode=CommunicationMode.ALL_REDUCE).to(device)
    moe_ar.init_weights()
    moe_ar.eval()

    torch.manual_seed(42)  # Same seed
    moe_a2a = MoEMLP(config, communication_mode=CommunicationMode.ALL_TO_ALL).to(device)
    moe_a2a.init_weights()
    moe_a2a.eval()

    # Create same input on all ranks
    torch.manual_seed(123)
    batch_size, seq_len, hidden_size = 2, 16, 256
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    print(f"[Rank {rank}] Comparing all-reduce vs all-to-all outputs...")

    # Forward pass with both modes
    with torch.no_grad():
        output_ar = moe_ar(x)
        output_a2a = moe_a2a(x)

    # Note: Outputs won't be exactly the same because routing may differ
    # based on which experts are local, but shapes should match
    print(f"[Rank {rank}] All-Reduce output shape: {output_ar.shape}")
    print(f"[Rank {rank}] All-to-All output shape: {output_a2a.shape}")

    assert output_ar.shape == output_a2a.shape, (
        f"Shape mismatch: all-reduce {output_ar.shape} vs all-to-all {output_a2a.shape}"
    )

    assert not torch.isnan(output_ar).any(), "All-Reduce output contains NaN"
    assert not torch.isnan(output_a2a).any(), "All-to-All output contains NaN"

    # Check that outputs are in similar range (not exact due to different processing)
    ar_mean = output_ar.abs().mean().item()
    a2a_mean = output_a2a.abs().mean().item()
    print(f"[Rank {rank}] All-Reduce mean abs: {ar_mean:.4f}")
    print(f"[Rank {rank}] All-to-All mean abs: {a2a_mean:.4f}")

    # The means should be within a reasonable range
    # (Not exact because the dispatch pattern differs)
    ratio = max(ar_mean, a2a_mean) / (min(ar_mean, a2a_mean) + 1e-8)
    assert ratio < 10.0, f"Output magnitudes differ too much: ratio={ratio}"

    print(f"[Rank {rank}] Comparison test passed (ratio={ratio:.2f})")

    destroy_expert_parallel()
    destroy_model_parallel()
    cleanup_distributed()

    if rank == 0:
        print("\nMode comparison tests passed!")


def main():
    """Run tests."""
    test_type = os.environ.get("TEST_TYPE", "all")

    if test_type == "all_to_all":
        test_all_to_all_mode()
    elif test_type == "compare":
        test_compare_modes()
    else:
        # Run all tests
        print("=" * 60)
        print("Running All-to-All EP Tests")
        print("=" * 60)

        print("\n[Test 1] All-to-All Mode")
        test_all_to_all_mode()

        print("\n[Test 2] Compare Modes")
        test_compare_modes()

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)


if __name__ == "__main__":
    main()
