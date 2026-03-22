# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU tests for gradient norm and clipping."""

import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ironcore.config import (
    DataConfig,
    InitConfig,
    MainConfig,
    ModelConfig,
    OperationConfig,
    OptimConfig,
    ParallelConfig,
    PEFTConfig,
    ProfilerConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.config.config_moe import MoEConfig
from ironcore.layers.moe import MoEMLP
from ironcore.parallel.expert_parallel import (
    destroy_expert_parallel,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
    initialize_expert_parallel,
)
from ironcore.parallel.parallel_states import (
    destroy_model_parallel,
    get_data_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    initialize_model_parallel,
)
from ironcore.parallel.grad_norm import clip_grad_norm


class SimpleModel(torch.nn.Module):
    """Simple test model."""

    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def setup_distributed():
    """Initialize distributed environment."""
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


# =============================================================================
# DP=2 Tests (Data Parallel)
# =============================================================================


def test_grad_norm_dp2():
    """Test gradient norm with DP=2."""
    rank, world_size = setup_distributed()

    # Initialize with TP=1 (DP=world_size)
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )

    device = torch.device(f"cuda:{rank}")

    # Create model with deterministic weights
    torch.manual_seed(42)
    model = SimpleModel().to(device)

    # Create deterministic input (same across DP ranks)
    torch.manual_seed(42)
    x = torch.randn(4, 10, device=device)
    y = torch.randn(4, 5, device=device)

    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()

    # Compute gradient norm
    max_norm = float("inf")
    norm = clip_grad_norm(model.parameters(), max_norm=max_norm, norm_type=2.0)

    # Verify: All ranks should get same norm value
    norm_value = norm.item()

    # Gather norms from all ranks
    norms = [torch.tensor(0.0, device=device) for _ in range(world_size)]
    dist.all_gather(norms, torch.tensor(norm_value, device=device))

    # All ranks should have identical norm
    for i in range(world_size):
        n = norms[i]
        assert torch.isclose(n, torch.tensor(norm_value, device=device), rtol=1e-5), (
            f"Rank {rank}: Norm mismatch with rank {i}: {n.item()} vs {norm_value}"
        )

    print(f"[Rank {rank}] ✅ DP=2 gradient norm test passed (norm={norm_value:.6f})")

    destroy_model_parallel()
    cleanup_distributed()


def test_grad_norm_dp2_clipping():
    """Test gradient clipping with DP=2."""
    rank, world_size = setup_distributed()

    initialize_model_parallel(
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )

    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42)
    model = SimpleModel().to(device)

    # Create input that produces large gradients
    torch.manual_seed(100)
    x = torch.randn(32, 10, device=device)
    y = torch.randn(32, 5, device=device)

    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y) * 100  # Large loss
    loss.backward()

    # Clip with aggressive max_norm
    max_norm = 0.5
    norm_before = clip_grad_norm(model.parameters(), max_norm=max_norm, norm_type=2.0)

    # Compute actual norm after clipping
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    norm_after = torch.norm(torch.cat([g.flatten() for g in grads]))

    # Clipped norm should be <= max_norm
    assert norm_after.item() <= max_norm + 1e-5, (
        f"Rank {rank}: Clipped norm {norm_after.item()} exceeds max_norm {max_norm}"
    )

    print(f"[Rank {rank}] ✅ DP=2 gradient clipping test passed (before={norm_before.item():.6f}, after={norm_after.item():.6f})")

    destroy_model_parallel()
    cleanup_distributed()


def test_param_norm_dp2():
    """Test parameter norm computation with DP=2."""
    rank, world_size = setup_distributed()

    initialize_model_parallel(
        tensor_model_parallel_size=1,
        timeout_in_minutes=10.0,
    )

    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42)
    model = SimpleModel().to(device)

    # Compute parameter norm (simulating base_trainer logic)
    param_norm_sq = 0.0
    for p in model.parameters():
        if p.data is not None:
            param_norm_sq += p.data.norm() ** 2

    # For DP without FSDP, average across DP group
    dp_size = get_data_parallel_world_size()
    param_norm_tensor = torch.tensor(param_norm_sq, device=device)
    if dp_size > 1:
        dist.all_reduce(param_norm_tensor, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        param_norm_tensor /= dp_size

    param_norm = param_norm_tensor.item() ** 0.5

    # Gather and verify all ranks have same param norm
    norms = [torch.tensor(0.0, device=device) for _ in range(world_size)]
    dist.all_gather(norms, torch.tensor(param_norm, device=device))

    for i in range(world_size):
        n = norms[i]
        assert torch.isclose(n, torch.tensor(param_norm, device=device), rtol=1e-5), (
            f"Rank {rank}: Param norm mismatch with rank {i}: {n.item()} vs {param_norm}"
        )

    print(f"[Rank {rank}] ✅ DP=2 parameter norm test passed (norm={param_norm:.6f})")

    destroy_model_parallel()
    cleanup_distributed()


# =============================================================================
# TP=2 Tests (Tensor Parallel)
# =============================================================================


def test_grad_norm_tp2():
    """Test gradient norm with TP=2."""
    rank, world_size = setup_distributed()

    if world_size < 2:
        print("[Rank 0] Skipping TP=2 test - needs 2 GPUs")
        cleanup_distributed()
        return

    # Initialize with TP=2
    initialize_model_parallel(
        tensor_model_parallel_size=2,
        timeout_in_minutes=10.0,
    )

    device = torch.device(f"cuda:{rank}")
    tp_size = get_tensor_model_parallel_world_size()

    assert tp_size == 2, f"Expected TP size 2, got {tp_size}"

    # Create simple model
    torch.manual_seed(42)
    model = SimpleModel().to(device)

    # Forward/backward
    torch.manual_seed(42)
    x = torch.randn(4, 10, device=device)
    y = torch.randn(4, 5, device=device)

    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()

    # Compute gradient norm
    norm = clip_grad_norm(model.parameters(), max_norm=float("inf"), norm_type=2.0)
    norm_value = norm.item()

    # All ranks should have same norm
    norms = [torch.tensor(0.0, device=device) for _ in range(world_size)]
    dist.all_gather(norms, torch.tensor(norm_value, device=device))

    for i in range(world_size):
        n = norms[i]
        assert torch.isclose(n, torch.tensor(norm_value, device=device), rtol=1e-5), (
            f"Rank {rank}: Norm mismatch with rank {i}: {n.item()} vs {norm_value}"
        )

    print(f"[Rank {rank}] ✅ TP=2 gradient norm test passed (norm={norm_value:.6f})")

    destroy_model_parallel()
    cleanup_distributed()


def test_grad_norm_tp2_inf_norm():
    """Test gradient norm with TP=2 and inf norm."""
    rank, world_size = setup_distributed()

    if world_size < 2:
        print("[Rank 0] Skipping TP=2 inf-norm test - needs 2 GPUs")
        cleanup_distributed()
        return

    initialize_model_parallel(
        tensor_model_parallel_size=2,
        timeout_in_minutes=10.0,
    )

    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42)
    model = SimpleModel().to(device)

    torch.manual_seed(42)
    x = torch.randn(4, 10, device=device)
    y = torch.randn(4, 5, device=device)

    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()

    # Compute inf-norm
    norm = clip_grad_norm(model.parameters(), max_norm=float("inf"), norm_type=float("inf"))
    norm_value = norm.item()

    # After TP all-reduce with MAX, all ranks should have global max
    norms = [torch.tensor(0.0, device=device) for _ in range(world_size)]
    dist.all_gather(norms, torch.tensor(norm_value, device=device))

    for i in range(world_size):
        n = norms[i]
        assert torch.isclose(n, torch.tensor(norm_value, device=device), rtol=1e-5), (
            f"Rank {rank}: Inf-norm mismatch with rank {i}: {n.item()} vs {norm_value}"
        )

    print(f"[Rank {rank}] ✅ TP=2 inf-norm test passed (norm={norm_value:.6f})")

    destroy_model_parallel()
    cleanup_distributed()


# =============================================================================
# EP=2 Tests (Expert Parallel)
# =============================================================================


def test_grad_norm_ep2_moe():
    """Test gradient norm with EP=2 using MoE model."""
    rank, world_size = setup_distributed()

    if world_size < 2:
        print("[Rank 0] Skipping EP=2 MoE test - needs 2 GPUs")
        cleanup_distributed()
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

    device = torch.device(f"cuda:{rank}")
    ep_size = get_expert_model_parallel_world_size()
    assert ep_size == 2

    # Create MoE model
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

    moe = MoEMLP(config).to(device)
    
    # Verify that routed experts are marked and shared experts are not
    for expert in moe.shared_experts:
        for p in expert.parameters():
            assert not getattr(p, "is_expert", False), "Shared expert should NOT be marked is_expert"
    
    for expert in moe.routed_experts:
        for p in expert.parameters():
            assert getattr(p, "is_expert", False), "Routed expert SHOULD be marked is_expert"

    # Forward/backward
    torch.manual_seed(42)
    x = torch.randn(4, 8, 64, device=device)
    y = torch.randn(4, 8, 64, device=device)

    output = moe(x)
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()

    # Compute gradient norm
    norm = clip_grad_norm(moe.parameters(), max_norm=float("inf"), norm_type=2.0)
    norm_value = norm.item()

    assert norm_value > 0
    
    # All ranks should have same norm
    norms = [torch.tensor(0.0, device=device) for _ in range(world_size)]
    dist.all_gather(norms, torch.tensor(norm_value, device=device))

    for i in range(world_size):
        n = norms[i]
        assert torch.isclose(n, torch.tensor(norm_value, device=device), rtol=1e-5), (
            f"Rank {rank}: EP norm mismatch with rank {i}"
        )

    print(f"[Rank {rank}] ✅ EP=2 MoE gradient norm test passed (norm={norm_value:.6f})")

    destroy_expert_parallel()
    destroy_model_parallel()
    cleanup_distributed()


def test_param_norm_ep2_moe():
    """Test parameter norm with EP=2 using MoE model."""
    rank, world_size = setup_distributed()

    if world_size < 2:
        print("[Rank 0] Skipping EP=2 MoE param norm test - needs 2 GPUs")
        cleanup_distributed()
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

    device = torch.device(f"cuda:{rank}")

    # Create MoE model
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

    moe = MoEMLP(config).to(device)

    # Compute parameter norm (simulating base_trainer logic)
    expert_params = [p for p in moe.parameters() if p.data is not None and getattr(p, "is_expert", False)]
    non_expert_params = [p for p in moe.parameters() if p.data is not None and not getattr(p, "is_expert", False)]

    expert_norm_sq = torch.stack([p.data.norm()**2 for p in expert_params]).sum() if expert_params else torch.tensor(0.0, device=device)
    non_expert_norm_sq = torch.stack([p.data.norm()**2 for p in non_expert_params]).sum() if non_expert_params else torch.tensor(0.0, device=device)

    # EP Reduction
    from ironcore.parallel.expert_parallel.parallel_states import get_expert_model_parallel_group
    dist.all_reduce(expert_norm_sq, op=dist.ReduceOp.SUM, group=get_expert_model_parallel_group())

    param_norm = (expert_norm_sq + non_expert_norm_sq).item() ** 0.5

    # All ranks should have same norm
    norms = [torch.tensor(0.0, device=device) for _ in range(world_size)]
    dist.all_gather(norms, torch.tensor(param_norm, device=device))

    for i in range(world_size):
        n = norms[i]
        assert torch.isclose(n, torch.tensor(param_norm, device=device), rtol=1e-5), (
            f"Rank {rank}: EP param norm mismatch with rank {i}"
        )

    print(f"[Rank {rank}] ✅ EP=2 MoE parameter norm test passed (norm={param_norm:.6f})")

    destroy_expert_parallel()
    destroy_model_parallel()
    cleanup_distributed()


# =============================================================================
# FSDP Tests
# =============================================================================


def test_grad_norm_fsdp_dp2():
    """Test FSDP gradient norm with DP=2."""
    rank, world_size = setup_distributed()

    if world_size < 2:
        print("[Rank 0] Skipping FSDP DP=2 test - needs 2 GPUs")
        cleanup_distributed()
        return

    device = torch.device(f"cuda:{rank}")

    # Create model
    torch.manual_seed(42)
    model = SimpleModel().to(device)

    # Wrap with FSDP
    model = FSDP(model)

    # Forward/backward
    torch.manual_seed(42)
    x = torch.randn(4, 10, device=device)
    y = torch.randn(4, 5, device=device)

    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()

    # Compute gradient norm using FSDP's clip_grad_norm_
    max_norm = 1.0
    norm = model.clip_grad_norm_(max_norm)

    # All ranks should have identical norm
    norms = [torch.tensor(0.0, device=device) for _ in range(world_size)]
    dist.all_gather(norms, torch.tensor(norm.item(), device=device))

    for i in range(world_size):
        n = norms[i]
        assert torch.isclose(n, torch.tensor(norm.item(), device=device), rtol=1e-4)

    print(f"[Rank {rank}] ✅ FSDP DP=2 gradient norm test passed (norm={norm.item():.6f})")

    cleanup_distributed()


def main():
    """Run tests based on arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all")
    args = parser.parse_args()

    if args.test == "dp2" or args.test == "all":
        test_grad_norm_dp2()
        test_grad_norm_dp2_clipping()
        test_param_norm_dp2()
    
    if args.test == "tp2" or args.test == "all":
        test_grad_norm_tp2()
        test_grad_norm_tp2_inf_norm()

    if args.test == "ep2" or args.test == "all":
        test_grad_norm_ep2_moe()
        test_param_norm_ep2_moe()

    if args.test == "fsdp" or args.test == "all":
        test_grad_norm_fsdp_dp2()


if __name__ == "__main__":
    main()
