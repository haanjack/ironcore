# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU TP=2 equivalence tests.

These directly validate the two fixes for the previously-reported TP=1 vs TP=2
training divergence (loss diff ~5e-3):

1. Weight init (ironcore/layers/module.py `_init_tp_weight`): gathering the TP=2
   shards must reconstruct the exact same full tensor that the deterministic
   per-parameter-seed formula produces analytically (equivalent to what a TP=1
   run would have initialized).
2. Muon optimizer (ironcore/optimizer/muon.py `_orthogonalize_tp_aware`): under
   TP=2, orthogonalizing a param must gather the full matrix across the TP
   group, run Newton-Schulz once, and re-shard — matching a TP=1 run that
   orthogonalizes the full matrix directly.

Run with:
    torchrun --nproc_per_node=2 tests/multi_gpu/test_tp_equivalence.py
"""

import zlib

import pytest
import torch
import torch.distributed as dist

import ironcore.layers  # noqa: F401  (import before tensor_parallel to avoid a circular import)
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
from ironcore.layers.module import BaseModule
from ironcore.optimizer.muon import _orthogonalize_tp_aware, zeropower_via_newtonschulz5
from ironcore.parallel.parallel_states import (
    destroy_model_parallel,
    get_tensor_model_parallel_world_size,
    initialize_model_parallel,
)
from ironcore.parallel.tensor_parallel import ColumnParallelLinear

pytestmark = pytest.mark.mp


def _make_config(seed: int, tp_size: int) -> MainConfig:
    return MainConfig(
        model=ModelConfig(d_model=64, num_layers=2),
        init=InitConfig(seed=seed, init_std=0.02, xavier_init=False),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        trainer=TrainerConfig(tensor_model_parallel_size=tp_size),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
        peft=PEFTConfig(),
    )


class _OneLayerModule(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.linear_q = ColumnParallelLinear(config, input_size=64, output_size=64)


def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_distributed():
    destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


def test_tp2_init_matches_analytic_full_tensor():
    """Gathering TP=2 shards must reconstruct the deterministic seeded full tensor."""
    rank, world_size = setup_distributed()
    if world_size < 2:
        print("[Rank 0] Skipping — needs 2 GPUs")
        return

    initialize_model_parallel(tensor_model_parallel_size=2, timeout_in_minutes=10.0)
    assert get_tensor_model_parallel_world_size() == 2

    device = torch.device(f"cuda:{rank}")
    config = _make_config(seed=4242, tp_size=2)

    module = _OneLayerModule(config).to(device)
    module.init_weights()

    local_shard = module.linear_q.weight.detach().clone()  # [64, 32] (sharded on dim 1)

    gathered = [torch.zeros_like(local_shard) for _ in range(world_size)]
    dist.all_gather(gathered, local_shard)
    reconstructed_full = torch.cat(gathered, dim=1)  # [64, 64]

    # Independently recompute the expected full tensor via the same
    # per-parameter-seed formula used by BaseModule._init_tp_weight.
    param_name = "linear_q.weight"
    seed = (config.init.seed + zlib.crc32(param_name.encode("utf-8"))) % (2**63)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    expected_full = torch.empty(64, 64, dtype=local_shard.dtype, device=device)
    torch.nn.init.normal_(expected_full, std=config.init.init_std, mean=0.0, generator=generator)

    assert torch.allclose(reconstructed_full, expected_full, atol=1e-6), (
        f"Rank {rank}: TP=2 gathered init does not match the analytic TP=1-equivalent "
        "full tensor — the per-parameter seed derivation regressed."
    )

    print(f"[Rank {rank}] TP=2 init equivalence test passed")
    destroy_model_parallel()


def test_tp2_muon_orthogonalization_matches_full_matrix():
    """Muon's TP-aware Newton-Schulz must match orthogonalizing the full matrix directly."""
    rank, world_size = setup_distributed()
    if world_size < 2:
        print("[Rank 0] Skipping — needs 2 GPUs")
        return

    initialize_model_parallel(tensor_model_parallel_size=2, timeout_in_minutes=10.0)
    device = torch.device(f"cuda:{rank}")

    # Same full matrix ([64, 64]) materialized identically on both ranks.
    torch.manual_seed(31337)
    full_grad = torch.randn(64, 64, device=device)

    shard_size = 32
    start = rank * shard_size
    local_shard = full_grad[:, start : start + shard_size].contiguous()

    fake_param = torch.empty_like(local_shard)
    fake_param.is_tp_sharded = True
    fake_param.tp_shard_dim = 1
    fake_param.tp_concatenated_weights = 1

    resharded, full_shape = _orthogonalize_tp_aware(local_shard, fake_param, newton_schulz_steps=5)
    assert full_shape == (64, 64)

    gathered = [torch.zeros_like(resharded) for _ in range(world_size)]
    dist.all_gather(gathered, resharded.contiguous())
    reconstructed_full = torch.cat(gathered, dim=1)

    expected_full = zeropower_via_newtonschulz5(full_grad, steps=5)

    assert torch.allclose(reconstructed_full, expected_full, atol=1e-4), (
        f"Rank {rank}: TP=2 Muon orthogonalization diverges from the TP=1-equivalent "
        "full-matrix Newton-Schulz result — orthogonalizing local shards independently "
        "is not equivalent to orthogonalizing the gathered full matrix."
    )

    print(f"[Rank {rank}] TP=2 Muon orthogonalization equivalence test passed")
    destroy_model_parallel()


def main():
    setup_distributed()
    try:
        test_tp2_init_matches_analytic_full_tensor()
        test_tp2_muon_orthogonalization_matches_full_matrix()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
