# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Guard CP contracts and compare its reference attention with an independent oracle."""

import pytest
import torch
from tests.fixtures.config_fixtures import create_small_test_config

from ironcore.config.config_context_parallel import validate_context_parallel
from ironcore.layers.context_parallel_attention import ContextParallelAttention
from ironcore.parallel import parallel_states as ps
from ironcore.parallel.context_parallel import partition_context_inputs


@pytest.mark.parametrize("heads,kv", [(4, 4), (4, 2), (4, 1), (9, 3)])
def test_reference_attention_value_and_gradient(monkeypatch, heads, kv):
    monkeypatch.setattr(ps, "_CONTEXT_PARALLEL_WORLD_SIZE", 1)
    torch.manual_seed(91)
    q = torch.randn(2, 7, heads, 8, requires_grad=True)
    k = torch.randn(2, 7, kv, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    oracle_inputs = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    oq, ok, ov = oracle_inputs
    ok, ov = (t.repeat_interleave(heads // kv, dim=2) for t in (ok, ov))
    scores = torch.einsum("bqhd,bkhd->bhqk", oq, ok) / 8**0.5
    mask = torch.ones(7, 7, dtype=torch.bool).tril()
    oracle = torch.einsum(
        "bhqk,bkhd->bqhd", scores.masked_fill(~mask, -torch.inf).softmax(-1), ov
    ).flatten(2)
    actual = ContextParallelAttention("sdpa")(q, k, v)
    gradient = torch.randn_like(actual)
    oracle.backward(gradient)
    actual.backward(gradient)
    torch.testing.assert_close(actual, oracle, atol=1e-6, rtol=1e-5)
    for a, b in zip((q, k, v), oracle_inputs, strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_partition_preserves_global_positions_and_shifted_labels(monkeypatch, rank):
    monkeypatch.setattr(ps, "_CONTEXT_PARALLEL_WORLD_SIZE", 3)
    monkeypatch.setattr(ps, "get_context_parallel_rank", lambda: rank)
    inputs = torch.tensor([[5, 6, 7, 8, 9]])
    labels = torch.tensor([[6, 7, -100, 9, 10]])
    positions = torch.arange(20, 25)[None]
    ids, targets, local_positions, length = partition_context_inputs(inputs, labels, positions)
    assert length == 5
    assert ids.tolist() == [[5, 6], [7, 8], [9, 0]][rank : rank + 1]
    assert targets.tolist() == [[6, 7], [-100, 9], [10, -100]][rank : rank + 1]
    assert local_positions.tolist() == [[20, 21], [22, 23], [24, 24]][rank : rank + 1]


@pytest.mark.parametrize("cp", [0, -1, True, 1.5])
def test_invalid_cp_size(cp):
    config = create_small_test_config()
    config.trainer.context_parallel_size = cp
    with pytest.raises(ValueError, match="positive integer"):
        validate_context_parallel(config)


@pytest.mark.parametrize(
    "path,value,match",
    [
        ("parallel.world_size", 3, "divisible"),
        ("trainer.context_parallel_backend", "bad", "backend"),
        ("model.moe.expert_model_parallel_size", 4, "EP=2"),
        ("parallel.use_fsdp", True, "FSDP"),
        ("parallel.use_distributed_optimizer", True, "optimizer"),
        ("offload.enabled", True, "offload"),
        ("data.task_type", "sft", "pretrain"),
        ("model.reset_attention_mask", True, "packed/reset"),
        ("model.reset_position_ids", True, "packed/reset"),
        ("model.kv_cache.use_paged", True, "paged"),
        ("model.dropout_mlp", 0.1, "dropout"),
        ("init.data_parallel_random_init", True, "initialization"),
    ],
)
def test_unsupported_contract_rejected(path, value, match):
    config = create_small_test_config()
    config.trainer.context_parallel_size = 2
    config.trainer.context_parallel_backend = "sdpa"
    config.model.reset_attention_mask = False
    config.model.reset_position_ids = False
    config.parallel.world_size = 2
    config.model.moe.use_moe = True
    owner = config
    fields = path.split(".")
    for field in fields[:-1]:
        owner = getattr(owner, field)
    setattr(owner, fields[-1], value)
    with pytest.raises(ValueError, match=match):
        validate_context_parallel(config)


def test_pretrain_ignores_unused_alignment_defaults():
    config = create_small_test_config()
    config.trainer.context_parallel_size = 2
    config.trainer.context_parallel_backend = "sdpa"
    config.model.reset_attention_mask = False
    config.model.reset_position_ids = False
    config.parallel.world_size = 2
    validate_context_parallel(config)


@pytest.mark.parametrize("world,tp,cp", [(8, 2, 2), (4, 1, 2), (4, 2, 1), (6, 1, 3)])
def test_parallel_mesh_axes_are_orthogonal(world, tp, cp):
    groups = ps.parallel_group_ranks(world, tp, cp)
    for axis in groups.values():
        assert sorted(rank for group in axis for rank in group) == list(range(world))
    for rank in range(world):
        containing = [
            next(set(group) for group in groups[axis] if rank in group)
            for axis in ("tp", "cp", "dp")
        ]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            assert containing[a] & containing[b] == {rank}
    assert groups["cp"][0] == list(range(0, tp * cp, tp))


def test_ring_rejects_fp32_compute_before_model_allocation():
    config = create_small_test_config()
    config.trainer.context_parallel_size = 2
    config.parallel.world_size = 2
    with pytest.raises(ValueError, match="FP16/BF16"):
        validate_context_parallel(config)


def test_cp_precision_is_restored_when_groups_are_destroyed(monkeypatch):
    matmul = torch.backends.cuda.matmul
    previous = (
        torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
        torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
    )
    monkeypatch.setattr(ps, "_CONTEXT_PARALLEL_ORIGINAL_PRECISION", None)
    try:
        # Newer backends return (allow_reduction, allow_splitk). Preserve both,
        # including a caller who deliberately disables split-K reductions.
        if isinstance(previous[0], tuple):
            matmul.allow_fp16_reduced_precision_reduction = (False, False)
            matmul.allow_bf16_reduced_precision_reduction = (False, False)
        expected = (
            torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
            torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
        )
        ps.configure_context_parallel_precision()
        assert not matmul.allow_fp16_reduced_precision_reduction
        assert not matmul.allow_bf16_reduced_precision_reduction
        ps._restore_context_parallel_precision()
        assert (
            torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
            torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
        ) == expected
    finally:
        (
            matmul.allow_fp16_reduced_precision_reduction,
            matmul.allow_bf16_reduced_precision_reduction,
        ) = previous
