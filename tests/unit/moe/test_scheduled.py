# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""First-order bounded recomputation and explicit configuration contracts."""

from __future__ import annotations

import pytest
import torch

from ironcore.config.config_blockwise import validate_blockwise_mlp
from ironcore.config.config_moe import MoEConfig
from ironcore.layers.moe.scheduled import _Block, _ScheduledExperts


@pytest.mark.parametrize("padded", [False, True])
def test_scheduler_gradcheck_and_saved_original_storage(padded):
    torch.manual_seed(31)
    hidden = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(6, dtype=torch.float64, requires_grad=True)
    # Uneven counts and a partial final padded block.
    order = torch.tensor([0, 2, 4, 5, 1, 3])
    counts = torch.tensor([4, 2])
    prefixes = torch.tensor([0, 4])
    blocks = (
        (_Block((0, 1), 0, (4, 2), 3), _Block((0, 1), 3, (4, 2), 1))
        if padded
        else (_Block((0,), 0, (3,)), _Block((0, 1), 3, (1, 2)))
    )
    parameters = tuple(
        torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        for shape in ((2, 3), (3, 2), (2, 3), (3, 2))
    )
    metadata = (blocks, torch.nn.SiLU(), False, padded, 2, False)

    def run(*inputs):
        x, w, *params = inputs
        return _ScheduledExperts.apply(x, w, order, prefixes, counts, metadata, *params)

    assert torch.autograd.gradcheck(run, (hidden, weights, *parameters))
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: (saved.append(t.data_ptr()), t)[1], lambda t: t
    ):
        result = run(hidden, weights, *parameters)
    assert saved == [t.data_ptr() for t in (hidden, weights, order, prefixes, counts, *parameters)]
    result.sum().backward()


@pytest.mark.parametrize("backend", ["scheduled", "triton"])
def test_configuration_rejects_unbounded_batched_and_loop(backend):
    from tests.fixtures.config_fixtures import create_moe_test_config

    with pytest.raises(ValueError, match="batched or grouped"):
        MoEConfig(expert_backend="loop", blockwise_backend=backend)
    cfg = create_moe_test_config()
    cfg.model.moe.expert_backend = "batched"
    cfg.model.moe.blockwise_backend = backend
    cfg.trainer.mlp_chunk_size = None
    with pytest.raises(ValueError, match="require mlp_chunk_size"):
        validate_blockwise_mlp(cfg)


def test_invalid_backend():
    with pytest.raises(ValueError, match="blockwise_backend"):
        MoEConfig(blockwise_backend="unknown")


@pytest.mark.parametrize("backend", ["scheduled", "triton"])
@pytest.mark.parametrize("expert_backend", ["batched", "grouped"])
def test_scheduled_backend_rejects_other_dispatch_modes(backend, expert_backend):
    from tests.fixtures.config_fixtures import create_moe_test_config

    from ironcore.layers.moe.moe_layer import CommunicationMode, MoEMLP

    cfg = create_moe_test_config()
    cfg.model.moe.expert_backend = expert_backend
    cfg.model.moe.blockwise_backend = backend
    cfg.trainer.mlp_chunk_size = 3
    with pytest.raises(ValueError, match="default dispatch mode"):
        MoEMLP(cfg, communication_mode=CommunicationMode.ALL_TO_ALL)


def test_scheduler_rejects_double_backward_explicitly():
    hidden = torch.randn(2, 2, requires_grad=True)
    weights = torch.randn(2, requires_grad=True)
    up = torch.randn(2, 3, requires_grad=True)
    down = torch.randn(3, 2, requires_grad=True)
    result = _ScheduledExperts.apply(
        hidden,
        weights,
        torch.arange(2),
        torch.tensor([0]),
        torch.tensor([2]),
        ((_Block((0,), 0, (2,), 2),), torch.nn.SiLU(), False, True, 1, False),
        up,
        down,
    )
    upstream = torch.ones_like(result, requires_grad=True)
    (first,) = torch.autograd.grad(result, hidden, upstream, create_graph=True)
    with pytest.raises(RuntimeError, match="once_differentiable"):
        first.sum().backward()
