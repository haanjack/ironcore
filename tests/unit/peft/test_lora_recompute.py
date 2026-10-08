# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint recomputation must replay tracked dropout and preserve its stream."""

import pytest
import torch

from ironcore.parallel import parallel_states
from ironcore.parallel.random import (
    checkpoint_with_tensor_parallel_rng,
    reset_tensor_parallel_rng_tracker,
    snapshot_tensor_parallel_rng_tracker,
    tensor_parallel_rng_fork,
)


@pytest.mark.parametrize("reentrant", [False, True])
def test_checkpoint_replays_tp_dropout_without_advancing_stream(monkeypatch, reentrant):
    monkeypatch.setattr(parallel_states, "get_data_parallel_group_rank", lambda: 0)
    monkeypatch.setattr(parallel_states, "get_tensor_model_parallel_rank", lambda: 0)
    x = torch.linspace(-1, 1, 64).reshape(8, 8).requires_grad_(True)
    other = x.detach().clone().requires_grad_(True)

    def operation(values):
        with tensor_parallel_rng_fork(42, values.device):
            return torch.nn.functional.dropout(values.sin(), p=0.3, training=True).square()

    reset_tensor_parallel_rng_tracker()
    expected = operation(other)
    expected.sum().backward()
    following = operation(other).detach()
    states = snapshot_tensor_parallel_rng_tracker()
    reset_tensor_parallel_rng_tracker()
    actual = checkpoint_with_tensor_parallel_rng(operation, x, use_reentrant=reentrant)
    actual.sum().backward()
    next_actual = operation(x).detach()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(x.grad, other.grad, atol=0, rtol=0)
    torch.testing.assert_close(next_actual, following, atol=0, rtol=0)
    actual_states = snapshot_tensor_parallel_rng_tracker()
    assert actual_states.keys() == states.keys()
    for key, state in states.items():
        torch.testing.assert_close(actual_states[key], state, atol=0, rtol=0)
    reset_tensor_parallel_rng_tracker()


def test_parameter_ratio_counts_sharded_base_once(monkeypatch):
    from tests.fixtures.lora_test_utils import count_parameters

    monkeypatch.setattr(parallel_states, "_TENSOR_MODEL_PARALLEL_WORLD_SIZE", 2)
    model = torch.nn.Module()
    model.base = torch.nn.Parameter(torch.zeros(10), requires_grad=False)
    model.base.is_tp_sharded = True
    model.lora_A = torch.nn.Parameter(torch.zeros(4))
    assert count_parameters(model) == (4, 24, 4)
