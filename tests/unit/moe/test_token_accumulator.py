# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Additive token accumulation derivatives and activation-storage contract."""

from __future__ import annotations

import pytest
import torch

from ironcore.layers.moe.batched import _AddTokenOutputs


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_repeated_ids_chained_additions_and_noncontiguous_gradient(dtype):
    torch.manual_seed(73)
    initial = torch.randn(5, 4, requires_grad=True)
    values = [torch.randn(3, 4, dtype=dtype, requires_grad=True) for _ in range(2)]
    ids = [torch.tensor([0, 2, 2]), torch.tensor([2, 4, 0])]
    reference_initial = initial.detach().clone().requires_grad_()
    reference_values = [v.detach().clone().requires_grad_() for v in values]
    destination = initial + 0
    original_ptr = destination.data_ptr()
    result = destination
    expected = reference_initial
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: (saved.append((t.dtype, tuple(t.shape))), t)[1], lambda t: t
    ):
        for value, index in zip(values, ids, strict=True):
            result = _AddTokenOutputs.apply(result, value, index)
    for value, index in zip(reference_values, ids, strict=True):
        expected = expected.index_add(0, index, value.float())
    assert result.data_ptr() == original_ptr
    assert saved == [(torch.int64, (3,)), (torch.int64, (3,))]
    torch.testing.assert_close(result, expected)
    gradient = torch.randn(4, 5).t()
    assert not gradient.is_contiguous()
    result.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(initial.grad, reference_initial.grad)
    for actual, reference in zip(values, reference_values, strict=True):
        torch.testing.assert_close(actual.grad, reference.grad)


def test_accumulator_gradcheck_and_gradgradcheck():
    ids = torch.tensor([0, 2, 2])

    def accumulate(initial, values):
        return _AddTokenOutputs.apply(initial + 0, values, ids)

    inputs = (
        torch.randn(4, 2, dtype=torch.float64, requires_grad=True),
        torch.randn(3, 2, dtype=torch.float64, requires_grad=True),
    )
    assert torch.autograd.gradcheck(accumulate, inputs)
    assert torch.autograd.gradgradcheck(accumulate, inputs)
