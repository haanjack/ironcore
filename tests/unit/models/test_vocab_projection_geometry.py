# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""True-vocab GEMM with padded TP storage and complete weight gradients."""

import pytest
import torch
import torch.nn.functional as F

from ironcore.layers.linear_cross_entropy import vocab_linear
from ironcore.parallel import parallel_states


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_padded_partition_preserves_real_entries_and_zero_dummy_gradients(
    monkeypatch, rank, transposed
):
    monkeypatch.setattr(parallel_states, "get_tensor_model_parallel_rank", lambda: rank)
    torch.manual_seed(51)
    x = torch.randn(2, 3, 8, requires_grad=True)
    weight = torch.randn((8, 4) if transposed else (4, 8), requires_grad=True)
    active = max(0, min(4, 7 - rank * 4))
    actual = vocab_linear(x, weight, 7, transposed=transposed, trim_padding=True)
    assert actual.shape == (2, 3, 4)
    assert torch.isneginf(actual[..., active:]).all()
    other = x.detach().clone().requires_grad_()
    expected_weight = (
        (weight[:, :active].T if transposed else weight[:active]).detach().clone().requires_grad_()
    )
    expected = F.linear(other, expected_weight)
    assert torch.equal(actual[..., :active], expected)
    coefficient = torch.randn_like(expected)
    actual[..., :active].backward(coefficient)
    expected.backward(coefficient)
    assert torch.equal(x.grad, other.grad)
    full_gradient = weight.grad.T if transposed else weight.grad
    assert torch.equal(full_gradient[:active], expected_weight.grad)
    assert torch.count_nonzero(full_gradient[active:]) == 0


def test_bf16_output_head_matches_true_vocab_backward(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("BF16 head geometry requires CUDA")
    monkeypatch.setattr(parallel_states, "get_tensor_model_parallel_rank", lambda: 0)
    torch.manual_seed(11)
    weight = torch.randn(640, 128, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = vocab_linear(x, weight, 513, trim_padding=True)
        expected = F.linear(other, weight[:513])
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual[:, :513], expected)
    coefficient = torch.randn_like(expected)
    actual[:, :513].backward(coefficient)
    expected.backward(coefficient)
    assert torch.equal(x.grad, other.grad)
