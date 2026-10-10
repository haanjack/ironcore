# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Model-dtype rounding, expert order, and direct low-precision output storage."""

import pytest
import torch
from tests.fixtures.gemma4 import gemma4_pair
from torch.utils._python_dispatch import TorchDispatchMode

from ironcore.config.config_moe import MoEConfig
from ironcore.layers.moe.grouped import grouped_experts
from ironcore.layers.moe.model_accumulation import _accumulate, _mixture_backward


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_mixture_writes_low_precision_without_full_fp32_product(fused, dtype):
    if not torch.cuda.is_available():
        pytest.skip("Direct CUDA mixture storage requires a GPU")
    pytest.importorskip("triton")
    torch.manual_seed(41)
    rows, width = 17, 2816
    values = torch.randn(rows * 2, width, device="cuda", dtype=dtype)
    weights = torch.rand(rows * 2, device="cuda")
    ids = torch.arange(rows, device="cuda").repeat(2)
    incoming = torch.randn_like(values)
    output = torch.zeros(rows, width, device="cuda", dtype=dtype)

    class Trace(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            for tensor in torch.utils._pytree.tree_leaves(result):
                if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32:
                    assert tensor.numel() < rows * width, (func, tuple(tensor.shape))
            return result

    with Trace():
        _accumulate(output, values, ids, (rows, rows), weights=weights, fused=fused)
        dv, dw = _mixture_backward(values, weights, incoming)
    expected = torch.zeros_like(output)
    for start in (0, rows):
        weighted = (values[start : start + rows].float() * weights[start : start + rows, None]).to(
            dtype
        )
        expected = expected + weighted
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(dv, (incoming.float() * weights[:, None]).to(dtype), atol=0, rtol=0)
    torch.testing.assert_close(
        dw, (values.float() * incoming.float()).sum(-1), atol=2e-5, rtol=2e-5
    )


def test_invalid_accumulation_precision():
    with pytest.raises(ValueError, match="expert_accumulation_precision"):
        MoEConfig(expert_accumulation_precision="invalid")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_products_round_before_expert_ordered_addition(dtype):
    values = torch.tensor([[1.0, -2.0], [0.01, 0.2], [-0.4, 0.3]], dtype=dtype)
    weights = torch.tensor([0.5001, 0.6011, 0.7117])
    ids = torch.tensor([0, 0, 0])
    actual = torch.zeros(1, 2, dtype=dtype)
    _accumulate(actual, values, ids, (1, 1, 1), weights=weights)
    expected = torch.zeros_like(actual)
    for row in range(3):
        expected = expected + (values[row].float() * weights[row]).to(dtype)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_weight_product_backward_preserves_mixed_dtypes():
    values = torch.randn(7, 16, dtype=torch.bfloat16)
    incoming = torch.randn_like(values)
    weights = torch.rand(7)
    dv, dw = _mixture_backward(values, weights, incoming)
    assert dv.dtype == torch.bfloat16 and dw.dtype == torch.float32
    torch.testing.assert_close(dv, (incoming.float() * weights[:, None]).bfloat16(), atol=0, rtol=0)
    torch.testing.assert_close(dw, (values.float() * incoming.float()).sum(-1), atol=0, rtol=0)


def test_model_accumulation_gradcheck_and_saved_storage(monkeypatch):
    torch.manual_seed(29)
    native, _, _ = gemma4_pair(monkeypatch, "A4B", lora=True)
    experts = native.model.layers[0].experts.double()
    experts[0].config.model.moe.grouped_token_budget = 3
    inputs = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    weights = torch.rand(3, 2, dtype=torch.float64, requires_grad=True)
    indices = torch.tensor([[2, 0], [1, 0], [0, 2]])
    assert torch.autograd.gradcheck(
        lambda x, w: grouped_experts(x, indices, w, experts), (inputs, weights)
    )
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: (saved.append(t), t)[1], lambda t: t):
        grouped_experts(inputs, indices, weights, experts)
    assert saved[0].data_ptr() == inputs.data_ptr()
    assert not any(t.ndim == 3 and t.shape[:2] == (3, 2) for t in saved)
