# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Independent HF gradients and absence of activation-sized FP32 norm tensors."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ironcore.layers.layernorm.rms_norm_kernel import frozen_scale_rms_norm


class FullPrecisionActivationTrace(TorchDispatchMode):
    def __init__(self, elements):
        super().__init__()
        self.elements = elements
        self.outputs = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for tensor in torch.utils._pytree.tree_leaves(result):
            if (
                isinstance(tensor, torch.Tensor)
                and tensor.dtype == torch.float32
                and tensor.numel() >= self.elements
            ):
                self.outputs.append((str(func), tuple(tensor.shape)))
        return result


@pytest.mark.parametrize("family", ["llama", "gemma4"])
@pytest.mark.parametrize("width", [16, 576, 2816])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_frozen_cuda_norm_matches_hf_without_full_fp32_activation(family, width, dtype):
    if not torch.cuda.is_available():
        pytest.skip("Frozen-scale fused RMSNorm requires CUDA")
    if family == "llama":
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        reference = LlamaRMSNorm(width, eps=1e-6)
    else:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

        reference = Gemma4RMSNorm(width, eps=1e-6)
    reference.to(device="cuda", dtype=dtype).requires_grad_(False)
    torch.manual_seed(31)
    with torch.no_grad():
        reference.weight.uniform_(0.5, 1.5)
    # Noncontiguous input also keeps any layout-copy allocation in model dtype.
    inputs = torch.randn(3, 7, width, device="cuda", dtype=dtype).transpose(0, 1).requires_grad_()
    other = inputs.detach().clone().requires_grad_()
    coefficient = torch.randn_like(inputs)
    expected = reference(other)
    expected.backward(coefficient)
    trace = FullPrecisionActivationTrace(inputs.numel())
    with trace:
        actual = frozen_scale_rms_norm(
            inputs, reference.weight, 1e-6, round_before_scale=family == "llama"
        )
        assert actual is not None and actual.dtype == dtype
        actual.backward(coefficient)
    assert not trace.outputs, trace.outputs
    # Independent FP32 reductions can differ at BF16/FP16 rounding boundaries.
    for a, b in ((actual, expected), (inputs.grad, other.grad)):
        error = (a.float() - b.float()).norm()
        assert error <= 2e-3 * b.float().norm(), (family, width, dtype, float(error))


@pytest.mark.parametrize("width,rows", [(1024, 1024), (1024, 1), (576, 21), (2816, 21), (128, 7)])
def test_granite_reference_reduction_without_fp32_activation(width, rows):
    if not torch.cuda.is_available():
        pytest.skip("Frozen-scale fused RMSNorm requires CUDA")
    from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeRMSNorm

    torch.manual_seed(42)
    reference = GraniteMoeRMSNorm(width, eps=1e-6).to("cuda", torch.bfloat16).requires_grad_(False)
    reference.weight.data.uniform_(0.5, 1.5)
    x = (torch.randn(rows, width, device="cuda", dtype=torch.bfloat16) * 7).requires_grad_()
    other = x.detach().clone().requires_grad_()
    coefficient = torch.randn_like(x)
    expected = reference(other)
    expected.backward(coefficient)
    trace = FullPrecisionActivationTrace(x.numel())
    with trace:
        actual = frozen_scale_rms_norm(
            x, reference.weight, 1e-6, round_before_scale=True, torch_compatible=True
        )
        actual.backward(coefficient)
    assert actual.dtype == torch.bfloat16
    assert not trace.outputs, trace.outputs
    assert torch.equal(actual, expected)
    assert torch.equal(x.grad, other.grad)
