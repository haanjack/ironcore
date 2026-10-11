# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Compare complete Gemma shared/routed LoRA gradients against HF + PEFT."""

import pytest
import torch
from tests.fixtures.gemma4 import gemma4_pair
from tests.fixtures.gemma4_lora import gemma4_peft_reference


@pytest.mark.parametrize("backend", ["loop", "torch", "scheduled"])
@pytest.mark.parametrize("recompute", [False, True])
def test_gemma4_a4b_lora_matches_hf_peft_gradients(monkeypatch, backend, recompute):
    pytest.importorskip("peft", reason="Independent LoRA oracle requires optional PEFT")
    torch.manual_seed(37)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    native, reference, config = gemma4_pair(monkeypatch, "A4B", lora=True, lora_targets=targets)
    reference, pairs = gemma4_peft_reference(native, reference)
    config.model.moe.expert_backend = "loop" if backend == "loop" else "grouped"
    config.model.moe.blockwise_backend = "torch" if backend == "loop" else backend
    config.model.moe.grouped_token_budget = 7
    config.operation.activation_recompute = recompute
    native.model.activation_recompute = recompute
    config.trainer.mlp_chunk_size = 3
    with torch.no_grad():
        for name, (p, q) in pairs.items():
            p.normal_(0, 0.02 if name.endswith(".A") else 0.005)
            q.copy_(p.T)
    tokens = torch.randint(2, 32, (2, 12))
    labels = torch.randint(2, 32, tokens.shape)
    labels[0, :3] = -100
    labels[1, :8] = -100
    native.train()
    reference.train()
    actual, _ = native(tokens)
    expected = reference(tokens, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def objective(logits):
        ce = torch.nn.functional.cross_entropy(
            logits.float().flatten(0, 1), labels.flatten(), reduction="none"
        ).view_as(labels)
        return torch.stack(
            [row[mask != -100].mean() for row, mask in zip(ce, labels, strict=True)]
        ).mean()

    objective(actual).backward()
    objective(expected).backward()
    for name, (p, q) in pairs.items():
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            error = (p.grad - q.grad.T).double().norm()
            scale = q.grad.double().norm()
            assert torch.isfinite(p.grad).all(), name
            # Bound every complete adapter, including entries near zero.
            assert error <= 2e-4 * scale + 2e-7, (name, float(error), float(scale))
    assert all(p.grad is None for p in native.parameters() if not p.requires_grad)
    assert all(p.grad is None for p in reference.parameters() if not p.requires_grad)


@pytest.mark.parametrize("backend", ["torch", "scheduled", "triton"])
@pytest.mark.parametrize("precision", ["model", "float32"])
def test_cuda_bf16_grouped_matches_hf_expert_accumulation(monkeypatch, backend, precision):
    """Default model rounding matches stock HF; FP32 remains an explicit option."""
    if not torch.cuda.is_available():
        pytest.skip("Grouped CUDA kernels require a GPU")
    pytest.importorskip("peft", reason="Independent LoRA oracle requires optional PEFT")
    torch.manual_seed(37)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    native, reference, config = gemma4_pair(monkeypatch, "A4B", lora=True, lora_targets=targets)
    reference.set_attn_implementation("sdpa")
    for model in (native, reference):
        frequencies = {
            name: value.clone() for name, value in model.named_buffers() if "inv_freq" in name
        }
        model.to(device="cuda", dtype=torch.bfloat16)
        for name, value in model.named_buffers():
            if name in frequencies:
                parent, _, key = name.rpartition(".")
                model.get_submodule(parent)._buffers[key] = frequencies[name].cuda()
    for parameter in native.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    reference, pairs = gemma4_peft_reference(
        native,
        reference,
        expert_accumulation_dtype=torch.float32 if precision == "float32" else None,
    )
    reference.cuda()
    config.model.precision = "bfloat16"
    config.model.moe.expert_backend = "grouped"
    config.model.moe.blockwise_backend = backend
    config.model.moe.expert_accumulation_precision = precision
    config.model.moe.grouped_token_budget = 19
    config.trainer.mlp_chunk_size = 11
    with torch.no_grad():
        for name, (p, q) in pairs.items():
            p.normal_(0, 0.02 if name.endswith(".A") else 0.005)
            q.copy_(p.T)
    tokens = torch.randint(2, 32, (2, 64), device="cuda")
    native.eval()
    reference.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual, _ = native(tokens)
        expected = reference(tokens, use_cache=False).logits
    # Test this fixture exactly; this is not a general bitwise GEMM guarantee.
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
