# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Full Granite decoder/parameter-LoRA parity and checkpoint contracts."""

import pytest
import torch
import torch.nn.functional as F
from tests.fixtures.granitemoe import adapter_bindings, granite_pair


def objective(logits, labels):
    ce = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="none").view_as(
        labels
    )
    return torch.stack([x[y != -100].mean() for x, y in zip(ce, labels, strict=True)]).mean()


@pytest.mark.parametrize("backend", ["loop", "grouped"])
@pytest.mark.parametrize("tied", [False, True])
def test_full_decoder_and_checkpoint_roundtrip(monkeypatch, backend, tied):
    pytest.importorskip("transformers.models.granitemoe")
    from ironcore.checkpointing.native import HFConfigManager
    from ironcore.checkpointing.weight_mapping import Architecture, WeightMapper

    torch.manual_seed(71)
    native, reference, config = granite_pair(monkeypatch, backend, tied=tied)
    tokens = torch.randint(2, 32, (2, 12))
    torch.testing.assert_close(native(tokens)[0], reference(tokens).logits, atol=2e-6, rtol=2e-5)
    mapper = WeightMapper(Architecture.GRANITEMOE, 2)
    exported = mapper.ironcore_to_hf(native.state_dict())
    reference.load_state_dict(
        exported, strict=False
    )  # Tied lm_head is an alias, not a second weight.
    assert all(
        torch.equal(native.state_dict()[k], v) for k, v in mapper.hf_to_ironcore(exported).items()
    )
    hf = HFConfigManager.get_hf_config(config)
    assert hf["model_type"] == "granitemoe"
    assert hf["hidden_act"] == "silu"
    assert hf["logits_scaling"] == 6.0
    # IBM's older public checkpoint names map to the same physical tensors.
    legacy = {}
    for key, value in exported.items():
        key = key.replace("experts.gate_up_proj", "input_linear.weight")
        key = key.replace("experts.down_proj", "output_linear.weight")
        key = key.replace("router.weight", "router.layer.weight")
        legacy[key] = value
    assert all(
        torch.equal(mapper.hf_to_ironcore(legacy)[k], v)
        for k, v in native.state_dict().items()
        if k != "rotary_pos_emb.theta"
    )


@pytest.mark.parametrize("backend", ["loop", "grouped"])
@pytest.mark.parametrize("recompute", [False, True])
def test_packed_parameter_lora_complete_gradients(monkeypatch, backend, recompute):
    pytest.importorskip("peft")
    from ironcore.peft.utils import merge_lora_weights

    torch.manual_seed(71)
    native, reference, config = granite_pair(monkeypatch, backend, lora=True)
    pairs = adapter_bindings(native, reference)
    with torch.no_grad():
        for name, pair in pairs.items():
            pair.native.normal_(0, 0.03 if name.endswith(".A") else 0.01)
            pair.reference_value().copy_(pair.native)
    config.operation.activation_recompute = recompute
    native.model.activation_recompute = recompute
    tokens = torch.randint(2, 32, (2, 12))
    labels = torch.randint(2, 32, tokens.shape)
    labels[0, :3] = -100
    labels[1, :8] = -100
    native.train()
    reference.train()
    actual, expected = native(tokens)[0], reference(tokens, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    objective(actual, labels).backward()
    objective(expected, labels).backward()
    for name, pair in pairs.items():
        p, q = pair.native.grad, pair.reference_value(gradient=True)
        assert p is not None and q is not None, name
        assert (p - q).double().norm() <= 3e-4 * q.double().norm() + 2e-8, name
    assert all(p.grad is None for p in native.parameters() if not p.requires_grad)
    # Recomputed CE must apply /6 after the projection and preserve its VJP.
    native.zero_grad(set_to_none=True)
    config.trainer.recompute_linear_ce = True
    config.trainer.loss_chunk_size = 5
    from ironcore.training_utils import loss_func_sft

    native.loss_fn = loss_func_sft
    loss = native(tokens, labels=labels)
    torch.testing.assert_close(loss, objective(expected.detach(), labels), atol=2e-6, rtol=2e-5)
    loss.backward()
    for name, pair in pairs.items():
        q = pair.reference_value(gradient=True)
        assert (pair.native.grad - q).double().norm() <= 3e-4 * q.double().norm() + 2e-8, name
    native.eval()
    with torch.no_grad():
        before = native(tokens)[0]
        merge_lora_weights(native)
        torch.testing.assert_close(native(tokens)[0], before, atol=2e-6, rtol=2e-5)


def test_unsupported_granite_paths_rejected(monkeypatch):
    from ironcore.config.config_granitemoe import validate_granitemoe_runtime

    _, _, config = granite_pair(monkeypatch)
    config.trainer.context_parallel_size = 2
    with pytest.raises(ValueError, match="CP"):
        validate_granitemoe_runtime(config)
    config.trainer.context_parallel_size = 1
    config.peft.method = "lora"
    config.peft.lora.target_modules = ["gate_proj"]
    with pytest.raises(ValueError, match="fused"):
        validate_granitemoe_runtime(config)
    config.peft.method = "none"
    config.model.moe.blockwise_backend = "triton"
    with pytest.raises(ValueError, match="grouped/torch"):
        validate_granitemoe_runtime(config)


@pytest.mark.parametrize("budget", [4096, 31])
def test_cuda_granite_grouped_bf16_output_and_packed_lora(monkeypatch, budget):
    if not torch.cuda.is_available():
        pytest.skip("BF16 grouped GEMM needs CUDA")
    pytest.importorskip("peft")
    torch.manual_seed(71)
    native, reference, config = granite_pair(
        monkeypatch, lora=True, experts_implementation="grouped_mm"
    )
    native.to(device="cuda", dtype=torch.bfloat16)
    reference.to(device="cuda", dtype=torch.bfloat16)
    pairs = adapter_bindings(native, reference)
    with torch.no_grad():
        for name, pair in pairs.items():
            pair.native.data = pair.native.data.float()
            pair.reference.data = pair.reference.data.float()
            pair.native.normal_(0, 0.03 if name.endswith(".A") else 0.01)
            pair.reference_value().copy_(pair.native)
    config.model.moe.grouped_token_budget = budget
    x = torch.randn(2, 64, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    other = x.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = native.model.layers[0].mlp(x)
        expected = reference.base_model.model.model.layers[0].block_sparse_moe(other)
    # Autocast sum() alone returns FP32 and promotes the residual. Explicit
    # sum dtype must write BF16 directly, matching HF's final BF16 rounding.
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert (actual.float() - expected.float()).norm() < 0.01 * expected.float().norm()
    coefficients = torch.randn_like(actual)
    (actual * coefficients).sum().backward()
    (expected * coefficients).sum().backward()
    assert (x.grad.float() - other.grad.float()).norm() < 0.03 * other.grad.float().norm()
    for name, pair in pairs.items():
        if not name.startswith("0.expert"):
            continue
        q = pair.reference_value(gradient=True)
        assert pair.native.grad is not None and q is not None
        assert (pair.native.grad - q).double().norm() <= 0.03 * q.double().norm() + 2e-6, name


def test_cuda_complete_bf16_trained_lora_matches_stock_hf(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("BF16 decoder parity requires CUDA")
    pytest.importorskip("peft")
    deterministic = torch.are_deterministic_algorithms_enabled()
    reduced = torch._C._get_cublas_allow_bf16_reduced_precision_reduction()
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.manual_seed(71)
        native, reference, config = granite_pair(
            monkeypatch,
            lora=True,
            experts_implementation="grouped_mm",
            hidden_size=128,
            intermediate_size=64,
            lora_rank=8,
        )
        # from_pretrained(dtype=BF16) keeps derived rotary frequencies FP32;
        # a blanket Module.to(BF16) would also quantize this oracle buffer.
        rope = reference.base_model.model.model.rotary_emb
        inv_freq = rope.inv_freq.detach().clone()
        original_inv_freq = rope.original_inv_freq.detach().clone()
        native.to("cuda", torch.bfloat16)
        reference.to("cuda", torch.bfloat16)
        rope.inv_freq = inv_freq.cuda()
        rope.original_inv_freq = original_inv_freq.cuda()
        pairs = adapter_bindings(native, reference)
        with torch.no_grad():
            for name, pair in pairs.items():
                pair.native.data = pair.native.data.float()
                pair.reference.data = pair.reference.data.float()
                pair.native.normal_(0, 0.03 if name.endswith(".A") else 0.01)
                pair.reference_value().copy_(pair.native)
        config.model.moe.grouped_token_budget = 4096
        tokens = torch.randint(2, 32, (2, 16), device="cuda")
        labels = torch.randint(2, 32, tokens.shape, device="cuda")
        labels[0, :3] = -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = native(tokens)[0]
            expected = reference(tokens, use_cache=False).logits
        assert actual.dtype == expected.dtype == torch.bfloat16
        assert torch.equal(actual, expected)
        objective(actual, labels).backward()
        objective(expected, labels).backward()
        error = target = 0.0
        for name, pair in pairs.items():
            q = pair.reference_value(gradient=True)
            assert q is not None and pair.native.grad is not None, name
            error += float((pair.native.grad - q).double().square().sum())
            target += float(q.double().square().sum())
        # Tiny head/GQA dimensions can choose different backward kernels;
        # the real H1024 checkpoint additionally has a bitwise GPU oracle.
        assert error <= (5e-4**2) * target
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced
