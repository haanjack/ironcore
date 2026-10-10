# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Bounded Gemma attention, MoE execution and softcapped linear CE parity."""

from __future__ import annotations

import pytest
import torch
from tests.fixtures.gemma4 import gemma4_pair


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("expert_backend", "batched"),
        ("num_shared_experts", 2),
        ("aux_loss_alpha", 0.01),
        ("router_jitter_noise", 0.1),
        ("router_bias", True),
        ("drop_tokens", True),
        ("expert_capacity_factor", 1.0),
    ],
)
def test_a4b_rejects_non_native_router_options(monkeypatch, option, value):
    from ironcore.config.config_gemma4 import validate_gemma4_runtime

    _, _, config = gemma4_pair(monkeypatch, "A4B")
    setattr(config.model.moe, option, value)
    with pytest.raises(ValueError, match="Gemma 4 MoE"):
        validate_gemma4_runtime(config)


@pytest.mark.parametrize("backend", ["loop", "grouped"])
@pytest.mark.parametrize("chunk", [None, 2])
def test_a4b_bounded_execution_matches_reference_logits_and_gradients(monkeypatch, backend, chunk):
    from ironcore.checkpointing.weight_mapping import Architecture, WeightMapper

    torch.manual_seed(42)
    native, reference, config = gemma4_pair(monkeypatch, "A4B")
    config.model.moe.expert_backend = backend
    config.model.moe.grouped_token_budget = 5
    config.model.moe.blockwise_backend = "scheduled" if backend == "grouped" else "torch"
    config.trainer.mlp_chunk_size = 2 if backend == "grouped" else None
    config.model.gemma4.attention_chunk_size = chunk
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7], [2, 8, 9, 10, 11, 12]])
    actual, _ = native(tokens)
    expected = reference(tokens, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    actual.square().mean().backward()
    expected.square().mean().backward()
    mapped = WeightMapper(Architecture.GEMMA4, 4).hf_to_ironcore(
        {n: p.grad for n, p in reference.named_parameters()}
    )
    for name, parameter in native.named_parameters():
        if parameter.grad is None:
            assert torch.count_nonzero(mapped[name]) == 0, name
            continue
        torch.testing.assert_close(parameter.grad, mapped[name], atol=5e-5, rtol=5e-4, msg=name)


@pytest.mark.parametrize("frozen", [False, True])
def test_softcapped_chunked_ce_matches_full_logits_with_masked_labels(monkeypatch, frozen):
    from ironcore.layers.linear_cross_entropy import linear_cross_entropy
    from ironcore.parallel import parallel_states
    from ironcore.parallel.tensor_parallel import vocab_parallel_cross_entropy

    monkeypatch.setattr(parallel_states, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_states, "get_tensor_model_parallel_rank", lambda: 0)
    torch.manual_seed(42)
    hidden = torch.randn(2, 7, 8, requires_grad=True)
    weight = torch.randn(32, 8, requires_grad=not frozen)
    other_hidden = hidden.detach().clone().requires_grad_()
    other_weight = weight.detach().clone().requires_grad_(not frozen)
    labels = torch.randint(0, 32, (2, 7))
    labels[:, :3] = -100
    mask = (labels != -100).float()
    logits = torch.nn.functional.linear(other_hidden, other_weight)
    expected = vocab_parallel_cross_entropy((logits / 2).tanh() * 2, labels)
    actual = linear_cross_entropy(hidden, weight, labels, 3, softcap=2)
    torch.testing.assert_close(actual, expected)
    (actual * mask).sum().backward()
    (expected * mask).sum().backward()
    torch.testing.assert_close(hidden.grad, other_hidden.grad)
    if frozen:
        assert weight.grad is None
    else:
        torch.testing.assert_close(weight.grad, other_weight.grad)


def test_unpacked_sft_uses_all_input_slots_without_quadratic_document_mask():
    from ironcore.dataloader.collator import UniversalCollator

    collator = UniversalCollator("sft", 4, pack_sequences=False, return_full_attention_mask=False)
    batch = collator(
        [{"token_ids": torch.tensor([2, 3, 4, 5, 6]), "metadata": {"mask_ranges": [[0, 3]]}}]
    )
    torch.testing.assert_close(batch["input_ids"], torch.tensor([[2, 3, 4, 5]]))
    torch.testing.assert_close(batch["labels"], torch.tensor([[-100, -100, 5, 6]]))
    assert "attention_mask" not in batch
    assert "loss_sample_ids" not in batch


@pytest.mark.parametrize("backend", ["loop", "grouped"])
def test_eval_chunked_sft_loss_matches_independent_full_logit_reference(monkeypatch, backend):
    """Held-out eval must use the same masked, per-conversation SFT objective."""
    from ironcore.training_utils import loss_func_sft

    native, reference, config = gemma4_pair(monkeypatch, "A4B", lora=True)
    config.model.moe.expert_backend = backend
    config.model.moe.grouped_token_budget = 5
    config.model.moe.blockwise_backend = "scheduled" if backend == "grouped" else "torch"
    config.trainer.recompute_linear_ce = True
    config.trainer.loss_chunk_size = 2
    native.loss_fn = loss_func_sft
    native.eval()
    reference.eval()
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7], [2, 8, 9, 10, 11, 12]])
    labels = torch.tensor([[-100, -100, -100, -100, 7, 1], [-100, -100, 10, 11, 12, 1]])
    with torch.no_grad():
        logits = reference(tokens, use_cache=False).logits
        per_token = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 32), labels.reshape(-1), reduction="none"
        ).view_as(labels)
        counts = (labels != -100).sum(1)
        expected = (per_token.sum(1) / counts).mean()
        actual = native(tokens, labels=labels)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert not actual.requires_grad
    assert all(parameter.grad is None for parameter in native.parameters())


def test_a4b_flop_estimate_counts_stored_experts_and_bounded_attention(monkeypatch):
    from ironcore.utils.mfu import MFUCalculator

    native, _, config = gemma4_pair(monkeypatch, "A4B")
    calculator = MFUCalculator.from_config(config.model, 32)
    assert calculator.get_num_parameters() == sum(p.numel() for p in native.parameters())
    unbounded = calculator.compute_tflops(1, 32, 1)
    config.model.gemma4.attention_chunk_size = 2
    bounded = calculator.compute_tflops(1, 32, 1)
    assert 0 < bounded < unbounded
    config.model.moe.num_experts_per_token = 1
    assert calculator.get_num_parameters() == sum(p.numel() for p in native.parameters())
    assert calculator.compute_tflops(1, 32, 1) < bounded


def test_unpacked_sample_loss_weights_answers_equally_and_handles_empty_batch(monkeypatch):
    from ironcore.parallel import parallel_states
    from ironcore.parallel.context_parallel import context_parallel_sample_mean

    monkeypatch.setattr(parallel_states, "get_context_parallel_world_size", lambda: 1)
    losses = torch.tensor([[2.0, 8.0], [3.0, 7.0]], requires_grad=True)
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    loss = context_parallel_sample_mean(losses, mask)
    torch.testing.assert_close(loss, torch.tensor(3.5))
    loss.backward()
    torch.testing.assert_close(losses.grad, torch.tensor([[0.5, 0.0], [0.25, 0.25]]))
    losses.grad = None
    empty = context_parallel_sample_mean(losses, torch.zeros_like(mask))
    torch.testing.assert_close(empty, torch.tensor(0.0))
    empty.backward()
    torch.testing.assert_close(losses.grad, torch.zeros_like(losses))
