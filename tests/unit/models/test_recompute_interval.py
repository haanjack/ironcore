# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Segment recompute must preserve gradients, KV sharing, and storage boundaries."""

import copy
from functools import partial

import pytest
import torch
from tests.fixtures.gemma4 import gemma4_pair

from ironcore.config import _config_validation
from ironcore.config.recompute import recompute_segments
from ironcore.offload.recompute import recompute_streamed_segment
from ironcore.parallel.random import reset_tensor_parallel_rng_tracker, tensor_parallel_rng_fork


@pytest.mark.parametrize("layers", [1, 4, 5, 30, 32])
def test_count_plan_preserves_exact_budget_and_bounds(layers):
    for skip in range(layers):
        segments = recompute_segments(layers, skip_count=skip)
        assert len(segments) == layers - skip
        assert segments[0][0] == 0 and segments[-1][1] == layers
        assert [i for start, end in segments for i in range(start, end)] == list(range(layers))
        sizes = [end - start for start, end in segments]
        assert max(sizes) - min(sizes) <= 1
    sizes = [end - start for start, end in recompute_segments(32, skip_count=8)]
    assert sizes.count(2) == 8 and sizes.count(1) == 16


@pytest.mark.parametrize("skip", [-1, True, 1.5, "2", 4])
def test_invalid_count_rejected(monkeypatch, skip):
    _, _, config = gemma4_pair(monkeypatch, "A4B")
    config.operation.recompute_skip_count = skip
    config.trainer.micro_batch_size = 1
    config.trainer.train_batch_size = 1
    config.trainer.gradient_accumulation_steps = 1
    with pytest.raises(ValueError, match="recompute_skip_count"):
        _config_validation(config)


@pytest.mark.parametrize("interval", [0, 2])
def test_count_and_interval_conflict(interval):
    with pytest.raises(ValueError, match="requires recompute_interval=1"):
        recompute_segments(32, interval, 8)


@pytest.mark.parametrize("unsupported", ["fsdp", "sub_layer"])
def test_count_rejects_unsupported_execution(monkeypatch, unsupported):
    _, _, config = gemma4_pair(monkeypatch, "A4B")
    config.trainer.micro_batch_size = 1
    config.trainer.train_batch_size = 1
    config.trainer.gradient_accumulation_steps = 1
    config.operation.recompute_skip_count = 1
    if unsupported == "fsdp":
        config.parallel.use_fsdp = True
    else:
        config.offload.enabled = True
        config.offload.activation_spill = True
        config.offload.activation_spill_granularity = "sub_layer"
    with pytest.raises(
        ValueError, match="Grouped recompute" if unsupported == "fsdp" else "full_layer"
    ):
        _config_validation(config)


def test_single_layer_segment_has_only_one_backward_replay():
    weight = torch.randn(3, 3, requires_grad=True)
    value = torch.randn(2, 3, requires_grad=True)
    calls = []

    def layer(x):
        calls.append(torch.is_grad_enabled())
        return x @ weight

    scheduler = HostBoundaryScheduler()
    result = recompute_streamed_segment([layer], scheduler, 0, value)
    result.sum().backward()
    assert calls == [False, True]
    torch.testing.assert_close(weight.grad, value.detach().sum(0)[:, None].expand_as(weight))
    assert not scheduler.saved


@pytest.mark.parametrize("variant", ["E2B", "A4B"])
@pytest.mark.parametrize(
    "interval,skip", [(0, None), (2, None), (3, None), (5, None), (1, 1), (1, 3)]
)
@pytest.mark.parametrize("reentrant", [False, True])
def test_resident_gemma_outputs_and_gradients(monkeypatch, variant, interval, skip, reentrant):
    native, _, config = gemma4_pair(monkeypatch, variant)
    expected = copy.deepcopy(native)
    config.operation.activation_recompute = interval > 0
    config.operation.recompute_interval = interval
    native.model.recompute_interval = interval
    native.model.recompute_skip_count = skip
    native.model.activation_recompute = interval > 0
    native.model.use_reentrant = reentrant
    # E2B exercises shared KV and per-layer embeddings across segment boundaries.
    tokens = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])
    actual = native(tokens)[0]
    reference = expected(tokens)[0]
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    cotangent = torch.randn_like(actual)
    actual.backward(cotangent)
    reference.backward(cotangent)
    for (name, p), (other_name, q) in zip(
        native.named_parameters(), expected.named_parameters(), strict=True
    ):
        assert name == other_name
        if q.grad is None:
            assert p.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-5, msg=name)


class HostBoundaryScheduler:
    """CPU substitute for DMA, with real detached checkpoint boundaries."""

    def __init__(self):
        self.saved = {}
        self.spills = []
        self.backwards = []

    def on_sublayer_forward(self, layer, sub, activation):
        self.saved[layer] = activation.detach().clone()
        self.spills.append(layer)

    def on_sublayer_backward(self, layer, sub, shape, dtype, device):
        return self.saved.pop(layer).to(device).reshape(shape)

    def on_layer_start(self, layer):
        pass

    def on_layer_end(self, layer):
        pass

    def on_backward_layer_start(self, layer):
        self.backwards.append(layer)

    def on_backward_layer_end(self, layer):
        pass


@pytest.mark.parametrize("interval,skip", [(2, None), (3, None), (8, None), (1, 1), (1, 3)])
@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize("offload_input", [False, True])
def test_host_boundaries_gradients_and_dropout(interval, skip, auxiliary, offload_input):
    # Nonuniform layers catch wrong closure binding and
    # lost cross-segment/auxiliary gradients. Dropout must replay both RNG types.
    weights = [torch.randn(7, 7, requires_grad=True) for _ in range(5)]
    reference_weights = [w.detach().clone().requires_grad_() for w in weights]
    x = torch.randn(2, 3, 7, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()

    def layer(weight, value):
        value = (value @ weight).sin()
        with tensor_parallel_rng_fork(42, value.device):
            value = torch.nn.functional.dropout(value, 0.15, training=True)
        value = torch.nn.functional.dropout(value, 0.1, training=True)
        return (value, value.square().mean()) if auxiliary else value

    torch.manual_seed(123)
    reset_tensor_parallel_rng_tracker()
    ref = reference_x
    ref_losses = []
    for weight in reference_weights:
        output = layer(weight, ref)
        if auxiliary:
            ref, loss = output
            ref_losses.append(loss)
        else:
            ref = output
    (ref.square().mean() + sum(ref_losses)).backward()
    expected_next_rng = torch.get_rng_state()
    torch.manual_seed(123)
    reset_tensor_parallel_rng_tracker()
    scheduler = HostBoundaryScheduler()
    actual = x
    actual_losses = []
    segments = recompute_segments(len(weights), interval, skip)
    for start, end in segments:
        output = recompute_streamed_segment(
            [partial(layer, weight) for weight in weights[start:end]],
            scheduler,
            start,
            actual,
            offload_input=offload_input,
        )
        if auxiliary:
            actual, *losses = output
            actual_losses.extend(losses)
        else:
            actual = output
    assert len(scheduler.saved) == (len(segments) if offload_input else 0)
    torch.testing.assert_close(actual, ref, atol=0, rtol=0)
    (actual.square().mean() + sum(actual_losses)).backward()
    assert not scheduler.saved
    assert scheduler.backwards == list(reversed(range(len(weights))))
    torch.testing.assert_close(torch.get_rng_state(), expected_next_rng, atol=0, rtol=0)
    torch.testing.assert_close(x.grad, reference_x.grad, atol=0, rtol=0)
    for weight, other in zip(weights, reference_weights, strict=True):
        torch.testing.assert_close(weight.grad, other.grad, atol=0, rtol=0)
    reset_tensor_parallel_rng_tracker()


@pytest.mark.parametrize("invalid", [-1, 1.5, True])
def test_invalid_interval_rejected(monkeypatch, invalid):
    _, _, config = gemma4_pair(monkeypatch, "A4B")
    config.operation.recompute_interval = invalid
    config.trainer.micro_batch_size = 1
    config.trainer.train_batch_size = 1
    config.trainer.gradient_accumulation_steps = 1
    with pytest.raises(ValueError, match="recompute_interval"):
        _config_validation(config)


@pytest.mark.parametrize("interval,skip", [(2, None), (3, None), (5, None), (1, 1), (1, 3)])
def test_offloaded_gemma_segment_matches_resident_model(monkeypatch, interval, skip):
    native, _, config = gemma4_pair(monkeypatch, "A4B")
    expected = copy.deepcopy(native)
    native.model.recompute_interval = interval
    native.model.recompute_skip_count = skip
    scheduler = HostBoundaryScheduler()
    scheduler.spill_manager = object()
    scheduler.weight_streaming_enabled = False
    native.model._offload_scheduler = scheduler
    tokens = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])
    actual = native(tokens)[0]
    reference = expected(tokens)[0]
    assert len(scheduler.saved) == len(native.model.get_recompute_segments())
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    cotangent = torch.randn_like(actual)
    actual.backward(cotangent)
    reference.backward(cotangent)
    assert not scheduler.saved
    for (name, p), (other_name, q) in zip(
        native.named_parameters(), expected.named_parameters(), strict=True
    ):
        assert name == other_name
        if q.grad is None:
            assert p.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-5, msg=name)


@pytest.mark.cuda
@pytest.mark.parametrize("variant", ["E2B", "A4B"])
@pytest.mark.parametrize("interval,skip", [(2, None), (3, None), (1, 1)])
def test_gpu_resident_segment_gradients(monkeypatch, variant, interval, skip):
    native, _, config = gemma4_pair(monkeypatch, variant)
    native.cuda()
    expected = copy.deepcopy(native)
    native.model.activation_recompute = True
    native.model.recompute_interval = interval
    native.model.recompute_skip_count = skip
    tokens = torch.tensor([[2, 3, 4, 5, 6]], device="cuda")
    actual = native(tokens)[0]
    reference = expected(tokens)[0]
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    actual.square().mean().backward()
    reference.square().mean().backward()
    for (name, p), (_, q) in zip(
        native.named_parameters(), expected.named_parameters(), strict=True
    ):
        if q.grad is None:
            assert p.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-5, msg=name)


def test_weight_streaming_keeps_activation_storage_explicit():
    from tests.unit.offload.test_config import _make_minimal_main_config

    config = _make_minimal_main_config()
    config.offload.enabled = True
    config.offload.weight_offload = True
    config.offload.activation_spill = False
    _config_validation(config)
    assert config.operation.activation_recompute
    assert not config.offload.activation_spill


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("spill", [False, True])
@pytest.mark.parametrize("interval,skip", [(2, None), (3, None), (1, 1), (1, 3)])
def test_generic_transformer_segment_auxiliary_gradients(monkeypatch, moe, spill, interval, skip):
    from tests.fixtures.config_fixtures import create_test_config

    from ironcore.models.transformer import TransformerModel

    gemma4_pair(monkeypatch, "A4B")  # Initialize the same CPU runtime stubs.
    config = create_test_config(
        d_model=16,
        num_attention_heads=4,
        num_attention_groups=4,
        head_dim=4,
        d_ffn=32,
        num_layers=5,
        max_seq_len=16,
        precision="float32",
    )
    if moe:
        config.model.moe.use_moe = True
        config.model.moe.num_routed_experts = 4
        config.model.moe.num_experts_per_token = 2
        config.model.moe.expert_intermediate_size = 16
        config.model.moe.aux_loss_alpha = 0.05
    native = TransformerModel(config)
    native.init_weights()
    expected = copy.deepcopy(native)
    native.recompute_interval = interval
    native.recompute_skip_count = skip
    native.activation_recompute = True
    if spill:
        scheduler = HostBoundaryScheduler()
        scheduler.spill_manager = object()
        scheduler.weight_streaming_enabled = False
        native._offload_scheduler = scheduler
    x = torch.randn(2, 6, 16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = native(x, None, None)
    reference = expected(reference_x, None, None)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    actual_loss = actual.square().mean()
    reference_loss = reference.square().mean()
    if moe:
        actual_loss = actual_loss + sum(layer.mlp.get_aux_loss() for layer in native.layers)
        reference_loss = reference_loss + sum(layer.mlp.get_aux_loss() for layer in expected.layers)
    actual_loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol=1e-5, rtol=1e-5)
    for (name, p), (_, q) in zip(
        native.named_parameters(), expected.named_parameters(), strict=True
    ):
        if q.grad is None:
            assert p.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-5, msg=name)
