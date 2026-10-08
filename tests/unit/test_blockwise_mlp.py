# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Block-wise MLP value/gradient and activation-retention regressions."""

from copy import deepcopy

import pytest
import torch
from tests.fixtures.config_fixtures import create_small_test_config
from tests.fixtures.utils import single_gpu_env

from ironcore.config.config_blockwise import validate_blockwise_mlp
from ironcore.config.config_moe import MoEConfig
from ironcore.layers.mlp import MLP
from ironcore.layers.moe import MoEMLP
from ironcore.parallel import parallel_states as ps
from ironcore.parallel.random import checkpoint_with_tensor_parallel_rng


@pytest.fixture(autouse=True)
def parallel_state():
    with single_gpu_env():
        ps.initialize_model_parallel(1, 10)
        yield
        ps.destroy_model_parallel()


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("expert_backend", ["loop", "batched", "grouped"])
@pytest.mark.parametrize("activation", ["gelu", "swiglu"])
@pytest.mark.parametrize("outer_checkpoint", [None, False, True])
def test_output_input_and_parameter_gradients(moe, expert_backend, activation, outer_checkpoint):
    torch.manual_seed(42)
    config = create_small_test_config(d_model=32, d_ffn=96, head_dim=8)
    config.model.activation_type = activation
    config.model.moe = MoEConfig(
        use_moe=moe,
        num_routed_experts=4,
        num_shared_experts=1,
        num_experts_per_token=2,
        aux_loss_alpha=0.03,
        expert_backend=expert_backend,
        virtual_block_size=3,
        grouped_token_budget=7,
    )
    reference = (MoEMLP if moe else MLP)(config)
    reference.init_weights()
    actual = deepcopy(reference)
    actual.config.trainer.mlp_chunk_size = 5
    # All descendants share the copied config, including shared/routed experts.
    x = torch.randn(2, 11, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected = reference(x)
    if outer_checkpoint is None:
        result = actual(y)
    else:
        result = checkpoint_with_tensor_parallel_rng(actual, y, use_reentrant=outer_checkpoint)
    torch.testing.assert_close(result, expected, atol=2e-7, rtol=2e-5)
    gradient = torch.randn_like(expected)
    expected_loss = (expected * gradient).sum()
    actual_loss = (result * gradient).sum()
    if moe and outer_checkpoint is not True:
        torch.testing.assert_close(actual.get_aux_loss(), reference.get_aux_loss())
        expected_loss = expected_loss + reference.get_aux_loss()
        actual_loss = actual_loss + actual.get_aux_loss()
    expected_loss.backward()
    actual_loss.backward()
    torch.testing.assert_close(y.grad, x.grad, atol=2e-7, rtol=3e-5)
    for (name, a), (_, b) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=5e-5, msg=name)


def test_frozen_input_keeps_parameter_gradients_and_eval_order():
    config = create_small_test_config(d_model=32, d_ffn=96, head_dim=8)
    config.trainer.mlp_chunk_size = 5
    model = MLP(config)
    model.init_weights()
    x = torch.randn(2, 11, 32)
    result = model(x)
    result.square().sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x), result)


def test_expanded_activations_are_not_retained():
    config = create_small_test_config(d_model=32, d_ffn=512, head_dim=8)
    model = MLP(config)
    model.init_weights()
    x = torch.randn(2, 31, 32, requires_grad=True)

    def saved_shapes(chunk):
        config.trainer.mlp_chunk_size = chunk
        shapes = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda t: (shapes.append(tuple(t.shape)), t)[1], lambda t: t
        ):
            model(x)
        return shapes

    full = saved_shapes(None)
    blocked = saved_shapes(7)
    assert any(512 in shape for shape in full)
    assert not any(512 in shape for shape in blocked)
    assert all(shape[0] <= 7 for shape in blocked if len(shape) == 2)


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_chunk_size(size):
    config = create_small_test_config()
    config.trainer.mlp_chunk_size = size
    with pytest.raises(ValueError, match="positive integer"):
        validate_blockwise_mlp(config)


def test_async_chunking_is_rejected():
    config = create_small_test_config()
    config.trainer.mlp_chunk_size = 5
    with pytest.raises(ValueError, match="synchronous"):
        MLP(config)(torch.randn(1, 9, 128), async_communication=True)


@pytest.mark.parametrize("reentrant", [False, True])
@pytest.mark.parametrize("expert_backend", ["loop", "batched", "grouped"])
def test_moe_checkpoint_preserves_auxiliary_gradient_and_counts(
    monkeypatch, reentrant, expert_backend
):
    from tests.fixtures.lora_tp import smollm2_lora_model

    from ironcore.training_utils import forward_step, loss_func

    torch.manual_seed(44)
    reference = smollm2_lora_model(monkeypatch, moe=True, lora=False, expert_backend=expert_backend)
    reference.loss_fn = loss_func
    actual = deepcopy(reference)
    actual.config.trainer.mlp_chunk_size = 3
    actual.model.activation_recompute = True
    actual.model.use_reentrant = reentrant
    batch = {
        "input_ids": torch.tensor([[2, 3, 5, 7, 11, 13, 17]]),
        "labels": torch.tensor([[3, 5, 7, 11, 13, 17, 19]]),
    }
    a = forward_step(reference, iter([batch]))
    b = forward_step(actual, iter([batch]))
    torch.testing.assert_close(a, b)
    a.backward()
    b.backward()
    for layer in actual.model.layers:
        assert layer.mlp._total_selections == 14
        assert layer.mlp.get_aux_loss() is None
    for (name, p), (_, q) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-7, rtol=5e-5, msg=name)


@pytest.mark.parametrize(
    "path,value,match",
    [
        ("model.dropout_mlp", 0.1, "dropout"),
        ("parallel.use_fsdp", True, "FSDP"),
        ("offload.enabled", True, "offload"),
    ],
)
def test_unsupported_chunk_contract(path, value, match):
    config = create_small_test_config()
    config.trainer.mlp_chunk_size = 3
    owner, name = path.split(".")
    setattr(getattr(config, owner), name, value)
    with pytest.raises(ValueError, match=match):
        validate_blockwise_mlp(config)
