# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Independent sparse-mixture forward/gradient and router precision checks."""

import pytest
import torch
import torch.nn.functional as F
from tests.fixtures.config_fixtures import create_moe_test_config
from tests.fixtures.utils import single_gpu_env

from ironcore.layers.moe import MoEMLP
from ironcore.parallel.parallel_states import destroy_model_parallel, initialize_model_parallel


@pytest.fixture(autouse=True)
def parallel_state():
    with single_gpu_env():
        initialize_model_parallel(1, timeout_in_minutes=1)
        yield
        destroy_model_parallel()


def config():
    cfg = create_moe_test_config(
        hidden_size=32,
        intermediate_size=32,
        num_shared_experts=1,
        num_routed_experts=4,
        num_experts_per_token=2,
        aux_loss_alpha=0.1,
        mlp_bias=False,
    )
    cfg.model.activation_type = "swiglu"
    return cfg


def test_router_preserves_fp32_under_bfloat16_autocast():
    torch.manual_seed(42)
    layer = MoEMLP(config())
    layer.init_weights()
    x = torch.randn(2, 8, 32).bfloat16()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = layer.router(x)
    expected = x.float() @ layer.router.weight.float()
    assert result.router_logits.dtype == torch.float32
    torch.testing.assert_close(result.router_logits, expected, rtol=0, atol=0)
    top_values, top_indices = torch.topk(expected, 2, dim=-1)
    assert torch.equal(result.topk_indices, top_indices)
    torch.testing.assert_close(result.topk_weights, F.softmax(top_values, dim=-1).bfloat16())


@pytest.mark.parametrize("blockwise", ["torch", "scheduled", "triton"])
@pytest.mark.parametrize("backend", ["batched", "grouped"])
def test_streaming_frozen_input_keeps_expert_and_router_derivatives(blockwise, backend):
    import copy

    torch.manual_seed(92)
    reference = MoEMLP(config())
    reference.init_weights()
    actual = copy.deepcopy(reference)
    actual.expert_backend = backend
    actual.config.trainer.mlp_chunk_size = 3
    actual.config.model.moe.blockwise_backend = blockwise
    actual.config.model.moe.virtual_block_size = 3
    actual.config.model.moe.grouped_token_budget = 7
    x = torch.randn(2, 13, 32)
    expected, result = reference(x), actual(x)
    torch.testing.assert_close(result, expected, atol=1e-7, rtol=1e-5)
    (expected.square().sum() + reference.get_aux_loss()).backward()
    (result.square().sum() + actual.get_aux_loss()).backward()
    for (name, p), (_, q) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-7, rtol=3e-5, msg=name)


def test_sparse_mixture_and_aux_gradient_match_independent_equations():
    torch.manual_seed(42)
    layer = MoEMLP(config())
    layer.init_weights()
    x = torch.randn(2, 8, 32, requires_grad=True)
    parameters = dict(layer.named_parameters())
    copies = {name: p.detach().clone().requires_grad_() for name, p in parameters.items()}
    oracle_x = x.detach().clone().requires_grad_()

    def expert(prefix):
        gate, up = (oracle_x @ copies[prefix + ".up_proj.weight"]).chunk(2, dim=-1)
        return (F.silu(gate) * up) @ copies[prefix + ".down_proj.weight"]

    logits = oracle_x @ copies["router.weight"]
    values, indices = logits.topk(2, dim=-1)
    weights = values.softmax(dim=-1)
    expected = expert("shared_experts.0")
    for i in range(4):
        token_weight = (weights * (indices == i)).sum(dim=-1, keepdim=True)
        expected = expected + token_weight * expert(f"routed_experts.{i}")
    fractions = torch.stack([(indices == i).float().sum() / indices.numel() for i in range(4)])
    expected_aux = 0.1 * 4 * (fractions * logits.softmax(dim=-1).mean(dim=(0, 1))).sum()
    actual = layer(x)
    aux = layer.get_aux_loss()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-8)
    torch.testing.assert_close(aux, expected_aux, rtol=0, atol=0)
    (actual.square().sum() + aux).backward()
    (expected.square().sum() + expected_aux).backward()
    torch.testing.assert_close(x.grad, oracle_x.grad, rtol=3e-5, atol=1e-8)
    for name, p in parameters.items():
        actual_grad = p.grad if p.grad is not None else torch.zeros_like(p)
        torch.testing.assert_close(actual_grad, copies[name].grad, rtol=3e-5, atol=1e-8, msg=name)


def test_unsupported_ep_topology_rejected_before_distributed_initialization(monkeypatch):
    from ironcore.trainers import LanguageModelTrainer

    trainer = object.__new__(LanguageModelTrainer)
    trainer.config = config()
    trainer.config.model.moe.expert_model_parallel_size = 2
    trainer._initialized = False
    initialized = []
    monkeypatch.setattr("ironcore.trainers.base_trainer.initialize_process", initialized.append)
    with pytest.raises(ValueError, match="EP>1 trainer requires"):
        trainer._initialize()
    assert not initialized


@pytest.mark.parametrize("idle", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("backend,chunk", [("batched", None), ("batched", 3), ("grouped", None)])
@pytest.mark.parametrize("blockwise", ["torch", "scheduled", "triton"])
def test_batched_experts_match_loop_gradients_and_unused_parameters(
    idle, bias, backend, chunk, blockwise
):
    import copy

    cfg = config()
    cfg.model.bias.up = cfg.model.bias.gate = cfg.model.bias.down = bias
    loop = MoEMLP(cfg)
    loop.init_weights()
    grouped = copy.deepcopy(loop)
    grouped.expert_backend = backend
    grouped.config.trainer.mlp_chunk_size = (
        3 if blockwise != "torch" and backend == "batched" else chunk
    )
    grouped.config.model.moe.blockwise_backend = blockwise
    grouped.config.model.moe.virtual_block_size = 3
    grouped.config.model.moe.grouped_token_budget = 7
    if idle:
        for layer in [loop, grouped]:
            with torch.no_grad():
                layer.router.weight.zero_()
                layer.router.weight[:, 0] = 2
                layer.router.weight[:, 1] = 1
                layer.router.weight[:, 2] = -2
                layer.router.weight[:, 3] = -3
    torch.manual_seed(52)
    x = torch.randn(2, 7, 32, requires_grad=True)
    if idle:
        x.data.abs_()
    gx = x.detach().clone().requires_grad_(True)
    a, b = loop(x), grouped(gx)
    torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(x.grad, gx.grad, atol=1e-7, rtol=1e-5)
    for (name, p), (gn, q) in zip(loop.named_parameters(), grouped.named_parameters(), strict=True):
        assert name == gn
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("task", ["dpo-concat", "dpo-separate", "grpo", "gspo"])
def test_alignment_adds_only_policy_auxiliary_loss_and_gradient(task):
    import copy
    from types import SimpleNamespace

    from ironcore.alignment.rollout import _build_rollout_output
    from ironcore.config.config_alignment import GenerationConfig
    from ironcore.trainers import DPOTrainer, GRPOTrainer

    class AuxModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.1))
            self.aux = None

        def forward(self, ids, **kwargs):
            self.aux = self.scale * ids.float().mean()
            logits = F.one_hot(ids % 8, 8).float() * self.scale
            return logits, None

        def get_aux_loss(self):
            return self.aux

        def clear_aux_loss(self):
            self.aux = None

    model = AuxModel()
    reference = copy.deepcopy(model).requires_grad_(False)
    trainer = object.__new__(DPOTrainer if task.startswith("dpo") else GRPOTrainer)
    trainer.model = model
    trainer.reference_model = reference
    trainer.config = SimpleNamespace(
        alignment=SimpleNamespace(
            moe_aux_loss="include",
            grpo_objective=task if task in ("grpo", "gspo") else "gspo",
            generation=GenerationConfig(top_p=1.0),
        )
    )
    trainer.beta = 0.1
    trainer.label_smoothing = 0.0
    trainer.clip_eps = 0.2
    trainer.entropy_coef = 0.0
    trainer.concat_forward_passes = task != "dpo-separate"
    ids = torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6]])
    if task.startswith("dpo"):
        batch = {
            "chosen_input_ids": ids,
            "rejected_input_ids": ids.flip(-1),
            "chosen_labels": ids,
            "rejected_labels": ids.flip(-1),
        }

        def objective():
            return trainer._dpo_forward_step(batch, compute_metrics=False)[0]

        expected_aux = model.scale * ids.float().mean()
    else:
        rollout = _build_rollout_output(
            ids,
            torch.tensor([[4, 5], [2, 3]]),
            [torch.full((2,), -1.0), torch.full((2,), -1.0)],
            torch.tensor([2, 2]),
            1,
            [{}, {}],
        )

        def objective():
            return trainer._compute_grpo_loss(
                rollout,
                torch.tensor([1.0, -1.0]),
                torch.zeros_like(rollout.completion_ids, dtype=torch.float),
                rollout.old_log_probs,
            )[0]

        expected_aux = model.scale * rollout.completion_ids.float().mean()
    included = objective()
    trainer.config.alignment.moe_aux_loss = "disable"
    disabled = objective()
    torch.testing.assert_close(included - disabled, expected_aux, atol=1e-7, rtol=1e-6)
    actual_grad = torch.autograd.grad(included - disabled, model.scale, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected_aux, model.scale)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-7, rtol=1e-6)
    assert reference.scale.grad is None and not reference.scale.requires_grad
    assert reference.aux is None and model.aux is None


def test_ep_autocast_scatter_matches_independent_sparse_gradient(monkeypatch):
    import copy

    from ironcore.parallel.expert_parallel import training

    monkeypatch.setattr(training, "get_expert_model_parallel_group", lambda: None)

    def exchange(output, input_tensor, *args, **kwargs):
        output.copy_(input_tensor)

    monkeypatch.setattr(training.dist, "all_to_all_single", exchange)
    torch.manual_seed(74)
    experts = torch.nn.ModuleList([torch.nn.Linear(5, 5, bias=False) for _ in range(4)])
    references = copy.deepcopy(experts)
    x = torch.randn(2, 7, 5, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    indices = torch.randint(0, 4, (2, 7, 2))
    weights = torch.rand(2, 7, 2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = training.route_expert_tokens(x, indices, weights, experts, 0, 4, 1)
        expected = sum(
            layer(reference_x).float() * (weights * (indices == i)).sum(-1, keepdim=True)
            for i, layer in enumerate(references)
        )
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol=0.02, rtol=0.02)
    for expert, reference in zip(experts, references, strict=True):
        torch.testing.assert_close(expert.weight.grad, reference.weight.grad, atol=0.04, rtol=0.02)
