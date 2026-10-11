# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Numerical regressions using real backward, optimizers and AMP on CPU."""

import copy
import logging
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ironcore.config import TrainerConfig, UtilsConfig
from ironcore.controller import TrainingControl
from ironcore.trainers import LanguageModelTrainer
from ironcore.training_utils import loss_func, loss_func_sft
from ironcore.utils import Timer


def make_trainer(model, batches, loss_fn=loss_func, accumulation=1, amp=False):
    trainer = object.__new__(LanguageModelTrainer)
    trainer.config = SimpleNamespace(
        trainer=TrainerConfig(gradient_accumulation_steps=accumulation),
        optim=SimpleNamespace(clip_grad=0.5),
        operation=SimpleNamespace(),
        utils=UtilsConfig(),
    )
    trainer.model = model
    trainer.loss_fn = loss_fn
    trainer.context = {"autocast": nullcontext()}
    trainer.scaler = torch.amp.GradScaler("cpu", enabled=amp)
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        trainer.optimizer, step_size=1, gamma=0.9
    )
    trainer.data_iterator = {"train": iter(batches)}
    trainer.timer = Timer()
    trainer.logger = logging.getLogger(__name__)
    trainer.control = TrainingControl(trainer.config)
    trainer._offload_scheduler = None

    def forward_step(module, iterator):
        batch = next(iterator)
        logits = module(batch["features"])
        labels = batch["labels"]
        losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none")
        return loss_fn(losses.view_as(labels), (labels != -100).float())

    trainer.forward_step_func = forward_step
    return trainer


@pytest.mark.parametrize("loss_fn", [loss_func, loss_func_sft])
def test_accumulation_matches_independent_full_batch_update(loss_fn):
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 7)
    reference = copy.deepcopy(model)
    features = torch.randn(8, 5, 3)
    labels = torch.randint(0, 7, (8, 5))
    if loss_fn is loss_func_sft:
        for row in range(8):
            labels[row, : row % 4] = -100
    batches = [
        {"features": features[i : i + 2], "labels": labels[i : i + 2]} for i in range(0, 8, 2)
    ]
    trainer = make_trainer(model, batches, loss_fn, accumulation=4)
    optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    logits = reference(features)
    token_losses = F.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), reduction="none"
    ).view_as(labels)
    if loss_fn is loss_func_sft:
        # Independent definition: average each response, then average samples.
        expected_loss = torch.stack(
            [token_losses[i][labels[i] != -100].mean() for i in range(8)]
        ).mean()
    else:
        expected_loss = token_losses.mean()
    expected_loss.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5)
    optimizer.step()
    actual_loss, actual_norm, _ = trainer.train_step(0)
    assert actual_loss == pytest.approx(expected_loss.item(), abs=1e-6)
    assert actual_norm == pytest.approx(expected_norm.item(), abs=1e-6)
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


def test_nonfinite_loss_cannot_update_parameters_or_scheduler():
    model = torch.nn.Linear(3, 7)
    trainer = make_trainer(
        model,
        [
            {
                "features": torch.full((2, 5, 3), float("nan")),
                "labels": torch.ones(2, 5, dtype=torch.long),
            }
        ],
    )
    before = copy.deepcopy(model.state_dict())
    scheduler_before = copy.deepcopy(trainer.lr_scheduler.state_dict())
    with pytest.raises(RuntimeError, match="NaN"):
        trainer.train_step(0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert trainer.lr_scheduler.state_dict() == scheduler_before
    assert not trainer.optimizer.state


def test_amp_overflow_skips_scheduler_and_clears_gradients():
    model = torch.nn.Linear(3, 7)
    trainer = make_trainer(model, [], amp=True)
    trainer.scaler.scale(model(torch.ones(2, 3)).sum()).backward()
    next(model.parameters()).grad.fill_(float("inf"))
    before = copy.deepcopy(model.state_dict())
    epoch_before = trainer.lr_scheduler.last_epoch
    scale_before = trainer.scaler.get_scale()
    trainer.scaler.unscale_(trainer.optimizer)
    trainer._optimizer_step()
    assert trainer.scaler.get_scale() < scale_before
    assert trainer.lr_scheduler.last_epoch == epoch_before
    assert all(p.grad is None for p in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


@pytest.mark.parametrize("loss_fn", [loss_func, loss_func_sft])
def test_empty_mask_has_finite_zero_loss_and_zero_gradient(loss_fn):
    losses = torch.randn(2, 5, requires_grad=True)
    loss = loss_fn(losses, torch.zeros_like(losses))
    loss.backward()
    assert loss.item() == 0
    torch.testing.assert_close(losses.grad, torch.zeros_like(losses))


@pytest.mark.parametrize("loss_fn", [loss_func, loss_func_sft])
def test_eval_uses_already_shifted_labels_and_training_objective(loss_fn, monkeypatch):
    logits = torch.tensor(
        [
            [[8.0, -8.0, -8.0], [-8.0, 8.0, -8.0], [-8.0, -8.0, 8.0]],
            [[8.0, -8.0, -8.0], [-8.0, 8.0, -8.0], [-8.0, -8.0, 8.0]],
        ]
    )
    labels = torch.tensor([[0, 1, 2], [-100, -100, 2]])

    class FixedLogits(torch.nn.Module):
        def forward(self, input_ids, labels=None):
            return logits, None

    # Pretend this is TP=2. Gathered logits must require no further collective.
    from ironcore.parallel import parallel_states

    monkeypatch.setattr(parallel_states, "_TENSOR_MODEL_PARALLEL_WORLD_SIZE", 2)
    trainer = object.__new__(LanguageModelTrainer)
    trainer.model = FixedLogits()
    trainer.context = {"autocast": nullcontext()}
    trainer.loss_fn = loss_fn
    actual_loss, accuracy = trainer._eval_step(
        {"input_ids": torch.zeros_like(labels), "labels": labels}
    )
    token_losses = F.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), reduction="none"
    ).view_as(labels)
    assert actual_loss == pytest.approx(loss_fn(token_losses, labels != -100).item())
    assert accuracy == 1.0


@pytest.mark.parametrize("loss_fn", [loss_func, loss_func_sft])
@pytest.mark.parametrize("chunk_size", [1, 7, 100])
def test_chunked_vocab_loss_matches_reference_and_gradient(loss_fn, chunk_size):
    from ironcore.language_model import LanguageModel

    torch.manual_seed(1337)
    logits = torch.randn(3, 5, 11, requires_grad=True)
    labels = torch.randint(0, 11, (3, 5))
    labels[0, :2] = -100
    labels[1, :4] = -100
    reference = logits.detach().clone().requires_grad_(True)
    reference_losses = F.cross_entropy(
        reference.flatten(0, 1), labels.flatten(), reduction="none"
    ).view_as(labels)
    expected = loss_fn(reference_losses, labels != -100)
    expected.backward()
    stub = SimpleNamespace(
        config=SimpleNamespace(trainer=TrainerConfig(loss_chunk_size=chunk_size)), loss_fn=loss_fn
    )
    actual = LanguageModel.compute_loss_from_logits(stub, logits, labels, labels != -100)
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(logits.grad, reference.grad, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("loss_fn", [loss_func, loss_func_sft])
def test_unequal_microbatches_and_valid_counts_match_global_objective(loss_fn):
    torch.manual_seed(57)
    model = torch.nn.Linear(3, 7)
    reference = copy.deepcopy(model)
    features = torch.randn(8, 5, 3)
    labels = torch.randint(0, 7, (8, 5))
    for row in range(8):
        labels[row, : row % 5] = -100
    labels[2].fill_(-100)
    batches = [
        {"features": features[a:b], "labels": labels[a:b]} for a, b in [(0, 1), (1, 4), (4, 8)]
    ]
    trainer = make_trainer(model, batches, loss_fn, accumulation=3)
    optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    logits = reference(features)
    losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").view_as(
        labels
    )
    if loss_fn is loss_func_sft:
        expected_loss = torch.stack(
            [losses[i][labels[i] != -100].mean() for i in range(8) if (labels[i] != -100).any()]
        ).mean()
    else:
        expected_loss = losses[labels != -100].mean()
    expected_loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5)
    optimizer.step()
    actual_loss, _, _ = trainer.train_step(0)
    assert actual_loss == pytest.approx(expected_loss.item(), abs=1e-6)
    for a, b in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_finite_forward_nonfinite_backward_cannot_update(bad):
    model = torch.nn.Linear(3, 7)
    trainer = make_trainer(
        model, [{"features": torch.ones(2, 5, 3), "labels": torch.ones(2, 5, dtype=torch.long)}]
    )
    next(model.parameters()).register_hook(lambda grad: torch.full_like(grad, bad))
    before = copy.deepcopy(model.state_dict())
    scheduler_before = copy.deepcopy(trainer.lr_scheduler.state_dict())
    with pytest.raises(RuntimeError, match="gradient detected"):
        trainer.train_step(0)
    assert not trainer.optimizer.state
    assert trainer.lr_scheduler.state_dict() == scheduler_before
    assert all(p.grad is None for p in model.parameters())
    for key in before:
        torch.testing.assert_close(model.state_dict()[key], before[key], atol=0, rtol=0)


def test_sft_document_average_is_independent_of_packed_rows():
    losses = torch.tensor([[2.0, 4.0, 10.0, 0.0], [8.0, 9.0, 12.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
    ids = torch.tensor([[0, 0, 1, -1], [2, 2, 2, -1]])
    expected = (losses[0, :2].mean() + losses[0, 2] + losses[1, :3].mean()) / 3
    actual = loss_func_sft(losses, mask, ids)
    torch.testing.assert_close(actual, expected)
    (actual_grad,) = torch.autograd.grad(actual, losses, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected, losses)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_repeated_evaluation_restarts_finite_data_and_uses_valid_counts(monkeypatch):
    batches = [
        {"labels": torch.tensor([[1, 2, -100]]), "loss": 2.0},
        {"labels": torch.tensor([[1, -100, -100]]), "loss": 8.0},
    ]
    model = torch.nn.Linear(3, 7)
    trainer = make_trainer(model, [])
    trainer.config.data = SimpleNamespace(task_type="pretrain")
    trainer.config.operation.eval_samples = 100
    trainer.config.trainer.eval_batch_size = 1
    trainer.data_iterator["eval"] = iter(batches)
    trainer._get_data_iterator = lambda: {"eval": iter(batches)}
    trainer._eval_step = lambda batch: (batch["loss"], 1.0)
    trainer.evaluators = []
    metrics = []
    monkeypatch.setattr(
        "ironcore.trainers.base_trainer.log_metrics", lambda row, step: metrics.append(row)
    )
    for step in [1, 2]:
        trainer.evaluate(step)
        assert metrics[-1]["eval_loss"] == pytest.approx(4.0)
        assert metrics[-1]["eval_accuracy"] == 1.0
    trainer._get_data_iterator = lambda: {"eval": iter([])}
    with pytest.raises(ValueError, match="no valid"):
        trainer.evaluate(3)


def test_unwarped_grpo_bf16_gradient_matches_single_fp32_loss_graph():
    """Policy/KL gradients must combine before a single cast back to BF16."""
    from ironcore.alignment.buffer import RolloutBuffer
    from ironcore.parallel import parallel_states
    from ironcore.trainers import GRPOTrainer

    class Logits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.randn(4, 6, 32, dtype=torch.bfloat16))

        def forward(self, *args, **kwargs):
            return self.logits, None

    parallel_states.initialize_model_parallel(1, 10)
    try:
        torch.manual_seed(47)
        model = Logits()
        trainer = object.__new__(GRPOTrainer)
        trainer.model = model
        trainer.config = SimpleNamespace(
            alignment=SimpleNamespace(
                grpo_objective="grpo",
                moe_aux_loss="include",
                generation=SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0),
            )
        )
        trainer.beta, trainer.clip_eps, trainer.entropy_coef = 0.1, 0.2, 0.0
        ids = torch.randint(0, 32, (4, 6))
        rollout = RolloutBuffer(
            prompt_ids=ids[:1, :2],
            prompt_attention_mask=torch.ones(1, 2),
            completion_ids=ids,
            response_ids=ids[:, 2:],
            old_log_probs=torch.zeros(4),
            old_token_log_probs=torch.zeros(4, 4),
            rewards=torch.zeros(4),
            advantages=torch.zeros(4),
            group_ids=torch.zeros(4, dtype=torch.long),
            metadata=[{}] * 4,
            response_lengths=torch.tensor([4, 3, 2, 4]),
        )
        labels, mask = trainer._prepare_labels_and_mask(rollout)
        selected_ids = labels.clamp(min=0).unsqueeze(-1)
        logp = model.logits.float().log_softmax(-1).gather(-1, selected_ids).squeeze(-1) * mask
        reference = logp.detach() + torch.randn_like(logp) * 0.7
        old = logp.detach() + torch.randn_like(logp) * 0.3
        rollout.old_token_log_probs = old[:, 1:5].clone()
        advantages = torch.tensor([1.0, -0.5, 0.75, -1.5])
        loss, _ = trainer._compute_grpo_loss(
            rollout, advantages, reference, old_log_probs=torch.zeros(4)
        )
        (actual_gradient,) = torch.autograd.grad(loss, model.logits)

        # Independent FP32 token objective, with variable lengths and signed clipping.
        fp32 = model.logits.detach().float().requires_grad_()
        token_logp = fp32.log_softmax(-1).gather(-1, selected_ids).squeeze(-1)
        ratio = (token_logp - old).exp()
        surrogate = torch.minimum(
            ratio * advantages[:, None], ratio.clamp(0.8, 1.2) * advantages[:, None]
        )
        delta = (reference - token_logp).clamp(-6, 6)
        kl = delta.exp() - delta - 1
        expected = (((-surrogate + 0.1 * kl) * mask).sum(-1) / mask.sum(-1)).mean()
        (fp32_gradient,) = torch.autograd.grad(expected, fp32)
        torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual_gradient, fp32_gradient.bfloat16(), atol=0, rtol=0)
    finally:
        parallel_states.destroy_model_parallel()
