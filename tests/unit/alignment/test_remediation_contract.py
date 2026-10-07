# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

from ironcore.alignment.loss.grpo import grpo_loss
from ironcore.alignment.rewards.base import RewardWorkerPool
from ironcore.layers.linear_cross_entropy import linear_cross_entropy
from ironcore.parallel import parallel_states


@pytest.mark.parametrize("chunk", [1, 3, 19])
@pytest.mark.parametrize("transposed", [True, False])
def test_recomputed_linear_ce_matches_independent_full_projection_gradient(chunk, transposed):
    parallel_states.initialize_model_parallel(1, 10)
    try:
        torch.manual_seed(96)
        hidden = torch.randn(2, 4, 5, requires_grad=True)
        weight = torch.randn(5, 11, requires_grad=True)
        labels = torch.tensor([[0, 1, 2, -100], [4, 5, 9, 10]])
        stored = weight if transposed else weight.T.detach().requires_grad_(True)
        expected = torch.nn.functional.cross_entropy(
            (hidden @ weight).reshape(-1, 11), labels.reshape(-1), reduction="none"
        ).view_as(labels)
        valid = labels != -100
        actual = linear_cross_entropy(hidden, stored, labels, chunk, transposed=transposed)
        torch.testing.assert_close(actual[valid], expected[valid])
        actual_grads = torch.autograd.grad(
            actual[valid].mean(), (hidden, stored), retain_graph=True
        )
        expected_grads = torch.autograd.grad(expected[valid].mean(), (hidden, weight))
        torch.testing.assert_close(actual_grads[0], expected_grads[0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(
            actual_grads[1],
            expected_grads[1] if transposed else expected_grads[1].T,
            atol=1e-6,
            rtol=1e-5,
        )
    finally:
        parallel_states.destroy_model_parallel()


def test_token_grpo_clipping_matches_independent_per_completion_oracle():
    new = torch.tensor(
        [[math.log(1.5), math.log(0.7), 0], [math.log(0.5), 0, 0]], requires_grad=True
    )
    old = torch.zeros_like(new)
    adv = torch.tensor([2.0, -3.0])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0.0]])
    loss, _ = grpo_loss(
        new.sum(-1),
        old.sum(-1),
        adv,
        torch.zeros(2),
        beta=0,
        old_log_probs=old.sum(-1),
        clip_eps=0.2,
        response_lengths=mask.sum(-1),
        objective="grpo",
        token_policy_log_probs=new,
        token_old_log_probs=old,
        response_mask=mask,
    )
    # Positive A: upper clip; negative A: lower clip. Pad tokens excluded.
    expected = (
        -(
            (
                torch.minimum(new[0, 0].exp() * 2, torch.tensor(2.4))
                + torch.minimum(new[0, 1].exp() * 2, torch.tensor(1.6))
            )
            / 2
            + torch.minimum(new[1, 0].exp() * -3, torch.tensor(-2.4))
        )
        / 2
    )
    torch.testing.assert_close(loss, expected)
    (actual_gradient,) = torch.autograd.grad(loss, new, retain_graph=True)
    (expected_gradient,) = torch.autograd.grad(expected, new)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.count_nonzero(actual_gradient[mask == 0]) == 0


@pytest.mark.parametrize("result", [float("nan"), float("inf"), "raise"])
def test_reward_error_policy_is_explicit_and_never_fabricates_success(result):
    class Backend:
        def compute(self, *args):
            if result == "raise":
                raise RuntimeError("backend failed")
            return result

    with RewardWorkerPool(Backend(), num_workers=1, failure_policy="error") as worker:
        scores = worker.score_batch(["p"], ["r"], [{}])
        assert torch.isnan(scores).all()
        assert worker.get_failure_count() == 1


@pytest.mark.parametrize("sampling", [(1.0, 1.0, 0), (0.7, 0.9, 8)])
def test_padded_native_rollouts_match_unpadded_and_behaviour_rescoring(sampling):
    from tests.fixtures.config_fixtures import create_test_config

    from ironcore.alignment.rollout import _filter_logits, generate_rollouts_batched
    from ironcore.config.config_model import KVCacheConfig, PositionalEmbeddingConfig
    from ironcore.global_vars import global_states_cleanup, set_global_states
    from ironcore.language_model import LanguageModel
    from ironcore.trainers import GRPOTrainer

    config = create_test_config(
        d_model=16,
        d_ffn=32,
        num_layers=2,
        num_attention_heads=2,
        num_attention_groups=1,
        head_dim=8,
        max_seq_len=32,
        dropout_attn=0,
        dropout_mlp=0,
        dropout_embd=0,
    )
    config.model.positional_embedding = PositionalEmbeddingConfig(type="rope")
    config.model.kv_cache = KVCacheConfig(enabled=False)
    set_global_states(config)
    parallel_states.initialize_model_parallel(1, 10)
    try:
        torch.manual_seed(29)
        model = LanguageModel(config).float().eval()
        ids = torch.tensor([[0, 0, 4, 5], [3, 4, 5, 6]])
        mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        temperature, top_p, top_k = sampling
        with torch.no_grad():
            rollout = generate_rollouts_batched(
                model,
                ids,
                1,
                [{}, {}],
                max_new_tokens=4,
                do_sample=False,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                prompt_attention_mask=mask,
            )
            for row in range(2):
                single = generate_rollouts_batched(
                    model,
                    ids[row][mask[row].bool()][None],
                    1,
                    [{}],
                    max_new_tokens=4,
                    do_sample=False,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                torch.testing.assert_close(
                    rollout.response_ids[row], single.response_ids[0], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    rollout.old_token_log_probs[row],
                    single.old_token_log_probs[0],
                    atol=1e-5,
                    rtol=1e-5,
                )
            trainer = object.__new__(GRPOTrainer)
            logits, _ = model(rollout.completion_ids, **trainer._rollout_forward_kwargs(rollout))
            logits = _filter_logits(logits[:, ids.size(1) - 1 : -1], temperature, top_p, top_k)
            scored = (
                logits.float()
                .log_softmax(-1)
                .gather(-1, rollout.response_ids[..., None])
                .squeeze(-1)
            )
            torch.testing.assert_close(scored, rollout.old_token_log_probs, atol=1e-5, rtol=1e-5)
    finally:
        global_states_cleanup()
        parallel_states.destroy_model_parallel()
