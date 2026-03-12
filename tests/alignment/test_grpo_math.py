# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GRPO mathematical correctness.

Tests:
1. Advantage normalization: sum(A) = 0, std(A) = 1 (for non-constant rewards)
2. KL divergence: numerical stability and parity
3. Distributed advantage computation
"""

import pytest
import torch


class TestAdvantageNormalization:
    """Tests for advantage normalization."""

    def test_advantage_sum_to_zero(self):
        """Verify that sum of advantages within a group is exactly 0."""
        from ironcore.alignment.loss.grpo import compute_advantages

        # Create rewards for 2 prompts with 4 completions each
        # Group 0: [1.0, 2.0, 3.0, 4.0]
        # Group 1: [0.5, 1.5, 2.5, 3.5]
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5])
        group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        advantages = compute_advantages(rewards, group_ids, distributed=False)

        # Sum of advantages within each group should be 0
        for g in group_ids.unique():
            group_advantages = advantages[group_ids == g]
            assert torch.allclose(group_advantages.sum(), torch.tensor(0.0), atol=1e-6), (
                f"Group {g}: sum of advantages = {group_advantages.sum()} (expected 0)"
            )

    def test_advantage_std_is_one(self):
        """Verify that std of advantages within a group is 1 (for non-constant rewards)."""
        from ironcore.alignment.loss.grpo import compute_advantages

        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5])
        group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        advantages = compute_advantages(rewards, group_ids, distributed=False)

        # Std of advantages within each group should be 1
        for g in group_ids.unique():
            group_advantages = advantages[group_ids == g]
            std = group_advantages.std()
            assert torch.allclose(std, torch.tensor(1.0), atol=1e-5), (
                f"Group {g}: std of advantages = {std} (expected 1)"
            )

    def test_identical_rewards_zero_advantage(self):
        """Verify that identical rewards produce exactly 0 advantage."""
        from ironcore.alignment.loss.grpo import compute_advantages

        # All rewards in each group are identical
        rewards = torch.tensor([5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 3.0])
        group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        advantages = compute_advantages(rewards, group_ids, distributed=False)

        # All advantages should be exactly 0
        assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-8), (
            f"Identical rewards should produce 0 advantage, got {advantages}"
        )

    def test_single_element_group_zero_advantage(self):
        """Verify that single-element groups have 0 advantage."""
        from ironcore.alignment.loss.grpo import compute_advantages

        # Single element per group
        rewards = torch.tensor([1.0, 2.0, 3.0])
        group_ids = torch.tensor([0, 1, 2])

        advantages = compute_advantages(rewards, group_ids, distributed=False)

        # All advantages should be 0 (can't normalize single element)
        assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-8)

    def test_advantage_formula_correctness(self):
        """Verify the exact formula: A_i = (R_i - mean) / (std + eps)."""
        from ironcore.alignment.loss.grpo import compute_advantages

        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        group_ids = torch.tensor([0, 0, 0, 0])

        advantages = compute_advantages(rewards, group_ids, distributed=False)

        # Manual calculation
        mean = rewards.mean()
        std = rewards.std()
        expected = (rewards - mean) / (std + 1e-8)

        assert torch.allclose(advantages, expected, atol=1e-6), (
            f"Expected {expected}, got {advantages}"
        )


class TestKLDivergence:
    """Tests for KL divergence computation."""

    def test_kl_zero_when_identical(self):
        """KL divergence should be exactly 0 when distributions are identical."""
        from ironcore.alignment.loss.kl import kl_divergence

        # Identical distributions
        log_probs = torch.randn(2, 10, 100)  # [batch, seq, vocab]
        log_probs = torch.log_softmax(log_probs, dim=-1)

        kl = kl_divergence(log_probs, log_probs.clone())

        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6), (
            f"KL should be 0 for identical distributions, got {kl}"
        )

    def test_kl_positive_when_different(self):
        """KL divergence should be positive when distributions differ."""
        from ironcore.alignment.loss.kl import kl_divergence

        torch.manual_seed(42)
        p = torch.randn(2, 10, 100)
        q = torch.randn(2, 10, 100)

        p_log_probs = torch.log_softmax(p, dim=-1)
        q_log_probs = torch.log_softmax(q, dim=-1)

        kl = kl_divergence(q_log_probs, p_log_probs)  # KL(P || Q)

        assert (kl >= -1e-6).all(), f"KL should be non-negative, got min={kl.min()}"

    def test_kl_with_mask(self):
        """Test KL divergence with sequence masking."""
        from ironcore.alignment.loss.kl import kl_divergence

        log_probs_p = torch.randn(2, 5, 100)
        log_probs_q = torch.randn(2, 5, 100)
        log_probs_p = torch.log_softmax(log_probs_p, dim=-1)
        log_probs_q = torch.log_softmax(log_probs_q, dim=-1)

        # Mask: only compute KL for first 3 tokens
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.float)

        kl_masked = kl_divergence(log_probs_q, log_probs_p, mask)

        # Unmasked KL should be larger (includes more tokens)
        kl_unmasked = kl_divergence(log_probs_q, log_probs_p)

        assert (kl_masked <= kl_unmasked + 1e-5).all(), "Masked KL should be <= unmasked KL"

    def test_kl_numerical_stability(self):
        """Test KL divergence with extreme log probabilities."""
        from ironcore.alignment.loss.kl import kl_divergence

        # Very peaked distributions (near-deterministic)
        p = torch.zeros(2, 5, 100)
        p[:, :, 0] = 100.0  # Strong preference for token 0
        p_log_probs = torch.log_softmax(p, dim=-1)

        q = torch.zeros(2, 5, 100)
        q_log_probs = torch.log_softmax(q, dim=-1)

        kl = kl_divergence(p_log_probs, q_log_probs)

        assert not torch.isnan(kl).any(), "KL should not be NaN"
        assert not torch.isinf(kl).any(), "KL should not be infinite"

    def test_kl_from_logits_matches_direct(self):
        """Verify kl_divergence_from_logits produces same result as kl_divergence."""
        from ironcore.alignment.loss.kl import kl_divergence, kl_divergence_from_logits

        torch.manual_seed(42)
        policy_logits = torch.randn(2, 5, 100)
        ref_logits = torch.randn(2, 5, 100)

        kl_from_logits = kl_divergence_from_logits(policy_logits, ref_logits)

        policy_log_probs = torch.log_softmax(policy_logits.float(), dim=-1)
        ref_log_probs = torch.log_softmax(ref_logits.float(), dim=-1)
        kl_direct = kl_divergence(policy_log_probs, ref_log_probs)

        assert torch.allclose(kl_from_logits, kl_direct, atol=1e-5), (
            f"KL from logits: {kl_from_logits}\nDirect KL: {kl_direct}"
        )


class TestGRPOLoss:
    """Tests for GRPO loss computation."""

    def test_loss_components(self):
        """Verify GRPO loss = policy_loss + beta * kl_loss."""
        from ironcore.alignment.loss.grpo import grpo_loss

        torch.manual_seed(42)
        batch_size = 8

        policy_log_probs = torch.randn(batch_size)
        ref_log_probs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        kl_per_seq = torch.abs(torch.randn(batch_size))

        beta = 0.1
        loss, metrics = grpo_loss(policy_log_probs, ref_log_probs, advantages, kl_per_seq, beta)

        # Manual calculation
        expected_policy_loss = -(advantages.detach() * policy_log_probs).mean()
        expected_kl_loss = beta * kl_per_seq.mean()
        expected_total = expected_policy_loss + expected_kl_loss

        assert torch.allclose(loss, expected_total, atol=1e-5), (
            f"Expected {expected_total}, got {loss}"
        )
        assert torch.allclose(torch.tensor(metrics["policy_loss"]), expected_policy_loss, atol=1e-5)
        assert torch.allclose(torch.tensor(metrics["kl_loss"]), expected_kl_loss, atol=1e-5)

    def test_loss_decreases_with_positive_advantage(self):
        """When advantage is positive, increasing log prob should decrease loss."""
        from ironcore.alignment.loss.grpo import grpo_loss

        batch_size = 4
        ref_log_probs = torch.zeros(batch_size)
        advantages = torch.ones(batch_size)  # Positive advantage
        kl_per_seq = torch.zeros(batch_size)
        beta = 0.0  # No KL penalty for this test

        # Higher log prob should give lower loss with positive advantage
        low_log_probs = torch.full((batch_size,), -1.0)
        high_log_probs = torch.full((batch_size,), -0.5)

        loss_low, _ = grpo_loss(low_log_probs, ref_log_probs, advantages, kl_per_seq, beta)
        loss_high, _ = grpo_loss(high_log_probs, ref_log_probs, advantages, kl_per_seq, beta)

        assert loss_high < loss_low, (
            f"With positive advantage, higher log prob should decrease loss. "
            f"Low: {loss_low}, High: {loss_high}"
        )

    def test_metrics_dict_values(self):
        """Verify all expected metrics are present."""
        from ironcore.alignment.loss.grpo import grpo_loss

        batch_size = 8
        policy_log_probs = torch.randn(batch_size)
        ref_log_probs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        kl_per_seq = torch.randn(batch_size)

        _, metrics = grpo_loss(policy_log_probs, ref_log_probs, advantages, kl_per_seq, beta=0.1)

        expected_keys = [
            "grpo_loss",
            "policy_loss",
            "kl_loss",
            "kl_per_seq",
            "mean_advantage",
            "std_advantage",
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
            assert isinstance(metrics[key], float), f"Metric {key} should be float"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTPSafeSoftmax:
    """Tests for tensor-parallel safe softmax operations."""

    def test_log_softmax_tp_safe_single_rank(self):
        """Test that TP-safe log_softmax works correctly on single rank."""
        from ironcore.alignment.loss.dpo import _compute_log_softmax_tp_safe

        torch.manual_seed(42)
        logits = torch.randn(2, 5, 100, device="cuda")

        # Our implementation
        log_probs_tp_safe = _compute_log_softmax_tp_safe(logits)

        # Reference implementation
        log_probs_ref = torch.log_softmax(logits.float(), dim=-1)

        assert torch.allclose(log_probs_tp_safe, log_probs_ref, atol=1e-5), (
            "TP-safe log_softmax should match reference on single rank"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
