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

from ironcore.parallel import parallel_states

pytestmark = pytest.mark.rlvr


@pytest.fixture(scope="module", autouse=True)
def setup_parallel_states():
    """Initialize parallel states for testing (TP=1 by default)."""
    parallel_states.initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=10.0)
    yield
    parallel_states.destroy_model_parallel()


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


class TestGrpoLossEntropy:
    """grpo_loss entropy bonus correctness."""

    @pytest.fixture
    def base_tensors(self):
        torch.manual_seed(0)
        B = 8
        policy_lp = -torch.rand(B)
        ref_lp = policy_lp + 0.05 * torch.randn(B)
        adv = torch.randn(B)
        kl = (policy_lp - ref_lp).abs()
        entropy = torch.rand(B) * 2.0 + 0.5  # in [0.5, 2.5]
        return policy_lp, ref_lp, adv, kl, entropy

    def test_no_entropy_args_succeeds(self, base_tensors):
        """grpo_loss with no entropy args must not raise TypeError."""
        from ironcore.alignment.loss.grpo import grpo_loss

        p, r, a, kl, _ = base_tensors
        loss, metrics = grpo_loss(p, r, a, kl)
        assert torch.isfinite(loss)
        assert "entropy" in metrics

    def test_entropy_coef_zero_metric_is_zero(self, base_tensors):
        """When entropy_coef=0, the entropy metric must be 0.0."""
        from ironcore.alignment.loss.grpo import grpo_loss

        p, r, a, kl, entropy = base_tensors
        _, metrics = grpo_loss(p, r, a, kl, entropy=entropy, entropy_coef=0.0)
        assert metrics["entropy"] == 0.0, (
            f"entropy metric should be 0.0 when entropy_coef=0, got {metrics['entropy']}"
        )

    def test_entropy_metric_is_raw_mean_not_scaled(self, base_tensors):
        """When entropy_coef>0, metric must be raw mean H (not coef*H)."""
        from ironcore.alignment.loss.grpo import grpo_loss

        p, r, a, kl, entropy = base_tensors
        coef = 0.01
        _, metrics = grpo_loss(p, r, a, kl, entropy=entropy, entropy_coef=coef)

        expected = entropy.mean().item()
        assert abs(metrics["entropy"] - expected) < 1e-5, (
            f"entropy metric should be raw mean ({expected:.4f}), got {metrics['entropy']:.4f}"
        )

    def test_entropy_bonus_reduces_loss(self, base_tensors):
        """Loss with entropy bonus must be lower than without."""
        from ironcore.alignment.loss.grpo import grpo_loss

        p, r, a, kl, entropy = base_tensors
        loss_no_entropy, _ = grpo_loss(p, r, a, kl)
        loss_with_entropy, _ = grpo_loss(p, r, a, kl, entropy=entropy, entropy_coef=0.1)
        assert loss_with_entropy.item() < loss_no_entropy.item(), (
            "Entropy bonus (subtracted) should reduce loss"
        )

    def test_entropy_none_skips_bonus(self, base_tensors):
        """entropy=None must produce same loss as entropy_coef=0."""
        from ironcore.alignment.loss.grpo import grpo_loss

        p, r, a, kl, entropy = base_tensors
        loss_none, _ = grpo_loss(p, r, a, kl, entropy=None, entropy_coef=0.1)
        loss_zero_coef, _ = grpo_loss(p, r, a, kl, entropy=entropy, entropy_coef=0.0)
        loss_baseline, _ = grpo_loss(p, r, a, kl)
        assert abs(loss_none.item() - loss_baseline.item()) < 1e-6
        assert abs(loss_zero_coef.item() - loss_baseline.item()) < 1e-6

    def test_required_metrics_present(self, base_tensors):
        """All expected metric keys must be present in returned dict."""
        from ironcore.alignment.loss.grpo import grpo_loss

        p, r, a, kl, entropy = base_tensors
        _, metrics = grpo_loss(p, r, a, kl, entropy=entropy, entropy_coef=0.01)
        required = {
            "grpo_loss",
            "policy_loss",
            "kl_loss",
            "kl_per_seq",
            "entropy",
            "mean_advantage",
            "std_advantage",
            "mean_ratio",
            "clip_fraction",
        }
        missing = required - set(metrics.keys())
        assert not missing, f"Missing metric keys: {missing}"


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
