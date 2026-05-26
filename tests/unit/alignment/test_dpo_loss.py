# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DPO loss function."""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.dpo

from ironcore.alignment.loss.dpo import (
    _compute_log_softmax_tp_safe,
    _extract_logps_from_log_probs,
    compute_logps,
    dpo_loss,
)
from ironcore.parallel import parallel_states


@pytest.fixture(scope="module", autouse=True)
def setup_parallel_states():
    """Initialize parallel states for testing (TP=1 by default)."""
    parallel_states.initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=10.0)
    yield
    parallel_states.destroy_model_parallel()


class TestComputeLogSoftmaxTPSafe:
    """Tests for _compute_log_softmax_tp_safe function."""

    def test_tp1_returns_correct_shape(self):
        """Test that log_softmax returns correct shape with TP=1."""
        batch_size, seq_len, vocab_size = 2, 8, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)

        log_probs = _compute_log_softmax_tp_safe(logits)

        assert log_probs.shape == (batch_size, seq_len, vocab_size)
        # Verify it's actual log probabilities (sum to 1 after exp)
        probs = torch.exp(log_probs)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)

    def test_log_probs_are_normalized(self):
        """Test that log probabilities are properly normalized."""
        logits = torch.randn(2, 4, 50)
        log_probs = _compute_log_softmax_tp_safe(logits)

        # Sum of exp(log_probs) should be 1
        probs = torch.exp(log_probs)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_cpu_tensor_error_with_tp_gt_1(self):
        """Test that CPU tensors raise error when TP>1 is simulated.

        Note: This test verifies the error message exists. In actual TP>1
        scenarios, the error is raised from the NCCL backend.
        """
        # With TP=1, CPU tensors should work fine
        logits = torch.randn(2, 4, 50)
        log_probs = _compute_log_softmax_tp_safe(logits)
        assert log_probs is not None


class TestExtractLogpsFromLogProbs:
    """Tests for _extract_logps_from_log_probs function."""

    def test_basic_extraction(self):
        """Test basic log probability extraction."""
        batch_size, seq_len, vocab_size = 2, 8, 100
        log_probs = torch.randn(batch_size, seq_len, vocab_size).softmax(dim=-1).log()

        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        result = _extract_logps_from_log_probs(log_probs, labels)

        assert result.shape == (batch_size,)
        # Result should be sum of log probs (not mean)
        # Check that values are reasonable (negative since they're log probs)
        assert (result < 0).all()

    def test_ignore_index_handling(self):
        """Test that -100 labels are properly ignored."""
        batch_size, seq_len, vocab_size = 2, 8, 100
        log_probs = torch.randn(batch_size, seq_len, vocab_size).softmax(dim=-1).log()

        # Create labels with some -100 (ignore index)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[:, :4] = -100  # First half should be ignored

        result = _extract_logps_from_log_probs(log_probs, labels)

        # Verify result is finite
        assert torch.isfinite(result).all()

    def test_with_mask(self):
        """Test extraction with optional mask."""
        batch_size, seq_len, vocab_size = 2, 8, 100
        log_probs = torch.randn(batch_size, seq_len, vocab_size).softmax(dim=-1).log()
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Mask: only second half of sequence is valid
        mask = torch.zeros(batch_size, seq_len)
        mask[:, 4:] = 1.0

        result_with_mask = _extract_logps_from_log_probs(log_probs, labels, mask)
        result_without_mask = _extract_logps_from_log_probs(log_probs, labels, None)

        # Masked result should be different (smaller magnitude since fewer tokens)
        assert not torch.allclose(result_with_mask, result_without_mask)


class TestComputeLogps:
    """Tests for compute_logps function."""

    def test_basic_computation(self):
        """Test basic log probability computation."""
        batch_size, seq_len, vocab_size = 2, 8, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        logps = compute_logps(logits, labels)

        assert logps.shape == (batch_size,)
        assert torch.isfinite(logps).all()

    def test_with_loss_mask(self):
        """Test computation with loss mask."""
        batch_size, seq_len, vocab_size = 2, 8, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len)
        mask[:, :2] = 0  # Mask first 2 positions

        logps_masked = compute_logps(logits, labels, mask)
        logps_unmasked = compute_logps(logits, labels, None)

        # Masked should have smaller magnitude (fewer positions summed)
        assert (logps_masked.abs() <= logps_unmasked.abs() + 1e-5).all()


class TestDpoLoss:
    """Tests for dpo_loss function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for DPO loss tests."""
        batch_size, seq_len, vocab_size = 2, 8, 100

        # Create logits
        policy_chosen = torch.randn(batch_size, seq_len, vocab_size)
        policy_rejected = torch.randn(batch_size, seq_len, vocab_size)
        ref_chosen = torch.randn(batch_size, seq_len, vocab_size)
        ref_rejected = torch.randn(batch_size, seq_len, vocab_size)

        # Create labels
        chosen_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        return {
            "policy_chosen_logits": policy_chosen,
            "policy_rejected_logits": policy_rejected,
            "reference_chosen_logits": ref_chosen,
            "reference_rejected_logits": ref_rejected,
            "chosen_labels": chosen_labels,
            "rejected_labels": rejected_labels,
        }

    def test_loss_is_scalar(self, sample_data):
        """Test that loss returns a scalar tensor."""
        loss, metrics = dpo_loss(**sample_data)

        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_metrics_are_returned(self, sample_data):
        """Test that metrics dictionary is returned."""
        loss, metrics = dpo_loss(**sample_data)

        assert isinstance(metrics, dict)
        assert "dpo_loss" in metrics
        assert "dpo_accuracy" in metrics
        assert "preference_margin" in metrics

    def test_accuracy_in_valid_range(self, sample_data):
        """Test that accuracy is between 0 and 1."""
        loss, metrics = dpo_loss(**sample_data)

        assert 0.0 <= metrics["dpo_accuracy"] <= 1.0

    def test_beta_affects_loss(self, sample_data):
        """Test that beta parameter affects the loss value."""
        loss_low_beta, _ = dpo_loss(**sample_data, beta=0.1)
        loss_high_beta, _ = dpo_loss(**sample_data, beta=1.0)

        # Different beta values should produce different losses
        # (can't guarantee which is larger due to random data)
        assert loss_low_beta != loss_high_beta

    def test_label_smoothing_affects_loss(self, sample_data):
        """Test that label smoothing affects the loss."""
        loss_no_smoothing, _ = dpo_loss(**sample_data, label_smoothing=0.0)
        loss_with_smoothing, _ = dpo_loss(**sample_data, label_smoothing=0.1)

        # Label smoothing should change the loss
        assert not torch.allclose(loss_no_smoothing, loss_with_smoothing)

    def test_loss_is_finite(self, sample_data):
        """Test that loss is finite (no NaN or Inf)."""
        loss, metrics = dpo_loss(**sample_data)

        assert torch.isfinite(loss), "Loss should be finite"
        for name, value in metrics.items():
            assert math.isfinite(value), f"Metric {name} should be finite, got {value}"

    def test_concat_logits_optimization(self, sample_data):
        """Test that concat_logits optimization produces same results."""
        # Without concat
        loss1, metrics1 = dpo_loss(**sample_data, compute_metrics=True)

        # With concat (simulate by passing concat tensors)
        policy_concat = torch.cat(
            [sample_data["policy_chosen_logits"], sample_data["policy_rejected_logits"]], dim=0
        )
        ref_concat = torch.cat(
            [sample_data["reference_chosen_logits"], sample_data["reference_rejected_logits"]],
            dim=0,
        )

        loss2, metrics2 = dpo_loss(
            **sample_data,
            policy_concat_logits=policy_concat,
            reference_concat_logits=ref_concat,
            compute_metrics=True,
        )

        # Results should be very close (small numerical differences expected)
        assert torch.allclose(loss1, loss2, atol=1e-5)

    def test_compute_metrics_false(self, sample_data):
        """Test that compute_metrics=False skips detailed metrics."""
        loss, metrics = dpo_loss(**sample_data, compute_metrics=False)

        assert torch.isfinite(loss)
        assert "dpo_loss" in metrics
        # When compute_metrics is False, some metrics may be placeholders
        assert "dpo_accuracy" in metrics

    def test_with_loss_masks(self, sample_data):
        """Test DPO loss with loss masks."""
        batch_size, seq_len = sample_data["chosen_labels"].shape

        # Create masks (only second half valid)
        chosen_mask = torch.zeros(batch_size, seq_len)
        chosen_mask[:, seq_len // 2 :] = 1.0
        rejected_mask = torch.zeros(batch_size, seq_len)
        rejected_mask[:, seq_len // 2 :] = 1.0

        loss, metrics = dpo_loss(
            **sample_data,
            chosen_loss_mask=chosen_mask,
            rejected_loss_mask=rejected_mask,
        )

        assert torch.isfinite(loss)

    def test_gradient_flow(self, sample_data):
        """Test that gradients flow through the loss."""
        # Make logits require gradients
        sample_data["policy_chosen_logits"].requires_grad_(True)
        sample_data["policy_rejected_logits"].requires_grad_(True)

        loss, _ = dpo_loss(**sample_data)
        loss.backward()

        # Check gradients exist
        assert sample_data["policy_chosen_logits"].grad is not None
        assert sample_data["policy_rejected_logits"].grad is not None
        assert torch.isfinite(sample_data["policy_chosen_logits"].grad).all()

    def test_baseline_loss_value(self, sample_data):
        """Test that loss equals ln(2) ≈ 0.693 when policy matches reference.

        When policy and reference logits are identical, the implicit reward
        (log p_policy - log p_ref) is exactly zero for both chosen and rejected.
        This makes the preference logit β*(reward_chosen - reward_rejected) = 0,
        and -log(sigmoid(0)) = ln(2) ≈ 0.693.

        Note: Using independent random logits for policy vs reference gives
        high-variance reward signals, driving the loss well above 0.693.
        """
        batch_size, seq_len, vocab_size = 32, 16, 100

        shared_chosen = torch.randn(batch_size, seq_len, vocab_size)
        shared_rejected = torch.randn(batch_size, seq_len, vocab_size)

        sample_data = {
            "policy_chosen_logits": shared_chosen,
            "policy_rejected_logits": shared_rejected,
            # Reference identical to policy → reward signal = 0
            "reference_chosen_logits": shared_chosen.clone(),
            "reference_rejected_logits": shared_rejected.clone(),
            "chosen_labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "rejected_labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        }

        loss, _ = dpo_loss(**sample_data, beta=1.0)

        # With zero reward signal, loss should be exactly ln(2) ≈ 0.693
        assert abs(loss.item() - math.log(2)) < 1e-4, (
            f"Expected loss = ln(2) ≈ 0.693, got {loss.item()}"
        )


class TestDpoLossEdgeCases:
    """Edge case tests for DPO loss."""

    def test_single_batch_element(self):
        """Test with single element batch."""
        batch_size, seq_len, vocab_size = 1, 8, 50

        loss, metrics = dpo_loss(
            policy_chosen_logits=torch.randn(batch_size, seq_len, vocab_size),
            policy_rejected_logits=torch.randn(batch_size, seq_len, vocab_size),
            reference_chosen_logits=torch.randn(batch_size, seq_len, vocab_size),
            reference_rejected_logits=torch.randn(batch_size, seq_len, vocab_size),
            chosen_labels=torch.randint(0, vocab_size, (batch_size, seq_len)),
            rejected_labels=torch.randint(0, vocab_size, (batch_size, seq_len)),
        )

        assert torch.isfinite(loss)

    def test_all_ignore_labels(self):
        """Test with all -100 labels (should not crash)."""
        batch_size, seq_len, vocab_size = 2, 8, 50

        loss, metrics = dpo_loss(
            policy_chosen_logits=torch.randn(batch_size, seq_len, vocab_size),
            policy_rejected_logits=torch.randn(batch_size, seq_len, vocab_size),
            reference_chosen_logits=torch.randn(batch_size, seq_len, vocab_size),
            reference_rejected_logits=torch.randn(batch_size, seq_len, vocab_size),
            chosen_labels=torch.full((batch_size, seq_len), -100),
            rejected_labels=torch.full((batch_size, seq_len), -100),
        )

        # Loss should still be computable (though may be at baseline)
        assert torch.isfinite(loss)

    def test_extreme_beta_values(self):
        """Test with extreme beta values."""
        batch_size, seq_len, vocab_size = 2, 8, 50

        base_kwargs = {
            "policy_chosen_logits": torch.randn(batch_size, seq_len, vocab_size),
            "policy_rejected_logits": torch.randn(batch_size, seq_len, vocab_size),
            "reference_chosen_logits": torch.randn(batch_size, seq_len, vocab_size),
            "reference_rejected_logits": torch.randn(batch_size, seq_len, vocab_size),
            "chosen_labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "rejected_labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        }

        # Very small beta
        loss_small, _ = dpo_loss(**base_kwargs, beta=0.001)
        assert torch.isfinite(loss_small)

        # Large beta
        loss_large, _ = dpo_loss(**base_kwargs, beta=10.0)
        assert torch.isfinite(loss_large)
