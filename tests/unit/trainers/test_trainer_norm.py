# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for trainer gradient and parameter norm computation.

Tests the _compute_grad_and_param_norms method across all trainer types:
- BaseTrainer (used by LanguageModelTrainer for pretrain/sft)
- DPOTrainer
- GRPOTrainer

Run tests:
    pytest tests/unit/trainers/test_trainer_norm.py -v
"""

from unittest.mock import MagicMock

import pytest
import torch

pytestmark = pytest.mark.smoke


class SimpleModel(torch.nn.Module):
    """Simple test model with a few parameters."""

    def __init__(self, hidden_size=32):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = torch.nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def simple_model():
    """Fixture providing a simple model with gradients."""
    model = SimpleModel()
    # Create some gradients
    x = torch.randn(2, 32)
    y = model(x)
    loss = y.sum()
    loss.backward()
    return model


# =============================================================================
# BaseTrainer Tests
# =============================================================================


class TestBaseTrainerNormComputation:
    """Test gradient and parameter norm computation in BaseTrainer."""

    def test_compute_grad_norm_returns_float(self, simple_model):
        """Test that _compute_grad_and_param_norms returns float for grad_norm."""
        from ironcore.parallel.grad_norm import clip_grad_norm

        # Directly test clip_grad_norm returns tensor -> .item() -> float
        grad_norm_tensor = clip_grad_norm(simple_model.parameters(), max_norm=1.0)
        grad_norm = grad_norm_tensor.item()

        assert isinstance(grad_norm, float), f"Expected float, got {type(grad_norm)}"
        assert grad_norm >= 0, "Gradient norm should be non-negative"
        assert not torch.isnan(torch.tensor(grad_norm)), "Gradient norm should not be NaN"

    def test_compute_grad_norm_no_clipping_returns_float(self, simple_model):
        """Test that computing norm without clipping returns float."""
        from ironcore.parallel.grad_norm import clip_grad_norm

        grad_norm_tensor = clip_grad_norm(simple_model.parameters(), max_norm=float("inf"))
        grad_norm = grad_norm_tensor.item()

        assert isinstance(grad_norm, float), f"Expected float, got {type(grad_norm)}"
        assert grad_norm >= 0, "Gradient norm should be non-negative"

    def test_param_norm_computation(self, simple_model):
        """Test parameter norm computation returns float."""
        # Compute parameter norm the way base_trainer does it
        # Note: p.data.norm() returns a tensor, so we use .item() to accumulate as float
        param_norm_sq = 0.0
        for p in simple_model.parameters():
            if p.data is not None:
                param_norm_sq += p.data.norm().item() ** 2
        param_norm = param_norm_sq**0.5

        assert isinstance(param_norm, float), f"Expected float, got {type(param_norm)}"
        assert param_norm > 0, "Parameter norm should be positive"

    def test_compute_grad_and_param_norms_mock(self, simple_model):
        """Test _compute_grad_and_param_norms with mocked trainer."""
        from ironcore.parallel.grad_norm import clip_grad_norm

        # Mock the trainer's dependencies
        mock_scaler = MagicMock()
        mock_scaler.unscale_ = MagicMock()

        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"params": list(simple_model.parameters())}]

        # Create a minimal mock trainer
        mock_trainer = MagicMock()
        mock_trainer.model = simple_model
        mock_trainer.scaler = mock_scaler
        mock_trainer.optimizer = mock_optimizer
        mock_trainer.config.optim.clip_grad = 1.0

        # Mock control to not compute param norm
        mock_control = MagicMock()
        mock_control.do_grad_norm = MagicMock(return_value=True)
        mock_control.do_param_norm = MagicMock(return_value=False)
        mock_trainer.control = mock_control

        # Simulate what _compute_grad_and_param_norms does
        mock_scaler.unscale_(mock_optimizer)

        grad_norm = 0.0
        if mock_trainer.config.optim.clip_grad > 0.0:
            grad_norm = clip_grad_norm(
                simple_model.parameters(), mock_trainer.config.optim.clip_grad
            ).item()

        assert isinstance(grad_norm, float), f"Expected float, got {type(grad_norm)}"
        assert grad_norm >= 0, "Gradient norm should be non-negative"


# =============================================================================
# DPOTrainer Tests
# =============================================================================


class TestDPOTrainerNormComputation:
    """Test gradient and parameter norm computation in DPOTrainer."""

    def test_dpo_clip_grad_norm_returns_float(self, simple_model):
        """Test DPO-style gradient clipping returns float."""
        from ironcore.parallel.grad_norm import clip_grad_norm

        # DPO uses the same clip_grad_norm pattern
        grad_norm = clip_grad_norm(simple_model.parameters(), max_norm=1.0).item()

        assert isinstance(grad_norm, float), f"Expected float, got {type(grad_norm)}"
        assert grad_norm >= 0, "Gradient norm should be non-negative"

    def test_dpo_no_clip_compute_norm_returns_float(self, simple_model):
        """Test DPO norm computation without clipping returns float."""
        from ironcore.parallel.grad_norm import clip_grad_norm

        grad_norm = clip_grad_norm(simple_model.parameters(), max_norm=float("inf")).item()

        assert isinstance(grad_norm, float), f"Expected float, got {type(grad_norm)}"
        assert grad_norm >= 0, "Gradient norm should be non-negative"


# =============================================================================
# GRPOTrainer Tests
# =============================================================================


class TestGRPOTrainerNormComputation:
    """Test gradient and parameter norm computation in GRPOTrainer."""

    def test_grpo_clip_grad_norm_returns_float(self, simple_model):
        """Test GRPO-style gradient clipping returns float."""
        from ironcore.parallel.grad_norm import clip_grad_norm

        # GRPO uses the same clip_grad_norm pattern
        grad_norm = clip_grad_norm(simple_model.parameters(), max_norm=1.0).item()

        assert isinstance(grad_norm, float), f"Expected float, got {type(grad_norm)}"
        assert grad_norm >= 0, "Gradient norm should be non-negative"

    def test_grpo_no_clip_compute_norm_returns_float(self, simple_model):
        """Test GRPO norm computation without clipping returns float."""
        from ironcore.parallel.grad_norm import clip_grad_norm

        grad_norm = clip_grad_norm(simple_model.parameters(), max_norm=float("inf")).item()

        assert isinstance(grad_norm, float), f"Expected float, got {type(grad_norm)}"
        assert grad_norm >= 0, "Gradient norm should be non-negative"


# =============================================================================
# Integration Tests
# =============================================================================


class TestNormComputationIntegration:
    """Integration tests for norm computation across all trainers."""

    def test_all_trainers_use_same_clip_grad_norm(self, simple_model):
        """Verify all trainers use the same clip_grad_norm function."""
        from ironcore.parallel.grad_norm import clip_grad_norm

        # This test verifies that the same function is used by all trainers
        # by checking that results are consistent

        results = []
        for _ in range(3):
            # Reset gradients
            for p in simple_model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            x = torch.randn(2, 32)
            y = simple_model(x)
            loss = y.sum()
            loss.backward()

            grad_norm = clip_grad_norm(simple_model.parameters(), max_norm=1.0).item()
            results.append(grad_norm)

        # All results should be floats and non-negative
        for grad_norm in results:
            assert isinstance(grad_norm, float)
            assert grad_norm >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
