# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""Unit tests for action components."""

import torch


class TestActionLoss:
    """Tests for ActionLoss."""

    def test_mse_loss(self):
        """Test MSE action loss."""
        from tests.fixtures.config_fixtures import MockVLAConfig

        from ironcore.action.loss import ActionLoss

        config = MockVLAConfig().action
        loss_fn = ActionLoss(config)

        pred = torch.randn(4, 7)
        target = torch.randn(4, 7)

        loss = loss_fn(pred, target)

        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0

    def test_l1_loss(self):
        """Test L1 action loss."""
        from tests.fixtures.config_fixtures import MockVLAConfig

        from ironcore.action.loss import ActionLoss

        config = MockVLAConfig().action
        config.loss_type = "l1"
        loss_fn = ActionLoss(config)

        pred = torch.randn(4, 7)
        target = torch.randn(4, 7)

        loss = loss_fn(pred, target)
        assert loss.ndim == 0

    def test_with_mask(self):
        """Test loss with action mask."""
        from tests.fixtures.config_fixtures import MockVLAConfig

        from ironcore.action.loss import ActionLoss

        config = MockVLAConfig().action
        loss_fn = ActionLoss(config)

        pred = torch.randn(4, 7)
        target = torch.randn(4, 7)
        mask = torch.ones(4, 7)
        mask[:, -3:] = 0  # Mask last 3 dimensions

        loss = loss_fn(pred, target, mask)
        assert loss.ndim == 0


class TestActionNormalizer:
    """Tests for ActionNormalizer."""

    def test_gaussian_normalization(self):
        """Test Gaussian normalization."""
        from ironcore.action.normalizer import ActionNormalizer

        normalizer = ActionNormalizer(action_dim=7, mode="gaussian")

        # Fit on data with known statistics
        actions = torch.randn(100, 7) * 2 + 1  # mean~1, std~2
        normalizer.fit(actions)

        normalized = normalizer(actions[:10])

        # Should be approximately zero-mean and unit-variance
        assert normalized.mean().abs() < 0.5

    def test_inverse_normalization(self):
        """Test that inverse recovers original values."""
        from ironcore.action.normalizer import ActionNormalizer

        normalizer = ActionNormalizer(action_dim=7, mode="gaussian")

        actions = torch.randn(100, 7) * 3 + 2
        normalizer.fit(actions)

        normalized = normalizer(actions)
        recovered = normalizer.inverse(normalized)

        assert torch.allclose(recovered, actions, atol=1e-5)

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        from ironcore.action.normalizer import ActionNormalizer

        normalizer = ActionNormalizer(action_dim=7, mode="minmax")

        actions = torch.randn(100, 7) * 5
        normalizer.fit(actions)

        normalized = normalizer(actions)

        # Should be in [-1, 1] range
        assert normalized.min() >= -1.0
        assert normalized.max() <= 1.0


class TestActionHead:
    """Tests for ActionHead."""

    def test_action_head_forward(self):
        """Test action head forward pass."""
        from tests.fixtures.config_fixtures import MockConfig, MockVLAConfig

        from ironcore.action.head import ActionHead
        from ironcore.parallel import parallel_states

        # Initialize parallel states
        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

        config = MockConfig()
        config.vla = MockVLAConfig()
        config.model.d_model = 1024

        head = ActionHead(config)

        hidden_states = torch.randn(2, 10, 1024)
        actions = head(hidden_states)

        assert actions.shape == (2, 7)  # action_dim * horizon

    def test_reshape_predictions(self):
        """Test reshaping predictions with horizon."""
        from tests.fixtures.config_fixtures import MockConfig, MockVLAConfig

        from ironcore.action.head import ActionHead
        from ironcore.parallel import parallel_states

        # Initialize parallel states
        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

        config = MockConfig()
        config.vla = MockVLAConfig()
        config.model.d_model = 1024

        # Set prediction horizon to 3
        config.vla.action.prediction_horizon = 3
        config.vla.action.action_dim = 7

        head = ActionHead(config)

        # Flat predictions
        predictions = torch.randn(2, 21)  # 7 * 3

        reshaped = head.reshape_predictions(predictions)

        assert reshaped.shape == (2, 3, 7)
