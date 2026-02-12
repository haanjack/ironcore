# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""Unit tests for VLA configuration."""


class TestVLAConfig:
    """Tests for VLA configuration classes."""

    def test_vision_config_defaults(self):
        """Test VisionConfig default values."""
        from ironcore.config.config_vla import VisionConfig

        config = VisionConfig()

        assert config.encoder_type == "siglip"
        assert config.image_size == 384
        assert config.hidden_size == 1152
        assert config.freeze_vision is True

    def test_action_config_defaults(self):
        """Test ActionConfig default values."""
        from ironcore.config.config_vla import ActionConfig

        config = ActionConfig()

        assert config.action_dim == 7
        assert config.loss_type == "mse"
        assert config.prediction_horizon == 1

    def test_fusion_config_defaults(self):
        """Test FusionConfig default values."""
        from ironcore.config.config_vla import FusionConfig

        config = FusionConfig()

        assert config.fusion_type == "gated_cross_attention"
        assert config.num_layers == 1

    def test_projector_config_defaults(self):
        """Test ProjectorConfig default values."""
        from ironcore.config.config_vla import ProjectorConfig

        config = ProjectorConfig()

        assert config.projector_type == "mlp"
        assert config.num_layers == 2

    def test_vla_config_aggregation(self):
        """Test that VLAConfig aggregates sub-configs."""
        from ironcore.config.config_vla import VLAConfig

        config = VLAConfig()

        assert hasattr(config, "vision")
        assert hasattr(config, "action")
        assert hasattr(config, "fusion")
        assert hasattr(config, "projector")

    def test_custom_config_values(self):
        """Test creating config with custom values."""
        from ironcore.config.config_vla import ActionConfig, VisionConfig

        vision = VisionConfig(
            image_size=224,
            freeze_vision=False,
        )

        assert vision.image_size == 224
        assert vision.freeze_vision is False

        action = ActionConfig(
            action_dim=14,
            prediction_horizon=16,
        )

        assert action.action_dim == 14
        assert action.prediction_horizon == 16
