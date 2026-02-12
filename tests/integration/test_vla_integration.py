# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""Integration tests for VLA model."""

import pytest
import torch


@pytest.mark.gpu
class TestVLAModelIntegration:
    """Integration tests for VLAModel."""

    @pytest.fixture
    def init_parallel(self):
        """Initialize parallel states."""
        from ironcore.parallel import parallel_states

        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

    @pytest.fixture
    def vla_config(self):
        """Create VLA config."""
        from tests.fixtures.config_fixtures import MockConfig, MockVLAConfig

        config = MockConfig()
        config.vla = MockVLAConfig()
        config.model.d_model = 1024
        config.model.d_ffn = 4096
        return config

    def test_vla_config_imports(self):
        """Test that all VLA config classes can be imported."""
        from ironcore.config.config_vla import (
            ActionConfig,
            FusionConfig,
            ProjectorConfig,
            VisionConfig,
            VLAConfig,
        )

        assert VisionConfig is not None
        assert ActionConfig is not None
        assert FusionConfig is not None
        assert ProjectorConfig is not None
        assert VLAConfig is not None

    def test_action_components_integration(self, vla_config):
        """Test action components work together."""
        from ironcore.action.head import ActionHead
        from ironcore.action.loss import ActionLoss
        from ironcore.action.normalizer import ActionNormalizer

        # Create components
        head = ActionHead(vla_config)
        loss_fn = ActionLoss(vla_config.vla.action)
        normalizer = ActionNormalizer(
            action_dim=vla_config.vla.action.action_dim,
            mode="gaussian",
        )

        # Simulate training loop
        hidden_states = torch.randn(4, 16, 1024)
        target_actions = torch.randn(4, 7)

        # Fit normalizer
        normalizer.fit(target_actions)

        # Predict and compute loss
        pred_actions = head(hidden_states)
        normalized_target = normalizer(target_actions)

        loss = loss_fn(pred_actions, normalized_target)

        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_cross_attention_integration(self, init_parallel, vla_config):
        """Test cross-attention with vision and language features."""
        from ironcore.layers.cross_attention import VisionLanguageFusion

        fusion = VisionLanguageFusion(
            config=vla_config,
            num_layers=1,
            fusion_type="gated_cross_attention",
        )

        batch_size = 2
        seq_len = 64
        vision_len = 32

        language_hidden = torch.randn(batch_size, seq_len, 1024)
        vision_hidden = torch.randn(batch_size, vision_len, 1152)

        fused = fusion(language_hidden, vision_hidden)

        assert fused.shape == (batch_size, seq_len, 1024)

    def test_projector_integration(self, init_parallel, vla_config):
        """Test projector with vision features."""
        from ironcore.layers.multimodal.projection import VisionLanguageProjector

        projector = VisionLanguageProjector(vla_config)

        vision_features = torch.randn(2, 729, 1152)  # 729 patches

        projected = projector(vision_features)

        assert projected.shape == (2, 729, 1024)

    @pytest.mark.slow
    def test_full_forward_pass(self, init_parallel, vla_config):
        """Test full VLA forward pass (requires more memory)."""
        pytest.skip("Full forward pass test - enable with --run-slow")

        # This test would create a full VLAModel and run a forward pass
        # Skipping by default as it requires significant memory
