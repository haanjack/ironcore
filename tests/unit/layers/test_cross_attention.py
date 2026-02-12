# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""Unit tests for cross-attention layers."""

import pytest
import torch


class TestCrossAttention:
    """Tests for CrossAttention layer."""

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
    def mock_config(self):
        """Create mock config."""
        from tests.fixtures.config_fixtures import MockConfig

        return MockConfig()

    def test_cross_attention_forward(self, init_parallel, mock_config):
        """Test basic cross-attention forward pass."""
        from ironcore.layers.cross_attention import CrossAttention

        cross_attn = CrossAttention(
            config=mock_config,
            hidden_size=512,
            num_heads=8,
            kv_hidden_size=512,
        )

        batch_size = 2
        seq_len = 64
        vision_len = 32

        hidden_states = torch.randn(batch_size, seq_len, 512)
        vision_features = torch.randn(batch_size, vision_len, 512)

        output = cross_attn(hidden_states, vision_features)

        assert output.shape == (batch_size, seq_len, 512)

    def test_cross_attention_with_different_kv_dim(self, init_parallel, mock_config):
        """Test cross-attention with different KV dimension."""
        from ironcore.layers.cross_attention import CrossAttention

        cross_attn = CrossAttention(
            config=mock_config,
            hidden_size=512,
            num_heads=8,
            kv_hidden_size=1152,  # Vision encoder dimension
        )

        hidden_states = torch.randn(2, 64, 512)
        vision_features = torch.randn(2, 32, 1152)

        output = cross_attn(hidden_states, vision_features)
        assert output.shape == (2, 64, 512)

    def test_cross_attention_with_mask(self, init_parallel, mock_config):
        """Test cross-attention with vision mask."""
        from ironcore.layers.cross_attention import CrossAttention

        cross_attn = CrossAttention(
            config=mock_config,
            hidden_size=512,
            num_heads=8,
        )

        hidden_states = torch.randn(2, 64, 512)
        vision_features = torch.randn(2, 32, 512)
        vision_mask = torch.ones(2, 32)
        vision_mask[:, -10:] = 0  # Mask out last 10 tokens

        output = cross_attn(hidden_states, vision_features, vision_mask)
        assert output.shape == (2, 64, 512)


class TestGatedCrossAttention:
    """Tests for GatedCrossAttention layer."""

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
    def mock_config(self):
        """Create mock config."""
        from tests.fixtures.config_fixtures import MockConfig

        return MockConfig()

    def test_gated_cross_attention_forward(self, init_parallel, mock_config):
        """Test gated cross-attention forward pass."""
        from ironcore.layers.cross_attention import GatedCrossAttention

        gated_ca = GatedCrossAttention(
            config=mock_config,
            hidden_size=512,
            ffn_hidden_size=2048,
            num_heads=8,
        )

        hidden_states = torch.randn(2, 64, 512)
        vision_features = torch.randn(2, 32, 512)

        output = gated_ca(hidden_states, vision_features)

        assert output.shape == (2, 64, 512)

    def test_gate_starts_at_zero(self, mock_config):
        """Test that gate parameter starts at zero for stable training."""
        from ironcore.layers.cross_attention import GatedCrossAttention

        gated_ca = GatedCrossAttention(
            config=mock_config,
            hidden_size=512,
            ffn_hidden_size=2048,
            num_heads=8,
        )

        assert gated_ca.gate.item() == 0.0


class TestVisionLanguageFusion:
    """Tests for VisionLanguageFusion module."""

    @pytest.fixture
    def init_parallel(self):
        """Initialize parallel states."""
        from ironcore.parallel import parallel_states

        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

    def test_gated_cross_attention_fusion(self, init_parallel):
        """Test gated cross-attention fusion."""
        from tests.fixtures.config_fixtures import MockConfig

        from ironcore.layers.cross_attention import VisionLanguageFusion

        mock_config = MockConfig()
        mock_config.vla = type("vla", (), {
            "vision": type("vision", (), {"hidden_size": 1152})(),
            "num_image_tokens": 729,
        })()

        fusion = VisionLanguageFusion(
            config=mock_config,
            num_layers=1,
            fusion_type="gated_cross_attention",
        )

        hidden_states = torch.randn(2, 64, 512)
        vision_features = torch.randn(2, 32, 1152)

        output = fusion(hidden_states, vision_features)
        assert output.shape == (2, 64, 512)

    def test_invalid_fusion_type_raises(self):
        """Test that invalid fusion type raises ValueError."""
        from tests.fixtures.config_fixtures import MockConfig

        from ironcore.layers.cross_attention import VisionLanguageFusion

        mock_config = MockConfig()
        mock_config.vla = type("vla", (), {
            "vision": type("vision", (), {"hidden_size": 512})(),
            "num_image_tokens": 32,
        })()

        with pytest.raises(ValueError, match="Unknown fusion type"):
            VisionLanguageFusion(
                config=mock_config,
                fusion_type="invalid_type",
            )
