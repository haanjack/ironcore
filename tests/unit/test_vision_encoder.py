# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""Unit tests for VisionEncoder on CPU."""

import pytest
import torch


class TestVisionEncoderInit:
    """Tests for VisionEncoder initialization."""

    def test_encoder_init_cpu(self):
        """Test VisionEncoder initialization on CPU."""
        from tests.fixtures.config_fixtures import MockConfig, MockVLAConfig

        from ironcore.parallel import parallel_states

        # Initialize parallel states
        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

        config = MockConfig()
        config.vla = MockVLAConfig()
        config.vla.vision.device = "cpu"

        from ironcore.vision.encoder import VisionEncoder

        encoder = VisionEncoder(config)

        # Check device
        assert encoder.vision_device.type == "cpu"
        assert next(encoder.parameters()).device.type == "cpu"

    def test_encoder_architecture(self):
        """Test VisionEncoder has correct architecture components."""
        from tests.fixtures.config_fixtures import MockConfig, MockVLAConfig

        from ironcore.parallel import parallel_states

        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

        config = MockConfig()
        config.vla = MockVLAConfig()

        from ironcore.vision.encoder import VisionEncoder

        encoder = VisionEncoder(config)

        # Check architecture
        assert hasattr(encoder, "embeddings")
        assert hasattr(encoder, "layers")
        assert hasattr(encoder, "norm")

        # Check number of layers
        assert len(encoder.layers) == config.vla.vision.num_hidden_layers


class TestVisionEncoderForward:
    """Tests for VisionEncoder forward pass."""

    @pytest.fixture
    def encoder(self):
        """Create VisionEncoder for testing."""
        from tests.fixtures.config_fixtures import MockConfig, MockVLAConfig

        from ironcore.parallel import parallel_states

        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

        config = MockConfig()
        config.vla = MockVLAConfig()
        config.vla.vision.device = "cpu"

        from ironcore.vision.encoder import VisionEncoder

        return VisionEncoder(config)

    def test_forward_shape_cpu(self, encoder):
        """Test forward pass produces correct output shape."""
        batch_size = 2
        image_size = encoder.vision_config.image_size
        hidden_size = encoder.vision_config.hidden_size

        # Expected num_patches = (image_size / patch_size)^2
        patch_size = encoder.vision_config.patch_size
        num_patches = (image_size // patch_size) ** 2

        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        output = encoder(pixel_values)

        assert output.shape == (batch_size, num_patches, hidden_size)

    def test_forward_shape_various_batch_sizes(self, encoder):
        """Test forward with various batch sizes."""
        image_size = encoder.vision_config.image_size
        hidden_size = encoder.vision_config.hidden_size
        patch_size = encoder.vision_config.patch_size
        num_patches = (image_size // patch_size) ** 2

        for batch_size in [1, 2, 4, 8]:
            pixel_values = torch.randn(batch_size, 3, image_size, image_size)
            output = encoder(pixel_values)
            assert output.shape == (batch_size, num_patches, hidden_size)

    def test_forward_output_dtype(self, encoder):
        """Test forward output dtype matches input dtype."""
        pixel_values = torch.randn(2, 3, 384, 384, dtype=torch.float32)
        output = encoder(pixel_values)
        assert output.dtype == torch.float32

        pixel_values_bf16 = torch.randn(2, 3, 384, 384, dtype=torch.bfloat16)
        output_bf16 = encoder(pixel_values_bf16.to(torch.float32))  # Encoder uses float32
        assert output_bf16.dtype == torch.float32

    def test_forward_deterministic(self, encoder):
        """Test forward is deterministic."""
        encoder.eval()
        pixel_values = torch.randn(2, 3, 384, 384)

        output1 = encoder(pixel_values)
        output2 = encoder(pixel_values)

        assert torch.allclose(output1, output2)


class TestVisionEncoderHelpers:
    """Tests for VisionEncoder helper methods."""

    @pytest.fixture
    def encoder(self):
        """Create VisionEncoder for testing."""
        from tests.fixtures.config_fixtures import MockConfig, MockVLAConfig

        from ironcore.parallel import parallel_states

        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

        config = MockConfig()
        config.vla = MockVLAConfig()

        from ironcore.vision.encoder import VisionEncoder

        return VisionEncoder(config)

    def test_num_patches(self, encoder):
        """Test num_patches is correct."""
        image_size = encoder.vision_config.image_size
        patch_size = encoder.vision_config.patch_size
        expected_patches = (image_size // patch_size) ** 2

        assert encoder.get_num_patches() == expected_patches
        assert encoder.get_num_patches() == 729  # (384/14)^2 for default config

    def test_hidden_size(self, encoder):
        """Test hidden_size returns correct value."""
        assert encoder.get_hidden_size() == encoder.vision_config.hidden_size
        assert encoder.get_hidden_size() == 1152  # Default SigLIP config


class TestVisionEncoderFrozen:
    """Tests for frozen encoder mode."""

    def test_frozen_encoder(self):
        """Test that frozen encoder has no gradients."""
        from tests.fixtures.config_fixtures import MockConfig, MockVLAConfig

        from ironcore.parallel import parallel_states

        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

        config = MockConfig()
        config.vla = MockVLAConfig()
        config.vla.vision.freeze_vision = True

        from ironcore.vision.encoder import VisionEncoder

        encoder = VisionEncoder(config)

        # All parameters should have requires_grad=False
        for param in encoder.parameters():
            assert param.requires_grad is False

    def test_trainable_encoder(self):
        """Test that trainable encoder has gradients."""
        from tests.fixtures.config_fixtures import MockConfig, MockVLAConfig

        from ironcore.parallel import parallel_states

        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

        config = MockConfig()
        config.vla = MockVLAConfig()
        config.vla.vision.freeze_vision = False

        from ironcore.vision.encoder import VisionEncoder

        encoder = VisionEncoder(config)

        # At least some parameters should have requires_grad=True
        trainable = [p for p in encoder.parameters() if p.requires_grad]
        assert len(trainable) > 0


class TestVisionEncoderHiddenStates:
    """Tests for hidden states output."""

    @pytest.fixture
    def encoder(self):
        """Create VisionEncoder for testing."""
        from tests.fixtures.config_fixtures import MockConfig, MockVLAConfig

        from ironcore.parallel import parallel_states

        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

        config = MockConfig()
        config.vla = MockVLAConfig()

        from ironcore.vision.encoder import VisionEncoder

        return VisionEncoder(config)

    def test_output_hidden_states(self, encoder):
        """Test output_hidden_states returns all layer outputs."""
        pixel_values = torch.randn(2, 3, 384, 384)
        num_layers = encoder.vision_config.num_hidden_layers

        output, hidden_states = encoder(pixel_values, output_hidden_states=True)

        # Should have num_layers + 1 hidden states (including embeddings)
        assert len(hidden_states) == num_layers + 1

        # Each hidden state should have correct shape
        for hs in hidden_states:
            assert hs.dim() == 3  # [batch, seq, hidden]

    def test_no_hidden_states_by_default(self, encoder):
        """Test that hidden states are not returned by default."""
        pixel_values = torch.randn(2, 3, 384, 384)

        output = encoder(pixel_values, output_hidden_states=False)

        # Should return just the output, not a tuple
        assert isinstance(output, torch.Tensor)


class TestVisionEncoderDevice:
    """Tests for device placement."""

    def test_cpu_device_explicit(self):
        """Test explicit CPU device."""
        from tests.fixtures.config_fixtures import MockConfig, MockVLAConfig

        from ironcore.parallel import parallel_states

        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

        config = MockConfig()
        config.vla = MockVLAConfig()
        config.vla.vision.device = "cpu"

        from ironcore.vision.encoder import VisionEncoder

        encoder = VisionEncoder(config)

        assert encoder.vision_device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_explicit(self):
        """Test explicit CUDA device."""
        from tests.fixtures.config_fixtures import MockConfig, MockVLAConfig

        from ironcore.parallel import parallel_states

        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

        config = MockConfig()
        config.vla = MockVLAConfig()
        config.vla.vision.device = "cuda:0"

        from ironcore.vision.encoder import VisionEncoder

        encoder = VisionEncoder(config)

        assert encoder.vision_device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_on_correct_device(self):
        """Test forward runs on correct device."""
        from tests.fixtures.config_fixtures import MockConfig, MockVLAConfig

        from ironcore.parallel import parallel_states

        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

        config = MockConfig()
        config.vla = MockVLAConfig()
        config.vla.vision.device = "cuda:0"

        from ironcore.vision.encoder import VisionEncoder

        encoder = VisionEncoder(config)

        pixel_values = torch.randn(2, 3, 384, 384, device="cuda:0")
        output = encoder(pixel_values)

        assert output.device.type == "cuda"
