# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""Integration tests for VLA pipeline.

Tests both synchronous (Vision GPU + Language GPU) and
asynchronous (Vision CPU + Language GPU) configurations.

Note: Full VLAModel tests require global_states initialization.
Component-level tests are provided for quick validation.
"""

import pytest
import torch


class TestVisionEncoderPipeline:
    """Tests for VisionEncoder pipeline (CPU and GPU)."""

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
        return config

    def test_vision_encoder_cpu_forward(self, init_parallel, vla_config):
        """Test VisionEncoder forward pass on CPU."""
        from ironcore.vision.encoder import VisionEncoder

        vla_config.vla.vision.device = "cpu"
        encoder = VisionEncoder(vla_config)

        pixel_values = torch.randn(2, 3, 384, 384)
        output = encoder(pixel_values)

        assert output.shape == (2, 729, 1152)
        assert output.device.type == "cpu"

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_vision_encoder_gpu_forward(self, init_parallel, vla_config):
        """Test VisionEncoder forward pass on GPU."""
        from ironcore.vision.encoder import VisionEncoder

        vla_config.vla.vision.device = "cuda:0"
        encoder = VisionEncoder(vla_config)

        pixel_values = torch.randn(2, 3, 384, 384, device="cuda:0")
        output = encoder(pixel_values)

        assert output.shape == (2, 729, 1152)
        assert output.device.type == "cuda"

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_vision_encoder_cpu_to_gpu_transfer(self, init_parallel, vla_config):
        """Test transferring vision features from CPU to GPU."""
        from ironcore.vision.encoder import VisionEncoder

        # CPU encoder
        vla_config.vla.vision.device = "cpu"
        encoder = VisionEncoder(vla_config)

        pixel_values = torch.randn(2, 3, 384, 384)
        cpu_output = encoder(pixel_values)

        # Transfer to GPU
        gpu_output = cpu_output.to("cuda:0")

        assert gpu_output.device.type == "cuda"
        assert torch.allclose(cpu_output, gpu_output.cpu())


class TestAsyncVisionPipeline:
    """Tests for async vision pipeline."""

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
    def mock_encoder(self):
        """Create mock vision encoder."""

        class MockEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_device = torch.device("cpu")
                self.linear = torch.nn.Linear(1152, 1152)

            def forward(self, x):
                # Simulate processing
                return self.linear(x.mean(dim=1, keepdim=True).expand(-1, 729, -1))

        return MockEncoder()

    def test_async_encoder_submit_get(self, init_parallel, mock_encoder):
        """Test submit and get with async encoder."""
        from tests.fixtures.config_fixtures import MockConfig

        from ironcore.vision.async_pipeline import AsyncVisionEncoder

        config = MockConfig()
        config.trainer.async_vision_queue_size = 2
        config.trainer.async_vision_workers = 1

        encoder = AsyncVisionEncoder(
            config,
            mock_encoder,
            num_workers=1,
            queue_size=2,
            target_device="cpu",
        )

        encoder.start()
        try:
            pixel_values = torch.randn(2, 729, 1152)
            batch_id = encoder.submit(pixel_values)

            features = encoder.get_features(batch_id=batch_id, timeout=5.0)

            assert features is not None
            assert features.shape[0] == 2

        finally:
            encoder.stop()

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_async_encoder_cpu_to_gpu(self, init_parallel, mock_encoder):
        """Test async encoder with CPU to GPU transfer."""
        from tests.fixtures.config_fixtures import MockConfig

        from ironcore.vision.async_pipeline import AsyncVisionEncoder

        config = MockConfig()
        config.trainer.async_vision_queue_size = 2
        config.trainer.async_vision_workers = 1

        encoder = AsyncVisionEncoder(
            config,
            mock_encoder,
            num_workers=1,
            queue_size=2,
            target_device="cuda:0",
        )

        encoder.start()
        try:
            pixel_values = torch.randn(2, 729, 1152)
            batch_id = encoder.submit(pixel_values)

            features = encoder.get_features(batch_id=batch_id, timeout=5.0)

            assert features is not None
            assert features.device.type == "cuda"

        finally:
            encoder.stop()


class TestCrossAttentionPipeline:
    """Tests for cross-attention in the pipeline."""

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
        config.model.d_model = 512
        return config

    def test_cross_attention_cpu(self, init_parallel, vla_config):
        """Test cross-attention on CPU."""
        from ironcore.layers.cross_attention import VisionLanguageFusion

        fusion = VisionLanguageFusion(
            config=vla_config,
            num_layers=1,
            fusion_type="gated_cross_attention",
        )

        text_hidden = torch.randn(2, 32, 512)
        vision_hidden = torch.randn(2, 729, 1152)

        fused = fusion(text_hidden, vision_hidden)

        assert fused.shape == (2, 32, 512)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cross_attention_gpu(self, init_parallel, vla_config):
        """Test cross-attention on GPU."""
        from ironcore.layers.cross_attention import VisionLanguageFusion

        fusion = VisionLanguageFusion(
            config=vla_config,
            num_layers=1,
            fusion_type="gated_cross_attention",
        ).cuda()

        text_hidden = torch.randn(2, 32, 512, device="cuda:0")
        vision_hidden = torch.randn(2, 729, 1152, device="cuda:0")

        fused = fusion(text_hidden, vision_hidden)

        assert fused.shape == (2, 32, 512)
        assert fused.device.type == "cuda"


class TestProjectorPipeline:
    """Tests for projector in the pipeline."""

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
        config.model.d_model = 512
        return config

    def test_projector_cpu(self, init_parallel, vla_config):
        """Test projector on CPU."""
        from ironcore.layers.multimodal.projection import VisionLanguageProjector

        projector = VisionLanguageProjector(vla_config)

        vision_features = torch.randn(2, 729, 1152)
        projected = projector(vision_features)

        assert projected.shape == (2, 729, 512)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_projector_gpu(self, init_parallel, vla_config):
        """Test projector on GPU."""
        from ironcore.layers.multimodal.projection import VisionLanguageProjector

        projector = VisionLanguageProjector(vla_config).cuda()

        vision_features = torch.randn(2, 729, 1152, device="cuda:0")
        projected = projector(vision_features)

        assert projected.shape == (2, 729, 512)
        assert projected.device.type == "cuda"


class TestActionHeadPipeline:
    """Tests for action head in the pipeline."""

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
        config.model.d_model = 512
        return config

    def test_action_head_cpu(self, init_parallel, vla_config):
        """Test action head on CPU."""
        from ironcore.action.head import ActionHead

        head = ActionHead(vla_config)

        hidden_states = torch.randn(2, 32, 512)
        actions = head(hidden_states)

        assert actions.shape == (2, 7)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_action_head_gpu(self, init_parallel, vla_config):
        """Test action head on GPU."""
        from ironcore.action.head import ActionHead

        head = ActionHead(vla_config).cuda()

        hidden_states = torch.randn(2, 32, 512, device="cuda:0")
        actions = head(hidden_states)

        assert actions.shape == (2, 7)
        assert actions.device.type == "cuda"


class TestVLAModelFullPipeline:
    """Full VLAModel tests (require global_states)."""

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
        config.model.d_model = 256
        config.model.num_layers = 1
        return config

    @pytest.mark.slow
    @pytest.mark.skip(reason="Requires global_states initialization (full training setup)")
    def test_full_forward_pass_sync(self, init_parallel, vla_config):
        """Test full forward pass with VLAModel."""
        # This test requires global_states to be initialized
        # with a tokenizer, which happens in the training script
        pytest.skip("Requires full training setup")

    @pytest.mark.slow
    @pytest.mark.skip(reason="Requires global_states initialization (full training setup)")
    def test_async_pipeline_full(self, init_parallel, vla_config):
        """Test async pipeline with full VLAModel."""
        pytest.skip("Requires full training setup")
