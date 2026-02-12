# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""Unit tests for async vision pipeline."""

import time

import pytest
import torch

# Skip all tests in this module if PIL is not installed
pytest.importorskip("PIL")


class TestVisionFeatureQueue:
    """Tests for VisionFeatureQueue."""

    def test_put_and_get(self):
        """Test basic put and get operations."""
        from ironcore.vision.async_pipeline import VisionFeatureQueue

        queue = VisionFeatureQueue(max_size=4)

        # Put features
        features = torch.randn(2, 32, 512)
        batch_id = queue.put(features)

        assert batch_id == 0
        assert queue.size() == 1

        # Get features
        batch = queue.get()
        assert batch.batch_id == 0
        assert batch.features.shape == (2, 32, 512)

        assert queue.size() == 0

    def test_queue_full_blocking(self):
        """Test blocking when queue is full."""
        from ironcore.vision.async_pipeline import VisionFeatureQueue

        queue = VisionFeatureQueue(max_size=2)

        # Fill queue
        for _ in range(2):
            queue.put(torch.randn(2, 32, 512))

        assert queue.is_full()

        # Non-blocking put should fail
        with pytest.raises(RuntimeError, match="Queue is full"):
            queue.put(torch.randn(2, 32, 512), block=False)

    def test_queue_empty_blocking(self):
        """Test blocking when queue is empty."""
        from ironcore.vision.async_pipeline import VisionFeatureQueue

        queue = VisionFeatureQueue(max_size=2)

        assert queue.is_empty()

        # Non-blocking get should fail
        with pytest.raises(RuntimeError, match="Queue is empty"):
            queue.get(block=False)

        # Get with timeout
        result = queue.get_nowait()
        assert result is None

    def test_shutdown(self):
        """Test queue shutdown."""
        from ironcore.vision.async_pipeline import VisionFeatureQueue

        queue = VisionFeatureQueue(max_size=2)

        queue.shutdown()

        with pytest.raises(RuntimeError, match="shutdown"):
            queue.put(torch.randn(2, 32, 512))

    def test_clear(self):
        """Test queue clear."""
        from ironcore.vision.async_pipeline import VisionFeatureQueue

        queue = VisionFeatureQueue(max_size=4)

        for _ in range(3):
            queue.put(torch.randn(2, 32, 512))

        assert queue.size() == 3

        queue.clear()

        assert queue.size() == 0
        assert queue.is_empty()


class TestVisionFeatureBatch:
    """Tests for VisionFeatureBatch."""

    def test_batch_creation(self):
        """Test batch creation."""
        from ironcore.vision.async_pipeline import VisionFeatureBatch

        features = torch.randn(2, 32, 512)
        batch = VisionFeatureBatch(
            features=features,
            batch_id=42,
        )

        assert batch.batch_id == 42
        assert batch.features is features
        assert batch.device.type == "cpu"

    def test_batch_to_device(self):
        """Test moving batch to device."""
        from ironcore.vision.async_pipeline import VisionFeatureBatch

        features = torch.randn(2, 32, 512)
        batch = VisionFeatureBatch(
            features=features,
            batch_id=0,
        )

        # Move to cuda if available
        if torch.cuda.is_available():
            new_batch = batch.to(torch.device("cuda:0"))
            assert new_batch.device.type == "cuda"
            assert new_batch.features.device.type == "cuda"
            # Original unchanged
            assert batch.device.type == "cpu"


class TestAsyncVisionEncoder:
    """Tests for AsyncVisionEncoder."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        from tests.fixtures.config_fixtures import MockConfig

        config = MockConfig()
        config.trainer.async_vision_queue_size = 2
        config.trainer.async_vision_workers = 1
        return config

    @pytest.fixture
    def mock_vision_encoder(self):
        """Create mock vision encoder."""

        class MockVisionEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_device = torch.device("cpu")
                # Input: [batch, num_patches, hidden] -> [batch, num_patches, hidden]
                self.linear = torch.nn.Linear(512, 512)

            def forward(self, x):
                # x is [batch, num_patches, hidden]
                # Simulate some computation
                time.sleep(0.01)
                return self.linear(x)

        return MockVisionEncoder()

    def test_submit_and_get(self, mock_config, mock_vision_encoder):
        """Test submit and get features."""
        from ironcore.vision.async_pipeline import AsyncVisionEncoder

        encoder = AsyncVisionEncoder(
            mock_config,
            mock_vision_encoder,
            num_workers=1,
            queue_size=2,
            target_device="cpu",
        )

        encoder.start()

        try:
            # Submit batch with correct dimensions [batch, num_patches, hidden]
            pixel_values = torch.randn(2, 32, 512)
            batch_id = encoder.submit(pixel_values)

            # Wait for encoding
            features = encoder.get_features(batch_id=batch_id, timeout=5.0)

            assert features is not None
            assert features.shape == (2, 32, 512)

        finally:
            encoder.stop()

    def test_statistics(self, mock_config, mock_vision_encoder):
        """Test encoding statistics."""
        from ironcore.vision.async_pipeline import AsyncVisionEncoder

        encoder = AsyncVisionEncoder(
            mock_config,
            mock_vision_encoder,
            num_workers=1,
            queue_size=2,
            target_device="cpu",
        )

        encoder.start()

        try:
            # Submit with correct dimensions
            encoder.submit(torch.randn(2, 32, 512))
            features = encoder.get_features(timeout=5.0)

            stats = encoder.get_stats()
            assert "encoded" in stats
            assert stats["encoded"] >= 1
            assert features is not None

        finally:
            encoder.stop()


class TestHybridAsyncVisionPipeline:
    """Tests for HybridAsyncVisionPipeline."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        from tests.fixtures.config_fixtures import MockConfig

        config = MockConfig()
        config.trainer.async_vision_queue_size = 2
        config.trainer.async_vision_workers = 1
        return config

    def test_cpu_strategy_selection(self, mock_config):
        """Test CPU strategy is selected for CPU vision device."""

        class CPUVisionEncoder(torch.nn.Module):
            vision_device = torch.device("cpu")

            def forward(self, x):
                return x.mean(dim=1)

        from ironcore.vision.async_pipeline import HybridAsyncVisionPipeline

        pipeline = HybridAsyncVisionPipeline(
            mock_config,
            CPUVisionEncoder(),
            queue_size=2,
            num_cpu_workers=1,
        )

        assert pipeline._strategy == "thread"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_strategy_selection(self, mock_config):
        """Test CUDA strategy is selected for GPU vision device."""

        class GPUVisionEncoder(torch.nn.Module):
            vision_device = torch.device("cuda:1")

            def forward(self, x):
                return x.mean(dim=1)

        from ironcore.vision.async_pipeline import HybridAsyncVisionPipeline

        pipeline = HybridAsyncVisionPipeline(
            mock_config,
            GPUVisionEncoder(),
            queue_size=2,
            num_cpu_workers=1,
        )

        assert pipeline._strategy == "cuda_stream"

    def test_pipeline_start_stop(self, mock_config):
        """Test pipeline start and stop."""

        class CPUVisionEncoder(torch.nn.Module):
            vision_device = torch.device("cpu")

            def forward(self, x):
                return x.mean(dim=1)

        from ironcore.vision.async_pipeline import HybridAsyncVisionPipeline

        pipeline = HybridAsyncVisionPipeline(
            mock_config,
            CPUVisionEncoder(),
            queue_size=2,
            num_cpu_workers=1,
        )

        # Start and stop should not raise
        pipeline.start()
        pipeline.stop()


class TestVLATrainingIterator:
    """Tests for VLATrainingIterator."""

    def test_iterator_basic(self):
        """Test basic iterator functionality."""
        pytest.skip("Requires full model initialization")
