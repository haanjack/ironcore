# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""Unit tests for device management."""

import torch


class TestDeviceManager:
    """Tests for DeviceManager."""

    def test_auto_select_with_no_gpu(self, monkeypatch):
        """Test device selection with no GPU."""
        from ironcore.vision.device_manager import DeviceManager

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        dm = DeviceManager(vision_device="auto", language_device="auto")

        assert dm.vision_device.type == "cpu"
        assert dm.language_device.type == "cpu"

    def test_explicit_device_selection(self):
        """Test explicit device selection."""
        from ironcore.vision.device_manager import DeviceManager

        dm = DeviceManager(vision_device="cpu", language_device="cuda:0")

        assert dm.vision_device.type == "cpu"
        assert dm.language_device.type == "cuda"

    def test_move_tensor(self):
        """Test tensor movement between devices."""
        from ironcore.vision.device_manager import DeviceManager

        dm = DeviceManager(vision_device="cpu", language_device="cpu")

        tensor = torch.randn(2, 3)
        moved = dm.move_tensor(tensor, "vision")

        assert moved.device.type == "cpu"


class TestGetOptimalDeviceConfig:
    """Tests for get_optimal_device_config."""

    def test_no_gpu_config(self, monkeypatch):
        """Test config with no GPU."""
        from ironcore.vision.device_manager import get_optimal_device_config

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

        config = get_optimal_device_config()

        assert config["vision_device"] == "cpu"
        assert config["language_device"] == "cpu"

    def test_prefer_cpu_with_avx512(self, monkeypatch):
        """Test preferring CPU when AVX-512 available."""
        from ironcore.vision.device_manager import get_optimal_device_config

        # Mock 2 GPUs
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

        config = get_optimal_device_config(
            tensor_parallel_size=2,
            prefer_cpu_for_vision=True,
        )

        # Should prefer CPU when explicitly requested
        assert config["vision_device"] == "cpu"

    def test_config_structure(self):
        """Test that config has all required keys."""
        from ironcore.vision.device_manager import get_optimal_device_config

        config = get_optimal_device_config()

        assert "vision_device" in config
        assert "language_device" in config
        assert "recommendation" in config
        assert "cpu_avx512" in config


class TestCPUCapabilities:
    """Tests for CPU capability detection."""

    def test_check_cpu_capabilities(self):
        """Test CPU capability check."""
        from ironcore.vision.device_manager import check_cpu_capabilities

        caps = check_cpu_capabilities()

        assert "avx512" in caps
        assert "avx2" in caps
        assert "num_threads" in caps
        assert isinstance(caps["num_threads"], int)
        assert caps["num_threads"] >= 1
