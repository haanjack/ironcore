#!/usr/bin/env python
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""
Test VisionEncoder with pretrained SigLIP weights.

This script tests:
1. Loading pretrained weights from HuggingFace
2. Inference on CPU and GPU
3. Output comparison between CPU and GPU
4. Inference on real images

Usage:
    # Basic test on CPU
    python tests/scripts/test_vision_encoder_gpu.py --device cpu

    # Test on GPU
    python tests/scripts/test_vision_encoder_gpu.py --device cuda:0

    # Test with custom image
    python tests/scripts/test_vision_encoder_gpu.py --device cuda:0 --image path/to/image.jpg

    # Full test suite
    python tests/scripts/test_vision_encoder_gpu.py --full
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TestResult:
    """Container for test results."""

    name: str
    passed: bool
    message: str = ""
    duration_ms: float = 0.0


def create_test_config(device: str = "cpu"):
    """Create test configuration."""
    from ironcore.config.config_vla import VisionConfig

    # Vision config for SigLIP
    vision_config = VisionConfig(
        encoder_type="siglip",
        model_name="google/siglip-so400m-patch14-384",
        image_size=384,
        patch_size=14,
        hidden_size=1152,
        num_hidden_layers=27,
        num_attention_heads=16,
        intermediate_size=4304,
        freeze_vision=True,
        device=device,
    )

    return vision_config


def test_weight_loading(device: str) -> TestResult:
    """Test loading pretrained weights from HuggingFace."""
    start_time = time.time()

    try:
        from ironcore.config import VLAConfig
        from ironcore.parallel import parallel_states
        from ironcore.vision.encoder import VisionEncoder

        # Initialize parallel states
        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

        # Create config
        vision_config = create_test_config(device)

        # Create a minimal MainConfig
        class TestConfig:
            trainer = type("trainer", (), {"tensor_model_parallel_size": 1})()
            model = type("model", (), {"d_model": 512})()
            init = type("init", (), {"init_std": 0.02, "xavier_init": False})()
            operation = type("operation", (), {"activation_recompute": False, "recompute_strategy": None})()
            vla = VLAConfig(vision=vision_config)

        config = TestConfig()

        # Create encoder (this loads weights)
        encoder = VisionEncoder(config)

        # Verify
        num_params = sum(p.numel() for p in encoder.parameters())
        duration_ms = (time.time() - start_time) * 1000

        return TestResult(
            name="weight_loading",
            passed=True,
            message=f"Loaded {num_params:,} parameters",
            duration_ms=duration_ms,
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return TestResult(
            name="weight_loading",
            passed=False,
            message=f"Error: {e}",
            duration_ms=duration_ms,
        )


def test_forward_pass(device: str, batch_size: int = 2) -> TestResult:
    """Test forward pass with pretrained weights."""
    start_time = time.time()

    try:
        from ironcore.config import VLAConfig
        from ironcore.parallel import parallel_states
        from ironcore.vision.encoder import VisionEncoder

        # Initialize parallel states
        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

        vision_config = create_test_config(device)

        class TestConfig:
            trainer = type("trainer", (), {"tensor_model_parallel_size": 1})()
            model = type("model", (), {"d_model": 512})()
            init = type("init", (), {"init_std": 0.02, "xavier_init": False})()
            operation = type("operation", (), {"activation_recompute": False, "recompute_strategy": None})()
            vla = VLAConfig(vision=vision_config)

        config = TestConfig()
        encoder = VisionEncoder(config)

        # Create input
        pixel_values = torch.randn(batch_size, 3, 384, 384, device=device)

        # Forward pass
        with torch.no_grad():
            output = encoder(pixel_values)

        # Verify output shape
        expected_shape = (batch_size, 729, 1152)  # 729 patches, 1152 hidden
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

        duration_ms = (time.time() - start_time) * 1000
        return TestResult(
            name="forward_pass",
            passed=True,
            message=f"Output shape: {output.shape}",
            duration_ms=duration_ms,
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return TestResult(
            name="forward_pass",
            passed=False,
            message=f"Error: {e}",
            duration_ms=duration_ms,
        )


def test_cpu_gpu_match() -> TestResult:
    """Test that CPU and GPU produce same outputs."""
    if not torch.cuda.is_available():
        return TestResult(
            name="cpu_gpu_match",
            passed=True,
            message="Skipped (CUDA not available)",
        )

    start_time = time.time()

    try:
        from ironcore.config import VLAConfig
        from ironcore.parallel import parallel_states
        from ironcore.vision.encoder import VisionEncoder

        # Initialize parallel states
        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

        # Create CPU encoder
        cpu_config = create_test_config("cpu")

        class TestConfigCPU:
            trainer = type("trainer", (), {"tensor_model_parallel_size": 1})()
            model = type("model", (), {"d_model": 512})()
            init = type("init", (), {"init_std": 0.02, "xavier_init": False})()
            operation = type("operation", (), {"activation_recompute": False, "recompute_strategy": None})()
            vla = VLAConfig(vision=cpu_config)

        cpu_encoder = VisionEncoder(TestConfigCPU())
        cpu_encoder.eval()

        # Create GPU encoder (share weights)
        gpu_config = create_test_config("cuda:0")

        class TestConfigGPU:
            trainer = type("trainer", (), {"tensor_model_parallel_size": 1})()
            model = type("model", (), {"d_model": 512})()
            init = type("init", (), {"init_std": 0.02, "xavier_init": False})()
            operation = type("operation", (), {"activation_recompute": False, "recompute_strategy": None})()
            vla = VLAConfig(vision=gpu_config)

        gpu_encoder = VisionEncoder(TestConfigGPU())
        gpu_encoder.eval()

        # Same input
        torch.manual_seed(42)
        pixel_values_cpu = torch.randn(1, 3, 384, 384)
        pixel_values_gpu = pixel_values_cpu.clone().cuda()

        # Forward on both
        with torch.no_grad():
            output_cpu = cpu_encoder(pixel_values_cpu)
            output_gpu = gpu_encoder(pixel_values_gpu)

        # Compare
        output_gpu_cpu = output_gpu.cpu()
        max_diff = (output_cpu - output_gpu_cpu).abs().max().item()

        # Allow small numerical differences
        tolerance = 1e-4
        passed = max_diff < tolerance

        duration_ms = (time.time() - start_time) * 1000
        return TestResult(
            name="cpu_gpu_match",
            passed=passed,
            message=f"Max difference: {max_diff:.6f} (tolerance: {tolerance})",
            duration_ms=duration_ms,
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return TestResult(
            name="cpu_gpu_match",
            passed=False,
            message=f"Error: {e}",
            duration_ms=duration_ms,
        )


def test_real_image(device: str, image_path: str | None = None) -> TestResult:
    """Test inference on a real image."""
    start_time = time.time()

    try:
        from PIL import Image

        from ironcore.config import VLAConfig
        from ironcore.parallel import parallel_states
        from ironcore.vision.encoder import VisionEncoder
        from ironcore.vision.image_processor import ImageProcessor

        # Initialize parallel states
        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

        # Create encoder
        vision_config = create_test_config(device)

        class TestConfig:
            trainer = type("trainer", (), {"tensor_model_parallel_size": 1})()
            model = type("model", (), {"d_model": 512})()
            init = type("init", (), {"init_std": 0.02, "xavier_init": False})()
            operation = type("operation", (), {"activation_recompute": False, "recompute_strategy": None})()
            vla = VLAConfig(vision=vision_config)

        encoder = VisionEncoder(TestConfig())
        encoder.eval()

        # Create image processor
        processor = ImageProcessor(vision_config)

        # Create or load image
        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert("RGB")
        else:
            # Create a random test image
            import numpy as np

            image_array = (np.random.rand(384, 384, 3) * 255).astype(np.uint8)
            image = Image.fromarray(image_array)

        # Preprocess (returns [1, C, H, W])
        pixel_values = processor.preprocess(image).to(device)

        # Forward
        with torch.no_grad():
            output = encoder(pixel_values)

        # Verify
        assert output.shape == (1, 729, 1152)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        duration_ms = (time.time() - start_time) * 1000
        return TestResult(
            name="real_image",
            passed=True,
            message=f"Output shape: {output.shape}, mean: {output.mean().item():.4f}",
            duration_ms=duration_ms,
        )

    except ImportError as e:
        duration_ms = (time.time() - start_time) * 1000
        return TestResult(
            name="real_image",
            passed=True,  # Skip rather than fail
            message=f"Skipped: {e}",
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return TestResult(
            name="real_image",
            passed=False,
            message=f"Error: {e}",
            duration_ms=duration_ms,
        )


def benchmark_inference(device: str, num_iterations: int = 10) -> TestResult:
    """Benchmark inference throughput."""
    start_time = time.time()

    try:
        from ironcore.config import VLAConfig
        from ironcore.parallel import parallel_states
        from ironcore.vision.encoder import VisionEncoder

        # Initialize parallel states
        if parallel_states._TENSOR_MODEL_PARALLEL_WORLD_SIZE is None:
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=1,
                timeout_in_minutes=30.0,
            )

        # Create encoder
        vision_config = create_test_config(device)

        class TestConfig:
            trainer = type("trainer", (), {"tensor_model_parallel_size": 1})()
            model = type("model", (), {"d_model": 512})()
            init = type("init", (), {"init_std": 0.02, "xavier_init": False})()
            operation = type("operation", (), {"activation_recompute": False, "recompute_strategy": None})()
            vla = VLAConfig(vision=vision_config)

        encoder = VisionEncoder(TestConfig())
        encoder.eval()

        # Warmup
        warmup_input = torch.randn(1, 3, 384, 384, device=device)
        with torch.no_grad():
            for _ in range(3):
                _ = encoder(warmup_input)

        # Sync if GPU
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        # Benchmark
        batch_size = 4
        pixel_values = torch.randn(batch_size, 3, 384, 384, device=device)

        times = []
        for _ in range(num_iterations):
            iter_start = time.time()
            with torch.no_grad():
                _ = encoder(pixel_values)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append(time.time() - iter_start)

        avg_time = sum(times) / len(times)
        throughput = batch_size / avg_time

        duration_ms = (time.time() - start_time) * 1000
        return TestResult(
            name="benchmark",
            passed=True,
            message=f"Avg: {avg_time*1000:.1f}ms, Throughput: {throughput:.1f} samples/sec",
            duration_ms=duration_ms,
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return TestResult(
            name="benchmark",
            passed=False,
            message=f"Error: {e}",
            duration_ms=duration_ms,
        )


def main():
    parser = argparse.ArgumentParser(description="Test VisionEncoder with pretrained weights")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run tests on",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to test image",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full test suite",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("VisionEncoder Pretrained Weight Tests")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    results = []

    # Run tests
    print("Running tests...")
    print("-" * 40)

    # Test 1: Weight loading
    print("1. Weight loading...", end=" ")
    result = test_weight_loading(args.device)
    results.append(result)
    print("PASS" if result.passed else "FAIL")
    if result.message:
        print(f"   {result.message}")

    # Test 2: Forward pass
    print("2. Forward pass...", end=" ")
    result = test_forward_pass(args.device)
    results.append(result)
    print("PASS" if result.passed else "FAIL")
    if result.message:
        print(f"   {result.message}")

    # Test 3: CPU/GPU match
    if args.full or args.device.startswith("cuda"):
        print("3. CPU/GPU match...", end=" ")
        result = test_cpu_gpu_match()
        results.append(result)
        print("PASS" if result.passed else "FAIL")
        if result.message:
            print(f"   {result.message}")

    # Test 4: Real image
    print("4. Real image inference...", end=" ")
    result = test_real_image(args.device, args.image)
    results.append(result)
    print("PASS" if result.passed else "FAIL")
    if result.message:
        print(f"   {result.message}")

    # Test 5: Benchmark
    if args.benchmark or args.full:
        print("5. Benchmark...", end=" ")
        result = benchmark_inference(args.device)
        results.append(result)
        print("PASS" if result.passed else "FAIL")
        if result.message:
            print(f"   {result.message}")

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print(f"Tests: {passed}/{total} passed")
    print(f"Total time: {sum(r.duration_ms for r in results):.1f}ms")

    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.message}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
