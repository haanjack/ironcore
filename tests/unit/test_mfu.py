# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MFU calculator."""

import pytest

from ironcore.mfu import MFUCalculator, MFUResult, compute_tflops


class TestMFUCalculator:
    """Tests for MFUCalculator class."""

    def test_init_basic(self):
        """Test basic initialization."""
        calc = MFUCalculator(
            num_layers=12,
            d_model=768,
            d_ffn=3072,
            vocab_size=50257,
            num_attention_heads=12,
        )
        assert calc.num_layers == 12
        assert calc.d_model == 768
        assert calc.num_attention_groups == 12
        assert calc.head_dim == 64

    def test_init_with_gqa(self):
        """Test initialization with grouped query attention."""
        calc = MFUCalculator(
            num_layers=32,
            d_model=4096,
            d_ffn=11008,
            vocab_size=32000,
            num_attention_heads=32,
            num_attention_groups=8,
        )
        assert calc.num_attention_groups == 8

    def test_get_num_parameters_gpt2_small(self):
        """Test parameter count for GPT-2 small like model."""
        calc = MFUCalculator(
            num_layers=12,
            d_model=768,
            d_ffn=3072,
            vocab_size=50257,
            num_attention_heads=12,
            tied_embeddings=True,
        )
        params = calc.get_num_parameters()
        assert 120_000_000 < params < 130_000_000

    def test_compute_tflops_basic(self):
        """Test basic TFLOPS computation."""
        calc = MFUCalculator(
            num_layers=12,
            d_model=768,
            d_ffn=3072,
            vocab_size=50257,
            num_attention_heads=12,
        )
        tflops = calc.compute_tflops(
            batch_size=8,
            seq_len=1024,
            step_time_seconds=0.1,
            num_gpus=1,
        )
        assert tflops > 0
        assert calc.result is not None
        assert calc.result.tokens_per_step == 8192

    def test_compute_tflops_multi_gpu(self):
        """Test TFLOPS computation with multiple GPUs."""
        calc = MFUCalculator(
            num_layers=12,
            d_model=768,
            d_ffn=3072,
            vocab_size=50257,
            num_attention_heads=12,
        )
        tflops_1gpu = calc.compute_tflops(
            batch_size=8,
            seq_len=1024,
            step_time_seconds=0.1,
            num_gpus=1,
        )
        tflops_4gpu = calc.compute_tflops(
            batch_size=8,
            seq_len=1024,
            step_time_seconds=0.1,
            num_gpus=4,
        )
        assert tflops_4gpu == pytest.approx(tflops_1gpu / 4)

    def test_from_config(self):
        """Test creating calculator from ModelConfig."""
        from ironcore.config import ModelConfig

        config = ModelConfig(
            num_layers=12,
            d_model=768,
            d_ffn=3072,
            num_attention_heads=12,
        )
        calc = MFUCalculator.from_config(config, vocab_size=50257)
        assert calc.num_layers == 12
        assert calc.d_model == 768


class TestConvenienceFunction:
    """Tests for compute_tflops convenience function."""

    def test_compute_tflops_function(self):
        """Test the convenience compute_tflops function."""
        from ironcore.config import ModelConfig

        config = ModelConfig(
            num_layers=12,
            d_model=768,
            d_ffn=3072,
            num_attention_heads=12,
        )
        tflops = compute_tflops(
            config=config,
            vocab_size=50257,
            batch_size=8,
            seq_len=1024,
            step_time_seconds=0.1,
        )
        assert tflops > 0


class TestMFUResult:
    """Tests for MFUResult dataclass."""

    def test_str_representation(self):
        """Test string representation."""
        result = MFUResult(
            tflops_per_gpu=60.74,
            model_flops_per_step=1e14,
            tokens_per_step=8192,
            step_time_seconds=0.1,
            num_parameters=124_000_000,
        )
        s = str(result)
        assert "60.74 TFLOPS/s/GPU" in s
        assert "8,192 tok/step" in s
