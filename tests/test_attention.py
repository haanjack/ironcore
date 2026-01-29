# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""
Comprehensive test suite for Attention layer implementations.

Tests:
1. Attention module (pure attention computation with Q, K, V inputs)
2. TransformerLayer (full attention with projections)
3. Numerical equivalence between standard and flash attention
4. GQA (Grouped Query Attention) support
5. MQA (Multi-Query Attention) support
6. Edge cases (varying sequence lengths, batch sizes, head dimensions)
7. Memory efficiency validation
"""

import os
import sys
import unittest
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironcore.config import (
    MainConfig, ModelConfig, TrainerConfig, InitConfig, OptimConfig,
    DataConfig, ParallelConfig, OperationConfig, UtilsConfig
)
from ironcore.layers.attention import Attention
from ironcore.models.transformer import TransformerLayer
from ironcore.parallel import parallel_states


# =============================================================================
# Test Configuration Helpers
# =============================================================================

# Initialize parallel states for testing (TP=1 by default)
parallel_states.initialize_model_parallel(
    tensor_model_parallel_size=1,
    timeout_in_minutes=10.0
)


def create_test_config(
    d_model=512,
    num_attention_heads=8,
    num_attention_groups=8,
    head_dim=64,
    max_seq_len=128,
    use_flash_attn=False,
    tensor_model_parallel_size=1,
    dropout_attn=0.0,
    num_layers=1,
    d_ffn=2048,
):
    """Create a test configuration."""
    model_config = ModelConfig(
        d_model=d_model,
        num_attention_heads=num_attention_heads,
        num_attention_groups=num_attention_groups,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        max_position_embeddings=max_seq_len,
        dropout_attn=dropout_attn,
        no_bias=False,
        num_layers=num_layers,
        d_ffn=d_ffn,
    )

    trainer_config = TrainerConfig(
        tensor_model_parallel_size=tensor_model_parallel_size,
        use_flash_attn=use_flash_attn,
    )

    init_config = InitConfig(seed=42, init_std=0.02)
    optim_config = OptimConfig(max_lr=1e-3, weight_decay=0.01)
    data_config = DataConfig()
    parallel_config = ParallelConfig()
    operation_config = OperationConfig(train_steps=100)
    utils_config = UtilsConfig()

    config = MainConfig(
        model=model_config,
        trainer=trainer_config,
        init=init_config,
        optim=optim_config,
        data=data_config,
        parallel=parallel_config,
        operation=operation_config,
        utils=utils_config,
    )
    return config


def create_causal_mask(batch_size, seq_len, device):
    """Create a causal attention mask with shape [b, 1, s, s]."""
    return torch.tril(
        torch.ones(seq_len, seq_len, device=device)
    ).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)


# =============================================================================
# Test Cases for Attention Module (Pure Attention Computation)
# =============================================================================

class TestAttentionBasics(unittest.TestCase):
    """Test basic attention functionality with Q, K, V inputs."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = create_test_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.seq_len = 64
        self.num_heads = self.config.model.num_attention_heads
        self.head_dim = self.config.model.head_dim

    def test_attention_initialization(self):
        """Test attention layer initialization."""
        attention = Attention(self.config).to(self.device)

        # Check that attention-specific attributes exist
        self.assertEqual(attention.num_attention_heads, 8)
        self.assertEqual(attention.head_dimension, 64)
        self.assertIsNotNone(attention.softmax)
        self.assertIsNotNone(attention.attn_dropout)

    def test_attention_forward_shape(self):
        """Test that attention forward pass produces correct output shape."""
        attention = Attention(self.config).to(self.device)

        # Create Q, K, V tensors with shape [b, s, heads, head_dim]
        query = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim,
            device=self.device
        )
        key = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim,
            device=self.device
        )
        value = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim,
            device=self.device
        )
        attention_mask = create_causal_mask(self.batch_size, self.seq_len, self.device)

        output = attention(query, key, value, attention_mask)

        # Output should have shape [b, s, heads * head_dim]
        expected_shape = (self.batch_size, self.seq_len, self.num_heads * self.head_dim)
        self.assertEqual(output.shape, expected_shape)

    def test_attention_with_causal_mask(self):
        """Test attention with causal masking."""
        attention = Attention(self.config).to(self.device)

        query = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim,
            device=self.device
        )
        key = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim,
            device=self.device
        )
        value = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim,
            device=self.device
        )
        attention_mask = create_causal_mask(self.batch_size, self.seq_len, self.device)

        output = attention(query, key, value, attention_mask)

        # Check that output is finite
        self.assertTrue(torch.isfinite(output).all())

    def test_attention_without_mask(self):
        """Test attention without mask (bidirectional)."""
        attention = Attention(self.config).to(self.device)

        query = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim,
            device=self.device
        )
        key = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim,
            device=self.device
        )
        value = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim,
            device=self.device
        )

        output = attention(query, key, value, attention_mask=None)

        # Check that output is finite
        self.assertTrue(torch.isfinite(output).all())


class TestAttentionWithDifferentConfigs(unittest.TestCase):
    """Test attention with various configurations (MHA, GQA, MQA)."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.seq_len = 64

    def test_attention_mha(self):
        """Test Multi-Head Attention (MHA) - num_groups == num_heads."""
        num_heads = 8
        head_dim = 64
        config = create_test_config(
            num_attention_heads=num_heads,
            num_attention_groups=num_heads,
            head_dim=head_dim,
        )
        attention = Attention(config).to(self.device)

        query = torch.randn(self.batch_size, self.seq_len, num_heads, head_dim, device=self.device)
        key = torch.randn(self.batch_size, self.seq_len, num_heads, head_dim, device=self.device)
        value = torch.randn(self.batch_size, self.seq_len, num_heads, head_dim, device=self.device)
        attention_mask = create_causal_mask(self.batch_size, self.seq_len, self.device)

        output = attention(query, key, value, attention_mask)

        expected_shape = (self.batch_size, self.seq_len, num_heads * head_dim)
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_attention_gqa(self):
        """Test Grouped Query Attention (GQA) - num_groups < num_heads."""
        num_heads = 8
        num_groups = 2
        head_dim = 64
        config = create_test_config(
            num_attention_heads=num_heads,
            num_attention_groups=num_groups,
            head_dim=head_dim,
        )
        attention = Attention(config).to(self.device)

        # Q has num_heads, K/V have num_groups
        query = torch.randn(self.batch_size, self.seq_len, num_heads, head_dim, device=self.device)
        key = torch.randn(self.batch_size, self.seq_len, num_groups, head_dim, device=self.device)
        value = torch.randn(self.batch_size, self.seq_len, num_groups, head_dim, device=self.device)
        attention_mask = create_causal_mask(self.batch_size, self.seq_len, self.device)

        output = attention(query, key, value, attention_mask)

        expected_shape = (self.batch_size, self.seq_len, num_heads * head_dim)
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_attention_mqa(self):
        """Test Multi-Query Attention (MQA) - num_groups == 1."""
        num_heads = 8
        num_groups = 1
        head_dim = 64
        config = create_test_config(
            num_attention_heads=num_heads,
            num_attention_groups=num_groups,
            head_dim=head_dim,
        )
        attention = Attention(config).to(self.device)

        # Q has num_heads, K/V have 1 group
        query = torch.randn(self.batch_size, self.seq_len, num_heads, head_dim, device=self.device)
        key = torch.randn(self.batch_size, self.seq_len, num_groups, head_dim, device=self.device)
        value = torch.randn(self.batch_size, self.seq_len, num_groups, head_dim, device=self.device)
        attention_mask = create_causal_mask(self.batch_size, self.seq_len, self.device)

        output = attention(query, key, value, attention_mask)

        expected_shape = (self.batch_size, self.seq_len, num_heads * head_dim)
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.isfinite(output).all())


class TestAttentionVaryingDimensions(unittest.TestCase):
    """Test attention with varying input dimensions."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.num_heads = 8
        self.head_dim = 64

    def test_different_sequence_lengths(self):
        """Test attention with different sequence lengths."""
        config = create_test_config()
        attention = Attention(config).to(self.device)

        for seq_len in [16, 32, 64, 128]:
            with self.subTest(seq_len=seq_len):
                query = torch.randn(
                    self.batch_size, seq_len, self.num_heads, self.head_dim,
                    device=self.device
                )
                key = torch.randn(
                    self.batch_size, seq_len, self.num_heads, self.head_dim,
                    device=self.device
                )
                value = torch.randn(
                    self.batch_size, seq_len, self.num_heads, self.head_dim,
                    device=self.device
                )
                attention_mask = create_causal_mask(self.batch_size, seq_len, self.device)

                output = attention(query, key, value, attention_mask)

                expected_shape = (self.batch_size, seq_len, self.num_heads * self.head_dim)
                self.assertEqual(output.shape, expected_shape)

    def test_different_batch_sizes(self):
        """Test attention with different batch sizes."""
        config = create_test_config()
        attention = Attention(config).to(self.device)

        seq_len = 64
        for batch_size in [1, 2, 4]:
            with self.subTest(batch_size=batch_size):
                query = torch.randn(
                    batch_size, seq_len, self.num_heads, self.head_dim,
                    device=self.device
                )
                key = torch.randn(
                    batch_size, seq_len, self.num_heads, self.head_dim,
                    device=self.device
                )
                value = torch.randn(
                    batch_size, seq_len, self.num_heads, self.head_dim,
                    device=self.device
                )
                attention_mask = create_causal_mask(batch_size, seq_len, self.device)

                output = attention(query, key, value, attention_mask)

                expected_shape = (batch_size, seq_len, self.num_heads * self.head_dim)
                self.assertEqual(output.shape, expected_shape)

    def test_different_head_dimensions(self):
        """Test attention with different head dimensions."""
        seq_len = 64

        for head_dim in [32, 64, 128]:
            with self.subTest(head_dim=head_dim):
                d_model = self.num_heads * head_dim
                config = create_test_config(d_model=d_model, head_dim=head_dim)
                attention = Attention(config).to(self.device)

                query = torch.randn(
                    self.batch_size, seq_len, self.num_heads, head_dim,
                    device=self.device
                )
                key = torch.randn(
                    self.batch_size, seq_len, self.num_heads, head_dim,
                    device=self.device
                )
                value = torch.randn(
                    self.batch_size, seq_len, self.num_heads, head_dim,
                    device=self.device
                )
                attention_mask = create_causal_mask(self.batch_size, seq_len, self.device)

                output = attention(query, key, value, attention_mask)

                expected_shape = (self.batch_size, seq_len, self.num_heads * head_dim)
                self.assertEqual(output.shape, expected_shape)


class TestAttentionGradients(unittest.TestCase):
    """Test gradient computation for attention."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.seq_len = 64

    def test_gradient_flow(self):
        """Test that gradients flow correctly through attention."""
        config = create_test_config(dropout_attn=0.0)
        attention = Attention(config).to(self.device)

        num_heads = config.model.num_attention_heads
        head_dim = config.model.head_dim

        query = torch.randn(
            self.batch_size, self.seq_len, num_heads, head_dim,
            device=self.device, requires_grad=True
        )
        key = torch.randn(
            self.batch_size, self.seq_len, num_heads, head_dim,
            device=self.device, requires_grad=True
        )
        value = torch.randn(
            self.batch_size, self.seq_len, num_heads, head_dim,
            device=self.device, requires_grad=True
        )
        attention_mask = create_causal_mask(self.batch_size, self.seq_len, self.device)

        # Forward pass
        output = attention(query, key, value, attention_mask)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check that gradients exist and are finite
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(key.grad).all())
        self.assertTrue(torch.isfinite(value.grad).all())


# =============================================================================
# Test Cases for TransformerLayer (Full Attention with Projections)
# =============================================================================

class TestTransformerLayerBasics(unittest.TestCase):
    """Test TransformerLayer which includes QKV projections."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = create_test_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.seq_len = 64

    def test_transformer_layer_initialization(self):
        """Test TransformerLayer initialization."""
        layer = TransformerLayer(self.config).to(self.device)
        layer.init_weights()

        # Check that projection layers exist
        self.assertIsNotNone(layer.linear_q)
        self.assertIsNotNone(layer.linear_kv)
        self.assertIsNotNone(layer.attn_output)
        self.assertIsNotNone(layer.self_attention)

    def test_gpt_layer_forward_shape(self):
        """Test that TransformerLayer forward pass produces correct output shape."""
        layer = TransformerLayer(self.config).to(self.device)
        layer.init_weights()

        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.config.model.d_model,
            device=self.device
        )
        attention_mask = create_causal_mask(self.batch_size, self.seq_len, self.device)

        output = layer(hidden_states, attention_mask, rotary_pos_emb=None)

        # Output should have same shape as input
        self.assertEqual(output.shape, hidden_states.shape)

    def test_gpt_layer_with_rope(self):
        """Test TransformerLayer with rotary positional embeddings."""
        from ironcore.layers.positional_embedding.rotary import RotaryPositionalEmbedding

        layer = TransformerLayer(self.config).to(self.device)
        layer.init_weights()

        rotary_emb = RotaryPositionalEmbedding(
            head_dim=self.config.model.head_dim,
            max_seq_len=self.config.model.max_seq_len,
        ).to(self.device)

        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.config.model.d_model,
            device=self.device
        )
        attention_mask = create_causal_mask(self.batch_size, self.seq_len, self.device)

        output = layer(hidden_states, attention_mask, rotary_pos_emb=rotary_emb)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_gpt_layer_gradient_flow(self):
        """Test that gradients flow correctly through TransformerLayer."""
        layer = TransformerLayer(self.config).to(self.device)
        layer.init_weights()

        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.config.model.d_model,
            device=self.device, requires_grad=True
        )
        attention_mask = create_causal_mask(self.batch_size, self.seq_len, self.device)

        # Forward pass
        output = layer(hidden_states, attention_mask, rotary_pos_emb=None)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(torch.isfinite(hidden_states.grad).all())

        # Check that all parameters have gradients
        for name, param in layer.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}")


class TestTransformerLayerWithDifferentConfigs(unittest.TestCase):
    """Test TransformerLayer with various attention configurations."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.seq_len = 64

    def test_gpt_layer_mha(self):
        """Test TransformerLayer with MHA."""
        config = create_test_config(
            num_attention_heads=8,
            num_attention_groups=8,
        )
        layer = TransformerLayer(config).to(self.device)
        layer.init_weights()

        hidden_states = torch.randn(
            self.batch_size, self.seq_len, config.model.d_model,
            device=self.device
        )
        attention_mask = create_causal_mask(self.batch_size, self.seq_len, self.device)

        output = layer(hidden_states, attention_mask, rotary_pos_emb=None)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_gpt_layer_gqa(self):
        """Test TransformerLayer with GQA."""
        config = create_test_config(
            num_attention_heads=8,
            num_attention_groups=2,
        )
        layer = TransformerLayer(config).to(self.device)
        layer.init_weights()

        hidden_states = torch.randn(
            self.batch_size, self.seq_len, config.model.d_model,
            device=self.device
        )
        attention_mask = create_causal_mask(self.batch_size, self.seq_len, self.device)

        output = layer(hidden_states, attention_mask, rotary_pos_emb=None)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_gpt_layer_mqa(self):
        """Test TransformerLayer with MQA."""
        config = create_test_config(
            num_attention_heads=8,
            num_attention_groups=1,
        )
        layer = TransformerLayer(config).to(self.device)
        layer.init_weights()

        hidden_states = torch.randn(
            self.batch_size, self.seq_len, config.model.d_model,
            device=self.device
        )
        attention_mask = create_causal_mask(self.batch_size, self.seq_len, self.device)

        output = layer(hidden_states, attention_mask, rotary_pos_emb=None)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertTrue(torch.isfinite(output).all())


# =============================================================================
# Test Cases for Flash Attention
# =============================================================================

class TestAttentionStandardVsFlash(unittest.TestCase):
    """Test numerical equivalence between standard and flash attention."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.seq_len = 64
        try:
            from flash_attn import flash_attn_varlen_func
            self.flash_available = True
        except ImportError:
            self.flash_available = False

    def test_flash_availability(self):
        """Test if flash attention is available."""
        # This is just an informational test
        if self.flash_available:
            print("Flash attention is available")
        else:
            print("Flash attention is NOT available")

    def test_standard_vs_flash_forward(self):
        """Compare standard and flash attention forward pass (GPU only)."""
        if not self.flash_available:
            self.skipTest("Flash attention library not available - install flash-attn")

        num_heads = 8
        head_dim = 64

        # Create two identical configs
        config_standard = create_test_config(
            use_flash_attn=False,
            dropout_attn=0.0,
            num_attention_heads=num_heads,
            head_dim=head_dim,
        )
        config_flash = create_test_config(
            use_flash_attn=True,
            dropout_attn=0.0,
            num_attention_heads=num_heads,
            head_dim=head_dim,
        )

        attention_standard = Attention(config_standard).to(self.device)
        attention_flash = Attention(config_flash).to(self.device)

        # Create identical Q, K, V inputs
        torch.manual_seed(42)
        query = torch.randn(
            self.batch_size, self.seq_len, num_heads, head_dim,
            device=self.device
        )
        key = torch.randn(
            self.batch_size, self.seq_len, num_heads, head_dim,
            device=self.device
        )
        value = torch.randn(
            self.batch_size, self.seq_len, num_heads, head_dim,
            device=self.device
        )

        # Create causal mask for standard attention
        attention_mask = create_causal_mask(self.batch_size, self.seq_len, self.device)

        # Forward pass
        with torch.no_grad():
            output_standard = attention_standard(query, key, value, attention_mask)
            output_flash = attention_flash(query, key, value, None)  # Flash uses causal=True

        # Check that outputs are close
        max_diff = (output_standard - output_flash).abs().max().item()
        mean_diff = (output_standard - output_flash).abs().mean().item()

        self.assertLess(mean_diff, 0.1, f"Mean difference too large: {mean_diff}")
        self.assertLess(max_diff, 1.0, f"Max difference too large: {max_diff}")

    def test_standard_vs_flash_forward_gqa(self):
        """Compare standard and flash attention forward pass for GQA (GPU only)."""
        if not self.flash_available:
            self.skipTest("Flash attention library not available - install flash-attn")

        num_heads = 8
        num_groups = 2  # GQA: fewer KV heads than Q heads
        head_dim = 64

        # Create two identical configs
        config_standard = create_test_config(
            use_flash_attn=False,
            dropout_attn=0.0,
            num_attention_heads=num_heads,
            num_attention_groups=num_groups,
            head_dim=head_dim,
        )
        config_flash = create_test_config(
            use_flash_attn=True,
            dropout_attn=0.0,
            num_attention_heads=num_heads,
            num_attention_groups=num_groups,
            head_dim=head_dim,
        )

        attention_standard = Attention(config_standard).to(self.device)
        attention_flash = Attention(config_flash).to(self.device)

        # Create identical Q, K, V inputs
        # Q has num_heads, K/V have num_groups
        torch.manual_seed(42)
        query = torch.randn(
            self.batch_size, self.seq_len, num_heads, head_dim,
            device=self.device
        )
        key = torch.randn(
            self.batch_size, self.seq_len, num_groups, head_dim,
            device=self.device
        )
        value = torch.randn(
            self.batch_size, self.seq_len, num_groups, head_dim,
            device=self.device
        )

        # Create causal mask for standard attention
        attention_mask = create_causal_mask(self.batch_size, self.seq_len, self.device)

        # Forward pass
        with torch.no_grad():
            output_standard = attention_standard(query, key, value, attention_mask)
            output_flash = attention_flash(query, key, value, None)  # Flash uses causal=True

        # Check that outputs are close
        max_diff = (output_standard - output_flash).abs().max().item()
        mean_diff = (output_standard - output_flash).abs().mean().item()

        self.assertLess(mean_diff, 0.1, f"GQA mean difference too large: {mean_diff}")
        self.assertLess(max_diff, 1.0, f"GQA max difference too large: {max_diff}")

    def test_standard_vs_flash_forward_mqa(self):
        """Compare standard and flash attention forward pass for MQA (GPU only)."""
        if not self.flash_available:
            self.skipTest("Flash attention library not available - install flash-attn")

        num_heads = 8
        num_groups = 1  # MQA: single KV head shared by all Q heads
        head_dim = 64

        # Create two identical configs
        config_standard = create_test_config(
            use_flash_attn=False,
            dropout_attn=0.0,
            num_attention_heads=num_heads,
            num_attention_groups=num_groups,
            head_dim=head_dim,
        )
        config_flash = create_test_config(
            use_flash_attn=True,
            dropout_attn=0.0,
            num_attention_heads=num_heads,
            num_attention_groups=num_groups,
            head_dim=head_dim,
        )

        attention_standard = Attention(config_standard).to(self.device)
        attention_flash = Attention(config_flash).to(self.device)

        # Create identical Q, K, V inputs
        # Q has num_heads, K/V have 1 group
        torch.manual_seed(42)
        query = torch.randn(
            self.batch_size, self.seq_len, num_heads, head_dim,
            device=self.device
        )
        key = torch.randn(
            self.batch_size, self.seq_len, num_groups, head_dim,
            device=self.device
        )
        value = torch.randn(
            self.batch_size, self.seq_len, num_groups, head_dim,
            device=self.device
        )

        # Create causal mask for standard attention
        attention_mask = create_causal_mask(self.batch_size, self.seq_len, self.device)

        # Forward pass
        with torch.no_grad():
            output_standard = attention_standard(query, key, value, attention_mask)
            output_flash = attention_flash(query, key, value, None)  # Flash uses causal=True

        # Check that outputs are close
        max_diff = (output_standard - output_flash).abs().max().item()
        mean_diff = (output_standard - output_flash).abs().mean().item()

        self.assertLess(mean_diff, 0.1, f"MQA mean difference too large: {mean_diff}")
        self.assertLess(max_diff, 1.0, f"MQA max difference too large: {max_diff}")

    def test_flash_attention_gqa_gradient_flow(self):
        """Test gradient flow through flash attention with GQA."""
        if not self.flash_available:
            self.skipTest("Flash attention library not available - install flash-attn")

        num_heads = 8
        num_groups = 2
        head_dim = 64

        config = create_test_config(
            use_flash_attn=True,
            dropout_attn=0.0,
            num_attention_heads=num_heads,
            num_attention_groups=num_groups,
            head_dim=head_dim,
        )
        attention = Attention(config).to(self.device)

        query = torch.randn(
            self.batch_size, self.seq_len, num_heads, head_dim,
            device=self.device, requires_grad=True
        )
        key = torch.randn(
            self.batch_size, self.seq_len, num_groups, head_dim,
            device=self.device, requires_grad=True
        )
        value = torch.randn(
            self.batch_size, self.seq_len, num_groups, head_dim,
            device=self.device, requires_grad=True
        )

        output = attention(query, key, value, None)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(key.grad).all())
        self.assertTrue(torch.isfinite(value.grad).all())

    def test_flash_attention_mqa_gradient_flow(self):
        """Test gradient flow through flash attention with MQA."""
        if not self.flash_available:
            self.skipTest("Flash attention library not available - install flash-attn")

        num_heads = 8
        num_groups = 1
        head_dim = 64

        config = create_test_config(
            use_flash_attn=True,
            dropout_attn=0.0,
            num_attention_heads=num_heads,
            num_attention_groups=num_groups,
            head_dim=head_dim,
        )
        attention = Attention(config).to(self.device)

        query = torch.randn(
            self.batch_size, self.seq_len, num_heads, head_dim,
            device=self.device, requires_grad=True
        )
        key = torch.randn(
            self.batch_size, self.seq_len, num_groups, head_dim,
            device=self.device, requires_grad=True
        )
        value = torch.randn(
            self.batch_size, self.seq_len, num_groups, head_dim,
            device=self.device, requires_grad=True
        )

        output = attention(query, key, value, None)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(key.grad).all())
        self.assertTrue(torch.isfinite(value.grad).all())


# =============================================================================
# Test Cases for Memory Efficiency
# =============================================================================

class TestAttentionMemory(unittest.TestCase):
    """Test memory efficiency of attention."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required for memory profiling")
    def test_memory_usage(self):
        """Test memory usage of attention."""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        num_heads = 12
        head_dim = 64
        d_model = num_heads * head_dim

        config = create_test_config(
            d_model=d_model,
            num_attention_heads=num_heads,
            num_attention_groups=num_heads,
            head_dim=head_dim,
        )
        attention = Attention(config).to(self.device)

        batch_size = 4
        seq_len = 256

        query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
        attention_mask = create_causal_mask(batch_size, seq_len, self.device)

        # Measure memory before forward
        mem_before = torch.cuda.memory_allocated()

        # Forward pass
        output = attention(query, key, value, attention_mask)

        # Measure memory after forward
        mem_after = torch.cuda.memory_allocated()
        mem_used = mem_after - mem_before

        # Memory should be reasonable (less than 1GB for this config)
        self.assertLess(mem_used, 1e9, f"Memory usage too high: {mem_used / 1e6:.2f} MB")


# =============================================================================
# Benchmark Utilities
# =============================================================================

def benchmark_attention(config, batch_size, seq_len, num_runs=100):
    """Benchmark attention forward and backward pass."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_heads = config.model.num_attention_heads
    num_groups = config.model.num_attention_groups
    head_dim = config.model.head_dim

    attention = Attention(config).to(device)

    query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    key = torch.randn(batch_size, seq_len, num_groups, head_dim, device=device)
    value = torch.randn(batch_size, seq_len, num_groups, head_dim, device=device)
    attention_mask = create_causal_mask(batch_size, seq_len, device)

    # Warmup
    for _ in range(10):
        query.requires_grad_(True)
        output = attention(query, key, value, attention_mask)
        loss = output.sum()
        loss.backward()
        query.grad = None

    torch.cuda.synchronize() if device.type == 'cuda' else None

    # Benchmark forward
    start_time = time.time()
    for _ in range(num_runs):
        output = attention(query, key, value, attention_mask)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    forward_time = (time.time() - start_time) / num_runs

    # Benchmark backward
    start_time = time.time()
    for _ in range(num_runs):
        query.requires_grad_(True)
        output = attention(query, key, value, attention_mask)
        loss = output.sum()
        loss.backward()
        query.grad = None
    if device.type == 'cuda':
        torch.cuda.synchronize()
    backward_time = (time.time() - start_time) / num_runs

    return {
        'forward_time_ms': forward_time * 1000,
        'backward_time_ms': backward_time * 1000,
        'total_time_ms': (forward_time + backward_time) * 1000,
    }


def run_benchmarks():
    """Run comprehensive benchmarks."""
    results = {}

    batch_size = 2
    seq_len = 256
    d_model = 512
    num_heads = 8
    head_dim = d_model // num_heads

    # Benchmark MHA
    print("\nBenchmarking MHA...")
    config_mha = create_test_config(
        d_model=d_model,
        num_attention_heads=num_heads,
        num_attention_groups=num_heads,
        head_dim=head_dim,
        use_flash_attn=False,
    )
    results['mha_standard'] = benchmark_attention(config_mha, batch_size, seq_len)

    # Benchmark GQA
    print("Benchmarking GQA...")
    config_gqa = create_test_config(
        d_model=d_model,
        num_attention_heads=num_heads,
        num_attention_groups=2,
        head_dim=head_dim,
        use_flash_attn=False,
    )
    results['gqa_standard'] = benchmark_attention(config_gqa, batch_size, seq_len)

    # Benchmark MQA
    print("Benchmarking MQA...")
    config_mqa = create_test_config(
        d_model=d_model,
        num_attention_heads=num_heads,
        num_attention_groups=1,
        head_dim=head_dim,
        use_flash_attn=False,
    )
    results['mqa_standard'] = benchmark_attention(config_mqa, batch_size, seq_len)

    # Print results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    for name, timings in results.items():
        print(f"\n{name}:")
        print(f"  Forward:  {timings['forward_time_ms']:.2f} ms")
        print(f"  Backward: {timings['backward_time_ms']:.2f} ms")
        print(f"  Total:    {timings['total_time_ms']:.2f} ms")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Attention Test Suite")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmarks"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    if args.benchmark:
        run_benchmarks()
    else:
        unittest.main(verbosity=2 if args.verbose else 1)
