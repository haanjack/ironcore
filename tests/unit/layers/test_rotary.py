# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test suite for Rotary Position Embedding (RoPE) implementation.

Tests:
1. Theta frequency computation correctness
2. Rotation formula correctness
3. Sin/cos cache generation
4. Position indexing
5. Edge cases (cache extension, custom position_ids)
6. Numerical precision
7. Gradient flow
8. Reference comparison with HuggingFace
"""

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironcore.layers.positional_embedding.rotary import RotaryPositionalEmbedding

# =============================================================================
# Helper Functions
# =============================================================================


def reference_rope_theta(head_dim: int, base: int = 10000) -> torch.Tensor:
    """Reference implementation of theta frequency computation.

    θᵢ = base^(-2i/d) for i = 0, 1, 2, ..., d/2 - 1

    Args:
        head_dim: Dimension of each attention head
        base: Base value for frequency computation (default: 10000)

    Returns:
        Tensor of shape [head_dim // 2] containing theta frequencies
    """
    return 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))


def reference_apply_rotary(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int,
    base: int = 10000,
) -> torch.Tensor:
    """Reference implementation of RoPE rotation (half/half split pattern).

    Uses the standard half/half split pattern matching HuggingFace/Megatron/Llama.

    Args:
        x: Input tensor [batch, seq_len, num_heads, head_dim]
        position_ids: Position indices [batch, seq_len]
        head_dim: Head dimension
        base: Base for theta computation

    Returns:
        Rotated tensor with same shape as input
    """
    batch_size, seq_len, num_heads, _ = x.shape
    device = x.device

    # Compute theta frequencies (on same device as x)
    theta = reference_rope_theta(head_dim, base).to(device)  # [head_dim // 2]

    # Compute position * theta for each position
    positions = position_ids.float()
    idx_theta = torch.einsum("bs,d->bsd", positions, theta)

    # Compute sin and cos
    cos_emb = torch.cos(idx_theta)  # [batch, seq_len, head_dim // 2]
    sin_emb = torch.sin(idx_theta)

    # Split x into first half and second half (standard pattern)
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Expand sin/cos for broadcasting over heads
    cos_emb = cos_emb.unsqueeze(2)  # [batch, seq_len, 1, head_dim // 2]
    sin_emb = sin_emb.unsqueeze(2)

    # Apply rotation formula: [x1*cos - x2*sin, x1*sin + x2*cos]
    x_rotated = torch.cat(
        [x1 * cos_emb - x2 * sin_emb, x1 * sin_emb + x2 * cos_emb],
        dim=-1,
    )

    return x_rotated


# =============================================================================
# Test Cases
# =============================================================================


class TestRoPEBasics(unittest.TestCase):
    """Test basic RoPE functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head_dim = 64
        self.max_seq_len = 128
        self.base = 10000

    def test_rope_initialization(self):
        """Test RoPE initialization creates correct attributes."""
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        self.assertEqual(rope.head_dim, self.head_dim)
        self.assertEqual(rope.max_seq_len, self.max_seq_len)
        self.assertEqual(rope.rope_base, self.base)
        self.assertEqual(rope.theta.shape, (self.head_dim // 2,))
        self.assertEqual(rope.sin_emb.shape, (self.max_seq_len, self.head_dim // 2))
        self.assertEqual(rope.cos_emb.shape, (self.max_seq_len, self.head_dim // 2))

    def test_theta_frequencies(self):
        """Test that theta frequencies match reference implementation."""
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        # Reference theta computation
        ref_theta = reference_rope_theta(self.head_dim, self.base).to(self.device)

        # Compare - cast to tensor for type checker
        theta = rope.theta
        sin_emb = rope.sin_emb
        cos_emb = rope.cos_emb
        assert isinstance(theta, torch.Tensor), "theta should be a tensor"
        assert isinstance(sin_emb, torch.Tensor), "sin_emb should be a tensor"
        assert isinstance(cos_emb, torch.Tensor), "cos_emb should be a tensor"

        self.assertTrue(
            torch.allclose(theta, ref_theta, rtol=1e-6, atol=1e-6),
            f"Theta mismatch:\nGot: {theta[:5]}\nExpected: {ref_theta[:5]}",
        )

    def test_sin_cos_cache_shape(self):
        """Test that sin/cos cache has correct shape."""
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        # sin_emb and cos_emb should have shape [max_seq_len, head_dim // 2]
        expected_shape = (self.max_seq_len, self.head_dim // 2)
        self.assertEqual(rope.sin_emb.shape, expected_shape)
        self.assertEqual(rope.cos_emb.shape, expected_shape)

    def test_sin_cos_values(self):
        """Test that sin/cos values are in valid range [-1, 1]."""
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        sin_emb = rope.sin_emb
        cos_emb = rope.cos_emb
        assert isinstance(sin_emb, torch.Tensor), "sin_emb should be a tensor"
        assert isinstance(cos_emb, torch.Tensor), "cos_emb should be a tensor"
        self.assertTrue((sin_emb >= -1).all() and (sin_emb <= 1).all())
        self.assertTrue((cos_emb >= -1).all() and (cos_emb <= 1).all())


class TestRoPERotation(unittest.TestCase):
    """Test RoPE rotation formula correctness."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head_dim = 64
        self.max_seq_len = 128
        self.base = 10000
        self.batch_size = 2
        self.seq_len = 16
        self.num_heads = 8

    def test_rotation_formula_single_element(self):
        """Test rotation formula on a single element pair."""
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        # Create simple input: x = [1, 0, 0, 0, ...] for position 0
        x = torch.zeros(1, 1, 1, self.head_dim, device=self.device)
        x[0, 0, 0, 0] = 1.0  # x_0 = 1, x_1 = 0

        position_ids = torch.tensor([[0]], device=self.device)

        # Apply rotation
        x_rotated = rope(x.clone(), position_ids)

        # At position 0: sin(0) = 0, cos(0) = 1
        # x'_0 = x_0 * cos(0) - x_1 * sin(0) = 1 * 1 - 0 * 0 = 1
        # x'_1 = x_0 * sin(0) + x_1 * cos(0) = 1 * 0 + 0 * 1 = 0
        self.assertAlmostEqual(x_rotated[0, 0, 0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(x_rotated[0, 0, 0, 1].item(), 0.0, places=5)

    def test_rotation_formula_position_one(self):
        """Test rotation at position 1 with known values (half/half split pattern)."""
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        # Create input: x = [1, 1, 0, 0, ...] where first half is [1, 1, ...] and second half is [0, 0, ...]
        x = torch.zeros(1, 1, 1, self.head_dim, device=self.device)
        x[0, 0, 0, 0] = 1.0  # First half, index 0
        x[0, 0, 0, 1] = 1.0  # First half, index 1
        # Second half (indices 32-63) remain 0

        position_ids = torch.tensor([[1]], device=self.device)

        # Get expected rotation values for theta at index 0
        theta_0 = 1.0 / (self.base ** (0 / self.head_dim))
        angle = 1 * theta_0  # position * theta
        expected_sin = math.sin(angle)
        expected_cos = math.cos(angle)

        # Apply rotation
        x_rotated = rope(x.clone(), position_ids)

        # Half/half split pattern:
        # x1 (first half) = [1, 1, 0, 0, ...], x2 (second half) = [0, 0, ...]
        # First half output: x1 * cos - x2 * sin = x1 * cos (since x2=0)
        # Second half output: x1 * sin + x2 * cos = x1 * sin (since x2=0)
        expected_first_half_0 = expected_cos  # x[0] * cos
        expected_second_half_0 = expected_sin  # x[0] * sin

        self.assertAlmostEqual(x_rotated[0, 0, 0, 0].item(), expected_first_half_0, places=5)
        self.assertAlmostEqual(
            x_rotated[0, 0, 0, self.head_dim // 2].item(), expected_second_half_0, places=5
        )

    def test_rotation_against_reference(self):
        """Test that rotation matches reference implementation."""
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        # Random input
        x = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim, device=self.device
        )
        position_ids = (
            torch.arange(self.seq_len, device=self.device).unsqueeze(0).expand(self.batch_size, -1)
        )

        # Our implementation
        x_rotated = rope(x.clone(), position_ids)

        # Reference implementation
        x_ref = reference_apply_rotary(x, position_ids, self.head_dim, self.base)

        # Compare
        self.assertTrue(
            torch.allclose(x_rotated, x_ref, rtol=1e-5, atol=1e-5),
            f"Rotation mismatch:\nMax diff: {(x_rotated - x_ref).abs().max()}",
        )

    def test_rotation_preserves_shape(self):
        """Test that rotation preserves input shape."""
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        x = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_dim, device=self.device
        )
        position_ids = (
            torch.arange(self.seq_len, device=self.device).unsqueeze(0).expand(self.batch_size, -1)
        )

        x_rotated = rope(x, position_ids)

        self.assertEqual(x_rotated.shape, x.shape)

    def test_rotation_different_positions(self):
        """Test that different positions produce different rotations."""
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        # Same input at different positions
        x = torch.randn(1, 1, 1, self.head_dim, device=self.device)

        pos_0 = torch.tensor([[0]], device=self.device)
        pos_1 = torch.tensor([[1]], device=self.device)
        pos_10 = torch.tensor([[10]], device=self.device)

        x_rot_0 = rope(x.clone(), pos_0)
        x_rot_1 = rope(x.clone(), pos_1)
        x_rot_10 = rope(x.clone(), pos_10)

        # Different positions should produce different outputs
        self.assertFalse(torch.allclose(x_rot_0, x_rot_1))
        self.assertFalse(torch.allclose(x_rot_1, x_rot_10))
        self.assertFalse(torch.allclose(x_rot_0, x_rot_10))


class TestRoPEEdgeCases(unittest.TestCase):
    """Test RoPE edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head_dim = 64
        self.base = 10000

    def test_cache_extension(self):
        """Test that cache extends when position exceeds max_seq_len."""
        initial_max_seq = 16
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=initial_max_seq,
            base=self.base,
        ).to(self.device)

        # Initial cache size
        self.assertEqual(rope.max_seq_len_cached, initial_max_seq)

        # Request position beyond initial max
        x = torch.randn(1, 32, 1, self.head_dim, device=self.device)
        position_ids = torch.arange(32, device=self.device).unsqueeze(0)

        x_rotated = rope(x, position_ids)

        # Cache should have been extended
        self.assertGreaterEqual(rope.max_seq_len_cached, 32)
        self.assertTrue(torch.isfinite(x_rotated).all())

    def test_custom_position_ids(self):
        """Test with custom (non-sequential) position IDs."""
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=128,
            base=self.base,
        ).to(self.device)

        x = torch.randn(1, 4, 1, self.head_dim, device=self.device)

        # Non-sequential positions
        position_ids = torch.tensor([[5, 10, 20, 30]], device=self.device)

        x_rotated = rope(x.clone(), position_ids)

        self.assertEqual(x_rotated.shape, x.shape)
        self.assertTrue(torch.isfinite(x_rotated).all())

    def test_offset_parameter(self):
        """Test that offset parameter shifts position indices."""
        offset = 10
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=128,
            base=self.base,
            offset=offset,
        ).to(self.device)

        # Without explicit position_ids, positions should start from offset
        x = torch.randn(1, 5, 1, self.head_dim, device=self.device)
        x_rotated_offset = rope(x.clone(), None)

        # Compare with a non-offset RoPE using explicit positions
        rope_no_offset = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=128,
            base=self.base,
            offset=0,
        ).to(self.device)
        explicit_position_ids = torch.arange(offset, offset + 5, device=self.device).unsqueeze(0)
        x_rotated_explicit = rope_no_offset(x.clone(), explicit_position_ids)

        self.assertTrue(
            torch.allclose(x_rotated_offset, x_rotated_explicit, rtol=1e-5, atol=1e-5),
            "RoPE with offset does not match explicit position IDs.",
        )

    def test_scale_parameter(self):
        """Test that scale parameter is applied to positions."""
        scale = 0.5
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=128,
            base=self.base,
            scale=scale,
        ).to(self.device)

        x = torch.randn(1, 1, 1, self.head_dim, device=self.device)
        position_ids = torch.tensor([[10]], device=self.device)

        # With scale=0.5, position 10 should behave like position 5
        rope_scaled = rope
        rope_unscaled = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=128,
            base=self.base,
            scale=1.0,
        ).to(self.device)

        x_rot_scaled = rope_scaled(x.clone(), position_ids)
        position_ids_half = torch.tensor([[5]], device=self.device)
        x_rot_unscaled = rope_unscaled(x.clone(), position_ids_half)

        # Should be approximately equal due to scale
        self.assertTrue(
            torch.allclose(x_rot_scaled, x_rot_unscaled, rtol=1e-5, atol=1e-5),
            f"Scale mismatch:\nScaled: {x_rot_scaled[0, 0, 0, :4]}\nUnscaled: {x_rot_unscaled[0, 0, 0, :4]}",
        )

    def test_odd_head_dim_handling(self):
        """Test that odd head_dim is handled (may produce warning or different behavior)."""
        # RoPE with odd head_dim will have mismatched x1/x2 sizes
        # This test documents the behavior rather than asserting it should work
        odd_head_dim = 63

        rope = RotaryPositionalEmbedding(
            head_dim=odd_head_dim,
            max_seq_len=128,
            base=self.base,
        ).to(self.device)

        x = torch.randn(1, 1, 1, odd_head_dim, device=self.device)
        position_ids = torch.tensor([[0]], device=self.device)

        # This will fail due to size mismatch between x1 (31) and sin_emb (32)
        # This test documents expected behavior - odd head_dim is not supported
        with self.assertRaises(RuntimeError):
            rope(x, position_ids)


class TestRoPENumericalPrecision(unittest.TestCase):
    """Test RoPE numerical precision."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head_dim = 64
        self.max_seq_len = 128
        self.base = 10000

    def test_fp32_precision(self):
        """Test precision in float32."""
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        x = torch.randn(2, 16, 8, self.head_dim, device=self.device, dtype=torch.float32)
        position_ids = torch.arange(16, device=self.device).unsqueeze(0).expand(2, -1)

        x_rotated = rope(x, position_ids)

        self.assertTrue(torch.isfinite(x_rotated).all())
        # Check dtype is preserved
        self.assertEqual(x_rotated.dtype, torch.float32)

    def test_fp16_precision(self):
        """Test precision in float16."""
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        x = torch.randn(2, 16, 8, self.head_dim, device=self.device, dtype=torch.float16)
        position_ids = torch.arange(16, device=self.device).unsqueeze(0).expand(2, -1)

        x_rotated = rope(x, position_ids)

        self.assertTrue(torch.isfinite(x_rotated).all())
        self.assertEqual(x_rotated.dtype, torch.float16)

    def test_bf16_precision(self):
        """Test precision in bfloat16."""
        if not torch.cuda.is_bf16_supported():
            self.skipTest("bfloat16 not supported on this device")

        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        x = torch.randn(2, 16, 8, self.head_dim, device=self.device, dtype=torch.bfloat16)
        position_ids = torch.arange(16, device=self.device).unsqueeze(0).expand(2, -1)

        x_rotated = rope(x, position_ids)

        self.assertTrue(torch.isfinite(x_rotated).all())
        self.assertEqual(x_rotated.dtype, torch.bfloat16)

    def test_large_values_stability(self):
        """Test numerical stability with large input values."""
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        # Large values
        x = torch.randn(2, 16, 8, self.head_dim, device=self.device) * 100
        position_ids = torch.arange(16, device=self.device).unsqueeze(0).expand(2, -1)

        x_rotated = rope(x, position_ids)

        self.assertTrue(torch.isfinite(x_rotated).all())
        # Norm should be approximately preserved
        input_norm = x.norm(dim=-1)
        output_norm = x_rotated.norm(dim=-1)
        relative_diff = (input_norm - output_norm).abs() / (input_norm + 1e-8)
        self.assertTrue((relative_diff < 1e-4).all())

    def test_norm_preservation(self):
        """Test that rotation approximately preserves vector norms."""
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        x = torch.randn(2, 16, 8, self.head_dim, device=self.device)
        position_ids = torch.arange(16, device=self.device).unsqueeze(0).expand(2, -1)

        x_rotated = rope(x, position_ids)

        input_norm = x.norm(dim=-1)
        output_norm = x_rotated.norm(dim=-1)

        # Rotation is orthogonal, so norms should be exactly preserved
        self.assertTrue(
            torch.allclose(input_norm, output_norm, rtol=1e-5, atol=1e-5),
            f"Norm not preserved:\nInput: {input_norm[0, 0]}\nOutput: {output_norm[0, 0]}",
        )


class TestRoPEGradients(unittest.TestCase):
    """Test RoPE gradient flow."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head_dim = 64
        self.max_seq_len = 128
        self.base = 10000

    def test_gradient_exists(self):
        """Test that gradients exist after backward pass."""
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        x = torch.randn(2, 16, 8, self.head_dim, device=self.device, requires_grad=True)
        position_ids = torch.arange(16, device=self.device).unsqueeze(0).expand(2, -1)

        x_rotated = rope(x, position_ids)
        loss = x_rotated.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        assert x.grad is not None  # For type checker
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_gradient_non_zero(self):
        """Test that gradients are non-zero."""
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        x = torch.randn(2, 16, 8, self.head_dim, device=self.device, requires_grad=True)
        position_ids = torch.arange(16, device=self.device).unsqueeze(0).expand(2, -1)

        x_rotated = rope(x, position_ids)
        loss = x_rotated.sum()
        loss.backward()

        self.assertTrue((x.grad != 0).any())

    def test_gradient_correctness(self):
        """Test gradient correctness against numerical gradient."""
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        x = torch.randn(1, 1, 1, self.head_dim, device=self.device, requires_grad=True)
        position_ids = torch.tensor([[0]], device=self.device)

        # Analytical gradient
        x_rotated = rope(x, position_ids)
        loss = x_rotated.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        assert x.grad is not None  # For type checker
        analytical_grad = x.grad.clone()

        # Numerical gradient (finite differences)
        eps = 1e-4
        x_np = x.detach().clone()
        numerical_grad = torch.zeros_like(x_np)

        for i in range(self.head_dim):
            x_plus = x_np.clone()
            x_plus[0, 0, 0, i] += eps
            with torch.no_grad():
                out_plus = rope(x_plus, position_ids).sum()

            x_minus = x_np.clone()
            x_minus[0, 0, 0, i] -= eps
            with torch.no_grad():
                out_minus = rope(x_minus, position_ids).sum()

            numerical_grad[0, 0, 0, i] = (out_plus - out_minus) / (2 * eps)

        # Compare (allow some tolerance due to numerical approximation)
        # Note: finite difference approximation can have small errors
        self.assertTrue(
            torch.allclose(analytical_grad, numerical_grad, rtol=1e-2, atol=1e-2),
            f"Gradient mismatch:\nAnalytical: {analytical_grad[0, 0, 0, :4]}\nNumerical: {numerical_grad[0, 0, 0, :4]}",
        )


class TestRoPEHuggingFaceComparison(unittest.TestCase):
    """Compare RoPE implementation with HuggingFace reference."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head_dim = 64
        self.max_seq_len = 128
        self.base = 10000

    def test_vs_huggingface_llama_rope(self):
        """Test against HuggingFace LlamaRotaryEmbedding.

        IronCore now uses the standard half/half split pattern matching
        HuggingFace/Megatron/Llama implementations.
        """
        try:
            from transformers.models.llama.configuration_llama import LlamaConfig
            from transformers.models.llama.modeling_llama import (
                LlamaRotaryEmbedding,
                apply_rotary_pos_emb,
            )
        except ImportError:
            self.skipTest("transformers library not installed")

        # Our implementation
        our_rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        # Create HuggingFace RoPE with config object
        try:
            config = LlamaConfig(
                hidden_size=self.head_dim * 8,
                num_attention_heads=8,
                num_key_value_heads=8,
                max_position_embeddings=self.max_seq_len,
                rope_theta=self.base,
            )
            hf_rope = LlamaRotaryEmbedding(config=config, device=self.device)
        except Exception as e:
            self.skipTest(f"Could not create LlamaRotaryEmbedding: {e}")

        # Test input
        batch_size, seq_len, num_heads = 2, 16, 8
        x = torch.randn(batch_size, seq_len, num_heads, self.head_dim, device=self.device)
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

        # Our output
        x_our = our_rope(x.clone(), position_ids)

        # HuggingFace output
        try:
            cos_hf, sin_hf = hf_rope(x, position_ids)
        except Exception as e:
            self.skipTest(f"HuggingFace RoPE forward failed: {e}")

        # Apply HF rotation
        x_hf = x.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        q_hf, _ = apply_rotary_pos_emb(x_hf, x_hf.clone(), cos_hf, sin_hf)
        q_hf = q_hf.transpose(1, 2)  # Back to [batch, seq, num_heads, head_dim]

        # Verify both outputs are finite
        self.assertTrue(torch.isfinite(x_our).all(), "IronCore output has NaN/Inf")
        self.assertTrue(torch.isfinite(q_hf).all(), "HF output has NaN/Inf")

        # Verify both preserve norms (rotation is orthogonal)
        input_norm = x.norm(dim=-1)
        our_norm = x_our.norm(dim=-1)
        hf_norm = q_hf.norm(dim=-1)

        self.assertTrue(
            torch.allclose(input_norm, our_norm, rtol=1e-5), "IronCore RoPE doesn't preserve norms"
        )
        self.assertTrue(
            torch.allclose(input_norm, hf_norm, rtol=1e-5), "HF RoPE doesn't preserve norms"
        )

        # Verify outputs match (now using same half/half split pattern)
        max_diff = (x_our - q_hf).abs().max().item()
        self.assertTrue(
            torch.allclose(x_our, q_hf, rtol=1e-4, atol=1e-4),
            f"HuggingFace mismatch:\nMax diff: {max_diff}\n"
            f"Our: {x_our[0, 0, 0, :4]}\nHF: {q_hf[0, 0, 0, :4]}",
        )
        print(f"\n[INFO] IronCore RoPE matches HuggingFace: max_diff = {max_diff:.2e}")


class TestRoPEIntegration(unittest.TestCase):
    """Test RoPE integration with attention."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head_dim = 64
        self.max_seq_len = 128
        self.base = 10000

    def test_rope_with_attention_qk_only(self):
        """Test that RoPE should only be applied to Q and K, not V."""
        rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=self.base,
        ).to(self.device)

        batch_size, seq_len = 2, 16
        num_heads = 8

        # Create Q, K, V
        query = torch.randn(batch_size, seq_len, num_heads, self.head_dim, device=self.device)
        key = torch.randn(batch_size, seq_len, num_heads, self.head_dim, device=self.device)
        torch.randn(batch_size, seq_len, num_heads, self.head_dim, device=self.device)

        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)

        # Apply RoPE to Q and K only (standard practice)
        query_rotated = rope(query.clone(), position_ids)
        key_rotated = rope(key.clone(), position_ids)
        # V is NOT rotated

        # Verify shapes
        self.assertEqual(query_rotated.shape, query.shape)
        self.assertEqual(key_rotated.shape, key.shape)

        # Verify Q and K are actually rotated (different from original)
        self.assertFalse(torch.allclose(query_rotated, query))
        self.assertFalse(torch.allclose(key_rotated, key))


if __name__ == "__main__":
    unittest.main()
