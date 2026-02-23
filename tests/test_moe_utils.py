# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for MoE utility functions."""

import pytest
import torch

from ironcore.layers.moe.utils import flatten_moe_inputs, validate_moe_input


class TestFlattenMoeInputs:
    """Tests for flatten_moe_inputs function."""

    def test_basic_flatten(self):
        """Test basic flattening of 3D tensors."""
        batch_size, seq_len, hidden_size = 2, 16, 64
        top_k = 4

        x = torch.randn(batch_size, seq_len, hidden_size)
        topk_weights = torch.randn(batch_size, seq_len, top_k)
        topk_indices = torch.randint(0, 8, (batch_size, seq_len, top_k))

        x_flat, weights_flat, indices_flat, num_tokens, hidden = flatten_moe_inputs(
            x, topk_weights, topk_indices
        )

        assert x_flat.shape == (batch_size * seq_len, hidden_size)
        assert weights_flat.shape == (batch_size * seq_len, top_k)
        assert indices_flat.shape == (batch_size * seq_len, top_k)
        assert num_tokens == batch_size * seq_len
        assert hidden == hidden_size

    def test_various_batch_sizes(self):
        """Test with various batch sizes."""
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 8, 64)
            weights = torch.randn(batch_size, 8, 2)
            indices = torch.randint(0, 4, (batch_size, 8, 2))

            x_flat, _, _, num_tokens, _ = flatten_moe_inputs(x, weights, indices)

            assert x_flat.shape[0] == batch_size * 8
            assert num_tokens == batch_size * 8

    def test_various_seq_lengths(self):
        """Test with various sequence lengths."""
        for seq_len in [1, 32, 128]:
            x = torch.randn(2, seq_len, 64)
            weights = torch.randn(2, seq_len, 2)
            indices = torch.randint(0, 4, (2, seq_len, 2))

            x_flat, _, _, num_tokens, _ = flatten_moe_inputs(x, weights, indices)

            assert x_flat.shape[0] == 2 * seq_len
            assert num_tokens == 2 * seq_len

    def test_values_preserved(self):
        """Test that values are preserved during flattening."""
        batch_size, seq_len, hidden_size = 2, 4, 8
        top_k = 2

        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, hidden_size)
        weights = torch.randn(batch_size, seq_len, top_k)
        indices = torch.randint(0, 4, (batch_size, seq_len, top_k))

        x_flat, weights_flat, indices_flat, _, _ = flatten_moe_inputs(x, weights, indices)

        # Verify values are preserved
        assert torch.allclose(x_flat, x.view(-1, hidden_size))
        assert torch.allclose(weights_flat, weights.view(-1, top_k))
        assert torch.equal(indices_flat, indices.view(-1, top_k))


class TestValidateMoeInput:
    """Tests for validate_moe_input function."""

    def test_valid_input_passes(self):
        """Test that valid input passes validation."""
        x = torch.randn(2, 16, 64)
        # Should not raise
        validate_moe_input(x, 64, "Test")

    def test_wrong_ndim_raises(self):
        """Test that wrong number of dimensions raises ValueError."""
        x_2d = torch.randn(16, 64)
        x_4d = torch.randn(2, 4, 16, 64)

        with pytest.raises(ValueError, match="3D input"):
            validate_moe_input(x_2d, 64, "Test")

        with pytest.raises(ValueError, match="3D input"):
            validate_moe_input(x_4d, 64, "Test")

    def test_wrong_hidden_size_raises(self):
        """Test that wrong hidden size raises ValueError."""
        x = torch.randn(2, 16, 64)

        with pytest.raises(ValueError, match="hidden_size=128"):
            validate_moe_input(x, 128, "Test")

    def test_empty_batch_raises(self):
        """Test that empty batch dimension raises ValueError."""
        x = torch.randn(0, 16, 64)

        with pytest.raises(ValueError, match="batch size"):
            validate_moe_input(x, 64, "Test")

    def test_empty_seq_raises(self):
        """Test that empty sequence dimension raises ValueError."""
        x = torch.randn(2, 0, 64)

        with pytest.raises(ValueError, match="sequence length"):
            validate_moe_input(x, 64, "Test")

    def test_nan_input_raises(self):
        """Test that NaN input raises ValueError."""
        x = torch.randn(2, 16, 64)
        x[0, 0, 0] = float("nan")

        with pytest.raises(ValueError, match="NaN"):
            validate_moe_input(x, 64, "Test")

    def test_inf_input_raises(self):
        """Test that Inf input raises ValueError."""
        x = torch.randn(2, 16, 64)
        x[0, 0, 0] = float("inf")

        with pytest.raises(ValueError, match="Inf"):
            validate_moe_input(x, 64, "Test")

        x[0, 0, 0] = float("-inf")
        with pytest.raises(ValueError, match="Inf"):
            validate_moe_input(x, 64, "Test")

    def test_custom_name_in_message(self):
        """Test that custom name appears in error message."""
        x = torch.randn(2, 16, 32)  # Wrong hidden size

        with pytest.raises(ValueError, match="CustomLayer"):
            validate_moe_input(x, 64, "CustomLayer")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
