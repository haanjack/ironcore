# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for FIM transformation algorithm.

Tests the core _apply_fim_transformation function including PSM structure,
token conservation, and randomness properties.
"""

from random import Random

import pytest
from tests.fixtures.utils import (
    get_section_lengths,
    reconstruct_original_sequence,
    validate_token_conservation,
    verify_psm_structure,
)


class TestFIMPSMFormat:
    """Test PSM (Prefix-Suffix-Middle) format structure."""

    def test_fim_psm_format_basic(self, serializer_with_fim, fim_token_ids):
        """Verify FIM sequence has correct PSM structure."""
        # Create a simple token sequence
        token_ids = list(range(10, 30))  # 20 tokens

        rng = Random(42)
        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        # Verify PSM structure
        assert verify_psm_structure(
            transformed, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )

    def test_fim_special_token_order(self, serializer_with_fim, fim_token_ids):
        """Test that special tokens appear in correct order: FP < FS < FM."""
        token_ids = list(range(10, 30))
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        fp_pos = transformed.index(fim_token_ids["prefix"])
        fs_pos = transformed.index(fim_token_ids["suffix"])
        fm_pos = transformed.index(fim_token_ids["middle"])

        assert fp_pos < fs_pos < fm_pos

    def test_fim_special_token_count(self, serializer_with_fim, fim_token_ids):
        """Test that each special token appears exactly once."""
        token_ids = list(range(10, 30))
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        assert transformed.count(fim_token_ids["prefix"]) == 1
        assert transformed.count(fim_token_ids["suffix"]) == 1
        assert transformed.count(fim_token_ids["middle"]) == 1


class TestFIMTokenConservation:
    """Test token conservation during FIM transformation."""

    def test_fim_token_conservation(self, serializer_with_fim, fim_token_ids):
        """Verify all original tokens are present after transformation."""
        token_ids = list(range(10, 30))
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        # Verify token conservation
        assert validate_token_conservation(
            token_ids,
            transformed,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
        )

    def test_fim_no_token_duplication(self, serializer_with_fim, fim_token_ids):
        """Verify no tokens are duplicated during transformation."""
        token_ids = list(range(10, 30))
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        # Remove special tokens
        special_tokens = [fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]]
        content_tokens = [t for t in transformed if t not in special_tokens]

        # Check no duplication (sorted lists should match)
        assert sorted(token_ids) == sorted(content_tokens)

    def test_fim_reconstruction(self, serializer_with_fim, fim_token_ids):
        """Test that original sequence can be reconstructed from FIM."""
        token_ids = list(range(10, 30))
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        # Reconstruct original
        reconstructed = reconstruct_original_sequence(
            transformed, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )

        assert reconstructed == token_ids


class TestFIMLengthInvariant:
    """Test length invariants of FIM transformation."""

    def test_fim_length_invariant(self, serializer_with_fim, fim_token_ids):
        """Verify transformed length = original + 3 special tokens."""
        token_ids = list(range(10, 30))
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        assert len(transformed) == len(token_ids) + 3

    def test_fim_length_various_sizes(self, serializer_with_fim, fim_token_ids):
        """Test length invariant holds for various sequence sizes."""
        rng = Random(42)

        for size in [10, 20, 50, 100, 500]:
            token_ids = list(range(size))
            transformed = serializer_with_fim._apply_fim_transformation(
                token_ids,
                fim_token_ids["prefix"],
                fim_token_ids["suffix"],
                fim_token_ids["middle"],
                rng,
            )

            assert len(transformed) == size + 3


class TestFIMSplitRandomness:
    """Test randomness properties of split point selection."""

    def test_fim_split_randomness_uniform(self, serializer_with_fim, fim_token_ids):
        """Test that split points are roughly uniformly distributed."""
        token_ids = list(range(100))  # 100 tokens
        rng = Random(42)

        # Generate many transformations
        splits = []
        for _ in range(100):
            transformed = serializer_with_fim._apply_fim_transformation(
                token_ids.copy(),
                fim_token_ids["prefix"],
                fim_token_ids["suffix"],
                fim_token_ids["middle"],
                rng,
            )

            prefix_len, suffix_len, middle_len = get_section_lengths(
                transformed,
                fim_token_ids["prefix"],
                fim_token_ids["suffix"],
                fim_token_ids["middle"],
            )

            # Calculate original split points
            split1 = prefix_len
            split2 = prefix_len + middle_len
            splits.append((split1, split2))

        # Basic check: splits should vary
        unique_splits = set(splits)
        # Should have good variety (at least 50 different split combinations)
        assert len(unique_splits) > 50

    def test_fim_split_different_seeds(self, serializer_with_fim, fim_token_ids):
        """Test that different seeds produce different splits."""
        token_ids = list(range(50))

        rng1 = Random(42)
        rng2 = Random(123)

        transformed1 = serializer_with_fim._apply_fim_transformation(
            token_ids.copy(),
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng1,
        )

        transformed2 = serializer_with_fim._apply_fim_transformation(
            token_ids.copy(),
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng2,
        )

        # Different seeds should likely produce different transformations
        # (very small chance they're the same)
        assert transformed1 != transformed2


class TestFIMDeterminism:
    """Test deterministic behavior with fixed seed."""

    def test_fim_deterministic_seed(self, serializer_with_fim, fim_token_ids):
        """Test that same seed produces same transformation."""
        token_ids = list(range(50))

        # Transform twice with same seed
        rng1 = Random(42)
        transformed1 = serializer_with_fim._apply_fim_transformation(
            token_ids.copy(),
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng1,
        )

        rng2 = Random(42)  # Same seed
        transformed2 = serializer_with_fim._apply_fim_transformation(
            token_ids.copy(),
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng2,
        )

        assert transformed1 == transformed2

    def test_fim_deterministic_multiple_calls(self, serializer_with_fim, fim_token_ids):
        """Test deterministic behavior across multiple transformations."""
        rng = Random(42)

        results = []
        for i in range(5):
            token_ids = list(range(20 + i * 10))  # Varying sizes
            transformed = serializer_with_fim._apply_fim_transformation(
                token_ids,
                fim_token_ids["prefix"],
                fim_token_ids["suffix"],
                fim_token_ids["middle"],
                rng,
            )
            results.append(transformed)

        # Transform again with same seed sequence
        rng2 = Random(42)
        results2 = []
        for i in range(5):
            token_ids = list(range(20 + i * 10))
            transformed = serializer_with_fim._apply_fim_transformation(
                token_ids,
                fim_token_ids["prefix"],
                fim_token_ids["suffix"],
                fim_token_ids["middle"],
                rng2,
            )
            results2.append(transformed)

        # All results should match
        for r1, r2 in zip(results, results2, strict=False):
            assert r1 == r2


class TestFIMNonEmptySections:
    """Test that all sections (prefix, suffix, middle) are non-empty."""

    def test_fim_all_sections_nonempty(self, serializer_with_fim, fim_token_ids):
        """Verify prefix, suffix, and middle are all non-empty."""
        token_ids = list(range(10, 30))
        rng = Random(42)

        # Test multiple times with different seeds
        for seed in range(10):
            rng = Random(seed)
            transformed = serializer_with_fim._apply_fim_transformation(
                token_ids.copy(),
                fim_token_ids["prefix"],
                fim_token_ids["suffix"],
                fim_token_ids["middle"],
                rng,
            )

            prefix_len, suffix_len, middle_len = get_section_lengths(
                transformed,
                fim_token_ids["prefix"],
                fim_token_ids["suffix"],
                fim_token_ids["middle"],
            )

            assert prefix_len > 0, f"Empty prefix with seed {seed}"
            assert suffix_len > 0, f"Empty suffix with seed {seed}"
            assert middle_len > 0, f"Empty middle with seed {seed}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
