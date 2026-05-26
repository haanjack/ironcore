# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for FIM edge cases and boundary conditions.

Tests behavior with short sequences, empty sequences, edge splits,
and other boundary conditions.
"""

from random import Random

import pytest
from tests.fixtures.utils import get_section_lengths, verify_psm_structure

pytestmark = pytest.mark.hf_hub


class TestFIMShortSequences:
    """Test FIM behavior with short sequences."""

    def test_fim_single_token(self, serializer_with_fim, fim_token_ids):
        """Test that single-token sequence is not transformed."""
        token_ids = [42]
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        # Should return unchanged (too short for FIM)
        assert transformed == token_ids

    def test_fim_length_below_threshold(self, serializer_with_fim, fim_token_ids):
        """Test sequences below 10 tokens are not transformed."""
        rng = Random(42)

        for length in range(1, 10):
            token_ids = list(range(length))
            transformed = serializer_with_fim._apply_fim_transformation(
                token_ids,
                fim_token_ids["prefix"],
                fim_token_ids["suffix"],
                fim_token_ids["middle"],
                rng,
            )

            # Should be unchanged
            assert transformed == token_ids, f"Length {length} was transformed"

    def test_fim_boundary_length(self, serializer_with_fim, fim_token_ids):
        """Test sequence at boundary (length=10) is transformed."""
        token_ids = list(range(10))  # Exactly 10 tokens
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        # Should be transformed (>= 10)
        assert verify_psm_structure(
            transformed, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )

    def test_fim_just_above_threshold(self, serializer_with_fim, fim_token_ids):
        """Test sequence just above threshold (length=11)."""
        token_ids = list(range(11))
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        assert verify_psm_structure(
            transformed, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )


class TestFIMEmptySequences:
    """Test FIM behavior with empty sequences."""

    def test_fim_empty_sequence(self, serializer_with_fim, fim_token_ids):
        """Test empty input is handled gracefully."""
        token_ids = []
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        # Should return unchanged (empty)
        assert transformed == []

    def test_fim_zero_length(self, serializer_with_fim, fim_token_ids):
        """Test zero-length sequence."""
        token_ids = []
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        assert len(transformed) == 0


class TestFIMRepeatedTokens:
    """Test FIM with repeated/identical tokens."""

    def test_fim_repeated_tokens(self, serializer_with_fim, fim_token_ids):
        """Test FIM with all-identical tokens."""
        token_ids = [42] * 50  # 50 identical tokens
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        # Should still have valid PSM structure
        assert verify_psm_structure(
            transformed, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )

        # Remove special tokens and verify all are 42
        special_tokens = [fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]]
        content_tokens = [t for t in transformed if t not in special_tokens]
        assert all(t == 42 for t in content_tokens)
        assert len(content_tokens) == 50

    def test_fim_alternating_tokens(self, serializer_with_fim, fim_token_ids):
        """Test FIM with alternating pattern."""
        token_ids = [10, 20] * 25  # [10, 20, 10, 20, ...]
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        assert verify_psm_structure(
            transformed, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )


class TestFIMEdgeSplits:
    """Test FIM with edge case split positions."""

    def test_fim_split_at_boundaries(self, serializer_with_fim, fim_token_ids):
        """Test splits near sequence boundaries."""
        token_ids = list(range(100))

        # Force specific split points using mock random
        from tests.fixtures.mocks import MockRandom

        # Test split near start: [1, 2]
        mock_rng = MockRandom(samples=[1, 2])
        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids.copy(),
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            mock_rng,
        )

        assert verify_psm_structure(
            transformed, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )

        prefix_len, suffix_len, middle_len = get_section_lengths(
            transformed, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )

        assert prefix_len > 0
        assert suffix_len > 0
        assert middle_len > 0

    def test_fim_split_near_end(self, serializer_with_fim, fim_token_ids):
        """Test splits near sequence end."""
        token_ids = list(range(100))
        from tests.fixtures.mocks import MockRandom

        # Split near end: [97, 98]
        mock_rng = MockRandom(samples=[97, 98])
        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids.copy(),
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            mock_rng,
        )

        assert verify_psm_structure(
            transformed, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )

        prefix_len, suffix_len, middle_len = get_section_lengths(
            transformed, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )

        assert prefix_len > 0
        assert suffix_len > 0
        assert middle_len > 0


class TestFIMEmptyMiddlePrevention:
    """Test that middle section is never empty."""

    def test_fim_empty_middle_handled(self, serializer_with_fim, fim_token_ids):
        """Test that empty middle is prevented/handled."""
        token_ids = list(range(20))

        # Try many random seeds to test edge cases
        for seed in range(100):
            rng = Random(seed)
            transformed = serializer_with_fim._apply_fim_transformation(
                token_ids.copy(),
                fim_token_ids["prefix"],
                fim_token_ids["suffix"],
                fim_token_ids["middle"],
                rng,
            )

            # If transformed (>= 10 tokens), check middle is non-empty
            if verify_psm_structure(
                transformed,
                fim_token_ids["prefix"],
                fim_token_ids["suffix"],
                fim_token_ids["middle"],
            ):
                prefix_len, suffix_len, middle_len = get_section_lengths(
                    transformed,
                    fim_token_ids["prefix"],
                    fim_token_ids["suffix"],
                    fim_token_ids["middle"],
                )

                assert middle_len > 0, f"Empty middle with seed {seed}"

    def test_fim_consecutive_splits_prevented(self, serializer_with_fim, fim_token_ids):
        """Test that consecutive split points are handled."""
        # The code uses random.sample which guarantees different values
        # This test documents that behavior
        token_ids = list(range(50))

        for seed in range(50):
            rng = Random(seed)
            transformed = serializer_with_fim._apply_fim_transformation(
                token_ids.copy(),
                fim_token_ids["prefix"],
                fim_token_ids["suffix"],
                fim_token_ids["middle"],
                rng,
            )

            if verify_psm_structure(
                transformed,
                fim_token_ids["prefix"],
                fim_token_ids["suffix"],
                fim_token_ids["middle"],
            ):
                # All sections should be non-empty
                prefix_len, suffix_len, middle_len = get_section_lengths(
                    transformed,
                    fim_token_ids["prefix"],
                    fim_token_ids["suffix"],
                    fim_token_ids["middle"],
                )

                assert prefix_len > 0
                assert middle_len > 0
                assert suffix_len > 0


class TestFIMUnicodeHandling:
    """Test FIM with unicode and special characters."""

    def test_fim_unicode_tokens(self, serializer_with_fim, test_tokenizer_with_fim, fim_token_ids):
        """Test FIM transformation with unicode text."""
        # Tokenize unicode text
        text = "Hello 世界! 🌍 émojis"
        token_ids = test_tokenizer_with_fim.encode(text, add_special_tokens=False)

        # Skip if too short
        if len(token_ids) < 10:
            token_ids = token_ids * 3  # Repeat to get enough tokens

        rng = Random(42)
        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        # Should have valid PSM structure
        assert verify_psm_structure(
            transformed, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )

    def test_fim_emoji_text(self, serializer_with_fim, test_tokenizer_with_fim, fim_token_ids):
        """Test FIM with emoji-heavy text."""
        text = "🔥" * 20 + " code " + "✨" * 20
        token_ids = test_tokenizer_with_fim.encode(text, add_special_tokens=False)

        if len(token_ids) >= 10:
            rng = Random(42)
            transformed = serializer_with_fim._apply_fim_transformation(
                token_ids,
                fim_token_ids["prefix"],
                fim_token_ids["suffix"],
                fim_token_ids["middle"],
                rng,
            )

            assert verify_psm_structure(
                transformed,
                fim_token_ids["prefix"],
                fim_token_ids["suffix"],
                fim_token_ids["middle"],
            )


class TestFIMLargeSequences:
    """Test FIM with very large sequences."""

    def test_fim_large_sequence(self, serializer_with_fim, fim_token_ids):
        """Test FIM with large sequence (1000+ tokens)."""
        token_ids = list(range(1000))
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        assert verify_psm_structure(
            transformed, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )

        assert len(transformed) == len(token_ids) + 3

    def test_fim_very_large_sequence(self, serializer_with_fim, fim_token_ids):
        """Test FIM with very large sequence (10000+ tokens)."""
        token_ids = list(range(10000))
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids,
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        assert verify_psm_structure(
            transformed, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
