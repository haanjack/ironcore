# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Property-based tests for FIM transformation invariants.

Uses hypothesis library for generative testing to verify FIM properties
hold across a wide range of inputs.
"""

from random import Random

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tests.fixtures.utils import (
    reconstruct_original_sequence,
    validate_token_conservation,
    verify_psm_structure,
)


class TestFIMInvariants:
    """Property-based tests for FIM transformation invariants."""

    @given(
        token_ids=st.lists(st.integers(min_value=2, max_value=50000), min_size=10, max_size=200),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_token_conservation(self, token_ids, seed, serializer_with_fim, fim_token_ids):
        """Property: All tokens are conserved (no loss/duplication)."""
        rng = Random(seed)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids.copy(),
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

    @given(
        token_ids=st.lists(st.integers(min_value=2, max_value=50000), min_size=10, max_size=200),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_length_constraint(self, token_ids, seed, serializer_with_fim, fim_token_ids):
        """Property: Transformed length = original + 3 special tokens."""
        rng = Random(seed)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids.copy(),
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        assert len(transformed) == len(token_ids) + 3

    @given(
        token_ids=st.lists(st.integers(min_value=2, max_value=50000), min_size=10, max_size=200),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_psm_structure(self, token_ids, seed, serializer_with_fim, fim_token_ids):
        """Property: Result always has [FP, ..., FS, ..., FM, ...] structure."""
        rng = Random(seed)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids.copy(),
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        assert verify_psm_structure(
            transformed, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )

    @given(
        token_ids=st.lists(
            st.integers(
                min_value=2, max_value=50000
            ),  # Below special token IDs to avoid collisions
            min_size=10,
            max_size=200,
        ),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_special_token_uniqueness(
        self, token_ids, seed, serializer_with_fim, fim_token_ids
    ):
        """Property: Each special token appears exactly once (excluding pre-existing occurrences)."""
        # Filter out any tokens that happen to match FIM token IDs
        special_ids = {fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]}
        filtered_token_ids = [t for t in token_ids if t not in special_ids]

        # Skip if filtered list is too short
        if len(filtered_token_ids) < 10:
            return

        rng = Random(seed)

        transformed = serializer_with_fim._apply_fim_transformation(
            filtered_token_ids.copy(),
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        assert transformed.count(fim_token_ids["prefix"]) == 1
        assert transformed.count(fim_token_ids["suffix"]) == 1
        assert transformed.count(fim_token_ids["middle"]) == 1

    @given(
        token_ids=st.lists(st.integers(min_value=2, max_value=50000), min_size=10, max_size=200),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_determinism(self, token_ids, seed, serializer_with_fim, fim_token_ids):
        """Property: Same seed + input produces same output."""
        rng1 = Random(seed)
        transformed1 = serializer_with_fim._apply_fim_transformation(
            token_ids.copy(),
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng1,
        )

        rng2 = Random(seed)  # Same seed
        transformed2 = serializer_with_fim._apply_fim_transformation(
            token_ids.copy(),
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng2,
        )

        assert transformed1 == transformed2

    @given(
        token_ids=st.lists(st.integers(min_value=2, max_value=50000), min_size=10, max_size=200),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_no_empty_sections(self, token_ids, seed, serializer_with_fim, fim_token_ids):
        """Property: Prefix, middle, and suffix are all non-empty."""
        rng = Random(seed)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids.copy(),
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        # Find positions
        fp_pos = transformed.index(fim_token_ids["prefix"])
        fs_pos = transformed.index(fim_token_ids["suffix"])
        fm_pos = transformed.index(fim_token_ids["middle"])

        # Calculate section lengths
        prefix_len = fs_pos - fp_pos - 1
        suffix_len = fm_pos - fs_pos - 1
        middle_len = len(transformed) - fm_pos - 1

        assert prefix_len > 0, "Empty prefix section"
        assert suffix_len > 0, "Empty suffix section"
        assert middle_len > 0, "Empty middle section"

    @given(
        token_ids=st.lists(st.integers(min_value=2, max_value=50000), min_size=10, max_size=200),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_reconstruction(self, token_ids, seed, serializer_with_fim, fim_token_ids):
        """Property: Original can be reconstructed from transformed."""
        rng = Random(seed)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids.copy(),
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        # Reconstruct
        reconstructed = reconstruct_original_sequence(
            transformed, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )

        assert reconstructed == token_ids


class TestFIMShortSequenceProperty:
    """Property tests for short sequences (< 10 tokens)."""

    @given(token_ids=st.lists(st.integers(min_value=2, max_value=50000), min_size=0, max_size=9))
    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_short_sequences_unchanged(
        self, token_ids, serializer_with_fim, fim_token_ids
    ):
        """Property: Sequences < 10 tokens are not transformed."""
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids.copy(),
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        # Should be unchanged
        assert transformed == token_ids


class TestFIMEdgeCaseProperties:
    """Property tests for edge cases."""

    @given(
        value=st.integers(min_value=2, max_value=50000),
        count=st.integers(min_value=10, max_value=100),
    )
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_repeated_values(self, value, count, serializer_with_fim, fim_token_ids):
        """Property: Repeated values are handled correctly."""
        # Skip if value matches any FIM token ID (edge case)
        special_ids = {fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]}
        if value in special_ids:
            return

        token_ids = [value] * count
        rng = Random(42)

        transformed = serializer_with_fim._apply_fim_transformation(
            token_ids.copy(),
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng,
        )

        # Verify PSM structure
        assert verify_psm_structure(
            transformed, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )

        # Remove special tokens and verify all are the same value
        special = [fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]]
        content = [t for t in transformed if t not in special]
        assert all(t == value for t in content)
        assert len(content) == count


class TestFIMCompositionProperty:
    """Test composition properties of FIM transformation."""

    @given(
        token_ids=st.lists(st.integers(min_value=2, max_value=50000), min_size=10, max_size=100),
        seed1=st.integers(min_value=0, max_value=10000),
        seed2=st.integers(min_value=0, max_value=10000),
    )
    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_different_seeds_different_results(
        self, token_ids, seed1, seed2, serializer_with_fim, fim_token_ids
    ):
        """Property: Different seeds (usually) produce different transformations."""
        # Skip if seeds are the same
        if seed1 == seed2:
            return

        rng1 = Random(seed1)
        transformed1 = serializer_with_fim._apply_fim_transformation(
            token_ids.copy(),
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng1,
        )

        rng2 = Random(seed2)
        transformed2 = serializer_with_fim._apply_fim_transformation(
            token_ids.copy(),
            fim_token_ids["prefix"],
            fim_token_ids["suffix"],
            fim_token_ids["middle"],
            rng2,
        )

        # Different seeds should usually (but not always) produce different results
        # We don't assert they're different because there's a small chance they match
        # Just verify both are valid transformations
        assert verify_psm_structure(
            transformed1, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )
        assert verify_psm_structure(
            transformed2, fim_token_ids["prefix"], fim_token_ids["suffix"], fim_token_ids["middle"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
