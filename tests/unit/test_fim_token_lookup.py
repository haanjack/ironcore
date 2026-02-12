"""
Unit tests for FIM token lookup functionality.

Tests the _get_token_id method for retrieving FIM special token IDs
from tokenizers, including error handling.
"""

import pytest
from transformers import AutoTokenizer

from ironcore.preprocessing.serializer import DataSerializer


class TestGetTokenIDSuccess:
    """Test successful token ID retrieval."""

    def test_get_token_id_success(self, serializer_with_fim, test_tokenizer_with_fim):
        """Test successful retrieval of FIM token IDs."""
        # Get token IDs for FIM tokens
        prefix_id = serializer_with_fim._get_token_id("<fim_prefix>")
        suffix_id = serializer_with_fim._get_token_id("<fim_suffix>")
        middle_id = serializer_with_fim._get_token_id("<fim_middle>")

        # Verify they are valid IDs (not unk)
        assert prefix_id is not None
        assert suffix_id is not None
        assert middle_id is not None

        # Verify they are different from each other
        assert prefix_id != suffix_id
        assert prefix_id != middle_id
        assert suffix_id != middle_id

        # Verify they match tokenizer's conversion
        assert prefix_id == test_tokenizer_with_fim.convert_tokens_to_ids("<fim_prefix>")
        assert suffix_id == test_tokenizer_with_fim.convert_tokens_to_ids("<fim_suffix>")
        assert middle_id == test_tokenizer_with_fim.convert_tokens_to_ids("<fim_middle>")

    def test_get_token_id_not_unk(self, serializer_with_fim, test_tokenizer_with_fim):
        """Verify returned token IDs are not the unknown token ID."""
        prefix_id = serializer_with_fim._get_token_id("<fim_prefix>")
        suffix_id = serializer_with_fim._get_token_id("<fim_suffix>")
        middle_id = serializer_with_fim._get_token_id("<fim_middle>")

        unk_id = test_tokenizer_with_fim.unk_token_id

        assert prefix_id != unk_id
        assert suffix_id != unk_id
        assert middle_id != unk_id


class TestGetTokenIDErrors:
    """Test error handling for missing tokens."""

    def test_get_token_id_missing_error(self, test_tokenizer_without_fim, test_config_fim_enabled):
        """Test that missing FIM token raises ValueError with clear message."""
        serializer = DataSerializer(
            test_config_fim_enabled, test_tokenizer_without_fim, verbose=False
        )

        with pytest.raises(ValueError) as exc_info:
            serializer._get_token_id("<fim_prefix>")

        error_msg = str(exc_info.value)
        assert "not found in tokenizer vocabulary" in error_msg
        assert "fim_prefix" in error_msg
        assert "add_special_tokens" in error_msg  # Should suggest solution

    def test_get_token_id_all_missing(self, test_tokenizer_without_fim, test_config_fim_enabled):
        """Test error for all three missing FIM tokens."""
        serializer = DataSerializer(
            test_config_fim_enabled, test_tokenizer_without_fim, verbose=False
        )

        # All three should raise errors
        for token in ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]:
            with pytest.raises(ValueError) as exc_info:
                serializer._get_token_id(token)

            error_msg = str(exc_info.value)
            assert token in error_msg or token.strip("<>") in error_msg

    def test_get_token_id_error_message_helpful(
        self, test_tokenizer_without_fim, test_config_fim_enabled
    ):
        """Test that error message provides helpful instructions."""
        serializer = DataSerializer(
            test_config_fim_enabled, test_tokenizer_without_fim, verbose=False
        )

        with pytest.raises(ValueError) as exc_info:
            serializer._get_token_id("<fim_prefix>")

        error_msg = str(exc_info.value)

        # Should contain helpful information
        assert "add_special_tokens" in error_msg
        assert "additional_special_tokens" in error_msg
        assert "save_pretrained" in error_msg


class TestGetTokenIDCustomTokens:
    """Test token lookup with custom token names."""

    def test_get_token_id_custom_tokens(self, test_config_fim_enabled, tmp_path):
        """Test token lookup with custom FIM token names."""
        # Create tokenizer with custom FIM tokens
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        custom_tokens = ["[PREFIX]", "[SUFFIX]", "[MIDDLE]"]
        tokenizer.add_special_tokens({"additional_special_tokens": custom_tokens})

        # Update config to use custom tokens
        test_config_fim_enabled.datasets[0].fim_prefix_token = "[PREFIX]"
        test_config_fim_enabled.datasets[0].fim_suffix_token = "[SUFFIX]"
        test_config_fim_enabled.datasets[0].fim_middle_token = "[MIDDLE]"

        serializer = DataSerializer(test_config_fim_enabled, tokenizer, verbose=False)

        # Get token IDs
        prefix_id = serializer._get_token_id("[PREFIX]")
        suffix_id = serializer._get_token_id("[SUFFIX]")
        middle_id = serializer._get_token_id("[MIDDLE]")

        # Verify they are valid and distinct
        assert prefix_id != tokenizer.unk_token_id
        assert suffix_id != tokenizer.unk_token_id
        assert middle_id != tokenizer.unk_token_id
        assert len({prefix_id, suffix_id, middle_id}) == 3


class TestGetTokenIDEdgeCases:
    """Test edge cases for token lookup."""

    def test_get_token_id_empty_string(self, serializer_with_fim):
        """Test behavior with empty token string."""
        # Empty string should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            serializer_with_fim._get_token_id("")

        assert "not found" in str(exc_info.value)

    def test_get_token_id_whitespace(self, serializer_with_fim):
        """Test behavior with whitespace-only token."""
        with pytest.raises(ValueError):
            serializer_with_fim._get_token_id("   ")

    def test_get_token_id_nonexistent(self, serializer_with_fim):
        """Test behavior with non-existent token."""
        with pytest.raises(ValueError) as exc_info:
            serializer_with_fim._get_token_id("<nonexistent_token>")

        assert "not found" in str(exc_info.value)


class TestTiktokenLimitation:
    """Test and document tiktoken limitations with FIM tokens."""

    def test_tiktoken_unsupported_documented(self):
        """
        Document that tiktoken doesn't support arbitrary special tokens.

        This is a limitation test - tiktoken tokenizers cannot easily add
        new special tokens like FIM tokens. Users must use HuggingFace tokenizers.
        """
        # This test serves as documentation
        # tiktoken.get_encoding("cl100k_base") doesn't support add_special_tokens

        # For FIM functionality, HuggingFace tokenizers are required
        # This is documented in the error message from _get_token_id
        pass

    def test_get_token_id_requires_hf_tokenizer(self, test_config_fim_enabled):
        """Test that _get_token_id requires HuggingFace tokenizer."""

        class FakeTiktokenTokenizer:
            """Mock tiktoken-like tokenizer without convert_tokens_to_ids."""

            def __init__(self):
                self.eos_token_id = 0

            def encode(self, text):
                return [1, 2, 3]

        fake_tokenizer = FakeTiktokenTokenizer()
        serializer = DataSerializer(test_config_fim_enabled, fake_tokenizer, verbose=False)

        # Should raise TypeError for unsupported tokenizer
        with pytest.raises(TypeError) as exc_info:
            serializer._get_token_id("<fim_prefix>")

        error_msg = str(exc_info.value)
        assert "Unsupported tokenizer type" in error_msg
        assert "HuggingFace" in error_msg


class TestGetTokenIDConsistency:
    """Test consistency of token ID retrieval."""

    def test_get_token_id_consistent_calls(self, serializer_with_fim):
        """Test that multiple calls return same token ID."""
        prefix_id_1 = serializer_with_fim._get_token_id("<fim_prefix>")
        prefix_id_2 = serializer_with_fim._get_token_id("<fim_prefix>")

        assert prefix_id_1 == prefix_id_2

    def test_get_token_id_matches_tokenizer(self, serializer_with_fim, test_tokenizer_with_fim):
        """Test that _get_token_id matches tokenizer's convert_tokens_to_ids."""
        tokens = ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]

        for token in tokens:
            serializer_id = serializer_with_fim._get_token_id(token)
            tokenizer_id = test_tokenizer_with_fim.convert_tokens_to_ids(token)

            assert serializer_id == tokenizer_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
