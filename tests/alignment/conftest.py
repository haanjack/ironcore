# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for GRPO integration tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture(scope="session")
def mock_tokenizer():
    """Provide a mock tokenizer for tests that need get_tokenizer()."""

    # Create mock tokenizer
    mock_tok = MagicMock()
    mock_tok.eos_token_id = 0
    mock_tok.pad_token_id = 0
    mock_tok.batch_decode = MagicMock(return_value=["decoded text"])

    def mock_encode(*args, **kwargs):
        # Return mock encoded output
        batch_size = len(args[0]) if args else 1
        return {
            "input_ids": MagicMock(shape=(batch_size, 32), to=MagicMock(return_value=torch.zeros(batch_size, 32, dtype=torch.long))),
            "attention_mask": MagicMock(shape=(batch_size, 32), to=MagicMock(return_value=torch.ones(batch_size, 32))),
        }
    mock_tok.side_effect = mock_encode
    mock_tok.return_value = mock_encode("", max_length=32, truncation=True, return_tensors="pt")

    # Mock vocab size
    mock_tok.vocab_size = 1000
    mock_tok.padded_vocab_size = 1024

    # Patch global states
    with patch("ironcore.global_vars.GLOBAL_STATES") as mock_gs:
        mock_gs.get_tokenizer.return_value = mock_tok

        yield mock_tok


@pytest.fixture(scope="session")
def temp_jsonl_file(tmp_path):
    """Create a temporary JSONL file with test data."""
    import json

    data = [
        {"prompt": "What is 1+1?", "answer": "2", "type": "math"},
        {"prompt": "What is 2+2?", "answer": "4", "type": "math"},
        {"prompt": "What is 3+3?", "answer": "6", "type": "math"},
    ]

    file_path = Path(tmp_path) / "test_data.jsonl"
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    yield file_path

    # Cleanup
    file_path.unlink(missing_ok=True)
