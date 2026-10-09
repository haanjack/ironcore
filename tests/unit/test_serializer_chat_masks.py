# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Assistant labels follow complete conversation prefixes, including one BOS."""

from __future__ import annotations

from collections import UserDict

import pytest
import torch

from ironcore.config import DataConfig
from ironcore.preprocessing.serializer import DataSerializer


def test_sft_streaming_memmap_can_be_collated_without_mutating_source(tmp_path):
    import numpy as np

    from ironcore.dataloader.collator import UniversalCollator

    path = tmp_path / "tokens.bin"
    np.array([2, 10, 11, 12, 13], dtype=np.uint32).tofile(path)
    tokens = np.memmap(path, dtype=np.uint32, mode="r")
    sample = {"token_ids": tokens, "metadata": {"mask_ranges": [[0, 3]]}}
    batch = UniversalCollator("sft", 8, return_full_attention_mask=True)([sample])
    torch.testing.assert_close(batch["input_ids"][0, :4], torch.tensor([2, 10, 11, 12]))
    torch.testing.assert_close(
        batch["labels"][0], torch.tensor([-100, -100, 12, 13, -100, -100, -100, -100])
    )
    assert path.read_bytes() == np.array([2, 10, 11, 12, 13], dtype=np.uint32).tobytes()


class ChatTokenizer:
    def __init__(self, mapping: bool) -> None:
        self.mapping = mapping

    def apply_chat_template(
        self, messages: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool
    ) -> list[int] | UserDict:
        assert tokenize
        text = "BOS"
        for message in messages:
            text += f"<{message['role']}>" + message["content"] + "END"
        if add_generation_prompt:
            text += "<assistant>"
        ids = list(text.encode())
        return UserDict(input_ids=ids, attention_mask=[1] * len(ids)) if self.mapping else ids


@pytest.mark.parametrize("mapping", [False, True])
def test_multiturn_masks_exclude_headers_and_keep_assistant_closures(tmp_path, mapping):
    serializer = DataSerializer(
        DataConfig(preprocessed_dir=tmp_path / "data", cache_dir=tmp_path / "cache"),
        ChatTokenizer(mapping),
        verbose=False,
    )
    messages = [
        {"role": "system", "content": "Instructions"},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Followup"},
        {"role": "assistant", "content": "Second answer"},
        {"role": "user", "content": "Unanswered"},
    ]
    ids, masks = serializer._apply_chat_template_and_get_masks(messages)
    supervised = bytes(v for i, v in enumerate(ids) if not any(a <= i < b for a, b in masks))
    assert supervised == b"First answerENDSecond answerEND"
    assert bytes(ids).count(b"BOS") == 1


def test_sentencepiece_preprocessing_uses_auto_tokenizer(monkeypatch):
    from transformers import AutoTokenizer

    from ironcore.preprocess import _resolve_tokenizer

    loaded = object()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda path: loaded)
    assert _resolve_tokenizer(DataConfig(tokenizer_type="sentencepiece")) is loaded


def test_rewritten_prefix_fails_instead_of_training_prompt_tokens(tmp_path):
    tokenizer = ChatTokenizer(False)
    original = tokenizer.apply_chat_template

    def rewrite(messages, **kwargs):
        ids = original(messages, **kwargs)
        return [99] + ids if len(messages) == 1 else ids

    tokenizer.apply_chat_template = rewrite
    serializer = DataSerializer(
        DataConfig(preprocessed_dir=tmp_path / "data", cache_dir=tmp_path / "cache"),
        tokenizer,
        verbose=False,
    )
    with pytest.raises(ValueError, match="rewrites conversation prefixes"):
        serializer._apply_chat_template_and_get_masks(
            [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]
        )
