# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from ironcore.tokenizer import tokenizer as tokenizer_module


class _FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token = None
    pad_token_id = None
    vocab_size = 100

    def __call__(self, text, **kwargs):
        return {"input_ids": [len(text)]}

    def decode(self, token_ids, **kwargs):
        return "decoded"

    def batch_decode(self, sequences, **kwargs):
        return ["decoded"] * len(sequences)


def _config(tokenizer_type: str = "bbpe"):
    return SimpleNamespace(
        model=SimpleNamespace(tokenizer_type=tokenizer_type, vocab_name_or_path="fake"),
        trainer=SimpleNamespace(
            model_path="",
            vocab_padding_unit=128,
            special_tokens_config_path=None,
        ),
    )


def test_build_bbpe_tokenizer_smoke(monkeypatch):
    monkeypatch.setattr(
        tokenizer_module.GPT2Tokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _FakeTokenizer(),
    )

    tokenizer = tokenizer_module.build_tokenizer(_config())

    assert tokenizer.eos_token_id == 2
    assert tokenizer.pad_token_id == 2
    assert tokenizer.padded_vocab_size == 128
    assert tokenizer("abc")["input_ids"] == [3]


def test_unknown_tokenizer_type_is_rejected(monkeypatch):
    monkeypatch.setattr(tokenizer_module.tiktoken, "list_encoding_names", lambda: [])
    with pytest.raises(NotImplementedError, match="not implemented"):
        tokenizer_module.build_tokenizer(_config("unknown"))
