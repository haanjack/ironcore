# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

from abc import ABC
from pathlib import Path
from typing import Dict, Optional, Union

from transformers import AutoTokenizer, GPT2Tokenizer

from ironcore.config import MainConfig
from ironcore.utils import load_yaml_config

try:
    import tiktoken
except:
    raise ImportError(f"tiktoken is not installed.")


class Tokenizer(ABC):

    def __init__(
        self,
        tokenizer,
        vocab_name_or_path: Union[str, Path],
        vocab_padding_unit: int = 128,
        special_tokens_config: Optional[Dict[str, str]] = None,
    ):
        self._tokenizer = tokenizer
        self.vocab_name_or_path = vocab_name_or_path

        if special_tokens_config:
            for token_name, token_value in special_tokens_config.items():
                setattr(self._tokenizer, token_name, token_value)

        # eos token
        self._eos_token = (
            getattr(self._tokenizer, "eot_token", None)
            or getattr(self._tokenizer, "eod_token", None)
            or getattr(self._tokenizer, "eos_token", None)
        )
        assert (
            self._eos_token is not None
        ), "eos_token not found in the tokenizer. Please check special_tokens_config or tokenizer settings."

        # pad token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        # vocab_size
        self._vocab_size = getattr(
            self._tokenizer, "vocab_size", getattr(
                self._tokenizer, "n_vocab", None)
        )
        assert (
            self._vocab_size is not None
        ), "Could not find vocab_size property in tokenizer"
        vocab_size = self._vocab_size
        if hasattr(self._tokenizer, "SPECIAL_TOKENS_ATTRIBUTES"):
            vocab_size += len(self._tokenizer.SPECIAL_TOKENS_ATTRIBUTES) + len(
                self._tokenizer.added_tokens_decoder
            )
        # padding to the nearest multiple of vocab_padding_unit
        self._padded_vocab_size = (
            (vocab_size + vocab_padding_unit - 1)
            // vocab_padding_unit
            * vocab_padding_unit
        )

    @property
    def eos_token(self):
        """eos token"""
        return self._eos_token

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value
        if hasattr(self._tokenizer, "eos_token"):
            self._tokenizer.eos_token = value

    @property
    def eos_token_id(self):
        return self._tokenizer.eos_token_id

    @property
    def eod_token_id(self):
        if hasattr(self._tokenizer, "eod_token_id"):
            return getattr(self._tokenizer, "eod_token_id")
        return self.eos_token_id

    @property
    def pad_token(self):
        return self._tokenizer.pad_token

    @property
    def pad_token_id(self):
        return self._tokenizer.pad_token_id

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def padded_vocab_size(self):
        return self._padded_vocab_size

    def encode(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)

    def decode(self, token_ids):
        try:
            if not token_ids:
                raise ValueError("Input token_ids is empty or invalid.")
            return self._tokenizer.decode(token_ids)
        except Exception as e:
            print(f"Error occured during decoding: {e}")
            return ""

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def __str__(self):
        return (
            f"Tokenizer type: {self.__class__.__name__}\n"
            f"Vocab path: {self.vocab_name_or_path}\n"
            f"Vocab size: {self._vocab_size}\n"
            f"Padded vocab size: {self._padded_vocab_size}\n"
            f"EOS token: {self._eos_token}\n"
        )


class BbpeTokenizer(Tokenizer):

    def __init__(
        self,
        vocab_name_or_path: Union[str, Path],
        merge_file_path: Optional[Union[str, Path]] = None,
        vocab_padding_unit: int = 128,
        special_tokens_config: Optional[Dict[str, str]] = None,
    ):

        if merge_file_path:
            tokenizer = tiktoken.get_bpe_tokenizer(
                vocab_name_or_path, merge_file_path
            )
        else:
            tokenizer = GPT2Tokenizer.from_pretrained(vocab_name_or_path)

        super().__init__(
            tokenizer,
            vocab_name_or_path,
            vocab_padding_unit,
            special_tokens_config,
        )


class SentencePieceTokenizer(Tokenizer):

    def __init__(
        self,
        vocab_name_or_path: Union[str, Path],
        vocab_padding_unit: int = 128,
        special_tokens_config: Optional[Dict[str, str]] = None,
    ):
        tokenizer = AutoTokenizer.from_pretrained(vocab_name_or_path)

        super().__init__(
            tokenizer,
            vocab_name_or_path,
            vocab_padding_unit,
            special_tokens_config,
        )


def build_tokenizer(config: MainConfig) -> Tokenizer:
    """Initialize trainer's tokenizer"""

    tokenizer_type = config.model.tokenizer_type.lower()

    # load special tokens config if exists
    if config.trainer.special_tokens_config_path:
        special_tokens_config = load_yaml_config(
            config.trainer.special_tokens_config_path
        )
    else:
        special_tokens_config = None

    kwargs = {
        "vocab_name_or_path": (
            config.model.vocab_name_or_path
            if config.model.vocab_name_or_path
            else config.trainer.model_path
        ),
        "vocab_padding_unit": config.trainer.vocab_padding_unit,
        "special_tokens_config": (
            special_tokens_config
            if hasattr(config.trainer, "special_tokens_config")
            else None
        ),
    }

    # tokenizer types
    bbpe_types = {"bbpe", "gpt2"}
    sentencepiece_tyeps = {"sentencepiece", "llama", "t5"}

    if tokenizer_type in bbpe_types or tokenizer_type in tiktoken.list_encoding_names():
        return BbpeTokenizer(**kwargs)
    elif tokenizer_type in sentencepiece_tyeps:
        return SentencePieceTokenizer(**kwargs)
    else:
        raise NotImplementedError(
            f"Given tokenizer ({tokenizer_type}) is not implemented."
        )
