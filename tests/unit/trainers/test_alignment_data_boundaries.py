# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
import torch
from scripts.benchmark_alignment import encode_chat_response
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


def test_response_mask_includes_bpe_token_crossing_template_boundary():
    backend = Tokenizer(BPE(unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.train_from_iterator(
        ["bos PROMPTsuffix end"] * 20, BpeTrainer(vocab_size=100, special_tokens=["<unk>"])
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.chat_template = "bos {{messages[0]['content']}}{% if messages|length > 1 %}{{messages[1]['content']}} end{% endif %}"
    messages = [{"role": "user", "content": "PROMPT"}, {"role": "assistant", "content": "suffix"}]
    prefix = tokenizer.encode(
        tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True),
        add_special_tokens=False,
    )
    full = tokenizer.encode(
        tokenizer.apply_chat_template(messages, tokenize=False), add_special_tokens=False
    )
    assert full[: len(prefix)] != prefix  # Actual BPE merge, not a mocked encoding.
    input_ids, labels = encode_chat_response(tokenizer, messages, 64)
    torch.testing.assert_close(input_ids, torch.tensor(full[:-1]))
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    offset = len(
        tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    )
    mapping = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)[
        "offset_mapping"
    ]
    crossing = [i for i, (start, end) in enumerate(mapping) if start < offset < end]
    assert len(crossing) == 1
    assert labels[crossing[0] - 1] == full[crossing[0]]
    assert (labels[: crossing[0] - 1] == -100).all()
    assert (labels[crossing[0] - 1 :] != -100).all()
