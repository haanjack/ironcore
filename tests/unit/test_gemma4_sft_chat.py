# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Check supervision of plain answers and structured Gemma 4 reasoning."""

import pytest
import torch
from jinja2.exceptions import TemplateError
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from ironcore.config import DataConfig
from ironcore.dataloader.collator import UniversalCollator
from ironcore.preprocessing.gemma4_chat import (
    gemma4_sft_chat_template,
    gemma4_sft_format_metadata,
)
from ironcore.preprocessing.serializer import DataSerializer

# A small rendition of the official template's mode and history behavior.
# Integration with the complete released tokenizer is audited separately.
INFERENCE_TEMPLATE = r"""
{{- bos_token -}}
{%- if enable_thinking -%}{{- '<|think|>' -}}{%- endif -%}
{%- set ns_turn = namespace(last_user_idx=-1) -%}
{%- for message in messages -%}
{%- if message.role == 'user' -%}{%- set ns_turn.last_user_idx = loop.index0 -%}{%- endif -%}
{%- endfor -%}
{%- for message in messages -%}
{%- set role = 'model' if message.role == 'assistant' else message.role -%}
{%- set continue_same_model_turn = false -%}
{{- '<|turn>' + role + '\n' -}}
{%- set thinking_text = message.get('reasoning') or message.get('reasoning_content') -%}
{%- set thinking_gate = (loop.index0 > ns_turn.last_user_idx) or (preserve_thinking and message.get('tool_calls')) -%}
{%- if thinking_text and thinking_gate -%}
{{- '<|channel>thought\n' + thinking_text + '\n<channel|>' -}}
{%- endif -%}
{{- message.content + '<turn|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|turn>model\n' -}}
{%- if not enable_thinking -%}{{- '<|channel>thought\n<channel|>' -}}{%- endif -%}
{%- endif -%}
"""


@pytest.fixture
def tokenizer():
    controls = ["[UNK]", "<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|think|>"]
    words = ["system", "user", "model", "thought", "Q1", "Q2", "A1", "A2", "R1", "R2"]
    backend = Tokenizer(
        models.WordLevel(dict(zip(controls + words, range(17), strict=True)), unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        unk_token="[UNK]",
        additional_special_tokens=controls[2:],
        chat_template=INFERENCE_TEMPLATE,
    )


def serialize(tmp_path, tokenizer, messages, thinking=False, template=None):
    serializer = DataSerializer(
        DataConfig(preprocessed_dir=tmp_path / "data", cache_dir=tmp_path / "cache"),
        tokenizer,
        verbose=False,
    )
    ids, masks = serializer._apply_chat_template_and_get_masks(
        messages, template, {"enable_thinking": thinking}
    )
    batch = UniversalCollator("sft", max_seq_len=len(ids) - 1, pack_sequences=False)(
        [{"token_ids": ids, "metadata": {"mask_ranges": masks}}]
    )
    return ids, batch["labels"][0]


def conversation(reasoning=False):
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    if reasoning:
        messages[1]["reasoning"] = "R1"
        messages[3]["reasoning_content"] = "R2"
    return messages


def test_plain_answers_use_empty_thought_without_supervising_channel_open(tmp_path, tokenizer):
    messages = conversation()
    with pytest.raises(ValueError, match="rewrites conversation prefixes"):
        serialize(tmp_path, tokenizer, messages)
    template = gemma4_sft_chat_template(tokenizer.chat_template)
    ids, labels = serialize(tmp_path, tokenizer, messages, template=template)
    assert tokenizer.chat_template == INFERENCE_TEMPLATE
    assert tokenizer.convert_tokens_to_ids("<|think|>") not in ids
    for index in (1, 3):
        prefix = tokenizer.apply_chat_template(
            messages[:index],
            tokenize=True,
            add_generation_prompt=True,
            chat_template=template,
            enable_thinking=False,
        )["input_ids"]
        assert ids[: len(prefix)] == prefix
        assert labels[len(prefix) - 1] == tokenizer.convert_tokens_to_ids(
            messages[index]["content"]
        )
    channel = tokenizer.convert_tokens_to_ids("<|channel>")
    assert ids.count(channel) == 2
    assert all(labels[i - 1] == -100 for i, token in enumerate(ids) if token == channel)
    assert labels[labels != -100].tolist() == [
        tokenizer.convert_tokens_to_ids(w) for w in ("A1", "<turn|>", "A2", "<turn|>")
    ]


def test_plain_template_without_thought_prefix_does_not_add_a_channel(tmp_path, tokenizer):
    # Released E2B generation ends at the model header; A4B appends an empty
    # thought channel. Plain teacher forcing must follow each actual prefix.
    inference = INFERENCE_TEMPLATE.replace(
        "{%- if not enable_thinking -%}{{- '<|channel>thought\\n<channel|>' -}}{%- endif -%}",
        "",
    )
    tokenizer.chat_template = inference
    messages = conversation()
    expected_ids, expected_labels = serialize(tmp_path, tokenizer, messages)
    ids, labels = serialize(
        tmp_path, tokenizer, messages, template=gemma4_sft_chat_template(inference)
    )
    assert ids == expected_ids
    torch.testing.assert_close(labels, expected_labels, atol=0, rtol=0)


def test_reasoning_channel_and_text_are_supervised(tmp_path, tokenizer):
    template = gemma4_sft_chat_template(tokenizer.chat_template)
    ids, labels = serialize(
        tmp_path, tokenizer, conversation(True)[:2], thinking=True, template=template
    )
    channel = tokenizer.convert_tokens_to_ids("<|channel>")
    for i, token in enumerate(ids):
        if token == channel:
            assert labels[i - 1] == channel
    assert labels[labels != -100].tolist() == [
        tokenizer.convert_tokens_to_ids(w)
        for w in (
            "<|channel>",
            "thought",
            "R1",
            "<channel|>",
            "A1",
            "<turn|>",
        )
    ]


def test_multiturn_reasoning_requires_separate_targets(tmp_path, tokenizer):
    with pytest.raises(TemplateError, match="one assistant target"):
        serialize(
            tmp_path,
            tokenizer,
            conversation(True),
            thinking=True,
            template=gemma4_sft_chat_template(tokenizer.chat_template),
        )


@pytest.mark.parametrize("thinking", [False, True])
def test_incompatible_data_and_thinking_mode_fail_before_training(tmp_path, tokenizer, thinking):
    template = gemma4_sft_chat_template(tokenizer.chat_template)
    with pytest.raises(TemplateError, match="requires"):
        serialize(
            tmp_path,
            tokenizer,
            conversation(not thinking)[:2],
            thinking=thinking,
            template=template,
        )


def test_old_objective_discourages_channel_open_but_new_plain_objective_masks_it(
    tmp_path, tokenizer
):
    messages = conversation()[:2]
    old_ids, old_labels = serialize(tmp_path, tokenizer, messages, thinking=True)
    old_prefix = tokenizer.apply_chat_template(
        messages[:1], tokenize=True, add_generation_prompt=True, enable_thinking=True
    )["input_ids"]
    new_ids, new_labels = serialize(
        tmp_path, tokenizer, messages, template=gemma4_sft_chat_template(tokenizer.chat_template)
    )
    channel = tokenizer.convert_tokens_to_ids("<|channel>")
    for ids, labels, decision, suppressed in (
        (old_ids, old_labels, len(old_prefix) - 1, True),
        (new_ids, new_labels, new_ids.index(channel) - 1, False),
    ):
        logits = torch.zeros(len(ids) - 1, len(tokenizer), requires_grad=True)
        torch.nn.functional.cross_entropy(logits, labels).backward()
        assert bool(logits.grad[decision, channel] > 0) == suppressed


def test_changed_template_or_mode_cannot_reuse_old_label_cache():
    template = gemma4_sft_chat_template(INFERENCE_TEMPLATE)
    plain = gemma4_sft_format_metadata(template, False)
    reasoning = gemma4_sft_format_metadata(template, True)
    changed = gemma4_sft_format_metadata(template + "\n", False)
    assert len({m["cache_namespace"] for m in (plain, reasoning, changed)}) == 3
    with pytest.raises(ValueError, match="Unsupported"):
        gemma4_sft_chat_template("{{ messages }}")
