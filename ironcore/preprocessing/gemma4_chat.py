# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 text SFT formatting, separate from the inference chat template."""

import hashlib
import json

SFT_FORMAT = "gemma4-sft-v3"
_REASONING = (
    "{%- set thinking_text = message.get('reasoning') or message.get('reasoning_content') -%}"
)
_GATE = "{%- set thinking_gate = (loop.index0 > ns_turn.last_user_idx) or (preserve_thinking and message.get('tool_calls')) -%}"
_GENERATION_GATE = "{%- if add_generation_prompt -%}"
_TRAINING_REASONING = r"""
    {%- if role == 'model' and not continue_same_model_turn and not message.get('tool_calls') -%}
        {%- if enable_thinking and not thinking_text -%}
            {{- raise_exception('Gemma 4 thinking SFT requires assistant reasoning or reasoning_content; use non-thinking SFT for plain answers.') -}}
        {%- elif not enable_thinking and thinking_text -%}
            {{- raise_exception('Gemma 4 reasoning data requires enable_thinking=True.') -}}
        {%- elif not enable_thinking -%}
            {{- '<|channel>thought\n<channel|>' -}}
        {%- endif -%}
    {%- endif -%}
"""
_THINKING_ROWS = r"""
{%- if enable_thinking and (messages | selectattr('role', 'equalto', 'assistant') | list | length) > 1 -%}
    {{- raise_exception('Gemma 4 reasoning SFT requires one assistant target per example; split multi-turn reasoning conversations before training.') -}}
{%- endif -%}
"""


def gemma4_sft_chat_template(inference_template: str) -> str:
    """Adapt the official template for prefix-consistent teacher forcing.

    Plain answers match the checkpoint's non-thinking generation prefix:
    A4B includes an empty thought channel, whereas E2B ends at the model header.
    Reasoning data uses structured assistant
    fields in one-assistant examples. Multi-turn reasoning must be split before
    training: the official template strips past thoughts, rewriting prefixes
    that cannot share a single teacher-forced token stream. The tokenizer's
    inference template and its reasoning-history rules are untouched.
    Unsupported template revisions fail rather than silently changing labels.
    """
    if not isinstance(inference_template, str) or any(
        inference_template.count(anchor) != 1 for anchor in (_REASONING, _GATE, _GENERATION_GATE)
    ):
        raise ValueError("Unsupported Gemma 4 chat template: reasoning anchors differ")
    generation = inference_template.split(_GENERATION_GATE, 1)[1]
    training_reasoning = _TRAINING_REASONING
    if "<|channel>thought\\n<channel|>" not in generation:
        training_reasoning = training_reasoning.replace(
            "{{- '<|channel>thought\\n<channel|>' -}}", ""
        )
    return _THINKING_ROWS + inference_template.replace(_REASONING, _REASONING + training_reasoning)


def gemma4_sft_format_metadata(template: str, enable_thinking: bool) -> dict:
    """Identify the exact training format, also used to isolate tokenizer caches."""
    fingerprint = hashlib.sha256(
        json.dumps({"template": template, "enable_thinking": enable_thinking}).encode()
    ).hexdigest()
    return {
        "format": SFT_FORMAT,
        "enable_thinking": enable_thinking,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "cache_namespace": f"{SFT_FORMAT}-{fingerprint[:16]}",
    }
