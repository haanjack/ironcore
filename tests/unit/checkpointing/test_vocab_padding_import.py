# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Odd HF vocabulary sizes must be padded globally before TP partitioning."""

import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from ironcore.checkpointing.hf_interop import load_from_huggingface
from ironcore.parallel import parallel_states


@pytest.mark.parametrize("rank", [0, 1])
def test_odd_vocab_global_padding_keeps_token_ids_and_untied_head(monkeypatch, tmp_path, rank):
    monkeypatch.setattr(parallel_states, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parallel_states, "get_tensor_model_parallel_rank", lambda: rank)
    model = nn.Module()
    model.embedding = nn.Module()
    model.embedding.word_embeddings = nn.Module()
    embed = model.embedding.word_embeddings
    embed.weight = nn.Parameter(torch.empty(2, 4))
    embed.column_parallel, embed.row_parallel, embed.concatenated_weights = False, True, 1
    model.output_layer = nn.Module()
    model.output_layer.weight = nn.Parameter(torch.empty(4, 2))
    model.output_layer.column_parallel, model.output_layer.row_parallel = True, False
    model.output_layer.concatenated_weights = 1
    embedding = torch.arange(12).float().reshape(3, 4)
    head = embedding + 100
    save_file(
        {"model.embed_tokens.weight": embedding, "lm_head.weight": head},
        tmp_path / "model.safetensors",
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "llama", "num_hidden_layers": 0})
    )
    result = load_from_huggingface(tmp_path, model, strict=True)
    assert not result["missing_keys"] and not result["unexpected_keys"]
    full_embedding = torch.cat([embedding, torch.zeros(1, 4)])
    full_head = torch.cat([head, torch.zeros(1, 4)])
    torch.testing.assert_close(embed.weight, full_embedding.chunk(2)[rank], atol=0, rtol=0)
    torch.testing.assert_close(
        model.output_layer.weight, full_head.chunk(2)[rank].T, atol=0, rtol=0
    )
