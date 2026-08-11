# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from ironcore.layers import attention as attention_module
from ironcore.layers.attention import Attention


def _make_attention(training: bool) -> Attention:
    attention = Attention.__new__(Attention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(
        init=SimpleNamespace(seed=123),
        model=SimpleNamespace(dropout_attn=0.25),
    )
    attention.num_local_attention_heads = 1
    attention.num_local_attention_groups = 1
    attention.head_dimension = 4
    attention.train(training)
    return attention


def test_sdpa_dropout_uses_sharded_tp_rng_stream(monkeypatch):
    calls = []

    @contextmanager
    def fake_rng_fork(seed, device, *, sharded=False):
        calls.append((seed, device, sharded))
        yield

    monkeypatch.setattr(attention_module, "tensor_parallel_rng_fork", fake_rng_fork)
    monkeypatch.setattr(
        attention_module.F,
        "scaled_dot_product_attention",
        lambda query, key, value, **kwargs: query,
    )

    attention = _make_attention(training=True)
    qkv = torch.zeros(1, 2, 1, 4)
    attention._attention(qkv, qkv, qkv, attention_mask=None, is_causal=True)

    assert calls == [(123, qkv.device, True)]


def test_flash_attention_disables_dropout_during_eval(monkeypatch):
    captured = []

    def fake_flash(query, key, value, cu_q, cu_k, max_q, max_k, dropout_p, **kwargs):
        captured.append(dropout_p)
        return query

    monkeypatch.setattr(attention_module, "flash_attn_varlen_func", fake_flash)

    attention = _make_attention(training=False)
    qkv = torch.zeros(1, 2, 1, 4)
    attention._flash_attention(qkv, qkv, qkv, 2, 2)
    assert captured[-1] == 0.0

    attention.train()
    attention._flash_attention(qkv, qkv, qkv, 2, 2)
    assert captured[-1] == 0.25
