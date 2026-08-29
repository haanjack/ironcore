#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
"""
KV cache tests with tensor_model_parallel_size=1.

TP=2 tests live in tests/multi_gpu/kvcache/test_kv_cache_tp.py.

Run:
    pytest tests/integration/kvcache/test_kv_cache.py -v
"""

import pytest
import torch

from ironcore.config import (
    DataConfig,
    InitConfig,
    KVCacheConfig,
    MainConfig,
    ModelConfig,
    OperationConfig,
    OptimConfig,
    ParallelConfig,
    PEFTConfig,
    PositionalEmbeddingConfig,
    ProfilerConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.config.config_model import BiasConfig
from ironcore.global_vars import global_states_cleanup, set_global_states
from ironcore.language_model import LanguageModel
from ironcore.parallel import parallel_states


def _tp_config(d_model: int = 512, num_layers: int = 2):
    """Create config for TP=1."""
    parallel_states.initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=10.0)

    kv_cache_config = KVCacheConfig(enabled=True, max_batch_size=4, max_seq_length=256)
    pos_emb_config = PositionalEmbeddingConfig(type="rope")

    model_config = ModelConfig(
        d_model=d_model,
        num_attention_heads=8,
        num_attention_groups=2,
        head_dim=d_model // 8,
        num_layers=num_layers,
        d_ffn=d_model * 4,
        max_seq_len=256,
        max_position_embeddings=256,
        dropout_attn=0.0,
        dropout_mlp=0.0,
        dropout_embd=0.0,
        positional_embedding=pos_emb_config,
        kv_cache=kv_cache_config,
        bias=BiasConfig.all_true(),
        layernorm_bias=True,
    )
    model_config.name = "GPT"

    trainer_config = TrainerConfig(
        tensor_model_parallel_size=1,
        use_flash_attn=False,
    )

    config = MainConfig(
        model=model_config,
        trainer=trainer_config,
        init=InitConfig(seed=42, init_std=0.02),
        optim=OptimConfig(max_lr=1e-3, weight_decay=0.01),
        data=DataConfig(),
        parallel=ParallelConfig(),
        operation=OperationConfig(train_steps=100, activation_recompute=False),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
        peft=PEFTConfig(),
    )

    set_global_states(config)
    return config


def _destroy_parallel():
    global_states_cleanup()
    parallel_states.destroy_model_parallel()


class TestKVCacheTP1:
    """KV cache tests with tensor_model_parallel_size=1."""

    @pytest.fixture(scope="class")
    def config(self):
        cfg = _tp_config()
        yield cfg
        _destroy_parallel()

    @pytest.fixture
    def model(self, config):
        model = LanguageModel(config)
        model.eval()
        yield model

    def test_end_to_end_generation(self, model):
        """Generate tokens one-by-one with cache, verify first-token logits match."""
        device = next(model.parameters()).device
        input_ids = torch.randint(0, 1000, (1, 10), device=device)

        with torch.no_grad():
            logits_0, _ = model(input_ids[:, :1], use_cache=True)
            logits_full = model(input_ids, use_cache=False)
            logits_full = logits_full[0] if isinstance(logits_full, tuple) else logits_full
        torch.testing.assert_close(logits_0, logits_full[:, :1], rtol=1e-4, atol=1e-5)

    def test_gqa_kv_group_shapes(self, model, config):
        """With 8 query heads and 2 KV groups, each TP=1 rank stores both groups."""
        device = next(model.parameters()).device
        input_ids = torch.randint(0, 1000, (1, 5), device=device)

        with torch.no_grad():
            _, past_kv = model(input_ids, use_cache=True)
        for layer_kv in past_kv:
            key, value = layer_kv
            expected_groups = (
                config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
            )
            assert key.shape == (1, 5, expected_groups, config.model.head_dim)
            assert value.shape == (1, 5, expected_groups, config.model.head_dim)

    def test_cached_vs_uncached_equivalence(self, model):
        """Generate sequence token-by-token with cache, verify against full-sequence output."""
        device = next(model.parameters()).device
        input_ids = torch.randint(0, 1000, (1, 5), device=device)

        with torch.no_grad():
            past_kv = None
            cached_logits = []
            for i in range(5):
                logits, past_kv = model(
                    input_ids[:, i : i + 1], use_cache=True, past_key_values=past_kv
                )
                cached_logits.append(logits)
            full_logits, _ = model(input_ids, use_cache=False)
        torch.testing.assert_close(
            torch.cat(cached_logits, dim=1), full_logits, rtol=1e-4, atol=1e-5
        )

    def test_cache_reuse_across_passes(self, model):
        """Process two sequences using cache; verify the second sequence's output matches."""
        device = next(model.parameters()).device
        ids_1 = torch.randint(0, 1000, (1, 10), device=device)
        ids_2 = torch.randint(0, 1000, (1, 10), device=device)

        with torch.no_grad():
            _, kv_1 = model(ids_1, use_cache=True)
            logits_2, _ = model(ids_2, use_cache=True, past_key_values=kv_1)
            full_logits, _ = model(torch.cat([ids_1, ids_2], dim=1), use_cache=False)
        torch.testing.assert_close(
            logits_2[:, -10:, :], full_logits[:, -10:, :], rtol=1e-4, atol=1e-5
        )
