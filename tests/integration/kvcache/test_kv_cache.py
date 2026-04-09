#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
"""
KV cache tests with tensor parallelism.

TP=1 tests run via torchrun --nproc_per_node=1.
TP=2 tests require torchrun --nproc_per_node=2.

Run:
    torchrun --nproc_per_node=1 -m pytest tests/integration/kvcache/test_kv_cache.py -v
    torchrun --nproc_per_node=2 -m pytest tests/integration/kvcache/test_kv_cache.py -v
"""

import os

import pytest
import torch
import torch.distributed as dist

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


def _tp_config(tp_size: int, d_model: int = 512, num_layers: int = 2):
    """Create config for the given TP size."""
    parallel_states.initialize_model_parallel(tensor_model_parallel_size=tp_size, timeout_in_minutes=10.0)

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
        tensor_model_parallel_size=tp_size,
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


# ──────────────────────────────────────────────────────────────────────
# TP=1 Tests
# ──────────────────────────────────────────────────────────────────────


class TestKVCacheTP1:
    """KV cache tests with tensor_model_parallel_size=1."""

    @pytest.fixture(scope="class")
    def config(self):
        _tp_config(tp_size=1)
        yield
        _destroy_parallel()

    @pytest.fixture
    def model(self, config):
        model = LanguageModel(config)
        model.eval()
        yield model

    def test_end_to_end_generation(self, model, config):
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
                expected_groups = config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
                assert key.shape == (1, 5, expected_groups, config.model.head_dim)
                assert value.shape == (1, 5, expected_groups, config.model.head_dim)

    def test_cached_vs_uncached_equivalence(self, model, config):
        """Generate sequence token-by-token with cache, verify against full-sequence output."""
        device = next(model.parameters()).device
        input_ids = torch.randint(0, 1000, (2, 15), device=device)

        with torch.no_grad():
            past_kv = None
            cached_logits = []
            for i in range(15):
                out = model(input_ids[:, i : i + 1], use_cache=True, past_key_values=past_kv)
                out = out[0] if isinstance(out, tuple) else out
                past_kv = out[1] if isinstance(out, tuple) else None
                cached_logits.append(out)
            full = model(input_ids, use_cache=False)
            full = full[0] if isinstance(full, tuple) else full
        torch.testing.assert_close(torch.cat(cached_logits, dim=1), full, rtol=1e-4, atol=1e-5)

    def test_cache_reuse_across_passes(self, model, config):
        """Process two sequences using cache; verify the second sequence's output matches."""
        device = next(model.parameters()).device
        ids_1 = torch.randint(0, 1000, (1, 10), device=device)
        ids_2 = torch.randint(0, 1000, (1, 10), device=device)

        with torch.no_grad():
            out_1 = model(ids_1, use_cache=True)
            kv_1 = out_1[1] if isinstance(out_1, tuple) else None
            out_2 = model(ids_2, use_cache=True, past_key_values=kv_1)
            logits_2 = out_2[0] if isinstance(out_2, tuple) else out_2
            full = model(torch.cat([ids_1, ids_2], dim=1), use_cache=False)
            full = full[0] if isinstance(full, tuple) else full
        torch.testing.assert_close(logits_2[:, -10:, :], full[:, -10:, :], rtol=1e-4, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────
# TP=2 Tests (require torchrun --nproc_per_node=2)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    "'RANK' not in os.environ or torch.cuda.device_count() < 2",
    reason="TP=2 tests require torchrun with 2 GPUs",
)
class TestKVCacheTP2:

    @pytest.fixture(scope="class")
    def config(self):
        _tp_config(tp_size=2, d_model=512, num_layers=2)
        yield
        _destroy_parallel()

    @pytest.fixture
    def model(self, config):
        model = LanguageModel(config)
        model.eval()
        yield model

    def test_cache_sharding(self, model, config):
        """Each rank allocates cache for num_groups / TP size."""
        from ironcore.layers.kv_cache import KVCacheManager

        device = next(model.parameters()).device
        cache_manager = KVCacheManager(config)
        cache_manager.initialize(
            batch_size=2, num_layers=config.model.num_layers, device=device,
        )
        expected_groups = config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        assert expected_groups == 1

        stats = cache_manager.get_statistics()
        assert stats["num_local_kv_groups"] == expected_groups
        assert stats["batch_size"] == 2
        assert stats["memory_mb"] > 0

    def test_gqa_cache_shape(self, config):
        """Verify cache dimensions for GQA with TP=2."""
        from ironcore.layers.kv_cache import KVCacheManager

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_manager = KVCacheManager(config)
        cache_manager.initialize(batch_size=1, num_layers=config.model.num_layers, device=device)

        expected_groups = config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        dummy_kv = torch.randn(1, 5, expected_groups, config.model.head_dim, device=device)
        for layer_idx in range(config.model.num_layers):
            key, value = cache_manager.update_layer(layer_idx, dummy_kv, dummy_kv, position=0)
            assert key.shape == (1, 5, expected_groups, config.model.head_dim)
            assert value.shape == (1, 5, expected_groups, config.model.head_dim)

    def test_cache_lifecycle(self, config):
        """Create, update, retrieve, and statistics of cache across all layers."""
        from ironcore.layers.kv_cache import KVCacheManager

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_manager = KVCacheManager(config)
        cache_manager.initialize(batch_size=2, num_layers=config.model.num_layers, device=device)

        expected_groups = config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        dummy_kv = torch.randn(2, 8, expected_groups, config.model.head_dim, device=device)

        # Update cache for all layers
        for layer_idx in range(config.model.num_layers):
            cache_manager.update_layer(layer_idx, dummy_kv, dummy_kv, position=0)
            cache_manager.update_cache_position(layer_idx, 8)

        # Verify position and retrieval
        assert cache_manager.get_cache_position(0) == 8
        for layer_idx in range(config.model.num_layers):
            key, value = cache_manager.get_layer_kv(layer_idx, start_pos=0, end_pos=8)
            assert key.shape == (2, 8, expected_groups, config.model.head_dim)
            assert value.shape == (2, 8, expected_groups, config.model.head_dim)

        stats = cache_manager.get_statistics()
        assert stats["num_local_kv_groups"] == expected_groups
        assert stats["batch_size"] == 2
        assert stats["memory_mb"] > 0

    def test_selective_reset(self, config):
        """Verify selective reset zeroes only specified positions."""
        from ironcore.layers.kv_cache import KVCacheManager

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_manager = KVCacheManager(config)
        cache_manager.initialize(batch_size=1, num_layers=config.model.num_layers, device=device)

        expected_groups = config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        dummy_kv = torch.randn(1, 10, expected_groups, config.model.head_dim, device=device)

        # Fill all positions
        for layer_idx in range(config.model.num_layers):
            for pos in range(10):
                cache_manager.update_layer(layer_idx, dummy_kv, dummy_kv, position=pos)

        # Reset positions 5 and 7
        cache_manager.update_cache_position(5, reset_value=0.0)
        cache_manager.update_cache_position(7, reset_value=0.0)

        # Verify only positions 5 and 7 were reset
        for layer_idx in range(config.model.num_layers):
            key, value = cache_manager.get_layer_kv(layer_idx, start_pos=0, end_pos=10)
            for pos in [5, 7]:
                assert torch.all(value[:, pos, :] == 0), (
                    f"Position {pos} was not reset in layer {layer_idx}"
                )
            # Non-reset positions should be non-zero
            for pos in [0, 1, 2, 3, 4, 6, 8, 9]:
                assert value[:, pos, :].abs().sum() > 0, (
                    f"Position {pos} was unexpectedly reset in layer {layer_idx}"
                )
