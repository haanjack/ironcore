#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
#
"""
KV cache tests with tensor_model_parallel_size=2.

Requires torchrun with 2 GPUs:
    torchrun --nproc_per_node=2 -m pytest tests/multi_gpu/kvcache/test_kv_cache_tp2.py -v
"""

import os  # noqa: F401 — used in pytest.mark.skipif expression

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


def _tp_config(tp_size: int, d_model: int = 512, num_layers: int = 2):
    """Create config for the given TP size."""
    parallel_states.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, timeout_in_minutes=10.0
    )

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


# run_distributed_tests.sh selects with `-m "mp"`; without the marker pytest
# collected nothing here and exited 5, which the runner reported as a failure.
@pytest.mark.cuda
@pytest.mark.mp
@pytest.mark.skipif(
    "'RANK' not in os.environ or torch.cuda.device_count() < 2",
    reason="TP=2 tests require torchrun with 2 GPUs",
)
class TestKVCacheTP2:
    """KV cache tests with tensor_model_parallel_size=2."""

    @pytest.fixture(scope="class")
    def config(self):
        if int(os.environ.get("WORLD_SIZE", "1")) < 2:
            pytest.skip("TP=2 requires world_size >= 2")
        cfg = _tp_config(tp_size=2, d_model=512, num_layers=2)
        yield cfg
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
            batch_size=2,
            num_layers=config.model.num_layers,
            device=device,
        )
        expected_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        )
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

        expected_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        )
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

        expected_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        )
        dummy_kv = torch.randn(2, 8, expected_groups, config.model.head_dim, device=device)

        for layer_idx in range(config.model.num_layers):
            cache_manager.update_layer(layer_idx, dummy_kv, dummy_kv, position=0)

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
        """Verify selective reset zeroes only specified sequences."""
        from ironcore.layers.kv_cache import KVCacheManager

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_manager = KVCacheManager(config)
        cache_manager.initialize(batch_size=2, num_layers=config.model.num_layers, device=device)

        expected_groups = (
            config.model.num_attention_groups // config.trainer.tensor_model_parallel_size
        )
        dummy_kv = torch.randn(2, 8, expected_groups, config.model.head_dim, device=device)

        for layer_idx in range(config.model.num_layers):
            cache_manager.update_layer(layer_idx, dummy_kv, dummy_kv, position=0)

        assert cache_manager.get_cache_position(0) == 8
        assert cache_manager.get_cache_position(1) == 8

        cache_manager.reset(batch_indices=torch.tensor([0], device=device))

        assert cache_manager.get_cache_position(0) == 0
        assert cache_manager.get_cache_position(1) == 8

        for layer_idx in range(config.model.num_layers):
            value_0 = cache_manager.get_layer_kv(layer_idx, start_pos=0, end_pos=8, batch_idx=0)[1]
            assert torch.all(value_0 == 0), f"Sequence 0 not reset in layer {layer_idx}"
            value_1 = cache_manager.get_layer_kv(layer_idx, start_pos=0, end_pos=8, batch_idx=1)[1]
            assert value_1.abs().sum() > 0, f"Sequence 1 unexpectedly reset in layer {layer_idx}"
