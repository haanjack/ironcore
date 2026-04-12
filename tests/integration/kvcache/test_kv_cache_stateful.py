# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for stateful KV cache path."""

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
from ironcore.global_vars import global_states_cleanup, set_global_states
from ironcore.language_model import LanguageModel
from ironcore.parallel import parallel_states


@pytest.fixture(scope="module")
def stateful_config():
    # Initialize parallel states first
    parallel_states.initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=10.0)

    kv_cache_config = KVCacheConfig(enabled=True, max_batch_size=4, max_seq_length=128)
    pos_emb_config = PositionalEmbeddingConfig(type="rope")
    model_config = ModelConfig(
        d_model=256,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=64,
        num_layers=2,
        d_ffn=512,
        max_seq_len=128,
        max_position_embeddings=128,
        dropout_attn=0.0,
        dropout_mlp=0.0,
        dropout_embd=0.0,
        positional_embedding=pos_emb_config,
        kv_cache=kv_cache_config,
    )
    model_config.name = "GPT"
    config = MainConfig(
        model=model_config,
        trainer=TrainerConfig(tensor_model_parallel_size=1, use_flash_attn=False),
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
    yield config
    global_states_cleanup()
    parallel_states.destroy_model_parallel()


@pytest.fixture
def stateful_model(stateful_config):
    model = LanguageModel(stateful_config)
    model.eval()
    return model


class TestStatefulKVCache:
    def test_stateful_vs_stateless_parity(self, stateful_model):
        batch_size, seq_len = 1, 5
        device = next(stateful_model.parameters()).device
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        with torch.no_grad():
            # Stateless: process full sequence with use_cache=True
            logits_stateless, _ = stateful_model(input_ids, use_cache=True, past_key_values=None)

            # Stateful: use KVCacheManager
            stateful_model.initialize_cache(batch_size=batch_size, device=device)
            # Process token by token using stateful cache
            past_kv = None
            all_logits = []
            for i in range(seq_len):
                token = input_ids[:, i : i + 1]
                logit, past_kv = stateful_model(token, use_cache=True, past_key_values=past_kv)
                all_logits.append(logit)
            logits_stateful = torch.cat(all_logits, dim=1)

            torch.testing.assert_close(logits_stateful, logits_stateless, rtol=1e-4, atol=1e-5)

    def test_stateful_multi_step(self, stateful_model):
        batch_size = 1
        device = next(stateful_model.parameters()).device
        tokens = torch.randint(0, 1000, (batch_size, 10), device=device)

        with torch.no_grad():
            # Process token by token with incremental cache
            past_kv = None
            all_logits = []
            for i in range(10):
                token = tokens[:, i : i + 1]
                logit, past_kv = stateful_model(token, use_cache=True, past_key_values=past_kv)
                all_logits.append(logit)

            # Compare with full sequence (without cache, returns tensor not tuple)
            full_logits, _ = stateful_model(tokens, use_cache=True, past_key_values=None)
            torch.testing.assert_close(
                torch.cat(all_logits, dim=1), full_logits, rtol=1e-4, atol=1e-5
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
