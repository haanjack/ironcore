# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions and and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""
Phase 6 validation tests for KV cache integration into evaluation.

Tests:
1. Evaluation with cache enabled
2. HellaSwag benchmark simulation
3. RLHF rollout simulation
4. Memory benchmarking

Note: These tests validate the integration infrastructure. Full end-to-end
performance benchmarks require actual datasets and more complex test setups.
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
from ironcore.global_vars import global_states_cleanup, set_global_states
from ironcore.language_model import LanguageModel
from ironcore.parallel import parallel_states

# Initialize parallel states for testing (TP=1)
parallel_states.initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=10.0)


@pytest.fixture(scope="module")
def eval_config():
    """Create and initialize config for evaluation testing."""
    # Create KV cache config
    kv_cache_config = KVCacheConfig(
        enabled=True,
        max_batch_size=4,
        max_seq_length=256,
    )

    # Create positional embedding config
    pos_emb_config = PositionalEmbeddingConfig(type="rope")

    # Create model config with GQA
    model_config = ModelConfig(
        d_model=256,
        num_attention_heads=4,
        num_attention_groups=2,  # GQA
        head_dim=64,
        num_layers=2,
        d_ffn=512,
        max_seq_len=256,
        max_position_embeddings=256,
        dropout_attn=0.0,
        dropout_mlp=0.0,
        dropout_embd=0.0,
        positional_embedding=pos_emb_config,
        kv_cache=kv_cache_config,
    )
    model_config.name = "GPT"

    # Trainer config with KV cache enabled for evaluation
    trainer_config = TrainerConfig(
        tensor_model_parallel_size=1,
        use_flash_attn=False,
        use_kv_cache_in_eval=True,  # Enable cache for evaluation
    )

    init_config = InitConfig(seed=42, init_std=0.02)
    optim_config = OptimConfig(max_lr=1e-3, weight_decay=0.01)
    data_config = DataConfig()
    parallel_config = ParallelConfig()
    operation_config = OperationConfig(
        train_steps=100,
        eval_samples=10,
        activation_recompute=False,
    )
    utils_config = UtilsConfig()

    config = MainConfig(
        model=model_config,
        trainer=trainer_config,
        init=init_config,
        optim=optim_config,
        data=data_config,
        parallel=parallel_config,
        operation=operation_config,
        utils=utils_config,
        profiler=ProfilerConfig(),
        peft=PEFTConfig(),
    )

    # Initialize global states
    set_global_states(config)
    yield config
    # Cleanup after all tests
    global_states_cleanup()


@pytest.fixture
def model(eval_config):
    """Create a language model."""
    model = LanguageModel(eval_config)
    model.eval()
    return model


def test_kv_cache_config_enabled(eval_config):
    """
    Test: KV cache config enabled
    - Verify use_kv_cache_in_eval is set correctly
    - Verify KVCacheConfig is properly configured
    """
    # Check trainer config
    assert eval_config.trainer.use_kv_cache_in_eval

    # Check model config
    assert eval_config.model.kv_cache.enabled
    assert eval_config.model.kv_cache.max_batch_size >= 4
    assert eval_config.model.kv_cache.max_seq_length >= 256


def test_model_forward_with_cache(model, eval_config):
    """
    Test: Model forward pass with cache
    - Run forward pass with use_cache=True
    - Verify cache state is returned
    - Verify subsequent forward pass uses cache correctly
    """
    batch_size = 2
    seq_len = 10
    device = next(model.parameters()).device

    # Create input
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    with torch.no_grad():
        # Forward pass with cache
        logits, past_kv = model(input_ids, use_cache=True, past_key_values=None)

        # Verify cache is returned
        assert past_kv is not None
        assert len(past_kv) == eval_config.model.num_layers

        # Verify cache has correct structure
        for layer_kv in past_kv:
            key, value = layer_kv
            assert key.shape[1] == seq_len  # Cached seq_len
            assert value.shape[1] == seq_len

        # Second forward pass with cache
        new_input = torch.randint(0, 1000, (batch_size, 1), device=device)
        logits_2, past_kv_2 = model(new_input, use_cache=True, past_key_values=past_kv)

        # Verify cache is extended
        assert past_kv_2 is not None
        for layer_kv in past_kv_2:
            key, value = layer_kv
            assert key.shape[1] == seq_len + 1  # Extended cache


def test_cache_manager_initialization(eval_config):
    """
    Test: Cache manager initialization
    - Verify KVCacheManager can be created and initialized
    - Check memory allocation
    """
    from ironcore.layers.kv_cache import KVCacheManager

    # Create cache manager
    cache_manager = KVCacheManager(eval_config)
    assert not cache_manager.is_initialized

    # Initialize cache
    cache_manager.initialize(
        batch_size=2,
        num_layers=eval_config.model.num_layers,
        device=torch.device("cpu"),  # Use CPU for testing
    )

    # Verify initialization
    assert cache_manager.is_initialized
    stats = cache_manager.get_statistics()
    assert stats["initialized"]
    assert stats["memory_mb"] > 0


def test_evaluation_accuracy_preservation(model, eval_config):
    """
    Test: Evaluation accuracy preservation
    - Run evaluation with cache enabled (autoregressive)
    - Run evaluation without cache (full sequence)
    - Verify outputs are identical (within numerical tolerance)

    The test verifies that using KV cache for autoregressive generation
    produces the same results as processing the full sequence at once.

    Note: This is a simplified test - full evaluation requires actual dataset.
    """
    batch_size = 2
    seq_len = 20
    device = next(model.parameters()).device

    # Create input (use same seed for reproducibility)
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    labels = input_ids.clone()  # Simple labels for testing

    with torch.no_grad():
        # Run WITHOUT cache: process full sequence at once
        logits_no_cache = model(input_ids, labels=None, use_cache=False)
        loss_no_cache = model.compute_loss_from_logits(
            logits=logits_no_cache,
            labels=labels,
            loss_mask=torch.ones_like(labels, dtype=torch.float),
            padding_start_idx=model.padding_start_idx,
        )

        # Run WITH cache: simulate autoregressive generation
        # Process prefix first, then generate token by token
        prefix_len = 10  # Process first 10 tokens as prefix
        gen_len = seq_len - prefix_len

        # Process prefix
        prefix_ids = input_ids[:, :prefix_len]
        logits_prefix, past_kv = model(
            prefix_ids, labels=None, use_cache=True, past_key_values=None
        )

        # Generate remaining tokens one at a time (simulating autoregressive generation)
        all_logits_cached = [logits_prefix]
        current_kv = past_kv

        for i in range(gen_len):
            next_token = input_ids[:, prefix_len + i : prefix_len + i + 1]
            logits_next, current_kv = model(
                next_token, labels=None, use_cache=True, past_key_values=current_kv
            )
            all_logits_cached.append(logits_next)

        # Concatenate all logits
        logits_cached = torch.cat(all_logits_cached, dim=1)

        # Compute loss from cached logits
        loss_cached = model.compute_loss_from_logits(
            logits=logits_cached,
            labels=labels,
            loss_mask=torch.ones_like(labels, dtype=torch.float),
            padding_start_idx=model.padding_start_idx,
        )

        # Verify losses are identical (within numerical tolerance)
        # Note: There might be small numerical differences due to different computation paths
        assert torch.allclose(loss_cached, loss_no_cache, rtol=1e-4, atol=1e-5), (
            f"Cached loss {loss_cached.item():.6f} != non-cached loss {loss_no_cache.item():.6f}"
        )

        # Also verify the logits are close
        assert torch.allclose(logits_cached, logits_no_cache, rtol=1e-4, atol=1e-5), (
            "Cached logits differ from non-cached logits"
        )


def test_rlhf_rollout_simulation(eval_config):
    """
    Test: RLHF rollout simulation
    - Simulate multiple generations with shared 128-token prompt
    - Enable prefix cache (simulated)
    - Measure speedup vs baseline

    Note: This is a simplified test showing the infrastructure is in place.
    Full RLHF rollout speedup testing requires actual generation loop.
    """
    # This test validates the infrastructure for prefix caching
    # Full RLHF testing would require:
    # 1. Actual prompt (128+ tokens)
    # 2. Generation loop with multiple rollouts
    # 3. Performance measurement

    # For now, just verify that the config supports prefix cache
    assert hasattr(eval_config.model.kv_cache, "use_prefix_cache")
    assert eval_config.model.kv_cache.prefix_cache_max_pages >= 1024
    assert eval_config.model.kv_cache.min_prefix_length >= 32


def test_memory_benchmarking(eval_config):
    """
    Test: Memory benchmarking
    - Compare peak memory: no cache vs cache
    - Verify paged cache reduces fragmentation (when enabled)

    Note: This is a simplified test - full memory benchmarking requires
    actual profiling with longer sequences and variable-length batches.
    """
    # Verify that paged attention config is available
    assert hasattr(eval_config.model.kv_cache, "use_paged_attention")
    assert eval_config.model.kv_cache.page_size >= 16
    assert eval_config.model.kv_cache.max_num_pages >= 4096


def test_cache_statistics_reporting(eval_config):
    """
    Test: Cache statistics reporting
    - Verify cache manager reports correct statistics
    - Check memory usage tracking
    - Verify utilization tracking
    """
    from ironcore.layers.kv_cache import KVCacheManager

    cache_manager = KVCacheManager(eval_config)
    cache_manager.initialize(
        batch_size=2,
        num_layers=eval_config.model.num_layers,
        device=torch.device("cpu"),
    )

    # Get statistics
    stats = cache_manager.get_statistics()

    # Verify statistics fields
    assert "initialized" in stats
    assert "memory_mb" in stats
    assert "utilization" in stats
    assert "num_layers" in stats
    assert "batch_size" in stats
    assert "num_local_kv_groups" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
