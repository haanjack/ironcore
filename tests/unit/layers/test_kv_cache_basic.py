# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

"""
Phase 1 validation tests for basic KV cache functionality.

Tests:
1. Single token caching - Process token 0, cache K/V, then process token 1 with cache
2. Multi-token sequence - Process tokens 0-9 one by one with cache
3. Batch independence - Multiple sequences in batch don't contaminate each other
4. Cache reset - Verify cache resets correctly
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
def global_config():
    """Create and initialize global states for the entire test module."""
    # Create KV cache config
    kv_cache_config = KVCacheConfig(
        enabled=True,
        max_batch_size=4,
        max_seq_length=128,
    )

    # Create positional embedding config
    pos_emb_config = PositionalEmbeddingConfig(type="rope")

    # Create model config
    model_config = ModelConfig(
        d_model=256,
        num_attention_heads=4,
        num_attention_groups=2,  # GQA
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
    # Set name attribute dynamically (required for model provider)
    model_config.name = "GPT"  # Use TransformerModel which supports KV cache

    trainer_config = TrainerConfig(
        tensor_model_parallel_size=1,
        use_flash_attn=False,  # Use standard attention for testing
    )

    init_config = InitConfig(seed=42, init_std=0.02)
    optim_config = OptimConfig(max_lr=1e-3, weight_decay=0.01)
    data_config = DataConfig()
    parallel_config = ParallelConfig()
    operation_config = OperationConfig(
        train_steps=100,
        activation_recompute=False,
    )
    utils_config = UtilsConfig()
    profiler_config = ProfilerConfig()

    config = MainConfig(
        model=model_config,
        trainer=trainer_config,
        init=init_config,
        optim=optim_config,
        data=data_config,
        parallel=parallel_config,
        operation=operation_config,
        utils=utils_config,
        profiler=profiler_config,
    )

    # Initialize global states
    set_global_states(config)
    yield config
    # Cleanup after all tests
    global_states_cleanup()


@pytest.fixture
def small_config(global_config):
    """Get the global config for each test."""
    return global_config


@pytest.fixture
def model(small_config):
    """Create a language model."""
    # Global states are already initialized by the global_config fixture
    model = LanguageModel(small_config)
    model.eval()  # Set to eval mode to disable dropout
    return model


def test_single_token_caching(model, small_config):
    """
    Test: Single token caching
    - Process token 0, cache K/V
    - Process token 1 with cache, verify output matches full sequence [0,1]
    """
    batch_size = 1
    device = next(model.parameters()).device

    # Create input: single token
    token_0 = torch.randint(0, 1000, (batch_size, 1), device=device)
    token_1 = torch.randint(0, 1000, (batch_size, 1), device=device)

    with torch.no_grad():
        # Step 1: Process token 0 with cache
        logits_0, past_kv = model(token_0, use_cache=True, past_key_values=None)
        assert logits_0.shape == (batch_size, 1, model.embedding.word_embeddings.weight.shape[0])
        assert len(past_kv) == small_config.model.num_layers

        # Check past_kv structure
        for layer_kv in past_kv:
            key, value = layer_kv
            assert key.shape[1] == 1  # Cached 1 token
            assert value.shape[1] == 1

        # Step 2: Process token 1 with cache
        logits_1, past_kv_2 = model(token_1, use_cache=True, past_key_values=past_kv)
        assert logits_1.shape == (batch_size, 1, model.embedding.word_embeddings.weight.shape[0])

        # Check updated past_kv
        for layer_kv in past_kv_2:
            key, value = layer_kv
            assert key.shape[1] == 2  # Now cached 2 tokens
            assert value.shape[1] == 2

        # Step 3: Compare with non-cached full sequence
        input_full = torch.cat([token_0, token_1], dim=1)
        logits_full = model(input_full, use_cache=False)

        # The last logit from cached should match the last logit from full
        # (both predict token 2 given [token_0, token_1])
        torch.testing.assert_close(
            logits_1[:, -1, :],
            logits_full[:, -1, :],
            rtol=1e-4,
            atol=1e-5,
        )


def test_multi_token_sequence(model, small_config):
    """
    Test: Multi-token sequence
    - Process tokens 0-9 one by one with cache
    - Compare final output with single forward pass on [0-9]
    """
    batch_size = 1
    seq_len = 10
    device = next(model.parameters()).device

    # Create full sequence
    full_input = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    with torch.no_grad():
        # Process one token at a time with cache
        past_kv = None
        cached_logits = []

        for i in range(seq_len):
            token = full_input[:, i : i + 1]
            logits, past_kv = model(token, use_cache=True, past_key_values=past_kv)
            cached_logits.append(logits)

        cached_logits_concat = torch.cat(cached_logits, dim=1)

        # Process full sequence without cache
        full_logits = model(full_input, use_cache=False)

        # Compare outputs - they should be identical (within numerical tolerance)
        torch.testing.assert_close(
            cached_logits_concat,
            full_logits,
            rtol=1e-4,
            atol=1e-5,
        )


def test_batch_independence(model, small_config):
    """
    Test: Batch independence
    - Batch size 4, different sequences per batch slot
    - Verify no cross-contamination in cache
    """
    batch_size = 4
    seq_len = 5
    device = next(model.parameters()).device

    # Create different sequences for each batch slot
    full_input = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    with torch.no_grad():
        # Process with cache (one token at a time)
        past_kv = None
        for i in range(seq_len):
            token = full_input[:, i : i + 1]
            _, past_kv = model(token, use_cache=True, past_key_values=past_kv)

        # Final forward pass with cache to get logits
        final_token = torch.randint(0, 1000, (batch_size, 1), device=device)
        logits_cached, _ = model(final_token, use_cache=True, past_key_values=past_kv)

        # Process full sequence + final token without cache
        full_input_with_final = torch.cat([full_input, final_token], dim=1)
        logits_full = model(full_input_with_final, use_cache=False)

        # Compare last logits for each batch element
        torch.testing.assert_close(
            logits_cached[:, -1, :],
            logits_full[:, -1, :],
            rtol=1e-4,
            atol=1e-5,
        )

        # Verify each batch element's output is different (they had different inputs)
        for i in range(batch_size - 1):
            assert not torch.allclose(logits_cached[i], logits_cached[i + 1], rtol=1e-3), (
                f"Batch elements {i} and {i + 1} should have different outputs"
            )


def test_cache_reset(model, small_config):
    """
    Test: Cache reset
    - Fill cache, reset, verify positions back to 0
    - New sequence produces correct output
    """
    from ironcore.layers.kv_cache import KVCacheManager

    batch_size = 2
    seq_len = 5
    device = next(model.parameters()).device

    # Create cache manager
    cache_manager = KVCacheManager(small_config)
    cache_manager.initialize(
        batch_size=batch_size,
        num_layers=small_config.model.num_layers,
        device=device,
    )

    # Fill cache with some data
    # Calculate local KV groups (sharded across TP ranks)
    num_local_kv_groups = (
        small_config.model.num_attention_groups // small_config.trainer.tensor_model_parallel_size
    )
    dummy_kv = torch.randn(
        batch_size,
        seq_len,
        num_local_kv_groups,
        small_config.model.head_dim,
        device=device,
    )

    # Update each layer with explicit position to avoid position increment
    for layer_idx in range(small_config.model.num_layers):
        cache_manager.update_layer(layer_idx, dummy_kv, dummy_kv, position=0)

    # Verify cache is filled (position should be seq_len since we set position explicitly)
    assert cache_manager.get_cache_position(0) == seq_len

    # Reset cache
    cache_manager.reset()

    # Verify positions are back to 0
    assert cache_manager.get_cache_position(0) == 0
    assert cache_manager.get_cache_position(1) == 0

    # Verify cache contents are zeroed
    for layer_idx in range(small_config.model.num_layers):
        key, value = cache_manager.get_layer_kv(layer_idx, start_pos=0, end_pos=seq_len)
        assert torch.all(key == 0)
        assert torch.all(value == 0)


def test_cache_statistics(model, small_config):
    """
    Test: Cache statistics
    - Verify statistics are correctly reported
    """
    from ironcore.layers.kv_cache import KVCacheManager

    batch_size = 2
    device = next(model.parameters()).device

    cache_manager = KVCacheManager(small_config)

    # Before initialization
    stats = cache_manager.get_statistics()
    assert not stats["initialized"]
    assert stats["memory_mb"] == 0

    # After initialization
    cache_manager.initialize(
        batch_size=batch_size,
        num_layers=small_config.model.num_layers,
        device=device,
    )

    stats = cache_manager.get_statistics()
    assert stats["initialized"]
    assert stats["num_layers"] == small_config.model.num_layers
    assert stats["batch_size"] == batch_size
    assert stats["memory_mb"] > 0
    assert stats["utilization"] == 0.0  # No tokens cached yet

    # After caching some tokens
    seq_len = 10
    num_local_kv_groups = (
        small_config.model.num_attention_groups // small_config.trainer.tensor_model_parallel_size
    )
    dummy_kv = torch.randn(
        batch_size,
        seq_len,
        num_local_kv_groups,
        small_config.model.head_dim,
        device=device,
    )

    # Update each layer with explicit position to avoid position increment
    for layer_idx in range(small_config.model.num_layers):
        cache_manager.update_layer(layer_idx, dummy_kv, dummy_kv, position=0)

    stats = cache_manager.get_statistics()
    expected_utilization = seq_len / small_config.model.kv_cache.max_seq_length
    assert abs(stats["utilization"] - expected_utilization) < 1e-6


def test_per_sequence_positions(small_config):
    """
    Test: Per-sequence position tracking
    - Different sequences at different positions
    - Update with positions tensor
    - Retrieve per-sequence KV
    """
    from ironcore.layers.kv_cache import KVCacheManager

    batch_size = 3
    seq_len = 5
    device = torch.device("cpu")

    cache_manager = KVCacheManager(small_config)
    cache_manager.initialize(
        batch_size=batch_size,
        num_layers=small_config.model.num_layers,
        device=device,
    )

    num_local_kv_groups = (
        small_config.model.num_attention_groups // small_config.trainer.tensor_model_parallel_size
    )

    # Initially all positions should be 0
    assert cache_manager.get_sequence_position(0) == 0
    assert cache_manager.get_sequence_position(1) == 0
    assert cache_manager.get_sequence_position(2) == 0

    # Set different positions for each sequence
    cache_manager.set_sequence_position(0, 10)
    cache_manager.set_sequence_position(1, 20)
    cache_manager.set_sequence_position(2, 5)

    assert cache_manager.get_sequence_position(0) == 10
    assert cache_manager.get_sequence_position(1) == 20
    assert cache_manager.get_sequence_position(2) == 5

    # Test per-sequence position update using positions tensor
    positions = torch.tensor([0, 10, 5], dtype=torch.long, device=device)
    dummy_kv = torch.randn(
        batch_size,
        seq_len,
        num_local_kv_groups,
        small_config.model.head_dim,
        device=device,
    )

    # Update with per-sequence positions
    cache_manager.update_layer(0, dummy_kv, dummy_kv, positions=positions)

    # Verify positions were updated correctly
    assert cache_manager.get_sequence_position(0) == 5  # 0 + seq_len
    assert cache_manager.get_sequence_position(1) == 15  # 10 + seq_len
    assert cache_manager.get_sequence_position(2) == 10  # 5 + seq_len


def test_position_parameter_validation(small_config):
    """
    Test: Validation of position parameters
    - Both position and positions should raise error
    """
    from ironcore.layers.kv_cache import KVCacheManager

    batch_size = 2
    device = torch.device("cpu")

    cache_manager = KVCacheManager(small_config)
    cache_manager.initialize(
        batch_size=batch_size,
        num_layers=small_config.model.num_layers,
        device=device,
    )

    num_local_kv_groups = (
        small_config.model.num_attention_groups // small_config.trainer.tensor_model_parallel_size
    )

    dummy_kv = torch.randn(
        batch_size,
        1,
        num_local_kv_groups,
        small_config.model.head_dim,
        device=device,
    )

    positions = torch.tensor([0, 5], dtype=torch.long, device=device)

    # Should raise error when both position and positions are specified
    with pytest.raises(ValueError, match="Cannot specify both"):
        cache_manager.update_layer(0, dummy_kv, dummy_kv, position=0, positions=positions)


def test_selective_cache_reset(small_config):
    """
    Test: Selective cache reset for specific sequences
    - Reset only specific batch indices
    - Verify other sequences unaffected
    """
    from ironcore.layers.kv_cache import KVCacheManager

    batch_size = 3
    seq_len = 5
    device = torch.device("cpu")

    cache_manager = KVCacheManager(small_config)
    cache_manager.initialize(
        batch_size=batch_size,
        num_layers=small_config.model.num_layers,
        device=device,
    )

    num_local_kv_groups = (
        small_config.model.num_attention_groups // small_config.trainer.tensor_model_parallel_size
    )

    # Fill cache for all sequences
    dummy_kv = torch.randn(
        batch_size,
        seq_len,
        num_local_kv_groups,
        small_config.model.head_dim,
        device=device,
    )
    cache_manager.update_layer(0, dummy_kv, dummy_kv, position=0)

    # Verify all positions are at seq_len
    assert cache_manager.get_sequence_position(0) == seq_len
    assert cache_manager.get_sequence_position(1) == seq_len
    assert cache_manager.get_sequence_position(2) == seq_len

    # Reset only sequence 0 and 2
    cache_manager.reset(batch_indices=torch.tensor([0, 2]))

    # Verify selective reset
    assert cache_manager.get_sequence_position(0) == 0
    assert cache_manager.get_sequence_position(1) == seq_len  # Unaffected
    assert cache_manager.get_sequence_position(2) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
