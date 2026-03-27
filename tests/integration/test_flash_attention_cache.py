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
Flash Attention integration tests with KV cache.

Tests:
1. Flash attention with cache concatenation
2. Flash attention numerical equivalence with standard attention
3. Flash attention with GQA and cache
4. Flash attention with varying sequence lengths

Note: Flash Attention requires fp16 or bf16 dtype. Tests will use bf16 if available,
otherwise fp16.
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
from ironcore.layers.attention import Attention, flash_attn_varlen_func
from ironcore.parallel import parallel_states

# Initialize parallel states for testing (TP=1)
parallel_states.initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=10.0)

# Check if Flash Attention is available
FLASH_ATTN_AVAILABLE = flash_attn_varlen_func is not None
CUDA_AVAILABLE = torch.cuda.is_available()

# Determine best dtype for Flash Attention (bf16 preferred, else fp16)
if CUDA_AVAILABLE and torch.cuda.is_bf16_supported():
    FLASH_DTYPE = torch.bfloat16
else:
    FLASH_DTYPE = torch.float16


def create_test_config(use_flash_attn: bool = True, num_attention_groups: int = 8) -> MainConfig:
    """Create a test configuration."""
    kv_cache_config = KVCacheConfig(
        enabled=True,
        max_batch_size=4,
        max_seq_length=256,
    )

    pos_emb_config = PositionalEmbeddingConfig(type="rope")

    model_config = ModelConfig(
        d_model=256,
        num_attention_heads=8,
        num_attention_groups=num_attention_groups,  # GQA or MHA
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

    trainer_config = TrainerConfig(
        tensor_model_parallel_size=1,
        use_flash_attn=use_flash_attn,
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

    return MainConfig(
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


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Flash Attention requires CUDA")
class TestFlashAttentionWithCache:
    """Test Flash Attention with KV cache."""

    def test_flash_attention_basic_with_cache(self):
        """Test: Basic Flash Attention with cache concatenation."""
        if not FLASH_ATTN_AVAILABLE:
            pytest.skip("flash-attn not installed")

        config = create_test_config(use_flash_attn=True)
        attention = Attention(config)
        device = torch.device("cuda")

        batch_size = 2
        num_heads = attention.num_local_attention_heads
        num_groups = attention.num_local_attention_groups
        head_dim = attention.head_dimension

        # Create past KV (cached) - use fp16/bf16 for Flash Attention
        past_len = 10
        past_key = torch.randn(
            batch_size, past_len, num_groups, head_dim, device=device, dtype=FLASH_DTYPE
        )
        past_value = torch.randn(
            batch_size, past_len, num_groups, head_dim, device=device, dtype=FLASH_DTYPE
        )

        # Create new Q, K, V - use fp16/bf16 for Flash Attention
        new_len = 5
        query = torch.randn(
            batch_size, new_len, num_heads, head_dim, device=device, dtype=FLASH_DTYPE
        )
        key = torch.randn(
            batch_size, new_len, num_groups, head_dim, device=device, dtype=FLASH_DTYPE
        )
        value = torch.randn(
            batch_size, new_len, num_groups, head_dim, device=device, dtype=FLASH_DTYPE
        )

        # Run attention with cache
        output, (cached_key, cached_value) = attention(
            query, key, value, attention_mask=None, use_cache=True, past_kv=(past_key, past_value)
        )

        # Verify output shape
        assert output.shape == (batch_size, new_len, num_heads * head_dim)
        assert output.dtype == FLASH_DTYPE

        # Verify KV was concatenated correctly
        assert cached_key.shape == (batch_size, past_len + new_len, num_groups, head_dim)
        assert cached_value.shape == (batch_size, past_len + new_len, num_groups, head_dim)

    def test_flash_vs_standard_attention_equivalence(self):
        """Test: Flash Attention produces equivalent results to standard attention."""
        if not FLASH_ATTN_AVAILABLE:
            pytest.skip("flash-attn not installed")

        device = torch.device("cuda")

        # Create configs for both standard and flash attention
        config_standard = create_test_config(use_flash_attn=False)
        config_flash = create_test_config(use_flash_attn=True)

        attention_standard = Attention(config_standard)
        attention_flash = Attention(config_flash)

        batch_size = 2
        num_heads = attention_standard.num_local_attention_heads
        num_groups = attention_standard.num_local_attention_groups
        head_dim = attention_standard.head_dimension

        # Use same random data - create in fp32 first then convert
        torch.manual_seed(42)
        past_len = 8
        past_key_fp32 = torch.randn(batch_size, past_len, num_groups, head_dim, device=device)
        past_value_fp32 = torch.randn(batch_size, past_len, num_groups, head_dim, device=device)

        new_len = 4
        query_fp32 = torch.randn(batch_size, new_len, num_heads, head_dim, device=device)
        key_fp32 = torch.randn(batch_size, new_len, num_groups, head_dim, device=device)
        value_fp32 = torch.randn(batch_size, new_len, num_groups, head_dim, device=device)

        # Create causal mask for standard attention
        total_len = past_len + new_len
        attention_mask = torch.tril(
            torch.ones(batch_size, total_len, total_len, device=device)
        ).unsqueeze(1)
        attention_mask = attention_mask[:, :, -new_len:, :]

        # Run standard attention with cache (fp32)
        output_standard, (cached_key_std, cached_value_std) = attention_standard(
            query_fp32,
            key_fp32,
            value_fp32,
            attention_mask=attention_mask,
            use_cache=True,
            past_kv=(past_key_fp32, past_value_fp32),
        )

        # Convert to fp16/bf16 for flash attention
        past_key = past_key_fp32.to(FLASH_DTYPE)
        past_value = past_value_fp32.to(FLASH_DTYPE)
        query = query_fp32.to(FLASH_DTYPE)
        key = key_fp32.to(FLASH_DTYPE)
        value = value_fp32.to(FLASH_DTYPE)

        # Run flash attention with cache (causal=True handles masking)
        output_flash, (cached_key_flash, cached_value_flash) = attention_flash(
            query,
            key,
            value,
            attention_mask=None,
            use_cache=True,
            past_kv=(past_key, past_value),
        )

        # Compare outputs (Flash Attention may have small numerical differences)
        # Convert flash output back to fp32 for comparison
        output_flash_fp32 = output_flash.float()
        torch.testing.assert_close(output_standard, output_flash_fp32, rtol=1e-2, atol=1e-2)

        # Verify cached KV has same values (convert back to fp32)
        torch.testing.assert_close(cached_key_std, cached_key_flash.float(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(
            cached_value_std, cached_value_flash.float(), rtol=1e-2, atol=1e-2
        )

    def test_flash_attention_gqa_with_cache(self):
        """Test: Flash Attention with GQA and cache."""
        if not FLASH_ATTN_AVAILABLE:
            pytest.skip("flash-attn not installed")

        # GQA: 8 query heads, 2 KV groups
        config = create_test_config(use_flash_attn=True, num_attention_groups=2)
        attention = Attention(config)
        device = torch.device("cuda")

        batch_size = 1
        num_heads = attention.num_local_attention_heads  # 8
        num_groups = attention.num_local_attention_groups  # 2
        head_dim = attention.head_dimension

        # Cached KV - use fp16/bf16
        past_len = 16
        past_key = torch.randn(
            batch_size, past_len, num_groups, head_dim, device=device, dtype=FLASH_DTYPE
        )
        past_value = torch.randn(
            batch_size, past_len, num_groups, head_dim, device=device, dtype=FLASH_DTYPE
        )

        # New Q, K, V - use fp16/bf16
        new_len = 8
        query = torch.randn(
            batch_size, new_len, num_heads, head_dim, device=device, dtype=FLASH_DTYPE
        )
        key = torch.randn(
            batch_size, new_len, num_groups, head_dim, device=device, dtype=FLASH_DTYPE
        )
        value = torch.randn(
            batch_size, new_len, num_groups, head_dim, device=device, dtype=FLASH_DTYPE
        )

        # Run attention with cache
        output, (cached_key, cached_value) = attention(
            query, key, value, attention_mask=None, use_cache=True, past_kv=(past_key, past_value)
        )

        # Verify shapes
        assert output.shape == (batch_size, new_len, num_heads * head_dim)
        assert output.dtype == FLASH_DTYPE
        assert cached_key.shape == (batch_size, past_len + new_len, num_groups, head_dim)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Flash Attention requires CUDA")
class TestFlashAttentionEdgeCases:
    """Test Flash Attention edge cases with cache."""

    def test_flash_attention_empty_cache(self):
        """Test: Flash Attention with empty cache (past_kv=None)."""
        if not FLASH_ATTN_AVAILABLE:
            pytest.skip("flash-attn not installed")

        config = create_test_config(use_flash_attn=True)
        attention = Attention(config)
        device = torch.device("cuda")

        batch_size = 2
        num_heads = attention.num_local_attention_heads
        num_groups = attention.num_local_attention_groups
        head_dim = attention.head_dimension

        seq_len = 10
        query = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=FLASH_DTYPE
        )
        key = torch.randn(
            batch_size, seq_len, num_groups, head_dim, device=device, dtype=FLASH_DTYPE
        )
        value = torch.randn(
            batch_size, seq_len, num_groups, head_dim, device=device, dtype=FLASH_DTYPE
        )

        # Run without cache
        output, (cached_key, cached_value) = attention(
            query, key, value, attention_mask=None, use_cache=True, past_kv=None
        )

        # Verify shapes
        assert output.shape == (batch_size, seq_len, num_heads * head_dim)
        assert output.dtype == FLASH_DTYPE
        assert cached_key.shape == (batch_size, seq_len, num_groups, head_dim)

    def test_flash_attention_single_token_generation(self):
        """Test: Flash Attention for single token generation (typical decoding)."""
        if not FLASH_ATTN_AVAILABLE:
            pytest.skip("flash-attn not installed")

        config = create_test_config(use_flash_attn=True)
        attention = Attention(config)
        device = torch.device("cuda")

        batch_size = 1
        num_heads = attention.num_local_attention_heads
        num_groups = attention.num_local_attention_groups
        head_dim = attention.head_dimension

        # Simulate decoding: large cache, single new token
        past_len = 100
        past_key = torch.randn(
            batch_size, past_len, num_groups, head_dim, device=device, dtype=FLASH_DTYPE
        )
        past_value = torch.randn(
            batch_size, past_len, num_groups, head_dim, device=device, dtype=FLASH_DTYPE
        )

        new_len = 1  # Single token
        query = torch.randn(
            batch_size, new_len, num_heads, head_dim, device=device, dtype=FLASH_DTYPE
        )
        key = torch.randn(
            batch_size, new_len, num_groups, head_dim, device=device, dtype=FLASH_DTYPE
        )
        value = torch.randn(
            batch_size, new_len, num_groups, head_dim, device=device, dtype=FLASH_DTYPE
        )

        # Run attention
        output, (cached_key, cached_value) = attention(
            query, key, value, attention_mask=None, use_cache=True, past_kv=(past_key, past_value)
        )

        # Verify
        assert output.shape == (batch_size, new_len, num_heads * head_dim)
        assert output.dtype == FLASH_DTYPE
        assert cached_key.shape == (batch_size, past_len + new_len, num_groups, head_dim)

    def test_flash_attention_multi_turn_generation(self):
        """Test: Multi-turn generation with incremental cache updates."""
        if not FLASH_ATTN_AVAILABLE:
            pytest.skip("flash-attn not installed")

        config = create_test_config(use_flash_attn=True)
        attention = Attention(config)
        device = torch.device("cuda")

        batch_size = 1
        num_heads = attention.num_local_attention_heads
        num_groups = attention.num_local_attention_groups
        head_dim = attention.head_dimension

        # Initial prompt
        prompt_len = 20
        query = torch.randn(
            batch_size, prompt_len, num_heads, head_dim, device=device, dtype=FLASH_DTYPE
        )
        key = torch.randn(
            batch_size, prompt_len, num_groups, head_dim, device=device, dtype=FLASH_DTYPE
        )
        value = torch.randn(
            batch_size, prompt_len, num_groups, head_dim, device=device, dtype=FLASH_DTYPE
        )

        # First forward: process prompt
        _, (cached_key, cached_value) = attention(
            query, key, value, attention_mask=None, use_cache=True, past_kv=None
        )

        current_len = prompt_len

        # Simulate multiple decoding steps
        num_decode_steps = 5
        for _ in range(num_decode_steps):
            # Single token query
            query = torch.randn(
                batch_size, 1, num_heads, head_dim, device=device, dtype=FLASH_DTYPE
            )
            key = torch.randn(batch_size, 1, num_groups, head_dim, device=device, dtype=FLASH_DTYPE)
            value = torch.randn(
                batch_size, 1, num_groups, head_dim, device=device, dtype=FLASH_DTYPE
            )

            _, (cached_key, cached_value) = attention(
                query,
                key,
                value,
                attention_mask=None,
                use_cache=True,
                past_kv=(cached_key, cached_value),
            )

            current_len += 1

        # Verify final cache length
        assert cached_key.shape[1] == prompt_len + num_decode_steps


class TestAttentionWithoutFlashAttn:
    """Test attention fallback when Flash Attention is not available."""

    def test_standard_attention_with_cache(self):
        """Test: Standard attention with cache works correctly."""
        config = create_test_config(use_flash_attn=False)
        attention = Attention(config)
        device = torch.device("cpu")

        batch_size = 2
        num_heads = attention.num_local_attention_heads
        num_groups = attention.num_local_attention_groups
        head_dim = attention.head_dimension

        # Cached KV
        past_len = 10
        past_key = torch.randn(batch_size, past_len, num_groups, head_dim, device=device)
        past_value = torch.randn(batch_size, past_len, num_groups, head_dim, device=device)

        # New Q, K, V
        new_len = 5
        query = torch.randn(batch_size, new_len, num_heads, head_dim, device=device)
        key = torch.randn(batch_size, new_len, num_groups, head_dim, device=device)
        value = torch.randn(batch_size, new_len, num_groups, head_dim, device=device)

        # Create causal mask
        total_len = past_len + new_len
        attention_mask = torch.tril(
            torch.ones(batch_size, total_len, total_len, device=device)
        ).unsqueeze(1)
        attention_mask = attention_mask[:, :, -new_len:, :]

        # Run attention with cache
        output, (cached_key, cached_value) = attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            use_cache=True,
            past_kv=(past_key, past_value),
        )

        # Verify shapes
        assert output.shape == (batch_size, new_len, num_heads * head_dim)
        assert cached_key.shape == (batch_size, past_len + new_len, num_groups, head_dim)

    def test_gqa_with_cache(self):
        """Test: GQA with cache works correctly with standard attention."""
        config = create_test_config(use_flash_attn=False, num_attention_groups=2)
        attention = Attention(config)
        device = torch.device("cpu")

        batch_size = 1
        num_heads = attention.num_local_attention_heads  # 8
        num_groups = attention.num_local_attention_groups  # 2
        head_dim = attention.head_dimension

        # Cached KV (2 groups, not 8)
        past_len = 16
        past_key = torch.randn(batch_size, past_len, num_groups, head_dim, device=device)
        past_value = torch.randn(batch_size, past_len, num_groups, head_dim, device=device)

        # New Q (8 heads), K, V (2 groups)
        new_len = 4
        query = torch.randn(batch_size, new_len, num_heads, head_dim, device=device)
        key = torch.randn(batch_size, new_len, num_groups, head_dim, device=device)
        value = torch.randn(batch_size, new_len, num_groups, head_dim, device=device)

        # Create causal mask
        total_len = past_len + new_len
        attention_mask = torch.tril(
            torch.ones(batch_size, total_len, total_len, device=device)
        ).unsqueeze(1)
        attention_mask = attention_mask[:, :, -new_len:, :]

        # Run attention
        output, (cached_key, cached_value) = attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            use_cache=True,
            past_kv=(past_key, past_value),
        )

        # Verify GQA: output has 8 heads worth, but cache stores only 2 groups
        assert output.shape == (batch_size, new_len, num_heads * head_dim)
        assert cached_key.shape == (batch_size, past_len + new_len, num_groups, head_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
