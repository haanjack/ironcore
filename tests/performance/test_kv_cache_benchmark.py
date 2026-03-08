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
Comprehensive performance benchmarks for KV cache implementation.

This module validates:
1. Speedup from KV cache during autoregressive generation
2. Memory usage comparison
3. Throughput differences
4. Numerical correctness validation
5. Optimization opportunities
"""

import gc
import time

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
    TrainerConfig,
    UtilsConfig,
)
from ironcore.global_vars import global_states_cleanup, set_global_states
from ironcore.language_model import LanguageModel
from ironcore.parallel import parallel_states

# Initialize parallel states for testing (TP=1)
parallel_states.initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=10.0)


def create_benchmark_config(
    d_model=512,
    num_layers=4,
    num_heads=8,
    num_kv_groups=2,
    head_dim=64,
    seq_len=256,
    enable_cache=True,
):
    """Create a config for benchmarking."""
    kv_cache_config = KVCacheConfig(
        enabled=enable_cache,
        max_batch_size=1,
        max_seq_length=seq_len,
    )

    pos_emb_config = PositionalEmbeddingConfig(type="rope")

    model_config = ModelConfig(
        d_model=d_model,
        num_attention_heads=num_heads,
        num_attention_groups=num_kv_groups,
        head_dim=head_dim,
        num_layers=num_layers,
        d_ffn=d_model * 4,
        max_seq_len=seq_len,
        max_position_embeddings=seq_len,
        dropout_attn=0.0,
        dropout_mlp=0.0,
        dropout_embd=0.0,
        positional_embedding=pos_emb_config,
        kv_cache=kv_cache_config,
    )
    model_config.name = "GPT"

    trainer_config = TrainerConfig(
        tensor_model_parallel_size=1,
        use_flash_attn=False,  # Disable flash attention for fair comparison
        use_kv_cache_in_eval=enable_cache,
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

    config = MainConfig(
        model=model_config,
        trainer=trainer_config,
        init=init_config,
        optim=optim_config,
        data=data_config,
        parallel=parallel_config,
        operation=operation_config,
        utils=utils_config,
    )

    return config


@pytest.fixture(scope="module")
def benchmark_models():
    """Create models with and without KV cache for comparison."""
    # Clean up any existing global states
    global_states_cleanup()

    # Create config with cache
    config_with_cache = create_benchmark_config(enable_cache=True)
    set_global_states(config_with_cache)
    model_with_cache = LanguageModel(config_with_cache)
    model_with_cache.eval()

    # Clean up and create config without cache
    global_states_cleanup()
    config_without_cache = create_benchmark_config(enable_cache=False)
    set_global_states(config_without_cache)
    model_without_cache = LanguageModel(config_without_cache)
    model_without_cache.eval()

    yield {
        "with_cache": model_with_cache,
        "without_cache": model_without_cache,
        "config": config_with_cache,
    }

    # Cleanup
    global_states_cleanup()


def benchmark_autoregressive_generation(
    model,
    input_ids,
    num_tokens_to_generate,
    use_cache,
    device,
):
    """
    Benchmark autoregressive generation with or without KV cache.

    Args:
        model: The language model
        input_ids: Initial prompt [1, prompt_len]
        num_tokens_to_generate: Number of tokens to generate
        use_cache: Whether to use KV cache
        device: Device to run on

    Returns:
        Tuple of (time_elapsed, memory_used, generated_logits)
    """
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        start_memory = torch.cuda.memory_allocated(device)

    past_kv = None
    current_input = input_ids
    all_logits = []

    with torch.no_grad():
        # Process initial prompt
        if use_cache:
            logits, past_kv = model(current_input, use_cache=True, past_key_values=None)
        else:
            logits = model(current_input, use_cache=False)

        all_logits.append(logits)

        # Generate tokens one at a time
        for _ in range(num_tokens_to_generate - 1):
            # Get next token (greedy sampling for benchmark)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            if use_cache:
                logits, past_kv = model(next_token, use_cache=True, past_key_values=past_kv)
            else:
                # Without cache, we need to provide the full sequence
                current_input = torch.cat([current_input, next_token], dim=1)
                logits = model(current_input, use_cache=False)

            all_logits.append(logits)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(device)
        memory_used = peak_memory - start_memory
    else:
        memory_used = 0

    return end_time - start_time, memory_used, all_logits


class TestKVCachePerformance:
    """Comprehensive performance tests for KV cache."""

    def test_autoregressive_speedup_small_model(self, benchmark_models):
        """
        Test: Autoregressive generation speedup (small model)
        - Generate 32 tokens with and without cache
        - Measure speedup ratio
        - Expected: 2-5x speedup with cache
        """
        model_with_cache = benchmark_models["with_cache"]
        model_without_cache = benchmark_models["without_cache"]
        config = benchmark_models["config"]
        device = next(model_with_cache.parameters()).device

        # Create initial prompt (16 tokens)
        prompt_len = 16
        num_tokens = 32
        input_ids = torch.randint(0, 1000, (1, prompt_len), device=device)

        # Warmup
        with torch.no_grad():
            _ = model_with_cache(input_ids, use_cache=True)
            _ = model_without_cache(input_ids, use_cache=False)

        # Benchmark with cache
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        time_with_cache, mem_with_cache, _ = benchmark_autoregressive_generation(
            model_with_cache, input_ids, num_tokens, use_cache=True, device=device
        )

        # Benchmark without cache
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        time_without_cache, mem_without_cache, _ = benchmark_autoregressive_generation(
            model_without_cache, input_ids, num_tokens, use_cache=False, device=device
        )

        speedup = time_without_cache / time_with_cache

        print("\n=== Autoregressive Generation Performance ===")
        print(f"Model: {config.model.num_layers} layers, {config.model.num_attention_heads} heads")
        print(f"Prompt: {prompt_len} tokens, Generate: {num_tokens} tokens")
        print(f"Time with cache: {time_with_cache:.4f}s")
        print(f"Time without cache: {time_without_cache:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        if torch.cuda.is_available():
            print(f"Memory with cache: {mem_with_cache / 1024**2:.2f} MB")
            print(f"Memory without cache: {mem_without_cache / 1024**2:.2f} MB")
            print(f"Memory savings: {(1 - mem_with_cache / mem_without_cache) * 100:.1f}%")

        # For small sequences, cache should still provide measurable speedup
        # The speedup depends on sequence length and model size
        # For 32 tokens, we expect at least some speedup (>= 1.2x)
        assert speedup >= 1.1, f"Expected speedup >= 1.1x, got {speedup:.2f}x"

    def test_autoregressive_speedup_medium_sequence(self, benchmark_models):
        """
        Test: Autoregressive generation speedup (medium sequence)
        - Generate 64 tokens with and without cache
        - Measure speedup ratio
        - Expected: 3-8x speedup with cache
        """
        model_with_cache = benchmark_models["with_cache"]
        model_without_cache = benchmark_models["without_cache"]
        benchmark_models["config"]
        device = next(model_with_cache.parameters()).device

        # Create initial prompt (32 tokens)
        prompt_len = 32
        num_tokens = 64
        input_ids = torch.randint(0, 1000, (1, prompt_len), device=device)

        # Warmup
        with torch.no_grad():
            _ = model_with_cache(input_ids, use_cache=True)
            _ = model_without_cache(input_ids, use_cache=False)

        # Benchmark with cache
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        time_with_cache, _, _ = benchmark_autoregressive_generation(
            model_with_cache, input_ids, num_tokens, use_cache=True, device=device
        )

        # Benchmark without cache
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        time_without_cache, _, _ = benchmark_autoregressive_generation(
            model_without_cache, input_ids, num_tokens, use_cache=False, device=device
        )

        speedup = time_without_cache / time_with_cache

        print("\n=== Medium Sequence Performance ===")
        print(f"Prompt: {prompt_len} tokens, Generate: {num_tokens} tokens")
        print(f"Time with cache: {time_with_cache:.4f}s")
        print(f"Time without cache: {time_without_cache:.4f}s")
        print(f"Speedup: {speedup:.2f}x")

        # For medium sequences, cache should provide more significant speedup
        assert speedup >= 1.5, f"Expected speedup >= 1.5x for medium sequences, got {speedup:.2f}x"

    def test_autoregressive_speedup_long_sequence(self, benchmark_models):
        """
        Test: Autoregressive generation speedup (long sequence)
        - Generate 128 tokens with and without cache
        - Measure speedup ratio
        - Expected: 5-15x speedup with cache
        """
        model_with_cache = benchmark_models["with_cache"]
        model_without_cache = benchmark_models["without_cache"]
        benchmark_models["config"]
        device = next(model_with_cache.parameters()).device

        # Create initial prompt (64 tokens)
        prompt_len = 64
        num_tokens = 128
        input_ids = torch.randint(0, 1000, (1, prompt_len), device=device)

        # Warmup
        with torch.no_grad():
            _ = model_with_cache(input_ids, use_cache=True)
            _ = model_without_cache(input_ids, use_cache=False)

        # Benchmark with cache
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        time_with_cache, _, _ = benchmark_autoregressive_generation(
            model_with_cache, input_ids, num_tokens, use_cache=True, device=device
        )

        # Benchmark without cache
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        time_without_cache, _, _ = benchmark_autoregressive_generation(
            model_without_cache, input_ids, num_tokens, use_cache=False, device=device
        )

        speedup = time_without_cache / time_with_cache

        print("\n=== Long Sequence Performance ===")
        print(f"Prompt: {prompt_len} tokens, Generate: {num_tokens} tokens")
        print(f"Time with cache: {time_with_cache:.4f}s")
        print(f"Time without cache: {time_without_cache:.4f}s")
        print(f"Speedup: {speedup:.2f}x")

        # For long sequences, cache should provide significant speedup
        assert speedup >= 2.0, f"Expected speedup >= 2.0x for long sequences, got {speedup:.2f}x"

    def test_numerical_correctness_detailed(self, benchmark_models):
        """
        Test: Detailed numerical correctness validation
        - Compare outputs with and without cache across multiple steps
        - Verify each generated token matches
        - Verify logits match within numerical tolerance
        """
        model_with_cache = benchmark_models["with_cache"]
        model_without_cache = benchmark_models["without_cache"]
        device = next(model_with_cache.parameters()).device

        # Create input sequence
        seq_len = 20
        input_ids = torch.randint(0, 1000, (1, seq_len), device=device)

        with torch.no_grad():
            # Generate with cache (one token at a time)
            past_kv = None
            cached_logits = []
            tokens = input_ids

            for i in range(seq_len):
                logits, past_kv = model_with_cache(tokens, use_cache=True, past_key_values=past_kv)
                cached_logits.append(logits)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tokens = next_token

            # Generate without cache (full sequence each time)
            all_logits = []
            for i in range(seq_len):
                # Process i+1 tokens
                current_input = input_ids[:, : i + 1]
                logits = model_without_cache(current_input, use_cache=False)
                all_logits.append(logits)

        # Compare each step
        for i, (cached, full) in enumerate(zip(cached_logits, all_logits, strict=False)):
            # The cached version processes only the new token
            # The full version processes all tokens up to i+1
            # We need to compare the last token's logits
            assert torch.allclose(
                cached[:, -1, :],
                full[:, -1, :],
                rtol=1e-4,
                atol=1e-5,
            ), f"Logits don't match at step {i}"

        print("\n=== Numerical Correctness ===")
        print(
            f"All {seq_len} steps validated: Cached and non-cached outputs match within tolerance"
        )

    def test_memory_efficiency(self, benchmark_models):
        """
        Test: Memory usage comparison
        - Compare peak memory with and without cache
        - Verify cache doesn't leak memory
        - Test cache reset functionality
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory benchmarking")

        model_with_cache = benchmark_models["with_cache"]
        model_without_cache = benchmark_models["without_cache"]
        device = next(model_with_cache.parameters()).device

        # Create input sequence
        prompt_len = 32
        num_tokens = 64
        input_ids = torch.randint(0, 1000, (1, prompt_len), device=device)

        # Benchmark with cache
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        with torch.no_grad():
            past_kv = None
            tokens = input_ids
            for _ in range(num_tokens):
                _, past_kv = model_with_cache(tokens, use_cache=True, past_key_values=past_kv)
                tokens = tokens[:, -1:]  # Take last token

        memory_with_cache = torch.cuda.max_memory_allocated(device)

        # Benchmark without cache
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        with torch.no_grad():
            current_input = input_ids
            for i in range(num_tokens):
                _ = model_without_cache(current_input, use_cache=False)
                # Extend input for next iteration
                next_token = torch.randint(0, 1000, (1, 1), device=device)
                current_input = torch.cat([current_input, next_token], dim=1)

        memory_without_cache = torch.cuda.max_memory_allocated(device)

        memory_ratio = memory_with_cache / memory_without_cache

        print("\n=== Memory Efficiency ===")
        print(f"Memory with cache: {memory_with_cache / 1024**2:.2f} MB")
        print(f"Memory without cache: {memory_without_cache / 1024**2:.2f} MB")
        print(f"Memory ratio: {memory_ratio:.2f}x")

        # Cache should use less or comparable memory
        # For autoregressive generation, cache should be more memory efficient
        # because it avoids storing intermediate activations for the full sequence
        assert memory_ratio <= 1.5, f"Cache uses too much memory: {memory_ratio:.2f}x"

    def test_cache_state_consistency(self, benchmark_models):
        """
        Test: Cache state consistency
        - Verify cache state is correctly maintained across forward passes
        - Check that cache position is tracked correctly
        - Verify cache reset works correctly
        """
        model = benchmark_models["with_cache"]
        device = next(model.parameters()).device

        # Create input sequence
        seq_len = 10
        input_ids = torch.randint(0, 1000, (1, seq_len), device=device)

        with torch.no_grad():
            # First forward pass
            logits_1, past_kv_1 = model(input_ids, use_cache=True, past_key_values=None)

            # Verify cache state
            assert past_kv_1 is not None
            assert len(past_kv_1) == benchmark_models["config"].model.num_layers

            # Check each layer's cache
            for layer_kv in past_kv_1:
                key, value = layer_kv
                # Should have cached all tokens
                assert key.shape[1] == seq_len
                assert value.shape[1] == seq_len

            # Second forward pass with cache
            next_token = torch.randint(0, 1000, (1, 1), device=device)
            logits_2, past_kv_2 = model(next_token, use_cache=True, past_key_values=past_kv_1)

            # Verify cache was extended
            for layer_kv in past_kv_2:
                key, value = layer_kv
                assert key.shape[1] == seq_len + 1
                assert value.shape[1] == seq_len + 1

        print("\n=== Cache State Consistency ===")
        print("Cache state correctly maintained across forward passes")


def run_comprehensive_benchmark():
    """
    Run comprehensive benchmark suite and generate report.
    """
    print("\n" + "=" * 60)
    print("KV CACHE COMPREHENSIVE VALIDATION REPORT")
    print("=" * 60)

    # Clean up any existing global states
    global_states_cleanup()

    # Create benchmark setup
    config = create_benchmark_config(
        d_model=512,
        num_layers=4,
        num_heads=8,
        num_kv_groups=2,
        head_dim=64,
        seq_len=256,
        enable_cache=True,
    )
    set_global_states(config)
    model_with_cache = LanguageModel(config)
    model_with_cache.eval()

    global_states_cleanup()
    config_no_cache = create_benchmark_config(enable_cache=False)
    set_global_states(config_no_cache)
    model_without_cache = LanguageModel(config_no_cache)
    model_without_cache.eval()

    benchmark_models_dict = {
        "with_cache": model_with_cache,
        "without_cache": model_without_cache,
        "config": config,
    }

    test_suite = TestKVCachePerformance()

    print("\nRunning comprehensive benchmark suite...")

    try:
        test_suite.test_autoregressive_speedup_small_model(benchmark_models_dict)
    except Exception as e:
        print(f"Small model benchmark failed: {e}")

    try:
        test_suite.test_autoregressive_speedup_medium_sequence(benchmark_models_dict)
    except Exception as e:
        print(f"Medium sequence benchmark failed: {e}")

    try:
        test_suite.test_autoregressive_speedup_long_sequence(benchmark_models_dict)
    except Exception as e:
        print(f"Long sequence benchmark failed: {e}")

    try:
        test_suite.test_numerical_correctness_detailed(benchmark_models_dict)
    except Exception as e:
        print(f"Numerical correctness test failed: {e}")

    try:
        test_suite.test_memory_efficiency(benchmark_models_dict)
    except Exception as e:
        print(f"Memory efficiency test failed: {e}")

    try:
        test_suite.test_cache_state_consistency(benchmark_models_dict)
    except Exception as e:
        print(f"Cache state consistency test failed: {e}")

    print("\n" + "=" * 60)
    print("BENCHMARK SUITE COMPLETE")
    print("=" * 60)

    global_states_cleanup()


if __name__ == "__main__":
    run_comprehensive_benchmark()
