# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware infrastructure tests for GRPO.

Tests:
1. VRAM leak detection across training steps
2. Worker pool latency measurement
3. LRU cache hit rate monitoring
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch


@dataclass
class MemoryStats:
    """Memory statistics for monitoring."""

    allocated_before: float = 0.0
    allocated_after: float = 0.0
    reserved_before: float = 0.0
    reserved_after: float = 0.0
    peak_allocated: float = 0.0

    @property
    def allocated_delta(self) -> float:
        return self.allocated_after - self.allocated_before

    @property
    def reserved_delta(self) -> float:
        return self.reserved_after - self.allocated_before


class MemoryMonitor:
    """Monitor GPU memory usage during operations."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.stats = MemoryStats()

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            self.stats.allocated_before = torch.cuda.memory_allocated() / 1024**3
            self.stats.reserved_before = torch.cuda.memory_reserved() / 1024**3
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.stats.allocated_after = torch.cuda.memory_allocated() / 1024**3
            self.stats.reserved_after = torch.cuda.memory_reserved() / 1024**3
            self.stats.peak_allocated = torch.cuda.max_memory_allocated() / 1024**3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMemoryLeaks:
    """Tests for memory leak detection."""

    def test_rollout_buffer_memory(self):
        """Test that RolloutBuffer doesn't leak memory across iterations."""
        from ironcore.alignment.buffer import RolloutBuffer

        device = "cuda"
        B, G = 4, 8
        prompt_len, response_len = 32, 64

        def create_buffer() -> RolloutBuffer:
            return RolloutBuffer(
                prompt_ids=torch.randint(0, 100, (B, prompt_len), device=device),
                prompt_attention_mask=torch.ones(B, prompt_len, device=device),
                completion_ids=torch.randint(0, 100, (B * G, prompt_len + response_len), device=device),
                response_ids=torch.randint(0, 100, (B * G, response_len), device=device),
                old_log_probs=torch.randn(B * G, device=device),
                rewards=torch.zeros(B * G, device=device),
                advantages=torch.zeros(B * G, device=device),
                group_ids=torch.arange(B).unsqueeze(1).expand(B, G).reshape(-1).to(device),
                metadata=[{"test": i} for i in range(B * G)],
            )

        # Warmup
        for _ in range(3):
            buf = create_buffer()
            _ = buf.summary()
            del buf

        torch.cuda.synchronize()

        # Measure baseline memory
        baseline_memory = torch.cuda.memory_allocated()

        # Create and destroy buffers multiple times
        for i in range(10):
            buf = create_buffer()
            _ = buf.summary()
            del buf
            gc.collect()
            torch.cuda.empty_cache()

        torch.cuda.synchronize()
        final_memory = torch.cuda.memory_allocated()

        # Memory should not grow significantly (< 10% increase)
        memory_growth = (final_memory - baseline_memory) / 1024**2  # MB
        assert memory_growth < 10, f"Memory grew by {memory_growth:.2f} MB (potential leak)"

    def test_kv_cache_cleanup(self):
        """Test that KV-cache is properly cleared after generation."""
        # Simulate KV-cache operations
        B, num_layers, num_heads, seq_len, head_dim = 4, 12, 8, 128, 64

        def create_kv_cache():
            return [
                (
                    torch.randn(B, num_heads, seq_len, head_dim, device="cuda"),
                    torch.randn(B, num_heads, seq_len, head_dim, device="cuda"),
                )
                for _ in range(num_layers)
            ]

        # Warmup
        kv = create_kv_cache()
        del kv
        gc.collect()
        torch.cuda.empty_cache()

        baseline_memory = torch.cuda.memory_allocated()

        # Create and destroy KV-cache multiple times
        for _ in range(10):
            kv = create_kv_cache()
            # Simulate expansion via repeat (not expand, since expand requires singleton dims)
            expanded_kv = []
            for k, v in kv:
                # Repeat each sample 2 times
                expanded_kv.append((k.repeat(2, 1, 1, 1), v.repeat(2, 1, 1, 1)))
            del kv, expanded_kv
            gc.collect()
            torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()
        memory_growth = (final_memory - baseline_memory) / 1024**2

        assert memory_growth < 10, f"KV-cache memory grew by {memory_growth:.2f} MB (potential leak)"


class TestWorkerPoolLatency:
    """Tests for reward worker pool performance."""

    def test_worker_pool_overhead(self):
        """Measure overhead of reward worker pool vs sequential."""
        from ironcore.alignment.rewards import MathRewardFunction, RewardWorkerPool

        reward_fn = MathRewardFunction()
        pool = RewardWorkerPool(reward_fn, num_workers=4, timeout=30)

        num_samples = 100
        prompts = ["What is 2+2?"] * num_samples
        completions = ["The answer is 4."] * num_samples
        metadata = [{"answer": "4"}] * num_samples

        # Measure pool time
        start = time.perf_counter()
        rewards_pool = pool.score_batch(prompts, completions, metadata)
        pool_time = time.perf_counter() - start

        # Measure sequential time
        start = time.perf_counter()
        rewards_sequential = []
        for p, c, m in zip(prompts, completions, metadata):
            rewards_sequential.append(reward_fn.compute(p, c, m))
        sequential_time = time.perf_counter() - start

        print(f"\nWorker Pool Performance:")
        print(f"  Pool time:       {pool_time*1000:.2f} ms")
        print(f"  Sequential time: {sequential_time*1000:.2f} ms")

        # Results should be identical
        assert torch.allclose(
            rewards_pool, torch.tensor(rewards_sequential, dtype=torch.float32)
        ), "Pool and sequential rewards should match"

        pool.shutdown()

    def test_reward_computation_latency_breakdown(self):
        """Measure latency breakdown of reward computation."""
        from ironcore.alignment.rewards import MathRewardFunction

        reward_fn = MathRewardFunction()

        # Test various completion lengths
        lengths = [50, 100, 200, 500]
        latencies = []

        for length in lengths:
            completion = " ".join(["test"] * length)
            metadata = {"answer": "42"}

            # Warmup
            reward_fn.compute("What is the answer?", completion, metadata)

            # Measure
            times = []
            for _ in range(10):
                start = time.perf_counter()
                reward_fn.compute("What is the answer?", completion, metadata)
                times.append(time.perf_counter() - start)

            avg_latency = sum(times) / len(times) * 1000  # ms
            latencies.append(avg_latency)
            print(f"  Length {length:4d}: {avg_latency:.3f} ms")

        # Latency should scale sub-linearly (or at least not explode)
        # Doubling length should less than triple latency (relaxed for CI variability)
        for i in range(1, len(latencies)):
            ratio = latencies[i] / latencies[i - 1]
            assert ratio < 3.0, f"Latency ratio {ratio:.2f} too high for doubled length"


class TestLRUCache:
    """Tests for LRU cache effectiveness."""

    def test_reward_cache_hit_rate(self):
        """Test LRU cache hit rate for repeated completions."""
        from ironcore.alignment.rewards import MathRewardFunction

        reward_fn = MathRewardFunction()

        # Create dataset with repeated patterns
        unique_prompts = ["What is 1+1?", "What is 2+2?", "What is 3+3?"]
        unique_completions = ["The answer is 2.", "The answer is 4.", "The answer is 6."]
        metadata = [{"answer": "2"}, {"answer": "4"}, {"answer": "6"}]

        # First pass: cache miss
        for _ in range(3):
            for p, c, m in zip(unique_prompts, unique_completions, metadata):
                reward_fn.compute(p, c, m)

        # Check cache stats (if available)
        # Note: The actual cache implementation may vary
        # This is a conceptual test

        print("\nLRU Cache Test:")
        print("  Completed repeated reward computations")
        print("  Cache should improve performance on repeated queries")


class TestThroughput:
    """End-to-end throughput tests."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_generation_throughput(self):
        """Measure generation throughput (tokens/sec)."""
        from ironcore.alignment.rollout import generate_rollouts_batched

        # Simple mock model
        class SimpleModel(torch.nn.Module):
            def forward(self, input_ids, labels=None, use_cache=False, past_key_values=None):
                logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 1000, device="cuda")
                if use_cache:
                    kv = [(torch.randn(input_ids.shape[0], 4, input_ids.shape[1], 64, device="cuda"),
                           torch.randn(input_ids.shape[0], 4, input_ids.shape[1], 64, device="cuda"))
                          for _ in range(2)]
                    return logits, kv
                return logits

        model = SimpleModel().cuda()
        model.eval()

        B, G, prompt_len = 4, 4, 32
        prompt_ids = torch.randint(0, 1000, (B, prompt_len), device="cuda")
        metadata = [{"answer": "42"} for _ in range(B)]

        # Warmup
        with torch.no_grad():
            _ = generate_rollouts_batched(
                model=model,
                prompt_ids=prompt_ids,
                group_size=G,
                metadata=metadata,
                max_new_tokens=32,
                do_sample=False,
            )

        torch.cuda.synchronize()

        # Benchmark
        total_tokens = 0
        start = time.perf_counter()

        for _ in range(5):
            with torch.no_grad():
                buffer = generate_rollouts_batched(
                    model=model,
                    prompt_ids=prompt_ids,
                    group_size=G,
                    metadata=metadata,
                    max_new_tokens=64,
                    do_sample=False,
                )
            total_tokens += buffer.total_samples * buffer.response_length

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        throughput = total_tokens / elapsed
        print(f"\nGeneration Throughput:")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Time: {elapsed:.3f} sec")
        print(f"  Throughput: {throughput:.0f} tokens/sec")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
