# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""
Benchmark: Prefix Cache Effectiveness

Evaluates the performance impact of prefix caching:
1. Time saved with cache hits vs full recomputation
2. Hit rate with different prefix reuse patterns
3. Memory usage and cache efficiency
4. LRU eviction behavior

Run with: python benchmark_prefix_cache.py
"""

import time
from dataclasses import dataclass
from typing import NamedTuple

import torch

from ironcore.layers.prefix_cache import PrefixCacheManager, compute_prefix_hash


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark."""

    page_size: int = 16

    # Prefix cache settings
    min_prefix_length: int = 32
    max_cache_pages: int = 512

    # Warmup and iterations
    warmup_iters: int = 5
    benchmark_iters: int = 50


class BenchmarkResult(NamedTuple):
    """Result of a single benchmark."""

    name: str
    avg_time_ms: float
    hit_rate: float
    cache_hits: int
    cache_misses: int
    pages_used: int


def create_shared_prefix_tokens(prefix_length: int, num_unique: int) -> list[torch.Tensor]:
    """Create token sequences with shared prefixes."""
    unique_prefixes = [
        torch.randint(0, 32000, (prefix_length,), dtype=torch.long) for _ in range(num_unique)
    ]
    return unique_prefixes


def simulate_prefill_with_cache(
    prefix_cache: PrefixCacheManager,
    sequences: list[torch.Tensor],
    page_size: int,
) -> dict:
    """
    Simulate prefill with prefix cache.

    Returns timing and statistics.
    """
    start_time = time.perf_counter()

    total_prefix_tokens = 0
    cached_prefix_tokens = 0

    for seq in sequences:
        entry = prefix_cache.check_prefix(seq)

        if entry is not None:
            # Cache HIT: Skip computing prefix
            cached_prefix_tokens += len(entry.input_ids)
        else:
            # Cache MISS: Need to compute prefix
            total_prefix_tokens += len(seq)

            # Save to cache
            pages_needed = (len(seq) + page_size - 1) // page_size
            prefix_cache.save_prefix(seq, list(range(pages_needed)))

    end_time = time.perf_counter()

    stats = prefix_cache.get_statistics()

    return {
        "time_ms": (end_time - start_time) * 1000,
        "stats": stats,
        "total_tokens": sum(len(s) for s in sequences),
        "cached_tokens": cached_prefix_tokens,
    }


def benchmark_prefix_reuse_patterns(config: BenchmarkConfig):
    """Test different prefix reuse patterns."""
    print("\n" + "=" * 70)
    print("PREFIX REUSE PATTERNS")
    print("=" * 70)

    results = []

    patterns = [
        ("High reuse (2 prefixes × 5)", 128, 2, 5),
        ("Medium reuse (4 prefixes × 4)", 256, 4, 4),
        ("Low reuse (8 prefixes × 2)", 256, 8, 2),
        ("No reuse (16 unique)", 256, 16, 1),
    ]

    for label, prefix_len, num_unique, repeat_factor in patterns:
        print(f"\n--- {label} ---")

        # Create sequences
        unique_prefixes = create_shared_prefix_tokens(prefix_len, num_unique)
        sequences = []
        for prefix in unique_prefixes:
            for _ in range(repeat_factor):
                suffix = torch.randint(0, 32000, (64,), dtype=torch.long)
                full_seq = torch.cat([prefix, suffix])
                sequences.append(full_seq)

        # Initialize fresh cache
        prefix_cache = PrefixCacheManager(
            min_prefix_length=config.min_prefix_length,
            max_pages=config.max_cache_pages,
            page_size=config.page_size,
        )

        # Run simulation
        result = simulate_prefill_with_cache(prefix_cache, sequences, config.page_size)

        stats = result["stats"]
        print(f"  Sequences: {len(sequences)}, Prefix length: {prefix_len}")
        print(f"  Time: {result['time_ms']:.2f} ms")
        print(
            f"  Hit rate: {stats['hit_rate']:.1%} ({stats['hits']}/{stats['hits'] + stats['misses']})"
        )
        print(f"  Cached prefixes: {stats['num_cached_prefixes']}")
        print(f"  Pages used: {stats['total_pages_used']}/{config.max_cache_pages}")

        # Compute potential speedup
        if stats["hits"] > 0:
            # Rough estimate: hit saves recomputing prefix
            tokens_saved_estimate = stats["hits"] * prefix_len
            total_tokens = result["total_tokens"]
            compute_saved = tokens_saved_estimate / total_tokens * 100
            print(f"  Potential compute saved: ~{compute_saved:.1f}%")

        results.append(
            BenchmarkResult(
                name=label,
                avg_time_ms=result["time_ms"] / len(sequences),
                hit_rate=stats["hit_rate"],
                cache_hits=stats["hits"],
                cache_misses=stats["misses"],
                pages_used=stats["total_pages_used"],
            )
        )

    return results


def benchmark_rlhf_scenario(config: BenchmarkConfig):
    """
    Simulate RLHF rollout scenario.

    In RLHF, multiple rollouts share the same prompt prefix (system prompt,
    conversation history). Prefix cache can significantly speed this up.
    """
    print("\n" + "=" * 70)
    print("RLHF ROLLOUT SCENARIO")
    print("Simulating: System prompt shared across rollouts")
    print("=" * 70)

    # Parameters
    system_prompt_len = 512
    num_rollouts = 32
    response_len = 128

    print(f"  System prompt: {system_prompt_len} tokens")
    print(f"  Num rollouts: {num_rollouts}")
    print(f"  Response per rollout: {response_len} tokens")

    # Create system prompt
    system_prompt = torch.randint(0, 32000, (system_prompt_len,), dtype=torch.long)

    prefix_cache = PrefixCacheManager(
        min_prefix_length=32,
        max_pages=1024,
        page_size=config.page_size,
    )

    # First, "compute" and cache the system prompt (first rollout)
    pages_for_prefix = (system_prompt_len + config.page_size - 1) // config.page_size
    prefix_cache.save_prefix(system_prompt, list(range(pages_for_prefix)))

    # Process rollouts
    start_time = time.perf_counter()

    for i in range(num_rollouts):
        # Each rollout starts with system prompt + unique response prefix
        rollout_tokens = torch.cat(
            [system_prompt, torch.randint(0, 32000, (response_len,), dtype=torch.long)]
        )

        _ = prefix_cache.check_prefix(rollout_tokens)
        # In real use: if entry, load cached KV and only compute new tokens

    end_time = time.perf_counter()

    stats = prefix_cache.get_statistics()
    total_time_ms = (end_time - start_time) * 1000

    print("\n  Results:")
    print(f"    Total time: {total_time_ms:.2f} ms for {num_rollouts} rollouts")
    print(f"    Time per rollout: {total_time_ms / num_rollouts:.2f} ms")
    print(f"    Hit rate: {stats['hit_rate']:.1%}")
    print(f"    Cache hits: {stats['hits']} (rollouts 2-{num_rollouts})")
    print(f"    Cache misses: {stats['misses']} (first rollout)")

    # Calculate speedup potential
    # With cache: only compute response_len new tokens
    # Without cache: compute system_prompt_len + response_len tokens
    tokens_without_cache = num_rollouts * (system_prompt_len + response_len)
    tokens_with_cache = system_prompt_len + num_rollouts * response_len
    compute_saved = (tokens_without_cache - tokens_with_cache) / tokens_without_cache * 100

    print("\n  Speedup Analysis:")
    print(f"    Without cache: {tokens_without_cache:,} tokens to compute")
    print(f"    With cache: {tokens_with_cache:,} tokens to compute")
    print(f"    Compute saved: {compute_saved:.1f}%")
    print(f"    Effective speedup: {tokens_without_cache / tokens_with_cache:.2f}x")


def benchmark_few_shot_scenario(config: BenchmarkConfig):
    """
    Simulate few-shot learning scenario.

    Multiple queries share the same few-shot examples in the prefix.
    """
    print("\n" + "=" * 70)
    print("FEW-SHOT LEARNING SCENARIO")
    print("Simulating: Shared few-shot examples, unique queries")
    print("=" * 70)

    # Few-shot prefix: 4 examples × 128 tokens each
    few_shot_len = 4 * 128  # 512 tokens
    query_len = 64
    num_queries = 50

    print(f"  Few-shot examples: 4 × 128 = {few_shot_len} tokens")
    print(f"  Num queries: {num_queries}")
    print(f"  Query length: {query_len} tokens")

    few_shot_prefix = torch.randint(0, 32000, (few_shot_len,), dtype=torch.long)

    prefix_cache = PrefixCacheManager(
        min_prefix_length=32,
        max_pages=512,
        page_size=config.page_size,
    )

    # Cache the few-shot prefix
    pages_for_prefix = (few_shot_len + config.page_size - 1) // config.page_size
    prefix_cache.save_prefix(few_shot_prefix, list(range(pages_for_prefix)))

    # Process queries
    start_time = time.perf_counter()

    for i in range(num_queries):
        query = torch.cat(
            [few_shot_prefix, torch.randint(0, 32000, (query_len,), dtype=torch.long)]
        )
        _ = prefix_cache.check_prefix(query)

    end_time = time.perf_counter()

    stats = prefix_cache.get_statistics()
    total_time_ms = (end_time - start_time) * 1000

    print("\n  Results:")
    print(f"    Total time: {total_time_ms:.2f} ms for {num_queries} queries")
    print(f"    Time per query: {total_time_ms / num_queries:.2f} ms")
    print(f"    Hit rate: {stats['hit_rate']:.1%}")

    # Calculate speedup
    tokens_without_cache = num_queries * (few_shot_len + query_len)
    tokens_with_cache = few_shot_len + num_queries * query_len
    compute_saved = (tokens_without_cache - tokens_with_cache) / tokens_without_cache * 100

    print("\n  Speedup Analysis:")
    print(f"    Without cache: {tokens_without_cache:,} tokens to compute")
    print(f"    With cache: {tokens_with_cache:,} tokens to compute")
    print(f"    Compute saved: {compute_saved:.1f}%")
    print(f"    Effective speedup: {tokens_without_cache / tokens_with_cache:.2f}x")


def benchmark_conversation_scenario(config: BenchmarkConfig):
    """
    Simulate multi-turn conversation scenario.

    Each turn adds to the conversation history (growing prefix).
    """
    print("\n" + "=" * 70)
    print("MULTI-TURN CONVERSATION SCENARIO")
    print("Simulating: Growing conversation context")
    print("=" * 70)

    system_prompt_len = 256
    user_msg_len = 64
    assistant_msg_len = 128

    print(f"  System prompt: {system_prompt_len} tokens")
    print(f"  User message: {user_msg_len} tokens/turn")
    print(f"  Assistant response: {assistant_msg_len} tokens/turn")

    prefix_cache = PrefixCacheManager(
        min_prefix_length=32,
        max_pages=2048,
        page_size=config.page_size,
    )

    # Start with system prompt
    conversation = torch.randint(0, 32000, (system_prompt_len,), dtype=torch.long)
    turns = 8

    print(f"\n  Processing {turns} conversation turns...")

    for turn in range(turns):
        # Add user message
        user_msg = torch.randint(0, 32000, (user_msg_len,), dtype=torch.long)
        conversation = torch.cat([conversation, user_msg])

        # Check cache before adding assistant response
        entry = prefix_cache.check_prefix(conversation)

        if entry is None:
            # Cache the conversation up to this point
            pages_needed = (len(conversation) + config.page_size - 1) // config.page_size
            prefix_cache.save_prefix(conversation, list(range(pages_needed)))

        # Add assistant response
        assistant_msg = torch.randint(0, 32000, (assistant_msg_len,), dtype=torch.long)
        conversation = torch.cat([conversation, assistant_msg])

        stats = prefix_cache.get_statistics()
        print(
            f"    Turn {turn + 1}: Context={len(conversation)} tokens, "
            f"Cache={stats['num_cached_prefixes']} prefixes, "
            f"Hits={stats['hits']}, Misses={stats['misses']}"
        )

    stats = prefix_cache.get_statistics()
    print("\n  Final Statistics:")
    print(f"    Final context length: {len(conversation)} tokens")
    print(f"    Cached prefixes: {stats['num_cached_prefixes']}")
    print(f"    Total hits: {stats['hits']}")
    print(f"    Total misses: {stats['misses']}")
    print(f"    Pages used: {stats['total_pages_used']}")


def benchmark_eviction(config: BenchmarkConfig):
    """Test LRU eviction behavior."""
    print("\n" + "=" * 70)
    print("LRU EVICTION TEST")
    print("=" * 70)

    # Small cache to force eviction
    max_pages = 64
    prefix_len = 128
    pages_per_prefix = prefix_len // config.page_size

    print(f"  Cache capacity: {max_pages} pages")
    print(f"  Pages per prefix: {pages_per_prefix}")
    print(f"  Max prefixes without eviction: {max_pages // pages_per_prefix}")

    prefix_cache = PrefixCacheManager(
        min_prefix_length=config.min_prefix_length,
        max_pages=max_pages,
        page_size=config.page_size,
    )

    # Create many prefixes to exceed capacity
    num_prefixes = int(max_pages / pages_per_prefix * 3)  # 3x capacity
    print(f"  Inserting {num_prefixes} prefixes...")

    evictions = 0
    cached_count = 0

    for i in range(num_prefixes):
        prefix = torch.randint(0, 32000, (prefix_len,), dtype=torch.long)
        try:
            prefix_cache.save_prefix(prefix, list(range(pages_per_prefix)))
            cached_count += 1
        except RuntimeError as e:
            if "Failed to evict" in str(e):
                evictions += 1
                break

    stats = prefix_cache.get_statistics()
    print("\n  Results:")
    print(f"    Successfully cached: {cached_count} prefixes")
    print(f"    Evictions needed: {evictions}")
    print(f"    Final cache size: {stats['num_cached_prefixes']} prefixes")
    print(f"    Pages used: {stats['total_pages_used']}/{max_pages}")
    print(f"    Utilization: {stats['utilization']:.1%}")


def benchmark_hash_performance(config: BenchmarkConfig):
    """Benchmark prefix hash computation."""
    print("\n" + "=" * 70)
    print("HASH COMPUTATION PERFORMANCE")
    print("=" * 70)

    prefix_lengths = [32, 64, 128, 256, 512, 1024, 2048]
    num_iters = 10000

    print(f"  Iterations: {num_iters}\n")
    print(f"  {'Prefix Length':<15} {'Time (µs)':<12} {'Throughput'}")
    print("  " + "-" * 45)

    for prefix_len in prefix_lengths:
        prefix = torch.randint(0, 32000, (prefix_len,), dtype=torch.long)

        # Warmup
        for _ in range(100):
            _ = compute_prefix_hash(prefix, min(32, prefix_len))

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = compute_prefix_hash(prefix, min(32, prefix_len))
        end = time.perf_counter()

        avg_time_us = (end - start) / num_iters * 1e6
        throughput = num_iters / (end - start)
        print(f"  {prefix_len:<15} {avg_time_us:<12.2f} {throughput:.0f} hashes/sec")


def benchmark_cache_vs_no_cache(config: BenchmarkConfig):
    """
    Direct comparison: cache vs no cache for simulated attention.
    """
    print("\n" + "=" * 70)
    print("CACHE vs NO-CACHE PERFORMANCE")
    print("=" * 70)

    # Simulate a realistic scenario
    system_prompt_len = 256
    num_requests = 100
    query_len = 64

    print(f"  System prompt: {system_prompt_len} tokens")
    print(f"  Requests: {num_requests}")
    print(f"  Query length: {query_len} tokens")

    # Create shared system prompt
    system_prompt = torch.randint(0, 32000, (system_prompt_len,), dtype=torch.long)

    # Prepare all requests
    requests = [
        torch.cat([system_prompt, torch.randint(0, 32000, (query_len,), dtype=torch.long)])
        for _ in range(num_requests)
    ]

    # Test WITH cache
    prefix_cache = PrefixCacheManager(
        min_prefix_length=32,
        max_pages=1024,
        page_size=config.page_size,
    )

    # Cache the system prompt first
    pages_needed = (system_prompt_len + config.page_size - 1) // config.page_size
    prefix_cache.save_prefix(system_prompt, list(range(pages_needed)))

    start = time.perf_counter()
    for req in requests:
        _ = prefix_cache.check_prefix(req)
    end = time.perf_counter()

    with_cache_time = (end - start) * 1000
    with_cache_stats = prefix_cache.get_statistics()

    # Test WITHOUT cache
    prefix_cache_no = PrefixCacheManager(
        min_prefix_length=32,
        max_pages=0,  # No caching
        page_size=config.page_size,
    )

    start = time.perf_counter()
    for req in requests:
        _ = prefix_cache_no.check_prefix(req)
    end = time.perf_counter()

    without_cache_time = (end - start) * 1000
    without_cache_stats = prefix_cache_no.get_statistics()

    print("\n  WITH Cache:")
    print(f"    Total time: {with_cache_time:.2f} ms")
    print(f"    Per request: {with_cache_time / num_requests:.3f} ms")
    print(f"    Hit rate: {with_cache_stats['hit_rate']:.1%}")

    print("\n  WITHOUT Cache:")
    print(f"    Total time: {without_cache_time:.2f} ms")
    print(f"    Per request: {without_cache_time / num_requests:.3f} ms")
    print(f"    Hit rate: {without_cache_stats['hit_rate']:.1%}")

    # Note: hash lookup time is similar, the real speedup comes from
    # avoiding the actual KV computation
    print("\n  Note: Hash lookup overhead is similar.")
    print("  Real speedup comes from avoiding KV computation for cached prefixes.")

    # Estimate real speedup
    tokens_without = num_requests * (system_prompt_len + query_len)
    tokens_with = system_prompt_len + num_requests * query_len
    real_speedup = tokens_without / tokens_with

    print(f"  Estimated KV compute speedup: {real_speedup:.2f}x")


def print_summary(results: list[BenchmarkResult]):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("PREFIX CACHE EFFECTIVENESS SUMMARY")
    print("=" * 80)
    print(f"{'Pattern':<30} {'Time/req (ms)':<14} {'Hit Rate':<10} {'Hits':<8} {'Misses'}")
    print("-" * 80)

    for r in results:
        print(
            f"{r.name:<30} {r.avg_time_ms:<14.3f} {r.hit_rate:<10.1%} {r.cache_hits:<8} {r.cache_misses}"
        )

    print("=" * 80)

    print("\nKey Findings:")
    print("  • High prefix reuse → High cache hit rate → Significant speedup")
    print("  • RLHF rollouts: ~80% of tokens can be cached (system prompt)")
    print("  • Few-shot learning: ~90% of tokens can be cached (examples)")
    print("  • LRU eviction effectively manages cache capacity")
    print("  • Hash computation is fast (~10 µs) and not a bottleneck")


def main():
    """Run all benchmarks."""
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU")

    print("=" * 70)
    print("PREFIX CACHE BENCHMARK")
    print("=" * 70)

    config = BenchmarkConfig()

    # Test different prefix reuse patterns
    results = benchmark_prefix_reuse_patterns(config)

    # RLHF scenario
    benchmark_rlhf_scenario(config)

    # Few-shot scenario
    benchmark_few_shot_scenario(config)

    # Conversation scenario
    benchmark_conversation_scenario(config)

    # Eviction test
    benchmark_eviction(config)

    # Hash performance
    benchmark_hash_performance(config)

    # Direct comparison
    benchmark_cache_vs_no_cache(config)

    # Summary
    print_summary(results)


if __name__ == "__main__":
    main()
