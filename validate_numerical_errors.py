# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT

"""
Numerical Error Validation for Paged Attention Implementations

Validates correctness across:
- Different sequence lengths
- Different dtypes (float16, float32, bfloat16)
- Edge cases (zeros, large values, mixed signs)
- Error statistics (max, mean, relative, distribution)

Run with: python validate_numerical_errors.py
"""

from dataclasses import dataclass

import torch

try:
    from ironcore.layers.triton_paged_attention import (
        TRITON_AVAILABLE,
        python_paged_attention,
        python_paged_attention_batched,
        triton_paged_attention,
    )
except ImportError:
    TRITON_AVAILABLE = False


@dataclass
class ValidationConfig:
    """Configuration for validation."""

    batch_size: int = 4
    num_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 64
    page_size: int = 16
    num_pages: int = 2048
    device: str = "cuda"


def create_test_data(
    config: ValidationConfig,
    seq_lengths: list[int],
    dtype: torch.dtype = torch.float16,
    seed: int = 42,
):
    """Create test tensors for validation."""
    torch.manual_seed(seed)

    batch_size = config.batch_size
    num_heads = config.num_heads
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim
    page_size = config.page_size
    num_pages = config.num_pages
    device = config.device

    # Query: [batch, 1, num_heads, head_dim]
    query = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=dtype)

    # Physical K/V cache: [num_pages, num_kv_heads, page_size, head_dim]
    key_cache = torch.randn(
        num_pages, num_kv_heads, page_size, head_dim, device=device, dtype=dtype
    )
    value_cache = torch.randn(
        num_pages, num_kv_heads, page_size, head_dim, device=device, dtype=dtype
    )

    # Block tables: [batch, max_pages_per_seq]
    max_seq_len = max(seq_lengths)
    max_pages_per_seq = (max_seq_len + page_size - 1) // page_size
    block_tables = torch.zeros(batch_size, max_pages_per_seq, device=device, dtype=torch.long)

    # Context lengths
    context_lens = torch.tensor(seq_lengths, device=device, dtype=torch.long)

    # Fill block tables with sequential physical page indices
    page_idx = 0
    for b, seq_len in enumerate(seq_lengths):
        pages_needed = (seq_len + page_size - 1) // page_size
        for p in range(pages_needed):
            block_tables[b, p] = page_idx
            page_idx += 1

    return query, key_cache, value_cache, block_tables, context_lens


def compute_error_stats(reference: torch.Tensor, test: torch.Tensor, name: str) -> dict:
    """Compute comprehensive error statistics."""
    diff = (reference - test).abs()

    # Handle dtype limits for machine epsilon
    if reference.dtype == torch.float16:
        eps = torch.finfo(torch.float16).eps
    elif reference.dtype == torch.float32:
        eps = torch.finfo(torch.float32).eps
    elif reference.dtype == torch.bfloat16:
        eps = torch.finfo(torch.bfloat16).eps
    else:
        eps = 1e-7

    max_abs_error = diff.max().item()
    mean_abs_error = diff.mean().item()
    std_abs_error = diff.std().item()

    # Relative error (avoid division by zero)
    ref_abs = reference.abs()
    relative_errors = diff / (ref_abs + eps)
    max_rel_error = relative_errors.max().item()
    mean_rel_error = relative_errors.mean().item()

    # Check if within dtype tolerance
    within_tolerance = max_abs_error < 10 * eps

    # Cosine similarity
    ref_flat = reference.flatten()
    test_flat = test.flatten()
    cosine_sim = torch.nn.functional.cosine_similarity(
        ref_flat.unsqueeze(0), test_flat.unsqueeze(0)
    ).item()

    return {
        "name": name,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "std_abs_error": std_abs_error,
        "max_rel_error": max_rel_error,
        "mean_rel_error": mean_rel_error,
        "cosine_similarity": cosine_sim,
        "within_tolerance": within_tolerance,
        "dtype": str(reference.dtype).split(".")[-1],
        "eps": eps,
    }


def validate_implementations(
    config: ValidationConfig,
    seq_lengths: list[int],
    dtype: torch.dtype,
    label: str,
):
    """Validate all implementations against each other."""
    print(f"\n{'=' * 70}")
    print(f"Validation: {label}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Dtype: {dtype}")
    print(f"{'=' * 70}")

    query, key_cache, value_cache, block_tables, context_lens = create_test_data(
        config, seq_lengths, dtype
    )

    # Run all implementations
    py_output = python_paged_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        config.num_heads,
        config.num_kv_heads,
        config.page_size,
    )

    py_batched_output = python_paged_attention_batched(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        config.num_heads,
        config.num_kv_heads,
        config.page_size,
    )

    results = []

    # Python loop vs batched
    stats = compute_error_stats(py_output, py_batched_output, "Python loop vs batched")
    results.append(stats)
    print(f"\n{stats['name']}:")
    print(f"  Max absolute error: {stats['max_abs_error']:.6e}")
    print(f"  Mean absolute error: {stats['mean_abs_error']:.6e}")
    print(f"  Max relative error: {stats['max_rel_error']:.6e}")
    print(f"  Cosine similarity: {stats['cosine_similarity']:.6f}")
    print(
        f"  Within tolerance (10*eps={10 * stats['eps']:.2e}): {'✅' if stats['within_tolerance'] else '⚠️'}"
    )

    if TRITON_AVAILABLE:
        triton_output = triton_paged_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            config.num_heads,
            config.num_kv_heads,
            config.page_size,
        )

        # Python loop vs Triton
        stats = compute_error_stats(py_output, triton_output, "Python loop vs Triton")
        results.append(stats)
        print(f"\n{stats['name']}:")
        print(f"  Max absolute error: {stats['max_abs_error']:.6e}")
        print(f"  Mean absolute error: {stats['mean_abs_error']:.6e}")
        print(f"  Max relative error: {stats['max_rel_error']:.6e}")
        print(f"  Cosine similarity: {stats['cosine_similarity']:.6f}")
        print(
            f"  Within tolerance (10*eps={10 * stats['eps']:.2e}): {'✅' if stats['within_tolerance'] else '⚠️'}"
        )

        # Python batched vs Triton
        stats = compute_error_stats(py_batched_output, triton_output, "Python batched vs Triton")
        results.append(stats)
        print(f"\n{stats['name']}:")
        print(f"  Max absolute error: {stats['max_abs_error']:.6e}")
        print(f"  Mean absolute error: {stats['mean_abs_error']:.6e}")
        print(f"  Max relative error: {stats['max_rel_error']:.6e}")
        print(f"  Cosine similarity: {stats['cosine_similarity']:.6f}")
        print(
            f"  Within tolerance (10*eps={10 * stats['eps']:.2e}): {'✅' if stats['within_tolerance'] else '⚠️'}"
        )
    else:
        print("\nTriton not available, skipping Triton comparisons")

    return results


def validate_edge_cases(config: ValidationConfig):
    """Validate edge cases."""
    print(f"\n{'=' * 70}")
    print("Edge Case Validation")
    print(f"{'=' * 70}")

    dtype = torch.float16

    # Test 1: All zeros in KV cache
    print("\n--- Test 1: Zero KV cache ---")
    seq_lengths = [64, 128, 256, 512]
    query, key_cache, value_cache, block_tables, context_lens = create_test_data(
        config, seq_lengths, dtype
    )
    key_cache.zero_()
    value_cache.zero_()

    py_output = python_paged_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        config.num_heads,
        config.num_kv_heads,
        config.page_size,
    )
    py_batched_output = python_paged_attention_batched(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        config.num_heads,
        config.num_kv_heads,
        config.page_size,
    )

    # When KV is zero, attention should produce near-zero output
    print(f"  Python loop output norm: {py_output.norm().item():.6e}")
    print(f"  Python batched output norm: {py_batched_output.norm().item():.6e}")
    print(f"  Difference norm: {(py_output - py_batched_output).norm().item():.6e}")

    # Test 2: Large values (test for overflow)
    print("\n--- Test 2: Large values ---")
    query, key_cache, value_cache, block_tables, context_lens = create_test_data(
        config, seq_lengths, dtype, seed=123
    )
    # Scale up values but stay within float16 range
    scale = 10.0
    query = query * scale
    key_cache = key_cache * scale
    value_cache = value_cache * scale

    py_output = python_paged_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        config.num_heads,
        config.num_kv_heads,
        config.page_size,
    )
    py_batched_output = python_paged_attention_batched(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        config.num_heads,
        config.num_kv_heads,
        config.page_size,
    )

    diff = (py_output - py_batched_output).abs()
    print(f"  Max absolute error: {diff.max().item():.6e}")
    print(f"  Mean absolute error: {diff.mean().item():.6e}")
    print(f"  Reference output range: [{py_output.min().item():.2f}, {py_output.max().item():.2f}]")

    # Test 3: Mixed signs
    print("\n--- Test 3: Mixed signs ---")
    query, key_cache, value_cache, block_tables, context_lens = create_test_data(
        config, seq_lengths, dtype, seed=456
    )
    # Alternate signs per kv head
    num_kv_heads = config.num_kv_heads
    sign_pattern = torch.tensor(
        [1 if i % 2 == 0 else -1 for i in range(num_kv_heads)], device=key_cache.device
    )
    key_cache = key_cache * sign_pattern.view(1, num_kv_heads, 1, 1)
    value_cache = value_cache * (-sign_pattern).view(1, num_kv_heads, 1, 1)

    py_output = python_paged_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        config.num_heads,
        config.num_kv_heads,
        config.page_size,
    )
    py_batched_output = python_paged_attention_batched(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        config.num_heads,
        config.num_kv_heads,
        config.page_size,
    )

    diff = (py_output - py_batched_output).abs()
    print(f"  Max absolute error: {diff.max().item():.6e}")
    print(f"  Mean absolute error: {diff.mean().item():.6e}")
    print(f"  Reference output range: [{py_output.min().item():.2f}, {py_output.max().item():.2f}]")

    # Test 4: Very short sequences
    print("\n--- Test 4: Very short sequences (1-4 tokens) ---")
    seq_lengths = [1, 2, 3, 4]
    query, key_cache, value_cache, block_tables, context_lens = create_test_data(
        config, seq_lengths, dtype, seed=789
    )

    py_output = python_paged_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        config.num_heads,
        config.num_kv_heads,
        config.page_size,
    )
    py_batched_output = python_paged_attention_batched(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        config.num_heads,
        config.num_kv_heads,
        config.page_size,
    )

    diff = (py_output - py_batched_output).abs()
    print(f"  Max absolute error: {diff.max().item():.6e}")
    print(f"  Mean absolute error: {diff.mean().item():.6e}")


def validate_dtypes(config: ValidationConfig):
    """Validate across different dtypes."""
    print(f"\n{'=' * 70}")
    print("Dtype Validation")
    print(f"{'=' * 70}")

    seq_lengths = [128, 256, 512, 1024]

    dtypes_to_test = [torch.float16, torch.float32]
    if torch.cuda.is_bf16_supported():
        dtypes_to_test.append(torch.bfloat16)

    all_results = []

    for dtype in dtypes_to_test:
        results = validate_implementations(config, seq_lengths, dtype, f"Dtype: {dtype}")
        all_results.extend(results)

    return all_results


def print_summary(all_results: list[dict]):
    """Print summary table of all validation results."""
    print(f"\n{'=' * 80}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"{'Comparison':<30} {'Dtype':<10} {'Max Abs Err':<12} {'Mean Abs Err':<12} {'Cos Sim':<8} {'Status':<6}"
    )
    print("-" * 80)

    for r in all_results:
        status = "✅" if r["within_tolerance"] else "⚠️"
        print(
            f"{r['name']:<30} {r['dtype']:<10} {r['max_abs_error']:<12.2e} {r['mean_abs_error']:<12.2e} {r['cosine_similarity']:<8.4f} {status:<6}"
        )

    print("=" * 80)

    # Check if all within tolerance
    all_pass = all(r["within_tolerance"] for r in all_results)
    if all_pass:
        print("\n✅ All validations passed within dtype tolerance!")
    else:
        print("\n⚠️  Some validations exceeded tolerance - review details above")


def main():
    """Run all validations."""
    if not torch.cuda.is_available():
        print("CUDA not available, validation requires GPU")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton available: {TRITON_AVAILABLE}")

    config = ValidationConfig()

    all_results = []

    # Validate different sequence lengths with float16
    test_cases = [
        ("Short (16-64)", [16, 32, 48, 64]),
        ("Medium (64-256)", [64, 128, 192, 256]),
        ("Long (256-1024)", [256, 512, 768, 1024]),
        ("Very long (1024-4096)", [1024, 2048, 3072, 4096]),
        ("Uniform 512", [512, 512, 512, 512]),
        ("Uniform 2048", [2048, 2048, 2048, 2048]),
    ]

    for label, seq_lengths in test_cases:
        results = validate_implementations(config, seq_lengths, torch.float16, label)
        all_results.extend(results)

    # Validate dtypes
    dtype_results = validate_dtypes(config)
    all_results.extend(dtype_results)

    # Validate edge cases
    validate_edge_cases(config)

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
