# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Tokenize input and show statistics."""

import sys
from argparse import Namespace
from pathlib import Path

from ironcore.train import load_full_config


def register_parser(subparsers) -> None:
    """Register the CLI subcommand arguments."""
    parser = subparsers.add_parser("tokenize", help="Tokenize input and show statistics")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--input", type=str, required=True, help="Text file path or literal string")
    parser.add_argument("--show-tokens", action="store_true", help="Display per-token breakdown")
    parser.add_argument("--histogram", action="store_true", help="Show sequence length histogram")


def run_tokenize(args: Namespace) -> None:
    """Tokenize input text or file and show statistics.

    Args:
        args: Command-line arguments.
    """
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}")
        sys.exit(1)

    # Build tokenizer
    config = load_full_config(config_path)
    from ironcore import global_vars

    if global_vars.GLOBAL_STATES is None:
        global_vars.set_global_states(config)

    from ironcore.tokenizer.tokenizer import build_tokenizer

    tokenizer = build_tokenizer(config)

    # Load input
    input_path = Path(args.input) if args.input else None
    if input_path and input_path.exists():
        with open(input_path) as f:
            lines = list(f)
    elif args.input:
        lines = [args.input]
    else:
        print("Error: --input is required (file path or text string)")
        sys.exit(1)

    # Tokenize
    all_lengths = []
    token_counts: dict[int, int] = {}
    total_tokens = 0

    for line in lines:
        ids = _encode_line(tokenizer, line.strip())
        total_tokens += len(ids)
        all_lengths.append(len(ids))
        for tid in ids:
            token_counts[tid] = token_counts.get(tid, 0) + 1

    if not all_lengths:
        print("No text to tokenize.")
        return

    # Statistics
    unique_tokens = len(token_counts)
    avg_length = total_tokens / len(lines)
    sorted_lengths = sorted(all_lengths)
    median_length = sorted_lengths[len(sorted_lengths) // 2]
    original_bytes = sum(len(line.encode("utf-8")) for line in lines)
    compression_ratio = original_bytes / total_tokens if total_tokens > 0 else 0

    # Print summary
    print(f"Tokenizer: {config.model.tokenizer_type}")
    print(f"Vocab size: {tokenizer.vocab_size:,}")
    print(f"Padded vocab size: {tokenizer.padded_vocab_size:,}")
    print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print()
    print(f"Input lines: {len(lines):,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Unique tokens: {unique_tokens:,}")
    print(
        f"Tokens/line: avg={avg_length:.1f}, min={min(all_lengths)}, max={max(all_lengths)}, median={median_length}"
    )
    print(f"Compression: {compression_ratio:.2f} bytes/token")

    # Show tokens
    if args.show_tokens:
        print("\nToken Breakdown:")
        for i, line in enumerate(lines):
            ids = _encode_line(tokenizer, line.strip())
            print(f"\n  Line {i} ({len(ids)} tokens):")
            for j, tid in enumerate(ids):
                text = tokenizer.decode([tid])
                print(f"    [{j:4d}] {tid:6d} -> {repr(text)}")

    # Histogram
    if args.histogram:
        _print_histogram(all_lengths)


def _encode_line(tokenizer, text: str) -> list[int]:
    """Encode a single line of text to token IDs."""
    if not text:
        return []
    encoded = tokenizer.encode(text)
    if isinstance(encoded, dict):
        ids = encoded.get("input_ids", encoded)
    else:
        ids = encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return list(ids) if not isinstance(ids, list) else ids


def _print_histogram(lengths: list[int], num_bins: int = 20) -> None:
    """Print ASCII histogram of sequence lengths."""
    if not lengths:
        return
    min_len = min(lengths)
    max_len = max(lengths)
    if min_len == max_len:
        print(f"\nAll sequences have length {min_len}")
        return

    bin_width = (max_len - min_len) / num_bins
    bins = [0] * num_bins
    for length in lengths:
        idx = min(int((length - min_len) / bin_width), num_bins - 1)
        bins[idx] += 1
    max_count = max(bins)

    print("\nSequence Length Distribution:")
    for i, count in enumerate(bins):
        lo = min_len + i * bin_width
        hi = lo + bin_width
        bar_len = int(40 * count / max_count) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  {lo:6.0f}-{hi:6.0f} | {bar} ({count})")
