# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Dataset inspection — integrity checks, statistics, and visual preview."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ironcore.config.config_data import DataConfig

_ANSI_RED = "\033[91m"
_ANSI_GREEN = "\033[92m"
_ANSI_BLUE = "\033[94m"
_ANSI_YELLOW = "\033[93m"
_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"


def inspect_dataset(
    config_path: str | Path,
    *,
    preview: int = 0,
) -> dict[str, Any]:
    """Inspect preprocessed datasets and return a structured report.

    Args:
        config_path: Path to data configuration YAML file.
        preview: Number of samples to preview visually (0 = no preview).

    Returns:
        Dict with keys: timestamp, config_path, preprocessed_dir, datasets.
        Each dataset entry contains integrity checks, statistics, and validity.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    data_config = DataConfig.from_yaml(config_path)
    preprocessed_dir = Path(data_config.preprocessed_dir)
    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}")

    report: dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(config_path),
        "preprocessed_dir": str(preprocessed_dir),
        "datasets": [],
    }

    for dataset_config in data_config.datasets:
        ds_report = _inspect_single_dataset(
            data_config, dataset_config, preprocessed_dir, preview=preview
        )
        report["datasets"].append(ds_report)

    return report


def _inspect_single_dataset(
    data_config: DataConfig,
    dataset_config,
    preprocessed_dir: Path,
    *,
    preview: int = 0,
) -> dict[str, Any]:
    """Inspect a single dataset, returning a report dict."""
    dataset_name = dataset_config.name
    print(f"\n{_ANSI_BOLD}Dataset: {dataset_name}{_ANSI_RESET}")
    print("-" * 80)

    ds_report: dict[str, Any] = {
        "name": dataset_name,
        "task_type": dataset_config.task_type,
        "valid": True,
        "integrity_checks": {},
        "statistics": {},
    }

    output_path = data_config.get_dataset_output_path(dataset_config)
    bin_path = output_path / "data.bin"
    idx_path = output_path / "data.idx.npy"

    # Integrity check
    print(f"\n  {_ANSI_BOLD}Integrity Check:{_ANSI_RESET}")

    ds_report["integrity_checks"]["bin_exists"] = bin_path.exists()
    ds_report["integrity_checks"]["idx_exists"] = idx_path.exists()

    if not bin_path.exists():
        print(f"  [X] Missing .bin file: {bin_path}")
        ds_report.update(valid=False, error="Missing .bin file")
        return ds_report

    if not idx_path.exists():
        print(f"  [X] Missing .idx file: {idx_path}")
        ds_report.update(valid=False, error="Missing .idx file")
        return ds_report

    print("  [V] Files exist")

    try:
        metadata = np.load(idx_path)
    except Exception as e:
        print(f"  [X] Failed to load metadata: {e}")
        ds_report.update(valid=False, error=str(e))
        return ds_report

    num_samples = len(metadata)
    print(f"  [V] Metadata loaded: {num_samples:,} samples")

    # Validate metadata structure
    expected_fields = {"offset", "length", "type", "group_id", "mask_ranges"}
    actual_fields = set(metadata.dtype.names)
    ds_report["integrity_checks"]["metadata_fields_valid"] = actual_fields == expected_fields

    if actual_fields != expected_fields:
        print("  [X] Metadata fields mismatch")
        print(f"    Expected: {expected_fields}")
        print(f"    Found: {actual_fields}")
        ds_report.update(valid=False, error="Metadata fields mismatch")
        return ds_report

    print("  [V] Metadata structure valid")

    offsets = metadata["offset"]
    lengths = metadata["length"]
    types = metadata["type"]

    offsets_monotonic = bool(np.all(offsets[1:] >= offsets[:-1]))
    lengths_positive = bool(np.all(lengths > 0))
    ds_report["integrity_checks"]["offsets_monotonic"] = offsets_monotonic
    ds_report["integrity_checks"]["lengths_positive"] = lengths_positive

    if not offsets_monotonic:
        print("  [X] Offsets are not monotonic")
        ds_report["valid"] = False
    if not lengths_positive:
        print("  [X] Found zero-length samples")
        ds_report["valid"] = False

    bin_data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
    total_tokens = len(bin_data)
    print(f"  [V] Binary data loaded: {total_tokens:,} tokens")

    max_offset = offsets[-1] + lengths[-1]
    offsets_valid = max_offset <= total_tokens
    ds_report["integrity_checks"]["offsets_within_bounds"] = bool(offsets_valid)

    if not offsets_valid:
        print("  [X] Metadata offsets exceed binary data size")
        print(f"    Max offset: {max_offset:,}, Binary size: {total_tokens:,}")
        ds_report["valid"] = False
    else:
        print("  [V] Offsets within bounds")

    if not ds_report["valid"]:
        return ds_report

    # Statistics
    print(f"\n  {_ANSI_BOLD}Statistics:{_ANSI_RESET}")
    print(f"    Task type: {dataset_config.task_type}")
    print(f"    Total samples: {num_samples:,}")
    print(f"    Total tokens: {total_tokens:,}")
    print(f"    Avg tokens/sample: {total_tokens / num_samples:.1f}")
    print(f"    Min length: {lengths.min():,}")
    print(f"    Max length: {lengths.max():,}")
    print(f"    Median length: {int(np.median(lengths)):,}")

    ds_report["statistics"] = {
        "total_samples": int(num_samples),
        "total_tokens": int(total_tokens),
        "avg_tokens_per_sample": float(total_tokens / num_samples),
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "median_length": int(np.median(lengths)),
    }

    unique_types = np.unique(types)
    print(f"    Sample types: {', '.join(unique_types)}")
    ds_report["statistics"]["sample_types"] = unique_types.tolist()

    # Masking statistics for SFT/DPO
    if dataset_config.task_type in ("sft", "dpo"):
        total_masked = 0
        for mask_ranges_str in metadata["mask_ranges"]:
            try:
                mask_ranges = json.loads(mask_ranges_str) if mask_ranges_str else []
                total_masked += sum(end - start for start, end in mask_ranges)
            except json.JSONDecodeError:
                pass

        total_trained = total_tokens - total_masked
        mask_ratio = total_masked / total_tokens if total_tokens > 0 else 0

        print(f"    Trained tokens: {total_trained:,} ({(1 - mask_ratio) * 100:.1f}%)")
        print(f"    Masked tokens: {total_masked:,} ({mask_ratio * 100:.1f}%)")

        ds_report["statistics"].update(
            trained_tokens=int(total_trained),
            masked_tokens=int(total_masked),
            mask_ratio=float(mask_ratio),
        )

    # Packing efficiency for SFT
    if dataset_config.task_type == "sft":
        packing_efficiency = _calculate_packing_efficiency(metadata, data_config.seq_length)
        print(f"    Packing efficiency: {packing_efficiency * 100:.1f}%")

        num_bins = (
            int(np.ceil(num_samples / packing_efficiency))
            if packing_efficiency > 0
            else num_samples
        )
        print(f"    Estimated packed batches: ~{num_bins:,} (from {num_samples:,} samples)")

        ds_report["statistics"].update(
            packing_efficiency=float(packing_efficiency),
            estimated_packed_batches=int(num_bins),
        )

    # Visual preview
    if preview > 0:
        _print_visual_preview(bin_path, idx_path, dataset_config, data_config, num_samples=preview)

    return ds_report


def _calculate_packing_efficiency(metadata: np.ndarray, max_seq_len: int) -> float:
    """Estimate packing efficiency using First-Fit Decreasing.

    Uses negated remainders with bisect for O(n log n) instead of O(n²).
    Remainders are stored as negative values in a sorted list so that
    bisect finds the tightest-fitting bin (smallest sufficient remainder).
    """
    import bisect

    lengths = metadata["length"]
    sorted_lengths = sorted(lengths, reverse=True)

    # Store negated remainders so the list stays ascending:
    # [-5, -10, -20] means bins with 5, 10, 20 remaining capacity.
    # bisect_left(-length) finds the first bin with remainder >= length.
    neg_remainders: list[int] = []

    for length in sorted_lengths:
        target = -length
        idx = bisect.bisect_left(neg_remainders, target)
        if idx < len(neg_remainders):
            old = neg_remainders.pop(idx)
            new_rem = old + length  # more negative = less remaining
            bisect.insort(neg_remainders, new_rem)
        else:
            bisect.insort(neg_remainders, -(max_seq_len - length))

    num_bins = len(neg_remainders)
    total_tokens = int(lengths.sum()) if hasattr(lengths, "sum") else sum(lengths)
    total_capacity = num_bins * max_seq_len
    return total_tokens / total_capacity if total_capacity > 0 else 0.0


def _print_visual_preview(
    bin_path: Path, idx_path: Path, dataset_config, data_config, num_samples: int = 5
) -> None:
    """Print decoded samples with masked tokens highlighted."""
    print(f"\n  {_ANSI_BOLD}Visual Preview:{_ANSI_RESET}")
    print(
        f"  {_ANSI_GREEN}Green = Trained tokens{_ANSI_RESET}, "
        f"{_ANSI_RED}Red = Masked tokens (labels=-100){_ANSI_RESET}"
    )
    print()

    metadata = np.load(idx_path)
    bin_data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")

    try:
        tokenizer = _load_tokenizer(data_config.vocab_name_or_path, data_config.tokenizer_type)
    except Exception as e:
        print(f"  {_ANSI_YELLOW}WARNING: Could not load tokenizer: {e}{_ANSI_RESET}")
        return

    total = len(metadata)
    num_to_show = min(num_samples, total)
    sample_indices = (
        list(range(total)) if total <= num_samples else random.sample(range(total), num_to_show)
    )

    for idx in sample_indices:
        offset = metadata["offset"][idx]
        length = metadata["length"][idx]
        sample_type = metadata["type"][idx]
        mask_ranges_str = metadata["mask_ranges"][idx]

        try:
            mask_ranges = json.loads(mask_ranges_str) if mask_ranges_str else []
        except json.JSONDecodeError:
            mask_ranges = []

        token_ids = bin_data[offset : offset + length]
        num_masked = sum(end - start for start, end in mask_ranges)
        num_trained = length - num_masked

        decoded = _decode_with_mask_highlighting(
            token_ids, mask_ranges, tokenizer, data_config.tokenizer_type
        )

        print(f"  {_ANSI_BLUE}[Sample {idx}]{_ANSI_RESET}")
        print(
            f"    Type: {sample_type} | Length: {length} tokens | "
            f"Trained: {num_trained} | Masked: {num_masked}"
        )
        print(f"    Text: {decoded}")
        print()


def _load_tokenizer(vocab_name_or_path: str, tokenizer_type: str):
    """Load tokenizer for decoding."""
    if tokenizer_type == "bbpe":
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(vocab_name_or_path)
    elif tokenizer_type == "tiktoken":
        import tiktoken

        return tiktoken.get_encoding(vocab_name_or_path)
    raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def _decode_with_mask_highlighting(
    token_ids: np.ndarray, mask_ranges: list[tuple[int, int]], tokenizer, tokenizer_type: str
) -> str:
    """Decode tokens with ANSI color highlighting for masked regions."""
    is_masked = np.zeros(len(token_ids), dtype=bool)
    for start, end in mask_ranges:
        is_masked[start:end] = True

    output_parts = []
    current_color = None

    for i, token_id in enumerate(token_ids):
        token_text = tokenizer.decode([int(token_id)])
        desired_color = _ANSI_RED if is_masked[i] else _ANSI_GREEN

        if desired_color != current_color:
            if current_color is not None:
                output_parts.append(_ANSI_RESET)
            output_parts.append(desired_color)
            current_color = desired_color

        output_parts.append(token_text)

    if current_color is not None:
        output_parts.append(_ANSI_RESET)

    return "".join(output_parts)


def save_report(report_data: dict[str, Any], output_dir: Path) -> None:
    """Save inspection report as JSON and Markdown."""
    output_dir = Path(output_dir)

    json_path = output_dir / "inspection_report.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"\nSaved JSON report to: {json_path}")
    except Exception as e:
        print(f"{_ANSI_RED}[X] Failed to save JSON report: {e}{_ANSI_RESET}")

    md_path = output_dir / "inspection_report.md"
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Data Inspection Report\n\n")
            f.write(f"**Date:** {report_data['timestamp']}\n\n")

            f.write("## Summary\n\n")
            f.write("| Dataset | Task | Status | Samples | Tokens | Efficiency |\n")
            f.write("|---|---|---|---|---|---|\n")

            for ds in report_data["datasets"]:
                valid_status = "PASSED" if ds["valid"] else "FAILED"
                stats = ds.get("statistics", {})
                samples = stats.get("total_samples", "N/A")
                tokens = stats.get("total_tokens", "N/A")
                efficiency = stats.get("packing_efficiency", "N/A")
                if isinstance(efficiency, float):
                    efficiency = f"{efficiency * 100:.1f}%"
                if isinstance(samples, int):
                    samples = f"{samples:,}"
                if isinstance(tokens, int):
                    tokens = f"{tokens:,}"

                f.write(
                    f"| {ds['name']} | {ds['task_type']} | {valid_status} | "
                    f"{samples} | {tokens} | {efficiency} |\n"
                )

            f.write("\n## Detailed Report\n\n")
            for ds in report_data["datasets"]:
                f.write(f"### Dataset: {ds['name']}\n\n")
                if not ds["valid"]:
                    f.write("**Status:** FAILED\n")
                    f.write(f"**Error:** {ds.get('error', 'Unknown error')}\n\n")
                    continue

                stats = ds["statistics"]
                f.write(f"- **Task Type:** {ds['task_type']}\n")
                f.write(f"- **Total Samples:** {stats['total_samples']:,}\n")
                f.write(f"- **Total Tokens:** {stats['total_tokens']:,}\n")
                f.write(f"- **Avg Tokens/Sample:** {stats['avg_tokens_per_sample']:.1f}\n")
                f.write(
                    f"- **Length Stats:** Min={stats['min_length']:,}, "
                    f"Max={stats['max_length']:,}, Median={stats['median_length']:,}\n"
                )

                if "packing_efficiency" in stats:
                    f.write(f"- **Packing Efficiency:** {stats['packing_efficiency'] * 100:.1f}%\n")
                    f.write(f"- **Est. Batches:** {stats['estimated_packed_batches']:,}\n")

                if "masked_tokens" in stats:
                    mask_pct = stats["masked_tokens"] / stats["total_tokens"] * 100
                    f.write(f"- **Masked Tokens:** {stats['masked_tokens']:,} ({mask_pct:.1f}%)\n")

                f.write("\n")

        print(f"Saved Markdown report to: {md_path}")
    except Exception as e:
        print(f"{_ANSI_RED}[X] Failed to save Markdown report: {e}{_ANSI_RESET}")
