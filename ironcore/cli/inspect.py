"""Dataset inspection CLI command."""

import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

from ironcore.dataloader.data_config import DataConfig


# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'      # Masked tokens (labels = -100)
    GREEN = '\033[92m'    # Trained tokens (labels = token_id)
    BLUE = '\033[94m'     # Metadata info
    YELLOW = '\033[93m'   # Warnings
    RESET = '\033[0m'     # Reset color
    BOLD = '\033[1m'


def _load_tokenizer(vocab_name_or_path: str, tokenizer_type: str):
    """Load tokenizer for decoding."""
    if tokenizer_type == "bbpe":
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(vocab_name_or_path)
    elif tokenizer_type == "tiktoken":
        import tiktoken
        return tiktoken.get_encoding(vocab_name_or_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def _decode_with_mask_highlighting(
    token_ids: np.ndarray,
    mask_ranges: List[Tuple[int, int]],
    tokenizer,
    tokenizer_type: str
) -> str:
    """Decode tokens and highlight masked regions.

    Args:
        token_ids: Array of token IDs
        mask_ranges: List of (start, end) ranges that are masked
        tokenizer: Tokenizer instance
        tokenizer_type: Type of tokenizer ("bbpe" or "tiktoken")

    Returns:
        String with ANSI color codes highlighting masked regions
    """
    # Create mask array
    is_masked = np.zeros(len(token_ids), dtype=bool)
    for start, end in mask_ranges:
        is_masked[start:end] = True

    # Decode tokens individually with color coding
    output_parts = []
    current_color = None

    for i, token_id in enumerate(token_ids):
        # Decode single token
        if tokenizer_type == "bbpe":
            token_text = tokenizer.decode([int(token_id)])
        else:  # tiktoken
            token_text = tokenizer.decode([int(token_id)])

        # Determine color
        desired_color = Colors.RED if is_masked[i] else Colors.GREEN

        # Change color if needed
        if desired_color != current_color:
            if current_color is not None:
                output_parts.append(Colors.RESET)
            output_parts.append(desired_color)
            current_color = desired_color

        output_parts.append(token_text)

    # Reset color at end
    if current_color is not None:
        output_parts.append(Colors.RESET)

    return ''.join(output_parts)


def _print_visual_preview(
    bin_path: Path,
    idx_path: Path,
    dataset_config,
    data_config,
    num_samples: int = 5
):
    """Print visual preview of decoded samples with masked tokens highlighted.

    Args:
        bin_path: Path to .bin file
        idx_path: Path to .idx file
        dataset_config: Dataset configuration
        data_config: Data configuration (for tokenizer)
        num_samples: Number of samples to preview
    """
    print(f"\n  {Colors.BOLD}Visual Preview:{Colors.RESET}")
    print(f"  {Colors.GREEN}Green = Trained tokens{Colors.RESET}, {Colors.RED}Red = Masked tokens (labels=-100){Colors.RESET}")
    print()

    # Load metadata and data
    metadata = np.load(idx_path)
    bin_data = np.memmap(str(bin_path), dtype=np.uint16, mode='r')

    # Load tokenizer
    try:
        tokenizer = _load_tokenizer(
            data_config.vocab_name_or_path,
            data_config.tokenizer_type
        )
    except Exception as e:
        print(f"  {Colors.YELLOW}WARNING: Could not load tokenizer: {e}{Colors.RESET}")
        print(f"  {Colors.YELLOW}Skipping visual preview{Colors.RESET}")
        return

    # Sample random indices
    total_samples = len(metadata)
    num_to_show = min(num_samples, total_samples)

    if total_samples <= num_samples:
        sample_indices = list(range(total_samples))
    else:
        sample_indices = random.sample(range(total_samples), num_to_show)

    # Display each sample
    for idx in sample_indices:
        offset = metadata['offset'][idx]
        length = metadata['length'][idx]
        sample_type = metadata['type'][idx]
        mask_ranges_str = metadata['mask_ranges'][idx]

        # Parse mask ranges
        try:
            mask_ranges = json.loads(mask_ranges_str) if mask_ranges_str else []
        except:
            mask_ranges = []

        # Extract tokens
        token_ids = bin_data[offset:offset+length]

        # Calculate statistics
        num_masked = sum(end - start for start, end in mask_ranges)
        num_trained = length - num_masked

        # Decode with highlighting
        decoded_text = _decode_with_mask_highlighting(
            token_ids,
            mask_ranges,
            tokenizer,
            data_config.tokenizer_type
        )

        # Print sample header
        print(f"  {Colors.BLUE}[Sample {idx}]{Colors.RESET}")
        print(f"    Type: {sample_type} | Length: {length} tokens | Trained: {num_trained} | Masked: {num_masked}")
        print(f"    Text: {decoded_text}")
        print()


def _calculate_packing_efficiency(metadata: np.ndarray, max_seq_len: int) -> float:
    """Calculate packing efficiency for SFT datasets.

    Simulates First-Fit Decreasing packing to estimate efficiency.

    Args:
        metadata: Metadata array
        max_seq_len: Maximum sequence length

    Returns:
        Packing efficiency as a fraction (0.0 to 1.0)
    """
    lengths = metadata['length']

    # Sort by length descending (First-Fit Decreasing)
    sorted_lengths = sorted(lengths, reverse=True)

    # Simulate packing
    bins = []
    for length in sorted_lengths:
        # Try to fit in existing bin
        placed = False
        for bin_items in bins:
            current_fill = sum(bin_items)
            if current_fill + length <= max_seq_len:
                bin_items.append(length)
                placed = True
                break

        # Create new bin if needed
        if not placed:
            bins.append([length])

    # Calculate efficiency
    total_tokens = sum(lengths)
    total_capacity = len(bins) * max_seq_len
    efficiency = total_tokens / total_capacity if total_capacity > 0 else 0.0

    return efficiency


def _save_report(report_data: Dict[str, Any], output_dir: Path):
    """Save inspection report to JSON and Markdown files.

    Args:
        report_data: Dictionary containing inspection data
        output_dir: Directory to save reports
    """
    # 1. Save JSON
    json_path = output_dir / "inspection_report.json"
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"\nSaved JSON report to: {json_path}")
    except Exception as e:
        print(f"{Colors.RED}[X] Failed to save JSON report: {e}{Colors.RESET}")

    # 2. Save Markdown
    md_path = output_dir / "inspection_report.md"
    try:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Data Inspection Report\n\n")
            f.write(f"**Date:** {report_data['timestamp']}\n\n")
            
            f.write("## Summary\n\n")
            f.write("| Dataset | Task | Status | Samples | Tokens | Efficiency |\n")
            f.write("|---|---|---|---|---|---|\n")
            
            for ds in report_data['datasets']:
                valid_status = "PASSED" if ds['valid'] else "FAILED"
                stats = ds.get('statistics', {})
                samples = stats.get('total_samples', 'N/A')
                tokens = stats.get('total_tokens', 'N/A')
                efficiency = stats.get('packing_efficiency', 'N/A')
                if isinstance(efficiency, float):
                    efficiency = f"{efficiency*100:.1f}%"
                
                # Format numbers
                if isinstance(samples, int): samples = f"{samples:,}"
                if isinstance(tokens, int): tokens = f"{tokens:,}"
                
                f.write(f"| {ds['name']} | {ds['task_type']} | {valid_status} | {samples} | {tokens} | {efficiency} |\n")
            
            f.write("\n## Detailed Report\n\n")
            
            for ds in report_data['datasets']:
                f.write(f"### Dataset: {ds['name']}\n\n")
                if not ds['valid']:
                    f.write(f"**Status:** FAILED\n")
                    f.write(f"**Error:** {ds.get('error', 'Unknown error')}\n\n")
                    continue
                
                stats = ds['statistics']
                f.write(f"- **Task Type:** {ds['task_type']}\n")
                f.write(f"- **Total Samples:** {stats['total_samples']:,}\n")
                f.write(f"- **Total Tokens:** {stats['total_tokens']:,}\n")
                f.write(f"- **Avg Tokens/Sample:** {stats['avg_tokens_per_sample']:.1f}\n")
                f.write(f"- **Length Stats:** Min={stats['min_length']:,}, Max={stats['max_length']:,}, Median={stats['median_length']:,}\n")
                
                if 'packing_efficiency' in stats:
                    f.write(f"- **Packing Efficiency:** {stats['packing_efficiency']*100:.1f}%\n")
                    f.write(f"- **Est. Batches:** {stats['estimated_packed_batches']:,}\n")
                
                if 'masked_tokens' in stats:
                    mask_pct = stats['masked_tokens'] / stats['total_tokens'] * 100
                    f.write(f"- **Masked Tokens:** {stats['masked_tokens']:,} ({mask_pct:.1f}%)\n")
                
                f.write("\n")
                
        print(f"Saved Markdown report to: {md_path}")
    except Exception as e:
        print(f"{Colors.RED}[X] Failed to save Markdown report: {e}{Colors.RESET}")


def run_inspect(args):
    """Run dataset inspection command.

    Args:
        args: Command-line arguments from argparse
            - config: Path to data configuration YAML file
            - preview: Number of samples to preview (default: 0, no preview)
    """
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    print(f"Loading configuration from: {config_path}")
    data_config = DataConfig.from_yaml(config_path)

    preprocessed_dir = Path(data_config.preprocessed_dir)
    if not preprocessed_dir.exists():
        print(f"Error: Preprocessed directory not found: {preprocessed_dir}")
        sys.exit(1)

    print(f"\nInspecting datasets in: {preprocessed_dir}")
    print("=" * 80)

    all_valid = True
    num_preview = getattr(args, 'preview', 0)
    
    # Initialize report data
    report_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(config_path),
        "preprocessed_dir": str(preprocessed_dir),
        "datasets": []
    }

    for dataset_config in data_config.datasets:
        dataset_name = dataset_config.name
        print(f"\n{Colors.BOLD}Dataset: {dataset_name}{Colors.RESET}")
        print("-" * 80)
        
        ds_report = {
            "name": dataset_name,
            "task_type": dataset_config.task_type,
            "valid": True,
            "integrity_checks": {},
            "statistics": {}
        }

        # Use same path logic as serializer
        output_path = data_config.get_dataset_output_path(dataset_config)
        bin_path = output_path / "data.bin"
        idx_path = output_path / "data.idx.npy"

        # ========================================
        # 1. INTEGRITY CHECK
        # ========================================
        print(f"\n  {Colors.BOLD}Integrity Check:{Colors.RESET}")

        # Check file existence
        ds_report["integrity_checks"]["bin_exists"] = bin_path.exists()
        ds_report["integrity_checks"]["idx_exists"] = idx_path.exists()

        if not bin_path.exists():
            print(f"  [X] Missing .bin file: {bin_path}")
            ds_report["valid"] = False
            ds_report["error"] = "Missing .bin file"
            all_valid = False
            report_data["datasets"].append(ds_report)
            continue

        if not idx_path.exists():
            print(f"  [X] Missing .idx file: {idx_path}")
            ds_report["valid"] = False
            ds_report["error"] = "Missing .idx file"
            all_valid = False
            report_data["datasets"].append(ds_report)
            continue

        print(f"  [V] Files exist")

        # Load and validate metadata
        try:
            metadata = np.load(idx_path)
            num_samples = len(metadata)
            print(f"  [V] Metadata loaded: {num_samples:,} samples")

            # Check metadata structure
            expected_fields = {'offset', 'length', 'type', 'group_id', 'mask_ranges'}
            actual_fields = set(metadata.dtype.names)
            
            ds_report["integrity_checks"]["metadata_fields_valid"] = (actual_fields == expected_fields)

            if actual_fields != expected_fields:
                print(f"  [X] Metadata fields mismatch")
                print(f"    Expected: {expected_fields}")
                print(f"    Found: {actual_fields}")
                ds_report["valid"] = False
                ds_report["error"] = "Metadata fields mismatch"
                all_valid = False
                report_data["datasets"].append(ds_report)
                continue

            print(f"  [V] Metadata structure valid")

            # Validate metadata values
            offsets = metadata['offset']
            lengths = metadata['length']
            types = metadata['type']
            
            offsets_monotonic = np.all(offsets[1:] >= offsets[:-1])
            lengths_positive = np.all(lengths > 0)
            
            ds_report["integrity_checks"]["offsets_monotonic"] = bool(offsets_monotonic)
            ds_report["integrity_checks"]["lengths_positive"] = bool(lengths_positive)

            if not offsets_monotonic:
                print(f"  [X] Offsets are not monotonic")
                ds_report["valid"] = False
                all_valid = False

            if not lengths_positive:
                print(f"  [X] Found zero-length samples")
                ds_report["valid"] = False
                all_valid = False

            # Load binary data
            bin_data = np.memmap(str(bin_path), dtype=np.uint16, mode='r')
            total_tokens = len(bin_data)
            print(f"  [V] Binary data loaded: {total_tokens:,} tokens")

            # Validate offsets match binary size
            max_offset = offsets[-1] + lengths[-1]
            offsets_valid = max_offset <= total_tokens
            ds_report["integrity_checks"]["offsets_within_bounds"] = bool(offsets_valid)
            
            if not offsets_valid:
                print(f"  [X] Metadata offsets exceed binary data size")
                print(f"    Max offset: {max_offset:,}, Binary size: {total_tokens:,}")
                ds_report["valid"] = False
                all_valid = False
            else:
                print(f"  [V] Offsets within bounds")
            
            if not ds_report["valid"]:
                 report_data["datasets"].append(ds_report)
                 continue

            # ========================================
            # 2. STATISTICS
            # ========================================
            print(f"\n  {Colors.BOLD}Statistics:{Colors.RESET}")
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
                "median_length": int(np.median(lengths))
            }

            # Type distribution
            unique_types = np.unique(types)
            print(f"    Sample types: {', '.join(unique_types)}")
            ds_report["statistics"]["sample_types"] = unique_types.tolist()

            # Calculate masking statistics for SFT/DPO
            if dataset_config.task_type in ['sft', 'dpo']:
                total_masked = 0
                for mask_ranges_str in metadata['mask_ranges']:
                    try:
                        mask_ranges = json.loads(mask_ranges_str) if mask_ranges_str else []
                        total_masked += sum(end - start for start, end in mask_ranges)
                    except:
                        pass

                total_trained = total_tokens - total_masked
                mask_ratio = total_masked / total_tokens if total_tokens > 0 else 0

                print(f"    Trained tokens: {total_trained:,} ({(1-mask_ratio)*100:.1f}%)")
                print(f"    Masked tokens: {total_masked:,} ({mask_ratio*100:.1f}%)")
                
                ds_report["statistics"]["trained_tokens"] = int(total_trained)
                ds_report["statistics"]["masked_tokens"] = int(total_masked)
                ds_report["statistics"]["mask_ratio"] = float(mask_ratio)

            # Calculate packing efficiency for SFT
            if dataset_config.task_type == 'sft':
                packing_efficiency = _calculate_packing_efficiency(metadata, data_config.seq_length)
                print(f"    Packing efficiency: {packing_efficiency*100:.1f}%")

                # Estimate number of batches
                num_bins = int(np.ceil(num_samples / packing_efficiency)) if packing_efficiency > 0 else num_samples
                print(f"    Estimated packed batches: ~{num_bins:,} (from {num_samples:,} samples)")
                
                ds_report["statistics"]["packing_efficiency"] = float(packing_efficiency)
                ds_report["statistics"]["estimated_packed_batches"] = int(num_bins)

            # ========================================
            # 3. VISUAL PREVIEW
            # ========================================
            if num_preview > 0:
                _print_visual_preview(
                    bin_path,
                    idx_path,
                    dataset_config,
                    data_config,
                    num_samples=num_preview
                )
            
            # Add to report data
            report_data["datasets"].append(ds_report)

        except Exception as e:
            print(f"  [X] Error inspecting dataset: {e}")
            import traceback
            traceback.print_exc()
            all_valid = False
            ds_report["valid"] = False
            ds_report["error"] = str(e)
            report_data["datasets"].append(ds_report)

    # Save reports
    _save_report(report_data, preprocessed_dir)

    print("\n" + "=" * 80)
    if all_valid:
        print(f"{Colors.GREEN}[V] All datasets passed inspection!{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}[X] Inspection failed for one or more datasets{Colors.RESET}")
        return 1
