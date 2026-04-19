# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Config validation and inspection."""

import sys
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import yaml

from .utils import load_full_config


def run_config_check(args: Namespace) -> None:
    """Validate, diff, and inspect training configs.

    Args:
        args: Command-line arguments.
    """
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}")
        sys.exit(1)

    config = load_full_config(config_path)

    # Run individual validation checks
    checks = _run_validation_checks(config)

    # Print results
    if not args.validate_only:
        print(f"Config: {config_path}")
        print()
        for check in checks:
            symbol = "[PASS]" if check["passed"] else "[FAIL]"
            msg = f" — {check['message']}" if check.get("message") else ""
            print(f"  {symbol} {check['name']}{msg}")
        print()
        all_passed = all(c["passed"] for c in checks)
        print(f"Overall: {'PASS' if all_passed else 'FAIL'}")

    # Diff mode
    if args.diff:
        _print_config_diff(config, args.diff)

    # Show mode
    if args.show:
        print("\nResolved Config:")
        print("-" * 40)
        config_dict = asdict(config)
        print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))

    if not all(c["passed"] for c in checks):
        sys.exit(1)


def _run_validation_checks(config) -> list[dict]:
    """Run each validation check individually, collecting results."""
    checks = []

    # Check 1: train_steps > 0
    try:
        if config.operation.train_steps <= 0:
            raise ValueError("train_steps must be > 0")
        checks.append({"name": "train_steps > 0", "passed": True})
    except ValueError as e:
        checks.append({"name": "train_steps > 0", "passed": False, "message": str(e)})

    # Check 2: world_size >= tp_size
    try:
        dp_group_size = config.trainer.tensor_model_parallel_size
        dp_world_size = config.parallel.world_size // dp_group_size
        if dp_world_size <= 0:
            raise ValueError(
                f"world_size ({config.parallel.world_size}) < tp_size ({dp_group_size})"
            )
        checks.append({"name": "world_size >= tp_size", "passed": True})
    except ValueError as e:
        checks.append({"name": "world_size >= tp_size", "passed": False, "message": str(e)})

    # Check 3: batch size consistency
    try:
        batch_fields = [
            config.trainer.micro_batch_size,
            config.trainer.train_batch_size,
            config.trainer.gradient_accumulation_steps,
        ]
        none_count = batch_fields.count(None)
        if none_count > 1:
            raise ValueError("At most one of micro_batch/train_batch/grad_accum can be None")
        if none_count == 0:
            dp_world_size = config.parallel.world_size // config.trainer.tensor_model_parallel_size
            expected = (
                config.trainer.micro_batch_size
                * config.trainer.gradient_accumulation_steps
                * dp_world_size
            )
            if expected != config.trainer.train_batch_size:
                raise ValueError(
                    f"micro_batch({config.trainer.micro_batch_size}) * grad_accum({config.trainer.gradient_accumulation_steps}) * dp({dp_world_size}) = {expected}, but train_batch = {config.trainer.train_batch_size}"
                )
        checks.append({"name": "batch size consistency", "passed": True})
    except ValueError as e:
        checks.append({"name": "batch size consistency", "passed": False, "message": str(e)})

    # Check 4: TP divisibility
    try:
        tp = config.trainer.tensor_model_parallel_size
        if tp > 1 and config.model.name != "dummy":
            if config.model.num_attention_heads % tp != 0:
                raise ValueError(
                    f"num_attention_heads ({config.model.num_attention_heads}) not divisible by tp ({tp})"
                )
            if config.model.num_attention_groups % tp != 0:
                raise ValueError(
                    f"num_attention_groups ({config.model.num_attention_groups}) not divisible by tp ({tp})"
                )
        checks.append({"name": "TP head divisibility", "passed": True})
    except ValueError as e:
        checks.append({"name": "TP head divisibility", "passed": False, "message": str(e)})

    # Check 5: positional embedding type
    try:
        valid = ["absolute", "rope", "none"]
        if config.model.positional_embedding.type.lower() not in valid:
            raise ValueError(
                f"positional_embedding must be one of {valid}, got '{config.model.positional_embedding.type}'"
            )
        checks.append({"name": "positional embedding type", "passed": True})
    except ValueError as e:
        checks.append({"name": "positional embedding type", "passed": False, "message": str(e)})

    # Check 6: distributed optimizer vs FSDP
    try:
        if config.parallel.use_distributed_optimizer and config.parallel.use_fsdp:
            raise ValueError("use_distributed_optimizer is incompatible with FSDP")
        checks.append({"name": "optimizer/FSDP compatibility", "passed": True})
    except ValueError as e:
        checks.append({"name": "optimizer/FSDP compatibility", "passed": False, "message": str(e)})

    return checks


def _print_config_diff(config_a, diff_path: str) -> None:
    """Print differences between two configs."""
    diff_config = Path(diff_path)
    if not diff_config.exists():
        print(f"\nError: diff config not found: {diff_config}")
        return

    config_b = load_full_config(diff_config)
    dict_a = asdict(config_a)
    dict_b = asdict(config_b)
    diffs = _find_diffs(dict_a, dict_b)

    if not diffs:
        print("\nConfigs are identical.")
        return

    print(f"\nDifferences ({diff_path}):")
    for path, val_a, val_b in diffs:
        print(f"  {path}: {val_a} -> {val_b}")


def _find_diffs(dict_a, dict_b, prefix="") -> list[tuple]:
    """Recursively find differences between two nested dicts."""
    diffs = []
    all_keys = set(dict_a.keys()) | set(dict_b.keys())
    for key in sorted(all_keys):
        path = f"{prefix}.{key}" if prefix else key
        val_a = dict_a.get(key, "<missing>")
        val_b = dict_b.get(key, "<missing>")
        if isinstance(val_a, dict) and isinstance(val_b, dict):
            diffs.extend(_find_diffs(val_a, val_b, path))
        elif val_a != val_b:
            diffs.append((path, val_a, val_b))
    return diffs
