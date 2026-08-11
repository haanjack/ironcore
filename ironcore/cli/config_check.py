# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Config validation and inspection."""

import sys
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import yaml


def register_parser(subparsers) -> None:
    """Register the CLI subcommand arguments."""
    parser = subparsers.add_parser("config-check", help="Validate and inspect training configs")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--diff", type=str, default=None, help="Second config to compare against")
    parser.add_argument("--show", action="store_true", help="Print resolved config as YAML")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, no output")


def run_config_check(args: Namespace) -> None:
    """Validate, diff, and inspect training configs.

    Args:
        args: Command-line arguments.
    """
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}")
        sys.exit(1)

    from ironcore.train import load_full_config

    # Load the config defensively. load_full_config runs _config_validation,
    # which raises ValueError on any hard failure. A config-check command
    # must REPORT those failures, not crash on them. (Fable issue #76.)
    try:
        config = load_full_config(config_path)
        checks = [{"name": "config_validation", "passed": True}]
    except ValueError as e:
        config = None
        checks = [
            {
                "name": "config_validation",
                "passed": False,
                "message": str(e),
            }
        ]
    except Exception as e:  # noqa: BLE001 — config-check must never crash
        config = None
        checks = [
            {
                "name": "config_loading",
                "passed": False,
                "message": f"{type(e).__name__}: {e}",
            }
        ]

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

    # Diff mode — requires a loaded config; skip if loading failed.
    if args.diff and config is not None:
        _print_config_diff(config, args.diff)

    # Show mode — requires a loaded config; skip if loading failed.
    if args.show and config is not None:
        print("\nResolved Config:")
        print("-" * 40)
        config_dict = asdict(config)
        print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))

    if not all(c["passed"] for c in checks):
        sys.exit(1)


def _print_config_diff(config_a, diff_path: str) -> None:
    """Print differences between two configs."""
    diff_config = Path(diff_path)
    if not diff_config.exists():
        print(f"\nError: diff config not found: {diff_config}")
        return

    from ironcore.train import load_full_config

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
