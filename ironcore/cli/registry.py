# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Subcommand registry — each CLI module registers its own parser.

Usage in ``__main__.py``::

    from ironcore.cli.registry import build_parser, dispatch

    parser, dispatch = build_parser()
    args = parser.parse_args()
    dispatch(args.command)(args)
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

# (command_name, module_path, help_text)
# Kept in import order so ``--help`` listing is deterministic.
_COMMANDS: list[tuple[str, str, str]] = [
    ("preprocess", "ironcore.cli.preprocess", "Preprocess and/or inspect datasets"),
    ("train", "ironcore.cli.train", "Run training"),
    ("track", "ironcore.cli.track", "Configure logging backends"),
    ("evaluate", "ironcore.cli.evaluate", "Run evaluation benchmarks"),
    ("gen-report", "ironcore.cli.gen_report", "Generate experiment reports"),
    ("profile", "ironcore.cli.profile", "Profile training runs"),
    ("verify-parity", "ironcore.cli.verify_parity", "Verify parallelism correctness"),
    ("verify-step", "ironcore.cli.verify_step", "Verify single-step training loss"),
    ("analyze-scaling", "ironcore.cli.analyze_scaling", "Run scaling analysis"),
    ("profile-mfu", "ironcore.cli.profile_mfu", "Profile Model FLOP Utilization"),
    ("config-check", "ironcore.cli.config_check", "Validate and inspect training configs"),
    ("tokenize", "ironcore.cli.tokenize", "Tokenize input and show statistics"),
    ("inspect-checkpoint", "ironcore.cli.inspect_checkpoint", "Inspect checkpoint contents"),
    ("export", "ironcore.cli.export", "Export checkpoint to HuggingFace format"),
    ("generate", "ironcore.cli.generate", "Generate text from a checkpoint"),
]


def build_parser():
    """Build the root argparse parser with all subcommands registered.

    Returns:
        Tuple of (parser, dispatch_dict) where dispatch_dict maps
        command names to ``run_*`` callables.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="ironcore",
        description="IronCore: High-Performance Research Platform for LLM Training",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    dispatch: dict[str, Callable] = {}

    for cmd_name, module_path, help_text in _COMMANDS:
        module = importlib.import_module(module_path)
        register_fn = getattr(module, "register_parser", None)
        if register_fn is not None:
            register_fn(subparsers)
        else:
            # Fallback: register a minimal parser
            subparsers.add_parser(cmd_name, help=help_text)

        # Convention: run_{command} where hyphens become underscores
        run_name = f"run_{cmd_name.replace('-', '_')}"
        dispatch[cmd_name] = getattr(module, run_name)

    return parser, dispatch
