# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""CLI wrapper for dataset inspection — delegates to ironcore.preprocessing.inspect."""

import sys
from argparse import Namespace
from pathlib import Path


def run_inspect(args: Namespace) -> None:
    """Inspect preprocessed datasets via :func:`ironcore.preprocessing.inspect.inspect_dataset`."""
    from ironcore.preprocessing.inspect import inspect_dataset, save_report

    try:
        report = inspect_dataset(args.config, preview=getattr(args, "preview", 0))
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    preprocessed_dir = Path(report["preprocessed_dir"])
    save_report(report, preprocessed_dir)

    print("\n" + "=" * 80)
    all_valid = all(ds["valid"] for ds in report["datasets"])
    if all_valid:
        print("\033[92m[V] All datasets passed inspection!\033[0m")
    else:
        print("\033[91m[X] Inspection failed for one or more datasets\033[0m")
        sys.exit(1)
