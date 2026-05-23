# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Training CLI command."""

import sys

from ironcore.train import load_full_config, train


def run_train(args):
    """Run training command.

    Args:
        args: Command-line arguments from argparse
            - config: Path to training configuration YAML file
    """
    try:
        config = load_full_config(args.config)
        train(config)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
