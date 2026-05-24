# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for IronCore."""

import sys

from ironcore.cli.registry import build_parser


def main():
    """Main CLI entry point."""
    parser, dispatch = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
