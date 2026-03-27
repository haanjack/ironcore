#!/usr/bin/env python3
"""
CLI entry point for the exploration phase of AI-driven kernel generation.

Usage:
    python -m experiments.generation.exploration.explorer_cli rmsnorm --provider glm
    python -m experiments.generation.exploration.explorer_cli layernorm --provider glm --debug
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

from experiments.generation.spec import get_spec, list_specs
from experiments.generation.exploration import KernelExplorer

# Import all specs to register them
from experiments.generation.specs import rmsnorm, layernorm, softmax, glu, cross_entropy  # noqa: F401


def main():
    parser = argparse.ArgumentParser(
        description="Exploration Phase for AI-Driven Triton Kernel Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s rmsnorm                    # Explore rmsnorm kernel
  %(prog)s layernorm --provider glm   # Use GLM-4.7 for analysis
  %(prog)s softmax --debug             # Enable debug output
  %(prog)s --list                      # List available kernels
        """
    )

    parser.add_argument("kernel", nargs="?", help="Kernel spec name to explore")
    parser.add_argument("--provider", "-p", default="glm", help="AI provider for analysis")
    parser.add_argument("--model", "-m", help="Model name (uses provider default if not specified)")
    parser.add_argument("--api-key", help="API key (overrides environment)")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug output")
    parser.add_argument("--list", action="store_true", help="List available kernel specs")

    args = parser.parse_args()

    if args.list:
        print("Available kernel specs for exploration:")
        for name in list_specs():
            print(f"  - {name}")
        return 0

    if not args.kernel:
        parser.error("kernel spec name is required (or use --list to see available specs)")

    # Get spec
    try:
        spec = get_spec(args.kernel)
    except KeyError as e:
        print(f"Error: {e}")
        return 1

    # Run exploration
    explorer = KernelExplorer(
        provider_name=args.provider,
        model=args.model,
        api_key=args.api_key,
    )

    try:
        result = explorer.explore(spec)

        # Print summary
        print(f"\n{'='*70}")
        print("EXPLORATION SUMMARY")
        print(f"{'='*70}")
        print(f"  Kernel: {result.spec_name}")
        print(f"  Timestamp: {result.timestamp}")
        print(f"  Total Tokens: {result.total_tokens_used}")
        print(f"  Code Generated: {len(result.initial_code)} characters")

        if result.graph_analysis:
            print(f"\n  Graph Analysis:")
            print(f"    Operations: {len(result.graph_analysis.operations)}")
            print(f"    Reductions: {result.graph_analysis.reductions}")
            print(f"    Can parallelize rows: {result.graph_analysis.can_parallelize_rows}")

        if result.tiling_strategy:
            print(f"\n  Tiling Strategy:")
            print(f"    Block size: {result.tiling_strategy.recommended_block_size}")
            print(f"    Register pressure: {result.tiling_strategy.register_pressure}")

        if result.conversion_plan:
            print(f"\n  Conversion Plan:")
            print(f"    Passes: {len(result.conversion_plan.passes)}")
            for i, p in enumerate(result.conversion_plan.passes):
                print(f"      {i+1}. {p.get('name', 'unnamed')} ({p.get('type', 'unknown')})")

        if result.compilation_errors:
            print(f"\n  Errors Encountered: {len(result.compilation_errors)}")
            for err in result.compilation_errors[:3]:
                print(f"    - {err[:100]}...")

        print(f"\n{'='*70}")
        return 0

    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error during exploration: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
