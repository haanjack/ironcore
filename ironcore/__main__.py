# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
CLI entry point for IronCore.

IronCore: High-Performance Research Platform for LLM Training

Supports subcommands:
    - preprocess: Preprocess and/or inspect datasets
    - train: Run training
    - generate: Generate DSL kernels using AI (single iteration)
    - explore: Exploration phase with multi-stage AI analysis
    - auto: Autonomous AI-driven kernel generation and optimization (default for production)
"""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ironcore", description="IronCore: High-Performance Research Platform for LLM Training"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ========================================
    # Subcommand: preprocess
    # ========================================
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Preprocess and/or inspect datasets"
    )
    preprocess_parser.add_argument(
        "--config", type=str, required=True, help="Path to data configuration YAML file"
    )
    preprocess_parser.add_argument(
        "--inspect",
        action="store_true",
        help="Run inspection (integrity checks, statistics, packing efficiency) after preprocessing",
    )
    preprocess_parser.add_argument(
        "--only-inspect",
        action="store_true",
        help="Skip preprocessing and only run inspection on existing files",
    )
    preprocess_parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Number of random samples to preview during inspection (implies --inspect)",
    )

    # ========================================
    # Subcommand: train
    # ========================================
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument(
        "--config", type=str, required=True, help="Path to training configuration YAML file"
    )

    # ========================================
    # Subcommand: generate
    # ========================================
    from ironcore.cli.generate import add_generate_subcommand
    add_generate_subcommand(subparsers)

    # ========================================
    # Subcommand: auto (autonomous generation)
    # ========================================
    auto_parser = subparsers.add_parser(
        "auto",
        help="Autonomous AI-driven kernel generation (production mode)",
        description="Automated kernel generation with AI analysis and optimization loops. "
                    "This is the default mode for production use."
    )
    auto_parser.add_argument(
        "kernel",
        help="Kernel spec name to generate"
    )
    auto_parser.add_argument(
        "--provider", "-p",
        default="glm",
        help="AI provider (default: glm)"
    )
    auto_parser.add_argument(
        "--model", "-m",
        help="Model name (uses provider default if not specified)"
    )
    auto_parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum optimization iterations (default: 5)"
    )
    auto_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce verbosity"
    )
    auto_parser.set_defaults(func=lambda args: _run_auto(args))

    # ========================================
    # Subcommand: explore (exploration phase)
    # ========================================
    explore_parser = subparsers.add_parser(
        "explore",
        help="Exploration phase: multi-stage AI kernel analysis and generation",
        description="Run exploration phase with 7 stages: algorithm analysis, graph analysis, "
                    "tiling strategy, conversion plan, code structure, initial implementation, "
                    "and diagnostic refinement."
    )
    explore_parser.add_argument(
        "kernel",
        help="Kernel spec name to explore"
    )
    explore_parser.add_argument(
        "--provider", "-p",
        default="glm",
        help="AI provider (default: glm)"
    )
    explore_parser.add_argument(
        "--model", "-m",
        help="Model name (uses provider default if not specified)"
    )
    explore_parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug output"
    )
    explore_parser.set_defaults(func=lambda args: _run_explore(args))

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "preprocess":
        from ironcore.cli.preprocess import run_preprocess

        run_preprocess(args)
    elif args.command == "train":
        from ironcore.cli.train import run_train

        run_train(args)
    elif args.command == "auto":
        _run_auto(args)
    elif args.command == "explore":
        _run_explore(args)
    elif hasattr(args, "func"):
        # Commands using the func pattern (e.g., generate)
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


def _run_auto(args):
    """Run autonomous kernel generation."""
    import os
    from experiments.generation.harness_autonomous import main as auto_main

    # Map args to sys.argv for the autonomous harness
    sys.argv = [
        "ironcore",
        args.kernel,
        "--provider", args.provider,
    ]

    if args.model:
        sys.argv.extend(["--model", args.model])
    if args.max_iterations != 5:
        sys.argv.extend(["--max-iterations", str(args.max_iterations)])
    if args.quiet:
        sys.argv.append("--quiet")

    # Get API key from environment based on provider
    from experiments.generation.ai_providers import resolve_provider_alias
    provider, base_url, _ = resolve_provider_alias(args.provider)

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")

    if api_key:
        sys.argv.extend(["--api-key", api_key])

    if base_url:
        sys.argv.extend(["--base-url", base_url])

    auto_main()


def _run_explore(args):
    """Run exploration phase for kernel generation."""
    import os
    from experiments.generation.exploration import KernelExplorer
    from experiments.generation.spec import get_spec

    # Get API key from environment based on provider
    from experiments.generation.ai_providers import resolve_provider_alias
    provider, base_url, _ = resolve_provider_alias(args.provider)

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")

    # Get spec
    try:
        spec = get_spec(args.kernel)
    except KeyError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Run exploration
    explorer = KernelExplorer(
        provider_name=args.provider,
        model=args.model,
        api_key=api_key,
    )

    try:
        result = explorer.explore(spec)

        # Print summary
        print(f"\n{'='*70}")
        print("EXPLORATION COMPLETE")
        print(f"{'='*70}")

        if result.refined_code:
            print(f"  Final code: {len(result.refined_code)} characters")
        if result.compilation_errors:
            print(f"  Errors: {len(result.compilation_errors)}")

        return 0

    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    main()
