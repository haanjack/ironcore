"""Generate DSL kernels using AI.

This module provides the CLI command for automated kernel generation
using AI providers (Claude, OpenAI, GLM, Kimi, vLLM, etc.).

Usage:
    ironcore generate rmsnorm
    ironcore generate --all --provider anthropic
    ironcore generate rmsnorm --provider glm
    ironcore generate rmsnorm --provider kimi
"""

import argparse
import os
import sys
from pathlib import Path

# Import spec modules to trigger registration
import experiments.generation.specs.rmsnorm  # noqa: F401
import experiments.generation.specs.layernorm  # noqa: F401
import experiments.generation.specs.softmax  # noqa: F401

from experiments.generation.spec import list_specs
from experiments.generation.pipeline import GenerationPipeline, PipelineConfig
from experiments.generation.ai_providers import resolve_provider_alias


# Default example kernel path for reference
EXAMPLE_KERNEL_PATH = Path(__file__).parent.parent / "kernels/triton/rmsnorm.py"


def run_generate(args):
    """Run the generate command.

    Args:
        args: Parsed CLI arguments
    """
    # Resolve provider alias and determine API key env var
    provider_name, base_url_from_alias, default_model = resolve_provider_alias(args.provider)

    # Determine API key environment variable
    if provider_name == "anthropic":
        api_key_env = "ANTHROPIC_API_KEY"
    else:  # openai (including all OpenAI-compatible providers)
        api_key_env = "OPENAI_API_KEY"

    # Use base_url from CLI argument, or from alias, or None
    base_url = args.base_url or base_url_from_alias

    # Use model from CLI argument, or default from alias, or a reasonable default
    model = args.model or default_model or ("gpt-4o" if provider_name == "openai" else "claude-opus-4-20250514")

    config = PipelineConfig(
        provider=provider_name,
        model=model,
        backend=args.backend,
        max_iterations=args.max_iterations,
        api_key_env=api_key_env,
        example_kernel_path=str(EXAMPLE_KERNEL_PATH) if args.use_example else None,
        base_url=base_url,
    )

    # Check if API key is set
    if not config.get_api_key():
        print(f"ERROR: API key not set. Please set {api_key_env} environment variable.")
        sys.exit(1)

    pipeline = GenerationPipeline(config)

    # Check provider availability
    if not pipeline.provider.is_available():
        print(f"ERROR: Provider '{args.provider}' is not available.")
        print(f"  - Check that the package is installed")
        print(f"  - Check that your API key is valid")
        sys.exit(1)

    # Determine which specs to generate
    if args.all:
        names = list_specs()
    elif args.kernel:
        names = [args.kernel]
    else:
        print("ERROR: Specify a kernel name or use --all")
        sys.exit(1)

    results = []
    for name in names:
        print(f"\n{'='*60}")
        print(f"Generating: {name}")
        print(f"{'='*60}")
        print(f"  Provider: {args.provider}")
        print(f"  Model: {model}")
        if base_url:
            print(f"  Base URL: {base_url}")
        print(f"  Backend: {args.backend}")

        result = pipeline.run(name)

        status = "SUCCESS" if result.success else "FAILED"
        print(f"\n  Status: {status}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Tokens used: {result.tokens_used}")
        if result.success:
            print(f"  Speedup: {result.final_result.speedup:.2f}x")
        else:
            print(f"  Error: {result.error_msg}")

        results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "OK" if r.success else "FAIL"
        speedup = f"{r.final_result.speedup:.2f}x" if r.success else "N/A"
        print(f"  {r.spec_name}: {status} ({r.iterations} iterations, {r.tokens_used} tokens, {speedup})")

    # Exit with error if any failed
    if not all(r.success for r in results):
        sys.exit(1)


def add_generate_subcommand(subparsers):
    """Add the generate subcommand to the CLI parser.

    Args:
        subparsers: ArgumentParser subparsers object
    """
    from experiments.generation.ai_providers import list_providers

    all_providers = list_providers()

    class CustomHelpFormatter(argparse.HelpFormatter):
        def __init__(self, prog):
            super().__init__(prog, max_help_position=40, width=120)

    parser = subparsers.add_parser(
        "generate",
        help="Generate DSL kernels using AI",
        description="Automated DSL kernel generation using AI APIs. Generates Triton/TileLang kernels from specifications.",
        epilog=f"""
Available providers: {', '.join(all_providers)}

Common aliases:
  --provider glm         Zhipu GLM (uses OPENAI_API_KEY)
  --provider kimi        Moonshot Kimi (uses OPENAI_API_KEY)
  --provider vllm        Local vLLM server (uses OPENAI_API_KEY)
  --provider ollama      Ollama (uses OPENAI_API_KEY)
  --provider openai      OpenAI GPT models
  --provider anthropic   Anthropic Claude models

Examples:
  ironcore generate rmsnorm --provider glm --model glm-4.7
  ironcore generate rmsnorm --provider kimi --model moonshot-v1-32k
  ironcore generate rmsnorm --provider openai --model gpt-4o
  ironcore generate rmsnorm --provider anthropic --model claude-opus-4-20250514
        """,
        formatter_class=CustomHelpFormatter,
    )
    parser.add_argument(
        "kernel",
        nargs="?",
        help="Kernel spec name to generate (or use --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all registered kernels",
    )
    parser.add_argument(
        "--provider", "-p",
        default="anthropic",
        metavar="PROVIDER",
        help=f"AI provider or alias to use (default: anthropic). Available: {', '.join(all_providers[:10])}...)",
    )
    parser.add_argument(
        "--model", "-m",
        default="",
        help="Model identifier (uses provider default if not specified)",
    )
    parser.add_argument(
        "--backend", "-b",
        default="triton",
        choices=["triton", "tilelang"],
        help="Kernel backend (default: triton)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum refinement iterations (default: 3)",
    )
    parser.add_argument(
        "--use-example",
        action="store_true",
        help="Include example kernel in prompt for reference",
    )
    parser.add_argument(
        "--base-url",
        help="Custom API base URL (overrides provider alias default)",
    )
    parser.set_defaults(func=run_generate)
