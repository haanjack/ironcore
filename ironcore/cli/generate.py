# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""CLI wrapper for interactive and one-shot text generation.

The core generation logic lives in ``ironcore.generate``.  This module
handles CLI parsing and the interactive REPL loop.
"""

import sys
from argparse import Namespace
from pathlib import Path

from .utils import load_full_config


def run_generate(args: Namespace) -> None:
    """Generate text from a loaded model checkpoint.

    Supports one-shot generation with --prompt or interactive REPL mode.

    Args:
        args: Command-line arguments.
    """
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}")
        sys.exit(1)

    # Load config, force single-GPU, enable KV cache
    config = load_full_config(config_path)
    config.parallel.rank = 0
    config.parallel.local_rank = 0
    config.parallel.world_size = 1
    config.trainer.tensor_model_parallel_size = 1
    config.model.kv_cache.enabled = True

    from ironcore.generate import generate

    if args.prompt:
        # One-shot generation — delegate to the programmatic API
        result = generate(
            config,
            args.prompt,
            checkpoint=args.checkpoint,
            max_new_tokens=args.max_new_tokens or 128,
            temperature=args.temperature if args.temperature is not None else 1.0,
            top_p=args.top_p if args.top_p is not None else 1.0,
            top_k=args.top_k if args.top_k is not None else 0,
            do_sample=not args.no_sample,
            chat=args.chat,
            system_prompt=args.system_prompt,
        )
        print(result)
    else:
        # Interactive REPL — build model once, loop over prompts
        _run_repl(config, args)


def _run_repl(config, args: Namespace) -> None:
    """Run an interactive REPL loop for multi-turn generation."""
    import torch

    from ironcore.generate import generate

    print("IronCore Generate (type 'quit' or Ctrl-C to exit)")
    print("-" * 40)
    try:
        while True:
            try:
                prompt = input(">>> ")
            except EOFError:
                break
            if prompt.strip().lower() in ("quit", "exit", "q"):
                break
            if not prompt.strip():
                continue
            try:
                result = generate(
                    config,
                    prompt,
                    checkpoint=args.checkpoint,
                    max_new_tokens=args.max_new_tokens or 128,
                    temperature=args.temperature if args.temperature is not None else 1.0,
                    top_p=args.top_p if args.top_p is not None else 1.0,
                    top_k=args.top_k if args.top_k is not None else 0,
                    do_sample=not args.no_sample,
                    chat=args.chat,
                    system_prompt=args.system_prompt,
                )
                print(result)
            except torch.cuda.OutOfMemoryError:
                print("Error: GPU out of memory. Try a shorter prompt or smaller model.")
            print()
    except KeyboardInterrupt:
        print("\nExiting.")
