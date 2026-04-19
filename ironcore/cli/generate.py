# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Interactive text generation from IronCore checkpoints."""

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

    # Set checkpoint path
    if args.checkpoint:
        config.trainer.model_path = args.checkpoint

    # Initialize global state and parallel
    from ironcore import global_vars
    from ironcore.parallel import parallel_states

    if global_vars.GLOBAL_STATES is None:
        global_vars.set_global_states(config)

    if parallel_states._TENSOR_MODEL_PARALLEL_GROUP is None:
        parallel_states.initialize_model_parallel(
            tensor_model_parallel_size=1, timeout_in_minutes=10.0
        )

    # Build tokenizer
    from ironcore.tokenizer.tokenizer import build_tokenizer

    tokenizer = build_tokenizer(config)

    # Build model
    import torch

    from ironcore.language_model import LanguageModel

    model = LanguageModel(config)
    model.eval()

    # Load checkpoint
    if config.trainer.model_path:
        from ironcore.checkpointing.native import load_checkpoint

        step = load_checkpoint(config, model)
        print(f"Loaded checkpoint from step {step}")
    else:
        print("Warning: no checkpoint loaded, using random weights")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.model.name} ({num_params:,} params)")

    # Determine device
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Initialize KV cache
    model.initialize_cache(batch_size=1, device=device, dtype=dtype)

    # Generation params
    do_sample = not args.no_sample
    max_new_tokens = args.max_new_tokens or 128
    temperature = args.temperature if args.temperature is not None else 1.0
    top_p = args.top_p if args.top_p is not None else 1.0
    top_k = args.top_k if args.top_k is not None else 0

    def generate_text(prompt_str: str) -> str:
        """Generate text from a prompt string."""
        # Tokenize
        if args.chat:
            messages = []
            if args.system_prompt:
                messages.append({"role": "system", "content": args.system_prompt})
            messages.append({"role": "user", "content": prompt_str})
            input_ids = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
        else:
            encoded = tokenizer.encode(prompt_str, return_tensors="pt")
            if isinstance(encoded, dict):
                input_ids = encoded["input_ids"]
            else:
                input_ids = encoded

        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor([input_ids], dtype=torch.long)
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        input_ids = input_ids.to(device)

        # Reset cache
        model.reset_cache()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only generated tokens
        generated_ids = output_ids[0, input_ids.shape[1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    # One-shot or interactive
    if args.prompt:
        print(generate_text(args.prompt))
    else:
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
                    print(generate_text(prompt))
                except torch.cuda.OutOfMemoryError:
                    print("Error: GPU out of memory. Try a shorter prompt or smaller model.")
                print()
        except KeyboardInterrupt:
            print("\nExiting.")
