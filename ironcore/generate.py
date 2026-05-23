# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Programmatic text generation from IronCore checkpoints.

Usage as a Python API::

    from ironcore.generate import generate

    text = generate(
        config,
        "Hello, world!",
        checkpoint="checkpoints/step_1000",
        max_new_tokens=256,
        temperature=0.8,
    )

Usage from the command line::

    python -m ironcore.generate --config configs/small.yaml --prompt "Hello"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ironcore.config import MainConfig

from ironcore.train import load_full_config


def generate(
    config: MainConfig,
    prompt: str,
    *,
    checkpoint: str | None = None,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    do_sample: bool = True,
    chat: bool = False,
    system_prompt: str | None = None,
) -> str:
    """Generate text from a model checkpoint.

    Args:
        config: Fully resolved MainConfig.  Callers should set
            ``parallel.world_size=1`` / ``trainer.tensor_model_parallel_size=1``
            and enable ``model.kv_cache.enabled`` before passing the config.
        prompt: Text prompt to feed the model.
        checkpoint: Optional checkpoint path override.  When provided, sets
            ``config.trainer.model_path``.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (1.0 = default).
        top_p: Top-p (nucleus) sampling threshold.
        top_k: Top-k sampling (0 = disabled).
        do_sample: If False, use greedy decoding.
        chat: Apply chat-template tokenization.
        system_prompt: System prompt injected when *chat* is True.

    Returns:
        The generated text string (new tokens only).
    """
    # Override checkpoint path when explicitly provided
    if checkpoint is not None:
        config.trainer.model_path = checkpoint

    # -- lazy imports for heavy modules ---------------------------------------
    from ironcore import global_vars
    from ironcore.parallel import parallel_states

    if global_vars.GLOBAL_STATES is None:
        global_vars.set_global_states(config)

    if parallel_states._TENSOR_MODEL_PARALLEL_GROUP is None:
        parallel_states.initialize_model_parallel(
            tensor_model_parallel_size=1, timeout_in_minutes=10.0
        )

    from ironcore.tokenizer.tokenizer import build_tokenizer

    tokenizer = build_tokenizer(config)

    import torch

    from ironcore.language_model import LanguageModel

    model = LanguageModel(config)
    model.eval()

    if config.trainer.model_path:
        from ironcore.checkpointing.native import load_checkpoint

        step = load_checkpoint(config, model)
        print(f"Loaded checkpoint from step {step}")
    else:
        print("Warning: no checkpoint loaded, using random weights")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.initialize_cache(batch_size=1, device=device, dtype=dtype)

    # -- tokenize -------------------------------------------------------------
    if chat:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
    else:
        encoded = tokenizer.encode(prompt, return_tensors="pt")
        if isinstance(encoded, dict):
            input_ids = encoded["input_ids"]
        else:
            input_ids = encoded

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor([input_ids], dtype=torch.long)
    elif input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    input_ids = input_ids.to(device)

    # -- generate -------------------------------------------------------------
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

    generated_ids = output_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# CLI entry point:  python -m ironcore.generate --config ... --prompt "..."
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ironcore.generate",
        description="Generate text from an IronCore checkpoint",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (overrides trainer.model_path)",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 = disabled)")
    parser.add_argument("--no-sample", action="store_true", help="Use greedy decoding")
    parser.add_argument(
        "--system-prompt", type=str, default=None, help="System prompt for chat mode"
    )
    parser.add_argument("--chat", action="store_true", help="Enable chat template mode")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}")
        sys.exit(1)

    config = load_full_config(config_path)
    config.parallel.rank = 0
    config.parallel.local_rank = 0
    config.parallel.world_size = 1
    config.trainer.tensor_model_parallel_size = 1
    config.model.kv_cache.enabled = True

    result = generate(
        config,
        args.prompt,
        checkpoint=args.checkpoint,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=not args.no_sample,
        chat=args.chat,
        system_prompt=args.system_prompt,
    )
    print(result)


if __name__ == "__main__":
    main()
