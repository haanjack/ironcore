#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Run a local pretrained Gemma 4 dense text decoder with TP=1 or TP=2.

python examples/gemma4_generate.py --checkpoint .local/models/gemma-4-E2B-it
torchrun --standalone --nproc_per_node=2 examples/gemma4_generate.py --checkpoint .local/models/gemma-4-E2B-it
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ironcore import global_vars
from ironcore.checkpointing.hf_interop import (
    load_from_huggingface,
    validate_imported_base_parameters,
)
from ironcore.config import (
    DataConfig,
    InitConfig,
    MainConfig,
    OperationConfig,
    OptimConfig,
    ParallelConfig,
    PEFTConfig,
    ProfilerConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.config.config_gemma4 import model_config_from_gemma4
from ironcore.language_model import LanguageModel
from ironcore.parallel import parallel_states
from ironcore.utils.memory import get_detailed_memory_breakdown, get_host_memory_usage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--prompt", default="What is 2 + 2? Answer briefly.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, help="Save timing and generated token IDs as JSON")
    parser.add_argument(
        "--logits-output", type=Path, help="Save prompt logits on rank 0 for TP comparisons"
    )
    args = parser.parse_args()
    tp_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
    if tp_size > 1 or args.device == "cuda":
        if "RANK" not in os.environ:
            os.environ.update(
                RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1", MASTER_PORT="29500"
            )
        dist.init_process_group("nccl" if args.device == "cuda" else "gloo")
    parallel_states.initialize_model_parallel(tp_size, timeout_in_minutes=10.0)
    try:
        checkpoint = args.checkpoint.resolve()
        hf_config = json.loads((checkpoint / "config.json").read_text())
        text_config = hf_config.get("text_config", hf_config)
        model_config = model_config_from_gemma4(hf_config)
        model_config.precision = args.precision
        model_config.max_seq_len = 2048
        model_config.vocab_name_or_path = str(checkpoint)
        config = MainConfig(
            model=model_config,
            init=InitConfig(),
            optim=OptimConfig(),
            data=DataConfig(
                vocab_size=text_config["vocab_size"],
                vocab_name_or_path=str(checkpoint),
                tokenizer_type="sentencepiece",
            ),
            parallel=ParallelConfig(rank=rank, local_rank=local_rank, world_size=tp_size),
            trainer=TrainerConfig(tensor_model_parallel_size=tp_size, recompute_linear_ce=False),
            operation=OperationConfig(activation_recompute=False),
            utils=UtilsConfig(),
            profiler=ProfilerConfig(),
            peft=PEFTConfig(),
        )
        global_vars.set_global_states(config)
        tokenizer = global_vars.get_tokenizer()
        started = time.perf_counter()
        print(
            f"rank={rank}: constructing Gemma 4 text decoder, TP={tp_size}, {args.precision}",
            flush=True,
        )
        model = LanguageModel(config)
        device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device("cpu")
        dtype = torch.bfloat16 if args.precision == "bfloat16" else torch.float32
        model.to(device=device, dtype=dtype).eval()
        info = load_from_huggingface(checkpoint, model, device="cpu")
        validate_imported_base_parameters(model, info["missing_keys"])
        if info["missing_keys"] or info["unexpected_keys"]:
            raise ValueError(
                f"Checkpoint mismatch: missing={info['missing_keys']}, unexpected={info['unexpected_keys']}"
            )
        load_seconds = time.perf_counter() - started
        print(f"rank={rank}: pretrained weights loaded in {load_seconds:.2f}s", flush=True)
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        input_ids = (
            encoded["input_ids"]
            if isinstance(encoded, dict) or hasattr(encoded, "input_ids")
            else encoded
        )
        input_ids = input_ids.to(device)
        if args.logits_output:
            with torch.inference_mode():
                prompt_logits, _ = model(input_ids)
            if rank == 0:
                args.logits_output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(prompt_logits.cpu(), args.logits_output)
            del prompt_logits
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        generation_path = checkpoint / "generation_config.json"
        generation_config = (
            json.loads(generation_path.read_text()) if generation_path.exists() else {}
        )
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=generation_config.get("eos_token_id", tokenizer.eos_token_id),
            )
        if args.device == "cuda":
            torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - started
        if rank == 0:
            generated = output_ids[0, input_ids.size(1) :].tolist()
            result = {
                "checkpoint": str(checkpoint),
                "tp_size": tp_size,
                "device": args.device,
                "precision": args.precision,
                "prompt": args.prompt,
                "input_tokens": input_ids.size(1),
                "generated_token_ids": generated,
                "text": tokenizer.decode(generated, skip_special_tokens=False),
                "load_seconds": load_seconds,
                "generation_seconds": generation_seconds,
                "tokens_per_second": len(generated) / generation_seconds,
                "local_parameters": sum(p.numel() for p in model.parameters()),
                "local_parameter_bytes": sum(
                    p.numel() * p.element_size() for p in model.parameters()
                ),
                "peak_cuda_bytes": torch.cuda.max_memory_allocated()
                if args.device == "cuda"
                else None,
                "memory_breakdown_mib": get_detailed_memory_breakdown(model),
                "host_memory_mib": get_host_memory_usage(),
            }
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    finally:
        global_vars.global_states_cleanup()
        parallel_states.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
