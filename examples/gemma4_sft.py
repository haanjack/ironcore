# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Native Gemma 4 A4B SFT with bounded execution and standalone LoRA weights.

torchrun --standalone --nproc_per_node=2 examples/gemma4_sft.py --checkpoint .local/models/gemma-4-26B-A4B-it --sequence-length 32768 --output .local/gemma4-a4b-32k
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import resource
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from ironcore.config import (
    DataConfig,
    InitConfig,
    MainConfig,
    OffloadConfig,
    OperationConfig,
    OptimConfig,
    ParallelConfig,
    PEFTConfig,
    ProfilerConfig,
    TrainerConfig,
    UtilsConfig,
    _config_validation,
)
from ironcore.config.config_data import DatasetConfig
from ironcore.config.config_gemma4 import model_config_from_gemma4
from ironcore.peft import load_lora_adapter, save_lora_adapter
from ironcore.preprocess import preprocess
from ironcore.trainers.language_model_trainer import LanguageModelTrainer
from ironcore.training_utils import forward_step, loss_func_sft
from ironcore.utils.memory import get_detailed_memory_breakdown, get_host_memory_usage


def create_tiny_checkpoint(path: Path) -> None:
    """Build a scaled reference checkpoint for offload/TP/CP correctness checks."""
    from safetensors.torch import save_file
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    if (path / "model.safetensors").exists():
        return
    path.mkdir(parents=True, exist_ok=True)
    config = Gemma4TextConfig(
        vocab_size=262144,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=32,
        num_global_key_value_heads=2,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=64,
        attention_k_eq_v=True,
        sliding_window=64,
        max_position_embeddings=65536,
        layer_types=["sliding_attention", "full_attention"] * 2,
        dtype="bfloat16",
    )
    config._attn_implementation = "eager"
    torch.manual_seed(42)
    model = Gemma4ForCausalLM(config)
    save_file(
        {n: p.contiguous() for n, p in model.state_dict().items() if n != "lm_head.weight"},
        str(path / "model.safetensors"),
    )
    (path / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")


def prepare_instruction(path: Path, tokenizer_path: Path, length: int) -> None:
    """Fill an actual instruction conversation to length+1 tokens, without padding."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    prefix = "Read this engineering log. Follow the final instruction.\n"
    suffix = "\nReturn only the code word COPPER."
    content = "".join(
        f"Record {i:04d}: temperature={i % 97}, memory={i % 83}, status=ready. Check routing and checkpoint integrity.\n"
        for i in range(length // 8 + 1)
    )
    filler = tokenizer.encode(content, add_special_tokens=False)

    def messages(budget):
        return [
            {
                "role": "user",
                "content": prefix
                + tokenizer.decode(filler[:budget], skip_special_tokens=False)
                + suffix,
            },
            {"role": "assistant", "content": "COPPER"},
        ]

    overhead = len(
        tokenizer.apply_chat_template(messages(0), tokenize=True, enable_thinking=True)["input_ids"]
    )
    budget = max(0, length + 1 - overhead)
    for _ in range(20):
        conversation = messages(budget)
        actual = len(
            tokenizer.apply_chat_template(conversation, tokenize=True, enable_thinking=True)[
                "input_ids"
            ]
        )
        if actual == length + 1:
            break
        budget += length + 1 - actual
    else:
        raise ValueError("Could not construct the requested real token length")
    path.write_text(json.dumps({"messages": conversation}, ensure_ascii=False) + "\n")


class ObservedSFTTrainer(LanguageModelTrainer):
    """Keep training in the production loop; observe updates, routing and memory."""

    def __init__(self, config: MainConfig, output: Path) -> None:
        super().__init__(config, forward_step, loss_func_sft)
        self.output = output
        self.records = []
        self.routing = {}

    def _post_checkpoint_load(self, last_step: int) -> None:
        if last_step:
            raise ValueError("Use a fresh output directory for this validation run")
        before = get_host_memory_usage()["rss_mb"]
        # Parameter storage has moved into pinned tiles. Release free glibc
        # arenas left by CPU construction/import before allocating activations.
        libc = ctypes.CDLL(None)
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
        self.host_initialization = {
            "rss_before_trim_mib": before,
            "rss_after_trim_mib": get_host_memory_usage()["rss_mb"],
        }
        print(
            f"rank={dist.get_rank()} HOST_INITIALIZATION {json.dumps(self.host_initialization)}",
            flush=True,
        )
        self.initial = {
            n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if p.requires_grad
        }
        assert self.initial
        assert all("lora_" in n for n, p in self.model.named_parameters() if p.requires_grad)
        for index, layer in enumerate(self.model.model.layers):

            def observe(module, inputs, result, index=index):
                if index not in self.routing:
                    counts = (
                        torch.bincount(
                            result[0].flatten(), minlength=self.config.model.moe.num_routed_experts
                        )
                        .cpu()
                        .tolist()
                    )
                    self.routing[index] = {
                        "active_experts": sum(v > 0 for v in counts),
                        "counts": counts,
                    }

            layer.router.register_forward_hook(observe)
        torch.cuda.reset_peak_memory_stats()

    def _prepare_gradients(self) -> None:
        super()._prepare_gradients()
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                assert parameter.grad is None, name

    def train_step(self, step: int) -> tuple[float, float, float]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = super().train_step(step)
        torch.cuda.synchronize()
        assert math.isfinite(result[0]) and math.isfinite(result[1]) and result[1] > 0
        record = {
            "step": step + 1,
            "loss": result[0],
            "grad_norm": result[1],
            "seconds": time.perf_counter() - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        self.records.append(record)
        print(f"rank={dist.get_rank()} SFT_RECORD {json.dumps(record)}", flush=True)
        return result

    def export_and_verify(self) -> dict:
        trained = {
            n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if p.requires_grad
        }
        changed = sum(not torch.equal(p, self.initial[n]) for n, p in trained.items())
        assert changed > 0
        replicas = [None] * dist.get_world_size()
        dist.all_gather_object(replicas, trained)
        for replica in replicas:
            for name, value in trained.items():
                torch.testing.assert_close(value, replica[name], atol=0, rtol=0)
        adapter = self.output / "adapter"
        if dist.get_rank() == 0:
            save_lora_adapter(self.model, adapter)
        dist.barrier()
        with torch.no_grad():
            for p in self.model.parameters():
                if p.requires_grad:
                    p.zero_()
        load_lora_adapter(self.model, adapter)
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(p.detach().cpu(), trained[name], atol=0, rtol=0)
        return {
            "changed_adapter_tensors": changed,
            "adapter_parameters": sum(p.numel() for p in trained.values()),
            "replicas_exact": True,
            "standalone_adapter_reload_exact": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(".local/models/gemma-4-26B-A4B-it"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--no-offload", action="store_true")
    parser.add_argument("--attention-chunk", type=int, default=128)
    parser.add_argument("--mlp-chunk", type=int, default=512)
    parser.add_argument("--grouped-budget", type=int, default=4096)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--verify-generation", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    rank, local, world = (
        int(os.environ.get(k, v))
        for k, v in [("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")]
    )
    if args.tp < 1 or world % args.tp:
        raise ValueError("TP must divide the torchrun world size")
    if args.verify_generation and world != args.tp:
        raise ValueError("Generation verification requires CP=1")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.checkpoint.resolve()
    checkpoint = (
        Path(".local/gemma4-a4b-study/tiny-model").resolve() if args.tiny else tokenizer_path
    )
    if args.tiny and rank == 0:
        create_tiny_checkpoint(checkpoint)
    dist.barrier()
    model_config = model_config_from_gemma4(json.loads((checkpoint / "config.json").read_text()))
    model_config.max_seq_len = args.sequence_length
    model_config.vocab_name_or_path = str(tokenizer_path)
    model_config.gemma4.attention_chunk_size = args.attention_chunk
    model_config.moe.expert_backend = "grouped"
    model_config.moe.blockwise_backend = "triton"
    model_config.moe.grouped_token_budget = args.grouped_budget
    data = DataConfig(
        task_type="sft",
        vocab_name_or_path=str(tokenizer_path),
        tokenizer_type="sentencepiece",
        vocab_size=262144,
        seq_length=args.sequence_length,
        sft_packing=False,
        pad_token_id=0,
        splits=[1.0, 0.0, 0.0],
        preprocessed_dir=output / "data",
        cache_dir=output / "cache",
        num_workers=0,
        datasets=[
            DatasetConfig(
                name="instruction",
                source=str(output / "instruction.jsonl"),
                task_type="sft",
                chat_template_kwargs={"enable_thinking": True},
            )
        ],
    )
    if rank == 0:
        prepare_instruction(output / "instruction.jsonl", tokenizer_path, args.sequence_length)
        preprocess(data)
    dist.barrier()
    config = MainConfig(
        model=model_config,
        init=InitConfig(seed=42),
        optim=OptimConfig(max_lr=1e-4, min_lr=1e-4, adam_eps=1e-4, weight_decay=0, clip_grad=1),
        data=data,
        parallel=ParallelConfig(rank=rank, local_rank=local, world_size=world),
        trainer=TrainerConfig(
            load_from_hf=str(checkpoint),
            model_path=str(output / "checkpoint"),
            tensor_model_parallel_size=args.tp,
            context_parallel_size=world // args.tp,
            context_parallel_backend="sdpa",
            parameter_precision="model",
            micro_batch_size=1,
            train_batch_size=1,
            gradient_accumulation_steps=1,
            eval_batch_size=1,
            recompute_linear_ce=True,
            loss_chunk_size=128,
            mlp_chunk_size=args.mlp_chunk,
            log_interval=1,
        ),
        operation=OperationConfig(
            train_steps=args.steps, no_save=True, activation_recompute=args.no_offload
        ),
        peft=PEFTConfig(method="lora"),
        profiler=ProfilerConfig(),
        utils=UtilsConfig(tensorboard_dir=str(output / "tensorboard")),
        offload=OffloadConfig(
            enabled=not args.no_offload,
            weight_offload=not args.no_offload,
            activation_spill=not args.no_offload,
            activation_spill_granularity="full_layer",
            weight_storage_precision="bf16",
            weight_prefetch_layers=1,
            backward_weight_prefetch_layers=1,
            pinned_memory_pool_gb=1 if args.tiny else (36 if args.tp > 1 else 55),
            pinned_chunk_gb=0.0625 if args.tiny else 1,
            gpu_staging_chunk_mb=1 if args.tiny else 256,
        ),
    )
    config.peft.lora.r, config.peft.lora.alpha = 8, 16
    config.peft.lora.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    config.peft.lora.parameter_precision = "float32"
    config.peft.lora.adapter_path = str(args.adapter_path.resolve()) if args.adapter_path else None
    _config_validation(config)
    if rank == 0:
        serialized = json.loads(json.dumps(asdict(config), default=str))
        serialized["data"]["train_datasets"] = serialized["data"].pop("datasets")
        (output / "config.yaml").write_text(yaml.safe_dump(serialized, sort_keys=False))
    with ObservedSFTTrainer(config, output) as trainer:
        trainer.train()
        verification = trainer.export_and_verify()
        if args.verify_generation:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Return only the code word COPPER."}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
                return_tensors="pt",
            )["input_ids"].cuda()
            trainer.model.eval()
            if trainer._offload_scheduler is not None:
                trainer._offload_scheduler.on_training_step_start()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = trainer.model.generate(prompt, max_new_tokens=8, do_sample=False)
            verification["generation_token_ids"] = generated[0, prompt.size(1) :].tolist()
            verification["generation_text"] = tokenizer.decode(verification["generation_token_ids"])
        local_result = {
            "rank": rank,
            "memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "routing": trainer.routing,
            "host_initialization": trainer.host_initialization,
            "memory_breakdown_bytes": get_detailed_memory_breakdown(
                trainer.model, trainer.optimizer, in_mib=False
            ),
        }
        ranks = [None] * world
        dist.all_gather_object(ranks, local_result)
        result = {
            "tiny": args.tiny,
            "checkpoint": str(checkpoint),
            "sequence_length": args.sequence_length,
            "tp": args.tp,
            "cp": world // args.tp,
            "weight_offload": not args.no_offload,
            "precision": "BF16 frozen weights + FP32 LoRA, BF16 autocast",
            "records": trainer.records,
            "by_rank": ranks,
            **verification,
        }
        if rank == 0:
            (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps({k: v for k, v in result.items() if k != "by_rank"}, indent=2),
                flush=True,
            )


if __name__ == "__main__":
    main()
