# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Native Gemma 4 A4B SFT with bounded execution and standalone LoRA weights.

torchrun --standalone --nproc_per_node=2 examples/gemma4_sft.py --checkpoint .local/models/gemma-4-26B-A4B-it --sequence-length 32768 --output .local/gemma4-a4b-32k
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
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
from ironcore.preprocessing.gemma4_chat import (
    gemma4_sft_chat_template,
    gemma4_sft_format_metadata,
)
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
    template = gemma4_sft_chat_template(tokenizer.chat_template)
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
        tokenizer.apply_chat_template(
            messages(0), tokenize=True, chat_template=template, enable_thinking=False
        )["input_ids"]
    )
    budget = max(0, length + 1 - overhead)
    for _ in range(20):
        conversation = messages(budget)
        actual = len(
            tokenizer.apply_chat_template(
                conversation, tokenize=True, chat_template=template, enable_thinking=False
            )["input_ids"]
        )
        if actual == length + 1:
            break
        budget += length + 1 - actual
    else:
        raise ValueError("Could not construct the requested real token length")
    path.write_text(json.dumps({"messages": conversation}, ensure_ascii=False) + "\n")


def estimate_host_budget(model, sequence_length: int, tp: int, world: int, attention_only=False):
    """Conservative example-only budget, before constructing pretrained weights.

    Account for replicated CP weights, full FP32 adapter replicas, FP32 Adam
    moments/gradients, spilled layer inputs and host/import headroom. This is a
    screening estimate, not a measured peak or a runtime memory reservation.
    """
    from ironcore.utils.mfu import MFUCalculator

    rank, hidden = 8, model.d_model
    adapters = 0
    for layer in range(model.num_layers):
        head, kv_heads = model.gemma4.head_layout(model, layer)
        query = model.num_attention_heads * head
        kv = kv_heads * head
        projections = [(hidden, query), (query, hidden), (hidden, kv)]
        if not (
            model.gemma4.attention_k_eq_v and model.gemma4.layer_types[layer] == "full_attention"
        ):
            projections.append((hidden, kv))
        if not attention_only:
            projections.extend([(hidden, model.d_ffn)] * 3)
            adapters += (
                3
                * (hidden + model.moe.expert_intermediate_size)
                * rank
                * model.moe.num_routed_experts
            )
        adapters += sum((inputs + outputs) * rank for inputs, outputs in projections)
    base = MFUCalculator.from_config(model, 262144).get_num_parameters()
    parts = {
        "frozen_weights_bytes": base * 2 * world // tp,
        "adapter_weights_gradients_adam_bytes": adapters * 16 * world,
        "spilled_inputs_bytes": sequence_length * hidden * model.num_layers * 2 * tp,
        "runtime_import_workspace_bytes": 4 * 1024**3 * world,
        "host_headroom_bytes": 16 * 1024**3,
    }
    return {
        **parts,
        "required_available_bytes": sum(parts.values()),
        "adapter_parameters": adapters,
    }


def check_host_budget(budget, available_bytes):
    if available_bytes < budget["required_available_bytes"]:
        raise RuntimeError(
            f"Insufficient host RAM for this offload example: estimated need including headroom "
            f"{budget['required_available_bytes'] / 1024**3:.1f} GiB, available "
            f"{available_bytes / 1024**3:.1f} GiB. CP replicates CPU weights; use TP2 "
            "or a host with more RAM. Model allocation has not started."
        )


def available_host_bytes():
    import psutil

    available = psutil.virtual_memory().available
    # Container limits can be much smaller than host RAM. Respect cgroup v2
    # remaining capacity as well as physical host availability.
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text())
            statistics = dict(
                line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()
            )
            # Both LRU lists contain regular file cache. Pinned / shared
            # offload storage is accounted separately as shmem/anon and must
            # not be treated as reclaimable checkpoint cache.
            reclaimable = int(statistics.get("inactive_file", 0)) + int(
                statistics.get("active_file", 0)
            )
            available = min(available, max(0, int(limit) - used + reclaimable))
    except (OSError, ValueError):
        pass
    return available


class ObservedSFTTrainer(LanguageModelTrainer):
    """Keep training in the production loop; observe updates, routing and memory."""

    def __init__(
        self, config: MainConfig, output: Path, evaluation_files=None, generation_prompts=None
    ) -> None:
        super().__init__(config, forward_step, loss_func_sft)
        self.output = output
        self.records = []
        self.routing = {}
        self.evaluation_files = evaluation_files or {}
        self.generation_prompts = generation_prompts
        self.evaluations = []
        self.generations = []

    def evaluate_files(self, phase: str, splits=None) -> None:
        """Measure assistant-only loss without materializing sequence-wide logits."""
        from transformers import AutoTokenizer

        from ironcore.dataloader.collator import UniversalCollator
        from ironcore.preprocessing.serializer import DataSerializer

        tokenizer = AutoTokenizer.from_pretrained(self.config.data.vocab_name_or_path)
        serializer = DataSerializer(self.config.data, tokenizer, verbose=False)
        was_training = self.model.training
        self.model.eval()
        for split, path in self.evaluation_files.items():
            if splits is not None and split not in splits:
                continue
            losses, token_counts = [], []
            for line in Path(path).read_text().splitlines():
                row = json.loads(line)
                tokens, masks = serializer._apply_chat_template_and_get_masks(
                    row["messages"],
                    self.config.data.datasets[0].chat_template,
                    self.config.data.datasets[0].chat_template_kwargs,
                )
                if len(tokens) > self.config.data.seq_length + 1:
                    raise ValueError("Evaluation conversation would be truncated")
                batch = UniversalCollator("sft", max_seq_len=len(tokens) - 1, pack_sequences=False)(
                    [{"token_ids": tokens, "metadata": {"mask_ranges": masks}}]
                )
                batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                count = int((batch["labels"] != -100).sum())
                if count == 0:
                    raise ValueError("Evaluation sample has no assistant targets")
                if self._offload_scheduler is not None:
                    self._offload_scheduler.on_training_step_start()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = self.model(
                        batch["input_ids"],
                        labels=batch["labels"],
                        position_ids=batch["position_ids"],
                    )
                losses.append(float(loss))
                token_counts.append(count)
                print(
                    f"rank={dist.get_rank()} EVAL_SAMPLE {phase} {split} "
                    f"{len(losses)} loss={losses[-1]:.6f} targets={count}",
                    flush=True,
                )
            record = {
                "phase": phase,
                "split": split,
                "samples": len(losses),
                "mean_sample_loss": sum(losses) / len(losses),
                "mean_token_loss": sum(a * b for a, b in zip(losses, token_counts, strict=True))
                / sum(token_counts),
                "assistant_tokens": sum(token_counts),
                "sample_losses": losses,
                "sample_target_counts": token_counts,
            }
            assert math.isfinite(record["mean_sample_loss"])
            self.evaluations.append(record)
            if dist.get_rank() == 0:
                (self.output / "evaluations.json").write_text(
                    json.dumps(self.evaluations, indent=2)
                )
                print(f"HELDOUT_RESULT {json.dumps(record)}", flush=True)
        self.model.train(was_training)

    def generate_probes(self, phase: str) -> None:
        if self.generation_prompts is None:
            return
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.config.data.vocab_name_or_path)
        self.model.eval()
        for probe in json.loads(Path(self.generation_prompts).read_text()):
            encoded = tokenizer.apply_chat_template(
                probe["messages"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
            prompt = encoded["input_ids"].cuda()
            if self._offload_scheduler is not None:
                self._offload_scheduler.on_training_step_start()
            limit = probe.get("max_new_tokens", 128)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = self.model.generate(
                    prompt,
                    max_new_tokens=limit,
                    do_sample=False,
                    eos_token_id=[
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<turn|>"),
                    ],
                )
            ids = generated[0, prompt.size(1) :].tolist()
            record = {
                "phase": phase,
                "probe": probe,
                "token_ids": ids,
                "text": tokenizer.decode(ids),
                "stopped_before_limit": len(ids) < limit,
                "enable_thinking": False,
            }
            self.generations.append(record)
            if dist.get_rank() == 0:
                (self.output / "generations.json").write_text(
                    json.dumps(self.generations, indent=2)
                )
                print(f"GENERATION_PROBE {json.dumps(record)}", flush=True)
        self.model.train()

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
        self.initial = self.adapter_fingerprints()
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
        if self.evaluation_files:
            self.evaluate_files("baseline")
        if self.generation_prompts is not None:
            self.generate_probes("baseline")
        if self.evaluation_files or self.generation_prompts is not None:
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
        if self.evaluation_files and (step + 1) % self.config.operation.eval_interval == 0:
            self.evaluate_files(f"step_{step + 1}", splits={"validation"})
        return result

    def adapter_fingerprints(self) -> dict[str, str]:
        # Full expert adapters exceed a GiB. Stream fingerprints instead of
        # retaining extra CPU snapshots or gathering tensor payloads on GPU.
        return {
            name: hashlib.sha256(
                parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    def export_and_verify(self) -> dict:
        trained = self.adapter_fingerprints()
        changed = sum(value != self.initial[name] for name, value in trained.items())
        assert changed > 0
        by_component = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            component = (
                "routed_experts"
                if ".experts." in name
                else "shared_mlp"
                if ".mlp." in name
                else "attention"
            )
            entry = by_component.setdefault(
                component, {"parameters": 0, "tensors": 0, "changed_tensors": 0}
            )
            entry["parameters"] += parameter.numel()
            entry["tensors"] += 1
            entry["changed_tensors"] += trained[name] != self.initial[name]
        digest = hashlib.sha256()
        for name, value in trained.items():
            digest.update(name.encode())
            digest.update(value.encode())
        replicas = [None] * dist.get_world_size()
        dist.all_gather_object(replicas, digest.hexdigest())
        assert len(set(replicas)) == 1, "Adapter replicas differ"
        adapter = self.output / "adapter"
        if dist.get_rank() == 0:
            save_lora_adapter(self.model, adapter)
        dist.barrier()
        with torch.no_grad():
            for p in self.model.parameters():
                if p.requires_grad:
                    p.zero_()
        load_lora_adapter(self.model, adapter)
        assert self.adapter_fingerprints() == trained, "Reloaded adapter weights differ"
        return {
            "changed_adapter_tensors": changed,
            "adapter_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "adapter_components": by_component,
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
    parser.add_argument(
        "--attention-only", action="store_true", help="Omit shared and routed MLP adapters"
    )
    parser.add_argument("--verify-generation", action="store_true")
    parser.add_argument("--training-data", type=Path)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Train structured assistant reasoning fields; plain-answer SFT defaults to non-thinking",
    )
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--test-data", type=Path)
    parser.add_argument("--train-probe-data", type=Path)
    parser.add_argument("--retention-data", type=Path, help="An earlier task's held-out data")
    parser.add_argument("--generation-prompts", type=Path)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=20)
    args = parser.parse_args()
    if args.enable_thinking and not args.training_data:
        raise ValueError(
            "Thinking SFT requires reasoning data; the COPPER fixture has plain answers"
        )
    if (args.validation_data or args.test_data or args.retention_data) and not args.training_data:
        raise ValueError("Learning validation requires an explicit training dataset")
    output = args.output.resolve()
    rank, local, world = (
        int(os.environ.get(k, v))
        for k, v in [("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")]
    )
    if args.tp < 1 or world % args.tp:
        raise ValueError("TP must divide the torchrun world size")
    if (
        args.validation_data
        or args.test_data
        or args.train_probe_data
        or args.retention_data
        or args.generation_prompts
    ) and world != args.tp:
        raise ValueError("Public-data learning validation currently requires CP=1")
    if args.eval_interval < 1 or args.gradient_accumulation < 1:
        raise ValueError("Evaluation interval and gradient accumulation must be positive")
    if args.verify_generation and world != args.tp:
        raise ValueError("Generation verification requires CP=1")
    tokenizer_path = args.checkpoint.resolve()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    training_template = gemma4_sft_chat_template(tokenizer.chat_template)
    training_format = gemma4_sft_format_metadata(training_template, args.enable_thinking)
    if not args.tiny and not args.no_offload:
        metadata = model_config_from_gemma4(
            json.loads((tokenizer_path / "config.json").read_text())
        )
        budget = estimate_host_budget(
            metadata, args.sequence_length, args.tp, world, args.attention_only
        )
        available = available_host_bytes()
        check_host_budget(budget, available)
        if rank == 0:
            print(
                f"HOST_RAM_BUDGET {json.dumps({**budget, 'available_bytes': available})}",
                flush=True,
            )
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
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
        # Changing mode/template must never reuse the old serialized labels.
        preprocessed_dir=output / "data" / training_format["cache_namespace"],
        cache_dir=output / "cache",
        num_workers=0,
        datasets=[
            DatasetConfig(
                name="instruction",
                source=str(
                    args.training_data.resolve()
                    if args.training_data
                    else output / "instruction.jsonl"
                ),
                task_type="sft",
                chat_template=training_template,
                chat_template_kwargs={"enable_thinking": args.enable_thinking},
            )
        ],
    )
    if rank == 0:
        if args.training_data is None:
            prepare_instruction(output / "instruction.jsonl", tokenizer_path, args.sequence_length)
        preprocess(data)
    dist.barrier()
    config = MainConfig(
        model=model_config,
        init=InitConfig(seed=42),
        optim=OptimConfig(
            max_lr=args.learning_rate,
            min_lr=args.learning_rate,
            warmup_steps=args.warmup_steps,
            adam_eps=1e-4,
            weight_decay=0,
            clip_grad=1,
        ),
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
            train_batch_size=args.gradient_accumulation,
            gradient_accumulation_steps=args.gradient_accumulation,
            eval_batch_size=1,
            recompute_linear_ce=True,
            loss_chunk_size=128,
            mlp_chunk_size=args.mlp_chunk,
            log_interval=1,
        ),
        operation=OperationConfig(
            train_steps=args.steps,
            no_save=True,
            activation_recompute=args.no_offload,
            eval_interval=args.eval_interval,
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
            # Thousands of rank-8 expert tensors are too small to amortize
            # the CPU thread-pool overhead of individual Adam operations.
            optimizer_cpu_threads=1,
        ),
    )
    config.peft.lora.r, config.peft.lora.alpha = 8, 16
    config.peft.lora.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if not args.attention_only:
        config.peft.lora.target_modules += ["gate_proj", "up_proj", "down_proj"]
    config.peft.lora.parameter_precision = "float32"
    config.peft.lora.adapter_path = str(args.adapter_path.resolve()) if args.adapter_path else None
    _config_validation(config)
    if rank == 0:
        serialized = json.loads(json.dumps(asdict(config), default=str))
        serialized["data"]["train_datasets"] = serialized["data"].pop("datasets")
        (output / "config.yaml").write_text(yaml.safe_dump(serialized, sort_keys=False))
    evaluation_files = {
        name: path.resolve()
        for name, path in {
            "train_probe": args.train_probe_data,
            "validation": args.validation_data,
            "test": args.test_data,
            "retention": args.retention_data,
        }.items()
        if path is not None
    }
    with ObservedSFTTrainer(config, output, evaluation_files, args.generation_prompts) as trainer:
        trainer.train()
        verification = trainer.export_and_verify()
        if evaluation_files:
            trainer.evaluate_files("reloaded_final")
        if args.generation_prompts is not None:
            trainer.generate_probes("reloaded_final")
        if args.verify_generation:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Return only the code word COPPER."}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
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
            "training_format": training_format,
            "checkpoint": str(checkpoint),
            "starting_adapter": str(args.adapter_path.resolve()) if args.adapter_path else None,
            "sequence_length": args.sequence_length,
            "tp": args.tp,
            "cp": world // args.tp,
            "weight_offload": not args.no_offload,
            "precision": "BF16 frozen weights + FP32 LoRA, BF16 autocast",
            "records": trainer.records,
            "evaluations": trainer.evaluations,
            "generations": trainer.generations,
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
