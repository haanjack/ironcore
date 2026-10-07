# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Controlled trainer validation, with real models, optimizers and checkpoints.

python scripts/validate_trainers.py --device cpu --output outputs/trainer_validation_cpu
python scripts/validate_trainers.py --device cuda --output outputs/trainer_validation_gpu

The tiny, local tokenizer and synthetic data deliberately isolate numerical
correctness. This is not a benchmark of real-corpus quality or GPU saturation.
GRPO equivalence uses a specified rollout fixture; online rollout is checked
separately. Failed comparisons return a nonzero exit status and retain logs.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def create_tokenizer(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    vocab.update({f"t{i}": i for i in range(3, 32)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    ).save_pretrained(path)


def make_config(args):
    from ironcore.config import (
        AlignmentConfig,
        DataConfig,
        InitConfig,
        MainConfig,
        ModelConfig,
        OperationConfig,
        OptimConfig,
        ParallelConfig,
        PEFTConfig,
        ProfilerConfig,
        TrainerConfig,
        UtilsConfig,
    )
    from ironcore.config.config_alignment import (
        GenerationConfig,
        RewardFunctionEntry,
        RewardManagerConfig,
    )

    world = int(os.environ["WORLD_SIZE"])
    tp = 2 if args.case == "tp2" else 1
    micro = 8 if args.case == "full" else 2
    config = MainConfig(
        model=ModelConfig(
            d_model=32,
            d_ffn=64,
            num_layers=2,
            max_seq_len=16,
            num_attention_heads=4,
            num_attention_groups=4,
            head_dim=8,
            dropout_embd=args.dropout,
            dropout_attn=args.dropout,
            dropout_mlp=args.dropout,
            precision=args.precision,
            tokenizer_type="sentencepiece",
            vocab_name_or_path=str(args.output / "tokenizer"),
        ),
        trainer=TrainerConfig(
            micro_batch_size=micro,
            num_workers=args.data_workers,
            train_batch_size=8,
            gradient_accumulation_steps=8 // micro // (world // tp),
            tensor_model_parallel_size=tp,
            parameter_precision=args.parameter_precision,
            loss_chunk_size=args.loss_chunk_size,
            recompute_linear_ce=args.recompute_linear_ce,
            use_flash_attn=False,
            vocab_padding_unit=2,
            eval_batch_size=2,
            log_interval=1,
            save_checkpoint_steps=args.steps // 2,
            model_path=str(args.run_dir / "checkpoint"),
        ),
        init=InitConfig(seed=42, init_std=0.02),
        optim=OptimConfig(
            max_lr=1e-3, min_lr=1e-4, annealing_steps=args.steps, clip_grad=1.0, weight_decay=0.01
        ),
        data=DataConfig(task_type=args.task, vocab_name_or_path=str(args.output / "tokenizer")),
        parallel=ParallelConfig(
            use_fsdp=args.fsdp and world > 1,
            use_distributed_optimizer=args.distributed_optimizer,
            fsdp_use_orig_params=True,
            rank=int(os.environ["RANK"]),
            local_rank=int(os.environ["LOCAL_RANK"]),
            world_size=world,
            timeout_minute=2,
            dist_backend="nccl" if args.device == "cuda" else "gloo",
        ),
        operation=OperationConfig(train_steps=args.steps, eval_samples=4, exit_interval=args.stop),
        utils=UtilsConfig(report_memory_usage=False),
        profiler=ProfilerConfig(),
        peft=PEFTConfig(),
        alignment=AlignmentConfig(
            method="grpo" if args.task == "grpo" else "dpo",
            metrics_interval=0,
            grpo_group_size=4,
            grpo_rollout_micro_group_size=2 if args.case != "full" else 4,
            grpo_num_epochs=2,
            grpo_beta=0.1,
            grpo_objective=args.grpo_objective,
            generation=GenerationConfig(max_new_tokens=4, temperature=1.0, top_p=1.0),
            reward_manager=RewardManagerConfig(
                functions=[RewardFunctionEntry(type="soft_keyword", keyword="t4")],
                num_workers=1,
            ),
        ),
    )
    if args.architecture == "cs336":
        from ironcore.config.config_model import BiasConfig, PositionalEmbeddingConfig

        config.model.ln_type = "rmsnorm"
        config.model.positional_embedding = PositionalEmbeddingConfig(type="rope")
        config.model.activation_type = "swiglu"
        config.model.bias = BiasConfig.none()
        config.model.untie_embed = True
    if args.moe:
        from ironcore.config.config_moe import MoEConfig

        config.model.moe = MoEConfig(
            use_moe=True,
            num_shared_experts=1,
            num_routed_experts=4,
            num_experts_per_token=2,
            expert_intermediate_size=32,
            aux_loss_alpha=args.moe_aux_alpha,
            expert_backend=args.moe_backend,
            expert_model_parallel_size=2 if args.case == "ep2" else 1,
            router_bias=args.moe_idle,
        )
    return config


def data_batches(config, split="train"):
    if not getattr(config, "_stateful_data", False):
        return _synthetic_data_batches(config, split)
    import torch
    from torchdata.stateful_dataloader import StatefulDataLoader

    from ironcore.dataloader.stateful import CheckpointableIterator
    from ironcore.parallel import parallel_states as ps

    root = Path(config.model.vocab_name_or_path).parent / "data"
    dp = ps.get_data_parallel_world_size()
    batch_size = (
        config.trainer.train_batch_size // dp
        if config.data.task_type == "grpo" and split == "train"
        else config.trainer.micro_batch_size
    )
    if config.data.task_type == "grpo":
        from ironcore.alignment.dataset import GRPODataset, collate_grpo_samples

        dataset = GRPODataset(root / "grpo.json", seed=81, shuffle=split == "train")
        collator = collate_grpo_samples
    else:
        from ironcore.dataloader.collator import UniversalCollator
        from ironcore.dataloader.dataset import StreamingBinaryDataset, StreamingDataset

        source = StreamingBinaryDataset(root / "data.bin", root / "data.idx.npy")
        dataset = StreamingDataset.__new__(StreamingDataset)
        dataset.datasets = [source]
        dataset.weights = [1.0]
        dataset.mode = config.data.task_type
        dataset.split = split
        dataset.seed = 81
        dataset.seq_length = 16
        dataset.shuffle_buffer_size = 13
        dataset.rank = ps.get_data_parallel_group_rank()
        dataset.world_size = dp
        dataset.split_ranges = {
            id(source): (0, source.total_tokens if dataset.mode == "pretrain" else len(source))
        }
        collator = UniversalCollator(dataset.mode, 16, return_full_attention_mask=True)
    workers = config.trainer.num_workers
    return CheckpointableIterator(
        StatefulDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            multiprocessing_context="spawn" if workers else None,
            snapshot_every_n_steps=1,
            collate_fn=collator,
            generator=torch.Generator().manual_seed(1337),
        ),
        cycle=split == "train",
    )


def create_stateful_fixture(root):
    import numpy as np

    root.mkdir()
    tokens = (4 + (np.arange(128)[:, None] + np.arange(17)[None, :]) % 16).astype(np.uint16)
    tokens.tofile(root / "data.bin")
    index = np.zeros(
        128,
        dtype=[
            ("offset", "i8"),
            ("length", "i8"),
            ("type", "U8"),
            ("group_id", "i8"),
            ("mask_ranges", "U32"),
        ],
    )
    index["offset"] = np.arange(128) * 17
    index["length"] = 17
    index["type"] = "sft"
    index["mask_ranges"] = "[[0, 3]]"
    np.save(root / "data.idx.npy", index)
    (root / "grpo.json").write_text(
        json.dumps(
            [
                {
                    "prompt": " ".join(f"t{token}" for token in tokens[i, : 2 + i % 4]),
                    "sample_id": i,
                }
                for i in range(128)
            ]
        )
    )


def _synthetic_data_batches(config, split="train"):
    import torch

    from ironcore.parallel import parallel_states as ps

    dp = ps.get_data_parallel_world_size()
    rank = ps.get_data_parallel_group_rank()
    batch_size = (
        (
            config.trainer.train_batch_size // dp
            if config.data.task_type == "grpo"
            else config.trainer.micro_batch_size
        )
        if split == "train"
        else config.trainer.eval_batch_size
    )
    cursor = 0 if split == "train" else 1000
    while True:
        # Striding gives DP workers disjoint IDs; all cases see the same
        # global batch at each update, even with different accumulation.
        ids = torch.arange(cursor, cursor + batch_size) * dp + rank
        cursor += batch_size
        tokens = 4 + (ids[:, None] + torch.arange(17)[None, :]) % 16
        if config.data.task_type == "grpo":
            yield {
                "input_ids": tokens[:, :3].contiguous(),
                "prompts": [" ".join(f"t{t}" for t in row) for row in tokens[:, :3].tolist()],
                "metadata": [{"sample_id": int(i)} for i in ids],
            }
        elif config.data.task_type == "dpo":
            rejected = 4 + (ids[:, None] - torch.arange(17)[None, :]) % 16
            chosen_labels, rejected_labels = tokens[:, 1:].clone(), rejected[:, 1:].clone()
            chosen_labels[:, :3] = -100
            rejected_labels[:, :3] = -100
            yield {
                "chosen_input_ids": tokens[:, :-1].contiguous(),
                "rejected_input_ids": rejected[:, :-1].contiguous(),
                "chosen_labels": chosen_labels,
                "rejected_labels": rejected_labels,
            }
        else:
            labels = tokens[:, 1:].clone()
            if config.data.task_type == "sft":
                for row, sample_id in enumerate(ids):
                    labels[row, : int(sample_id) % 7 + 1] = -100
            yield {"input_ids": tokens[:, :-1].contiguous(), "labels": labels}


def collect_parameters(model, gradients=False):
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(model, FSDP):
        with FSDP.summon_full_params(model, with_grads=gradients, writeback=False):
            return collect_parameters(model.module, gradients)

    from ironcore.parallel import parallel_states as ps

    model = getattr(model, "module", model)
    result = {}
    for name, param in model.named_parameters():
        # An unselected expert contributes a zero gradient to this objective.
        value = (
            (param.grad if param.grad is not None else torch.zeros_like(param))
            if gradients
            else param
        )
        value = value.detach().contiguous()
        module = dict(model.named_modules())[name.rsplit(".", 1)[0]]
        should_gather = (
            getattr(module, "column_parallel", False) and name.endswith(("weight", "bias"))
        ) or (getattr(module, "row_parallel", False) and name.endswith("weight"))
        if ps.get_tensor_model_parallel_world_size() > 1 and should_gather:
            from ironcore.parallel.tensor_parallel.comm import gather_from_model_parallel_workers

            value = gather_from_model_parallel_workers(
                value,
                {
                    "column_parallel": module.column_parallel,
                    "row_parallel": module.row_parallel,
                    "concatenated_weights": module.concatenated_weights,
                },
            )
        result[name.replace("_fsdp_wrapped_module.", "")] = value.float().cpu().clone()
    if model.config.model.moe.expert_model_parallel_size > 1:
        import torch.distributed as dist

        from ironcore.checkpointing.expert import global_parameter_names

        names = global_parameter_names(model)
        result = {names[name]: value for name, value in result.items()}
        peers = [None] * dist.get_world_size()
        dist.all_gather_object(peers, result)
        result = {name: value for peer in peers for name, value in peer.items()}
    return result


def replay_rollouts(model, prompt_ids, group_size, metadata, **kwargs):
    """Specified completions with genuine behaviour-policy log probabilities."""
    import torch
    import torch.nn.functional as F

    from ironcore.alignment.rollout import _build_rollout_output

    count = prompt_ids.size(0) * group_size
    generated = torch.full((count, 4), 5, device=prompt_ids.device, dtype=torch.long)
    generated[torch.arange(count, device=prompt_ids.device) % group_size % 2 == 0] = 4
    expanded = prompt_ids.repeat_interleave(group_size, dim=0)
    completion = torch.cat([expanded, generated], dim=1)
    scoring_kwargs = {}
    mask = kwargs.get("prompt_attention_mask")
    if mask is not None and not bool(mask.all()):
        keys = torch.cat(
            [mask.repeat_interleave(group_size, 0), torch.ones_like(generated)], dim=1
        ).bool()
        scoring_kwargs = {
            "position_ids": (keys.long().cumsum(-1) - 1).clamp(min=0),
            "attention_mask": keys[:, None, None, :].expand(-1, 1, keys.size(1), -1),
        }
    logits, _ = model(completion, labels=None, **scoring_kwargs)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    start = prompt_ids.size(1) - 1
    selected = log_probs[:, start : start + 4].gather(-1, generated.unsqueeze(-1)).squeeze(-1)
    return _build_rollout_output(
        prompt_ids,
        generated,
        list(selected.unbind(dim=1)),
        torch.full((count,), 4, device=prompt_ids.device),
        group_size,
        metadata,
        prompt_attention_mask=mask,
    )


def worker(args):
    import torch
    import torch.distributed as dist

    from ironcore.trainers import DPOTrainer, GRPOTrainer, LanguageModelTrainer
    from ironcore.training_utils import forward_step, get_loss_func

    torch.set_num_threads(1)
    if args.device == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    trainer_cls = {
        "pretrain": LanguageModelTrainer,
        "sft": LanguageModelTrainer,
        "dpo": DPOTrainer,
        "grpo": GRPOTrainer,
    }[args.task]

    class ExperimentTrainer(trainer_cls):
        def _get_data_iterator(self):
            return {split: data_batches(self.config, split) for split in ("train", "eval")}

        def _compute_grad_and_param_norms(self, step):
            self._prepare_gradients()
            if not hasattr(self, "first_gradient"):
                self.first_gradient = collect_parameters(self.model, gradients=True)
                self.first_gradient = {name: value for name, value in self.first_gradient.items()}
            norms = super()._compute_grad_and_param_norms(step)
            return norms

        def log_training(self, step, loss, *args):
            self.losses.append(loss)
            super().log_training(step, loss, *args)

    if args.task == "grpo" and not args.online:
        import ironcore.trainers.grpo_trainer as grpo_module

        grpo_module.generate_rollouts_batched = replay_rollouts
    config = make_config(args)
    config._stateful_data = args.stateful_data
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with ExperimentTrainer(config, forward_step, get_loss_func(args.task)) as trainer:
        trainer.losses = []
        if args.moe_idle:
            from ironcore.layers.moe import TopKRouter

            with torch.no_grad():
                for module in trainer.model.modules():
                    if isinstance(module, TopKRouter):
                        module.weight.zero_()
                        module.bias.copy_(module.bias.new_tensor([2.0, 1.0, -2.0, -3.0]))
        initial = collect_parameters(trainer.model)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        trainer.train()
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        final = collect_parameters(trainer.model)
        reference = (
            collect_parameters(trainer.reference_model) if args.task in ("dpo", "grpo") else {}
        )
        if reference:
            assert all(
                not p.requires_grad and p.grad is None for p in trainer.reference_model.parameters()
            )
        if args.task == "grpo":
            evaluation = trainer.evaluate(config.operation.train_steps)
        else:
            trainer.model.eval()
            evaluation = dict(
                zip(
                    ("loss", "accuracy"),
                    trainer._eval_step(next(trainer.data_iterator["eval"])),
                    strict=True,
                )
            )
        state = {
            "initial": initial,
            "final": final,
            "gradient": trainer.first_gradient,
            "reference": reference,
            "scheduler": trainer.lr_scheduler.state_dict(),
            "scaler": trainer.scaler.state_dict(),
        }
        torch.save(state, args.run_dir / f"rank{dist.get_rank()}.pt")
        summary = {
            "task": args.task,
            "case": args.case,
            "precision": args.precision,
            "device": str(next(trainer.model.parameters()).device),
            "parameter_count_full": sum(t.numel() for t in initial.values()),
            "losses": trainer.losses,
            "evaluation": evaluation,
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()
            if args.device == "cuda"
            else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()
            if args.device == "cuda"
            else None,
            "max_weight_change": max((final[k] - initial[k]).abs().max().item() for k in initial),
        }
        (args.run_dir / f"rank{dist.get_rank()}.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )


def compare_tensors(left, right, atol, rtol):
    import torch

    assert left.keys() == right.keys(), "Parameter keys differ"
    largest = 0.0
    for name in left:
        assert left[name].shape == right[name].shape, name
        largest = max(largest, (left[name] - right[name]).abs().max().item())
        torch.testing.assert_close(left[name], right[name], atol=atol, rtol=rtol, msg=name)
    return largest


def run(args):
    import torch

    if args.output.exists():
        raise ValueError("Use a new output directory to avoid mixing experiment artifacts")
    if args.device == "cuda" and torch.cuda.device_count() < 2:
        raise RuntimeError(
            "CUDA validation requires two accessible GPUs; no CPU fallback is allowed"
        )
    args.output.mkdir(parents=True)
    create_tokenizer(args.output / "tokenizer")
    if args.stateful_data:
        create_stateful_fixture(args.output / "data")
    report = {
        "status": "running",
        "device": args.device,
        "torch": torch.__version__,
        "steps": args.steps,
        "precision": args.precision,
        "parameter_precision": args.parameter_precision,
        "architecture": args.architecture,
        "loss_chunk_size": args.loss_chunk_size,
        "moe": args.moe,
        "moe_idle": args.moe_idle,
        "moe_aux_alpha": args.moe_aux_alpha,
        "grpo_objective": args.grpo_objective,
        "stateful_data": args.stateful_data,
        "data_workers": args.data_workers,
        "fsdp": args.fsdp,
        "distributed_optimizer": args.distributed_optimizer,
        "recompute_linear_ce": args.recompute_linear_ce,
        "moe_backend": args.moe_backend,
        "baseline_case": args.cases.split(",")[0],
        "checks": [],
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    tasks = args.tasks.split(",")
    cases = args.cases.split(",")
    atol, rtol = (2e-5, 2e-4) if args.precision == "float32" else (5e-3, 5e-2)

    def launch(task, case, run_dir, stop=None, online=False):
        count = 2 if case in ("dp2", "tp2", "ep2") else 1
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={count}",
            str(Path(__file__).resolve()),
            "--worker",
            "--task",
            task,
            "--case",
            case,
            "--device",
            args.device,
            "--precision",
            args.precision,
            "--architecture",
            args.architecture,
            "--parameter-precision",
            args.parameter_precision,
            "--grpo-objective",
            args.grpo_objective,
            "--steps",
            str(args.steps),
            "--output",
            str(args.output),
            "--run-dir",
            str(run_dir),
        ]
        if args.stateful_data:
            cmd += ["--stateful-data", "--data-workers", str(args.data_workers)]
        if stop:
            cmd += ["--stop", str(stop)]
        if online:
            cmd += ["--online"]
        if args.loss_chunk_size is not None:
            cmd += ["--loss-chunk-size", str(args.loss_chunk_size)]
        if args.moe:
            cmd += ["--moe", "--moe-aux-alpha", str(args.moe_aux_alpha)]
        if args.moe_idle:
            cmd += ["--moe-idle"]
        if args.recompute_linear_ce:
            cmd += ["--recompute-linear-ce"]
        cmd += ["--moe-backend", args.moe_backend]
        if args.fsdp:
            cmd += ["--fsdp"]
        if args.distributed_optimizer:
            cmd += ["--distributed-optimizer"]
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "run.log").open("a") as log:
            completed = subprocess.run(
                cmd, cwd=ROOT, env=env, stdout=log, stderr=log, timeout=args.timeout, check=False
            )
        if completed.returncode:
            raise RuntimeError(
                f"{task}/{case} failed ({completed.returncode}); see {run_dir / 'run.log'}"
            )
        return torch.load(run_dir / "rank0.pt", weights_only=True)

    try:
        for task in tasks:
            reference = None
            for case in cases:
                run_dir = args.output / task / case
                print(f"Running {task}/{case}", flush=True)
                state = launch(task, case, run_dir)
                summary = json.loads((run_dir / "rank0.json").read_text())
                assert summary["max_weight_change"] > 0, f"{task}/{case}: policy did not update"
                assert all(torch.isfinite(torch.tensor(summary["losses"]))), "Nonfinite losses"
                if state["reference"]:
                    compare_tensors(state["initial"], state["reference"], 0, 0)
                if reference is None:
                    reference = state
                    reference_summary = summary
                else:
                    errors = {
                        key: compare_tensors(reference[key], state[key], atol, rtol)
                        for key in ("initial", "gradient", "final", "reference")
                    }
                    torch.testing.assert_close(
                        torch.tensor(reference_summary["losses"]),
                        torch.tensor(summary["losses"]),
                        atol=atol,
                        rtol=rtol,
                    )
                    assert state["scheduler"] == reference["scheduler"]
                    report["checks"].append({"task": task, "case": case, "max_abs_errors": errors})
                # Check replicated DP weights on every worker, not rank 0 only.
                if case == "dp2":
                    peer = torch.load(run_dir / "rank1.pt", weights_only=True)
                    compare_tensors(state["final"], peer["final"], 0, 0)
                resumed_dir = args.output / task / (case + "_resume")
                launch(task, case, resumed_dir, stop=args.steps // 2)
                resumed = launch(task, case, resumed_dir)
                resume_error = compare_tensors(state["final"], resumed["final"], 0, 0)
                compare_tensors(state["reference"], resumed["reference"], 0, 0)
                assert state["scheduler"] == resumed["scheduler"]
                assert state["scaler"] == resumed["scaler"]
                resumed_summary = json.loads((resumed_dir / "rank0.json").read_text())
                assert summary["losses"][args.steps // 2 :] == resumed_summary["losses"]
                report["checks"].append(
                    {"task": task, "case": case, "resume_max_abs_error": resume_error}
                )
            if task == "grpo":
                for case in (c for c in ("full", "dp2", "tp2", "ep2") if c in cases):
                    run_dir = args.output / task / (case + "_online")
                    launch(task, case, run_dir, stop=2, online=True)
                    summary = json.loads((run_dir / "rank0.json").read_text())
                    assert summary["max_weight_change"] > 0, "Online GRPO produced no policy update"
                    report["checks"].append({"task": task, "case": case, "online": summary})
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validation passed: {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--parameter-precision", choices=["float32", "model"], default="float32")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--distributed-optimizer", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/trainer_validation"))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--architecture", choices=["gpt2", "cs336"], default="gpt2")
    parser.add_argument("--loss-chunk-size", type=int)
    parser.add_argument(
        "--moe", action="store_true", help="Test 4 routed/1 shared expert, top-2; aux alpha=0"
    )
    parser.add_argument("--recompute-linear-ce", action="store_true")
    parser.add_argument("--moe-backend", choices=["loop", "batched"], default="loop")
    parser.add_argument("--moe-idle", action="store_true", help="Start with two unselected experts")
    parser.add_argument("--moe-aux-alpha", type=float, default=0.0)
    parser.add_argument(
        "--cases",
        default="full,accum,dp2,tp2",
        help="Comma-separated cases; first is comparison baseline",
    )
    parser.add_argument("--stateful-data", action="store_true")
    parser.add_argument("--data-workers", type=int, default=0)
    parser.add_argument("--tasks", default="pretrain,sft,dpo,grpo")
    parser.add_argument("--grpo-objective", choices=["gspo", "grpo"], default="gspo")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--task", choices=["pretrain", "sft", "dpo", "grpo"], help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--case", choices=["full", "accum", "dp2", "tp2", "ep2"], help=argparse.SUPPRESS
    )
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stop", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--online", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dropout", type=float, default=0.0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stateful_data and (
        (args.worker and args.task == "dpo") or (not args.worker and "dpo" in args.tasks.split(","))
    ):
        parser.error("Stateful fixture currently supports pretrain/sft/grpo")
    if args.moe_idle and not args.moe:
        parser.error("--moe-idle requires --moe")
    if "ep2" in args.cases.split(",") and not args.moe:
        parser.error("ep2 requires --moe")
    if not set(args.cases.split(",")) <= {"full", "accum", "dp2", "tp2", "ep2"}:
        parser.error("Unknown case in --cases")
    args.output = args.output.resolve()
    if args.steps < 4 or args.steps % 2:
        parser.error("--steps must be an even number >= 4")
    if not args.worker and not set(args.tasks.split(",")) <= {"pretrain", "sft", "dpo", "grpo"}:
        parser.error("Unknown task in --tasks")
    (worker if args.worker else run)(args)


if __name__ == "__main__":
    main()
