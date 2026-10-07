# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Compare actual IronCore/TRL training loops with a shared completion fixture.

Optional reference: pip install trl==0.29.0. No TRL dependency in production.
Launch with torchrun, including for CPU/world=1. Use a new output directory.
Scripted generation isolates the update path; it is not an on-policy sampler
benchmark or a demonstration of reasoning quality.
"""

import argparse
import hashlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def canonical(model, gradients=False):
    from ironcore.checkpointing.weight_mapping import WeightMapper, get_architecture

    model = getattr(model, "module", model)
    state = {
        name: (parameter.grad if gradients else parameter).detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if not gradients or parameter.grad is not None
    }
    return WeightMapper(get_architecture("llama"), model.config.model.num_layers).ironcore_to_hf(
        state, strict=False
    )


def make_checkpoint(path):
    from scripts.validate_trainers import create_tokenizer
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(83)
    path.mkdir(parents=True)
    create_tokenizer(path)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-5,
        rope_theta=100000,
        max_position_embeddings=8192,
        tie_word_embeddings=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=None,
        attention_dropout=0.0,
    )
    LlamaForCausalLM(config).save_pretrained(path)


def configuration(args, model_dir):
    from scripts.benchmark_alignment import config_for

    cfg = config_for(
        argparse.Namespace(
            model_dir=model_dir,
            context=32,
            micro_batch=1,
            global_batch=int(os.environ["WORLD_SIZE"]),
            output=args.output,
            steps=args.rollouts,
            task="grpo",
            lr=args.lr,
            objective="grpo",
            eval_samples=4,
            max_new_tokens=4,
            seed=83,
        )
    )
    if args.model_dir is None:
        cfg.model.d_model = 64
        cfg.model.d_ffn = 128
        cfg.model.num_layers = 2
        cfg.model.num_attention_heads = 4
        cfg.model.num_attention_groups = 2
        cfg.model.head_dim = 16
    cfg.model.precision = args.precision
    cfg.trainer.vocab_padding_unit = 1
    cfg.trainer.log_interval = 1000000
    cfg.parallel.dist_backend = "nccl" if args.device == "cuda" else "gloo"
    cfg.alignment.grpo_eps = 1e-4
    cfg.alignment.grpo_beta = args.beta
    cfg.alignment.grpo_clip_eps = 0.2
    cfg.alignment.grpo_rollout_micro_group_size = 4
    cfg.alignment.metrics_interval = 0
    cfg.alignment.generation.temperature = 1.0
    cfg.alignment.generation.top_k = 0
    cfg.alignment.generation.top_p = 1.0
    cfg.alignment.reward_manager.functions[0].type = "soft_keyword"
    cfg.alignment.reward_manager.functions[0].keyword = "yes"
    return cfg


def run(args):
    import inspect
    import types

    import trl
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig
    from trl import GRPOTrainer as ReferenceTrainer

    from ironcore.alignment.rewards.builtin import SoftKeywordRewardFunction
    from ironcore.alignment.rollout import _build_rollout_output
    from ironcore.trainers import GRPOTrainer
    from ironcore.training_utils import loss_func_sft

    if trl.__version__ != "0.29.0":
        raise ValueError("This reference protocol requires trl==0.29.0")
    if args.output.exists():
        raise ValueError("Use a new output directory")
    torch.set_num_threads(2)
    if args.device == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.backends.cuda.matmul.allow_tf32 = False
    model_dir = args.model_dir or args.output.parent / (args.output.name + "-initial-model")
    if args.model_dir is None and int(os.environ["RANK"]) == 0:
        make_checkpoint(model_dir)
    # Trainer initialization establishes the process group before model loading;
    # waiting for the initial checkpoint avoids reading a partially written file.
    if int(os.environ["RANK"]) != 0:
        import time

        while not (model_dir / "model.safetensors").exists():
            time.sleep(0.1)
    cfg = configuration(args, model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompt = "t3 t6" if args.model_dir is None else "Answer yes or no: Is two plus two four?"
    prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False)).unsqueeze(0)
    candidates = ["t4 t4 t4", "t5 t5 t5"] if args.model_dir is None else ["yes", "no"]
    encoded = [tokenizer.encode(text, add_special_tokens=False) for text in candidates]
    # Alternate the two candidates in every group. Include EOS and variable lengths.
    responses = [encoded[index % 2] + [tokenizer.eos_token_id] for index in range(4)]
    width = max(map(len, responses))
    response_ids = torch.tensor(
        [ids + [tokenizer.pad_token_id] * (width - len(ids)) for ids in responses]
    )
    lengths = torch.tensor(list(map(len, responses)))
    keyword = candidates[0].split()[0]
    cfg.alignment.reward_manager.functions[0].keyword = keyword
    reward_fn = SoftKeywordRewardFunction(keyword=keyword)
    snapshots = {"ironcore": [], "trl": []}
    loss_snapshots = {"ironcore": [], "trl": []}
    scores = {"ironcore": {}, "trl": {}}
    hf = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="sdpa"
    )
    initial = {name: value.detach().cpu().clone() for name, value in hf.named_parameters()}

    def score_candidates(model, native=False, context=None):
        device = next(model.parameters()).device
        ids = torch.cat([prompt_ids.repeat_interleave(4, 0), response_ids], dim=1).to(device)
        with torch.no_grad(), context or nullcontext():
            output = model(ids, labels=None) if native else model(ids, use_cache=False)
            logits = output[0] if native else output.logits
            selected = (
                logits[:, prompt_ids.size(1) - 1 : -1]
                .float()
                .log_softmax(-1)
                .gather(-1, response_ids.to(device).unsqueeze(-1))
                .squeeze(-1)
            )
            valid = torch.arange(width, device=device)[None, :] < lengths.to(device)[:, None]
            sequence = (selected * valid).sum(-1)
            return {
                "positive_sequence_log_probability": sequence[0].item(),
                "negative_sequence_log_probability": sequence[1].item(),
                "positive_mass_conditional_on_two_candidates": sequence[:2].softmax(0)[0].item(),
            }

    def fixed_rollout(model, prompt_ids, group_size, metadata, **kwargs):
        assert group_size == 4 and prompt_ids.size(0) == 1
        generated = response_ids.to(prompt_ids.device)
        full = torch.cat([prompt_ids.repeat_interleave(4, 0), generated], dim=1)
        logits = model(full, labels=None)[0].float()
        selected = (
            logits[:, prompt_ids.size(1) - 1 : -1]
            .log_softmax(-1)
            .gather(-1, generated.unsqueeze(-1))
            .squeeze(-1)
        )
        return _build_rollout_output(
            prompt_ids,
            generated,
            list(selected.unbind(1)),
            lengths.to(prompt_ids.device),
            group_size,
            metadata,
            prompt_attention_mask=kwargs.get("prompt_attention_mask"),
        )

    class TargetTrainer(GRPOTrainer):
        def _get_data_iterator(self):
            def batches():
                while True:
                    yield {
                        "input_ids": prompt_ids,
                        "attention_mask": torch.ones_like(prompt_ids),
                        "prompts": [prompt],
                        "metadata": [{}],
                    }

            return {"train": batches()}

        def _post_checkpoint_load(self, step):
            super()._post_checkpoint_load(step)
            state = canonical(self.model)
            for name, value in initial.items():
                torch.testing.assert_close(state[name], value, atol=0, rtol=0, msg=name)
            scores["ironcore"]["before"] = score_candidates(
                self.model, native=True, context=self.context["autocast"]
            )

        def _compute_grpo_loss(self, *items, **kwargs):
            loss, metrics = super()._compute_grpo_loss(*items, **kwargs)
            loss_snapshots["ironcore"].append(loss.detach().item())
            return loss, metrics

        def _optimizer_step(self):
            gradient = canonical(self.model, gradients=True)
            super()._optimizer_step()
            snapshots["ironcore"].append({"gradient": gradient, "weights": canonical(self.model)})

    class RecordingReference(ReferenceTrainer):
        def compute_loss(self, *items, **kwargs):
            loss = super().compute_loss(*items, **kwargs)
            loss_snapshots["trl"].append(loss.detach().item())
            return loss

    class Capture(TrainerCallback):
        def on_optimizer_step(self, args, state, control, model=None, **kwargs):
            snapshots["trl"].append(
                {
                    "gradient": {
                        k: v.grad.detach().cpu().clone()
                        for k, v in model.named_parameters()
                        if v.grad is not None
                    },
                    "weights": {k: v.detach().cpu().clone() for k, v in model.named_parameters()},
                }
            )

    def fixed_generate(model, input_ids, **kwargs):
        assert input_ids.size(0) == 4
        return torch.cat([input_ids, response_ids.to(input_ids.device)], dim=1)

    def rewards(prompts, completions, **kwargs):
        return [reward_fn.compute(p, c, {}) for p, c in zip(prompts, completions, strict=True)]

    with patch("ironcore.trainers.grpo_trainer.generate_rollouts_batched", fixed_rollout):
        with TargetTrainer(cfg, None, loss_func_sft) as target:
            target.train()
            scores["ironcore"]["after"] = score_candidates(
                target.model, native=True, context=target.context["autocast"]
            )
            hf.generate = types.MethodType(fixed_generate, hf)
            reference = RecordingReference(
                hf,
                reward_funcs=rewards,
                args=GRPOConfig(
                    output_dir=str(args.output / "trl"),
                    max_steps=2 * args.rollouts,
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=1,
                    num_generations=4,
                    num_iterations=2,
                    max_completion_length=width,
                    learning_rate=args.lr,
                    lr_scheduler_type="constant",
                    adam_beta1=cfg.optim.adam_beta1,
                    adam_beta2=cfg.optim.adam_beta2,
                    adam_epsilon=cfg.optim.adam_eps,
                    weight_decay=cfg.optim.weight_decay,
                    max_grad_norm=cfg.optim.clip_grad,
                    optim="adamw_torch",
                    beta=args.beta,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    loss_type="grpo",
                    importance_sampling_level="token",
                    scale_rewards="group",
                    epsilon=0.2,
                    epsilon_high=0.2,
                    use_bias_correction_kl=False,
                    report_to=[],
                    disable_tqdm=True,
                    save_strategy="no",
                    logging_strategy="no",
                    use_cpu=args.device == "cpu",
                    bf16=args.precision == "bfloat16",
                    dataloader_num_workers=0,
                ),
                train_dataset=Dataset.from_dict(
                    {"prompt": [prompt] * max(8, cfg.parallel.world_size)}
                ),
                processing_class=tokenizer,
                callbacks=[Capture()],
            )
            reference_context = (
                torch.autocast(device_type=args.device, dtype=torch.bfloat16)
                if args.precision == "bfloat16"
                else nullcontext()
            )
            scores["trl"]["before"] = score_candidates(hf, context=reference_context)
            reference.train()
            scores["trl"]["after"] = score_candidates(hf, context=reference_context)
            references = [canonical(target.reference_model)]
            if reference.ref_model is not None:
                references.append(
                    {name: p.detach().cpu() for name, p in reference.ref_model.named_parameters()}
                )
            for frozen in references:
                for name, value in initial.items():
                    torch.testing.assert_close(frozen[name], value, atol=0, rtol=0, msg=name)
            assert len(snapshots["ironcore"]) == len(snapshots["trl"]) == args.rollouts * 2
            errors = []
            atol, rtol = (2e-5, 2e-4) if args.precision == "float32" else (5e-3, 5e-2)
            for step, (left, right) in enumerate(
                zip(snapshots["ironcore"], snapshots["trl"], strict=True)
            ):
                error = {"optimizer_step": step + 1}
                for kind in ["gradient", "weights"]:
                    assert left[kind].keys() == right[kind].keys(), (
                        kind,
                        left[kind].keys(),
                        right[kind].keys(),
                    )
                    error[kind + "_max_abs_error"] = max(
                        (left[kind][name] - right[kind][name]).abs().max().item()
                        for name in left[kind]
                    )
                    for name in left[kind]:
                        torch.testing.assert_close(
                            left[kind][name], right[kind][name], atol=atol, rtol=rtol, msg=name
                        )
                error["loss_abs_error"] = abs(
                    loss_snapshots["ironcore"][step] - loss_snapshots["trl"][step]
                )
                if error["loss_abs_error"] > atol:
                    raise AssertionError(error)
                errors.append(error)
            args.output.mkdir(parents=True, exist_ok=True)
            reference_file = Path(inspect.getfile(ReferenceTrainer))
            result = {
                "status": "passed",
                "protocol": "shared scripted completions; real trainer loops",
                "trl_version": trl.__version__,
                "trl_source_sha256": hashlib.sha256(reference_file.read_bytes()).hexdigest(),
                "ironcore_commit": "aaec6b925e47ef9257c5b819ae0b3853fbd7badf",
                "world_size": cfg.parallel.world_size,
                "precision": args.precision,
                "parameters": sum(p.numel() for p in hf.parameters()),
                "rollouts": args.rollouts,
                "optimizer_updates": args.rollouts * 2,
                "atol": atol,
                "rtol": rtol,
                "errors": errors,
                "losses": loss_snapshots,
                "candidate_scores": scores,
                "initial_weights_exact": True,
                "reference_weights_exact_frozen": True,
                "command": [sys.executable, *sys.argv],
                "experiment_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "reward_fixture": [
                    reward_fn.compute(prompt, text, {})
                    for text in tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                ],
                "normalization": {
                    "group_std_correction": 1,
                    "group_eps": 1e-4,
                    "loss_type": "grpo",
                    "temperature": 1.0,
                },
                "limitations": [
                    "Scripted completions are not samples from the current policy; this isolates numerical update equivalence",
                    "No reasoning benchmark quality claim",
                ],
            }
            (args.output / f"rank{dist.get_rank()}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.1)
    run(parser.parse_args())
