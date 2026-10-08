# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""LoRA TP=1/2 numerical parity on a scaled SmolLM2 decoder.

torchrun --standalone --nproc_per_node=2 -m tests.multi_gpu.test_lora_tp --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from tests.fixtures.lora_tp import shard_like, smollm2_lora_model

from ironcore.parallel.parallel_states import destroy_model_parallel, initialize_model_parallel

pytestmark = [pytest.mark.mp, pytest.mark.skipif("RANK" not in os.environ, reason="Needs torchrun")]


def _model(patch, variant: str, tp_size: int, checkpoint: Path | None, dropout: float):
    if variant == "SmolLM2":
        return smollm2_lora_model(patch, tp_size, checkpoint, dropout)
    from tests.fixtures.gemma4 import gemma4_pair

    targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj",
        "per_layer_input_gate",
        "per_layer_projection",
    ]
    return gemma4_pair(
        patch, variant, lora=True, tp_size=tp_size, lora_targets=targets, lora_dropout=dropout
    )[0]


def _trainer(model, tokens: torch.Tensor, learning_rate: float = 1e-3):
    from tests.unit.trainers.test_training_correctness import make_trainer

    from ironcore.training_utils import forward_step, loss_func

    model.train()
    model.loss_fn = loss_func
    batch = {"input_ids": tokens[:, :-1], "labels": tokens[:, 1:]}
    trainer = make_trainer(model, [batch] * 6, loss_func, accumulation=2)
    trainer.forward_step_func = forward_step
    # Tiny Gemma norms amplify nearly-zero gradient reduction noise in AdamW.
    # Use the same supported epsilon in both trainers; direct gradient checks
    # remain stricter and use nonzero adapters to expose communication errors.
    trainer.optimizer.param_groups[0]["eps"] = 1e-4
    trainer.optimizer.param_groups[0]["lr"] = learning_rate
    return trainer


def _adapters(model) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}


def check_lora_tp(
    variant: str,
    device: torch.device,
    checkpoint: Path | None = None,
    dropout: float = 0.0,
    recompute: str | None = None,
) -> dict:
    """Compare nonzero adapter gradients and three production trainer updates."""
    from ironcore.parallel.random import reset_tensor_parallel_rng_tracker

    tokens = torch.tensor([[2, 3, 4, 5], [18, 19, 20, 21]], device=device)
    with pytest.MonkeyPatch.context() as single_patch:
        torch.manual_seed(42)
        single = _model(single_patch, variant, 1, checkpoint, dropout).to(device).train()
        with torch.no_grad():
            for name, p in single.named_parameters():
                if name.endswith("lora_B"):
                    p.copy_(torch.linspace(-0.01, 0.01, p.numel(), device=device).reshape_as(p))
        # Keep full pretrained snapshots off GPU; the TP=1 reference and TP=2
        # model already coexist on each rank during the numerical comparison.
        initial = {
            name: value.detach().cpu().clone() for name, value in single.state_dict().items()
        }
        reset_tensor_parallel_rng_tracker()
        expected, _ = single(tokens)
        expected.square().mean().backward()
        gradients = {
            name: p.grad.clone() for name, p in single.named_parameters() if p.requires_grad
        }
        single.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, p in single.named_parameters():
                if name.endswith("lora_B"):
                    p.zero_()
        train_start = {
            name: value.detach().cpu().clone() for name, value in single.state_dict().items()
        }
        single_trainer = _trainer(single, tokens, learning_rate=1e-4 if checkpoint else 1e-3)
        reset_tensor_parallel_rng_tracker()
        records = []
        for step in range(3):
            loss, norm, _ = single_trainer.train_step(step)
            records.append((loss, norm, _adapters(single)))
    with pytest.MonkeyPatch.context() as patch:
        torch.manual_seed(42)
        native = _model(patch, variant, 2, checkpoint, dropout).to(device).train()
        native.load_state_dict(
            {name: shard_like(native, name, full) for name, full in initial.items()}, strict=True
        )
        if recompute:
            native.config.operation.activation_recompute = True
            native.config.operation.recompute_strategy = recompute
            native.model.activation_recompute = True
            native.model.use_reentrant = recompute == "optimized"
        reset_tensor_parallel_rng_tracker()
        actual, _ = native(tokens)
        # Full pretrained decoders accumulate more FP32 reduction error than
        # the two-layer fixture. The gradient comparison remains independent.
        logit_atol = 2e-4 if checkpoint else 2e-5
        torch.testing.assert_close(actual, expected, atol=logit_atol, rtol=2e-5)
        actual.square().mean().backward()
        max_gradient_error = 0.0
        max_relative_gradient_error = 0.0
        for name, p in native.named_parameters():
            if p.requires_grad:
                assert p.shape == gradients[name].shape, name
                assert not getattr(p, "is_tp_sharded", False), name
                assert p.grad is not None, name
                max_gradient_error = max(
                    max_gradient_error, float((p.grad - gradients[name]).abs().max())
                )
                if checkpoint:
                    # Deep pretrained FP32 reductions can differ near zero.
                    # Bound each complete adapter's relative L2 error instead
                    # of assigning a large absolute tolerance to every entry.
                    error = (p.grad - gradients[name]).norm()
                    scale = gradients[name].norm()
                    assert torch.isfinite(p.grad).all()
                    assert error <= 1e-4 * scale + 1e-6, (
                        f"{name}: gradient error {error}, norm {scale}"
                    )
                    max_relative_gradient_error = max(
                        max_relative_gradient_error, float(error / scale.clamp_min(1e-7))
                    )
                else:
                    torch.testing.assert_close(
                        p.grad,
                        gradients[name],
                        atol=1e-6,
                        rtol=1e-4,
                        msg=lambda info, key=name: f"rank={dist.get_rank()} {key}: {info}",
                    )
            else:
                assert p.grad is None, name
        native.load_state_dict(
            {name: shard_like(native, name, full) for name, full in train_start.items()},
            strict=True,
        )
        native.zero_grad(set_to_none=True)
        trainer = _trainer(native, tokens, learning_rate=1e-4 if checkpoint else 1e-3)
        reset_tensor_parallel_rng_tracker()
        loss_differences = []
        for step in range(3):
            loss, norm, _ = trainer.train_step(step)
            expected_loss, expected_norm, updated = records[step]
            if checkpoint:
                print(
                    f"rank={dist.get_rank()} step={step}: loss={loss} vs {expected_loss}, grad_norm={norm} vs {expected_norm}",
                    flush=True,
                )
            loss_differences.append(abs(loss - expected_loss))
            assert loss == pytest.approx(expected_loss, abs=1e-4 if checkpoint else 2e-5), (
                f"{variant} {recompute} dropout={dropout} step={step}: loss={loss}, expected={expected_loss}"
            )
            assert norm == pytest.approx(expected_norm, rel=1e-4, abs=1e-6), (
                f"step={step}: norm={norm}, expected={expected_norm}"
            )
            for name, p in native.named_parameters():
                if p.requires_grad:
                    torch.testing.assert_close(
                        p,
                        updated[name],
                        atol=2e-5,
                        rtol=2e-4,
                        msg=lambda info, key=name, index=step: (
                            f"rank={dist.get_rank()} step={index} {key}: {info}"
                        ),
                    )
                    others = [torch.empty_like(p) for _ in range(2)]
                    dist.all_gather(others, p.detach())
                    torch.testing.assert_close(others[0], others[1], atol=0, rtol=0)
                else:
                    assert p.grad is None, name
        if checkpoint is None and dropout == 0.0 and recompute is None:
            _checkpoint_roundtrip(native, trainer, tokens, single, single_trainer)
        from ironcore.peft.utils import merge_lora_weights

        native.eval()
        before, _ = native(tokens)
        merge_lora_weights(native)
        after, _ = native(tokens)
        torch.testing.assert_close(after, before, atol=logit_atol, rtol=2e-5)
        result = {
            "variant": variant,
            "device": str(device),
            "precision": str(native.dtype),
            "tp_size": 2,
            "steps": 3,
            "learning_rate": 1e-4 if checkpoint else 1e-3,
            "adam_epsilon": 1e-4,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "dropout": dropout,
            "recompute": recompute,
            "max_gradient_error": max_gradient_error,
            "max_relative_gradient_l2_error": max_relative_gradient_error,
            "losses_tp1": [r[0] for r in records],
            "max_loss_difference": max(loss_differences),
            "adapter_parameters": sum(p.numel() for p in single.parameters() if p.requires_grad),
        }
        print(f"rank={dist.get_rank()} LoRA parity passed: {result}", flush=True)
        return result


def _checkpoint_roundtrip(model, trainer, tokens: torch.Tensor, single, single_trainer) -> None:
    from ironcore.checkpointing.native import load_checkpoint, save_checkpoint

    paths = [tempfile.mkdtemp(prefix="ironcore-lora-tp-") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(paths)
    root = paths[0]
    model.config.trainer.model_path = root
    model.config.operation.save_full_model = True
    expected = _adapters(model)
    moments = {
        name: trainer.optimizer.state[p]["exp_avg"].clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    for distributed in [False, True]:
        model.config.operation.save_dist_ckpt = distributed
        save_checkpoint(model.config, model, trainer.optimizer, trainer.lr_scheduler, step=2)
        if not distributed:
            from ironcore.parallel import parallel_states

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(parallel_states, "_TENSOR_MODEL_PARALLEL_WORLD_SIZE", 1)
                patch.setattr(parallel_states, "_DATA_PARALLEL_WORLD_SIZE", 1)
                patch.setattr(parallel_states, "get_tensor_model_parallel_rank", lambda: 0)
                single.config.trainer.model_path = root
                single.config.operation.save_dist_ckpt = False
                assert (
                    load_checkpoint(
                        single.config,
                        single,
                        single_trainer.optimizer,
                        single_trainer.lr_scheduler,
                        step=2,
                    )
                    == 2
                )
                for name, p in single.named_parameters():
                    if p.requires_grad:
                        torch.testing.assert_close(p, expected[name], atol=0, rtol=0)
                        torch.testing.assert_close(
                            single_trainer.optimizer.state[p]["exp_avg"],
                            moments[name],
                            atol=0,
                            rtol=0,
                        )
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.fill_(1.0)
        trainer.optimizer.state.clear()
        assert (
            load_checkpoint(model.config, model, trainer.optimizer, trainer.lr_scheduler, step=2)
            == 2
        )
        for name, p in model.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(p, expected[name], atol=0, rtol=0)
                torch.testing.assert_close(
                    trainer.optimizer.state[p]["exp_avg"], moments[name], atol=0, rtol=0
                )
    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(root)


def check_lora_bf16(
    variant: str,
    device: torch.device,
    checkpoint: Path | None = None,
    dropout: float = 0.0,
    recompute: str | None = None,
) -> dict:
    """Exercise BF16 autocast with FP32 master weights and native TP=2 training.

    This checks finite gradients, actual updates and exact rank replication;
    FP32 TP=1/2 numerical parity is checked separately by ``check_lora_tp``.
    """
    from ironcore.parallel.random import reset_tensor_parallel_rng_tracker

    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 validation requires a BF16-capable CUDA GPU")
    tokens = torch.tensor([[2, 3, 4, 5], [18, 19, 20, 21]], device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with pytest.MonkeyPatch.context() as patch:
        torch.manual_seed(42)
        model = _model(patch, variant, 2, checkpoint, dropout).to(device).train()
        if recompute:
            model.config.operation.activation_recompute = True
            model.config.operation.recompute_strategy = recompute
            model.model.activation_recompute = True
            model.model.use_reentrant = recompute == "optimized"
        # Nonzero B makes the initial A gradients observable too.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("lora_B"):
                    parameter.copy_(
                        torch.linspace(-0.01, 0.01, parameter.numel(), device=device).reshape_as(
                            parameter
                        )
                    )
        reset_tensor_parallel_rng_tracker()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(tokens)
            objective = logits.float().square().mean()
        assert logits.dtype == torch.bfloat16
        assert torch.isfinite(logits).all()
        objective.backward()
        adapter_count = 0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
                adapter_count += parameter.numel()
            else:
                assert parameter.grad is None, name
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("lora_B"):
                    parameter.zero_()
        initial = _adapters(model)
        trainer = _trainer(model, tokens, learning_rate=1e-4 if checkpoint else 1e-3)
        trainer.context["autocast"] = torch.autocast("cuda", dtype=torch.bfloat16)
        reset_tensor_parallel_rng_tracker()
        losses, norms = [], []
        for step in range(3):
            loss, norm, _ = trainer.train_step(step)
            assert 0 < loss < float("inf")
            assert 0 < norm < float("inf")
            losses.append(loss)
            norms.append(norm)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    assert torch.isfinite(parameter).all(), name
                    replicas = [torch.empty_like(parameter) for _ in range(2)]
                    dist.all_gather(replicas, parameter.detach())
                    torch.testing.assert_close(replicas[0], replicas[1], atol=0, rtol=0)
                else:
                    assert parameter.grad is None, name
        assert losses[-1] < losses[0], losses
        assert any(
            not torch.equal(parameter, initial[name])
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        result = {
            "variant": variant,
            "device": str(device),
            "precision": "bfloat16 autocast",
            "parameter_precision": "float32",
            "backend": dist.get_backend(),
            "tp_size": 2,
            "steps": 3,
            "learning_rate": 1e-4 if checkpoint else 1e-3,
            "adam_epsilon": 1e-4,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "dropout": dropout,
            "recompute": recompute,
            "losses": losses,
            "grad_norms": norms,
            "adapter_parameters": adapter_count,
            "rank_replicas_exact": True,
            "tp1_numerical_parity_checked": False,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        }
        print(f"rank={dist.get_rank()} BF16 LoRA training passed: {result}", flush=True)
        return result


@pytest.mark.parametrize("variant", ["SmolLM2", "E2B", "E4B", "31B"])
@pytest.mark.parametrize("recompute", [None, "standard", "optimized"])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_smollm2_lora_tp2(variant: str, recompute: str | None, dropout: float) -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    initialize_model_parallel(2, timeout_in_minutes=5.0)
    try:
        check_lora_tp(variant, torch.device("cuda", rank), dropout=dropout, recompute=recompute)
    finally:
        destroy_model_parallel()


@pytest.mark.parametrize("variant", ["SmolLM2", "E2B", "E4B", "31B"])
@pytest.mark.parametrize("recompute", [None, "standard", "optimized"])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_lora_tp2_bf16(variant: str, recompute: str | None, dropout: float) -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16-capable GPU required")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    initialize_model_parallel(2, timeout_in_minutes=5.0)
    try:
        check_lora_bf16(variant, torch.device("cuda", rank), dropout=dropout, recompute=recompute)
    finally:
        destroy_model_parallel()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--checkpoint", type=Path, help="Use actual local SmolLM2 weights instead of tiny layouts"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.precision == "bfloat16" and args.device != "cuda":
        parser.error("BF16 validation requires --device cuda")
    rank = int(os.environ["LOCAL_RANK"])
    if args.device == "cuda":
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if args.device == "cuda" else torch.device("cpu")
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo")
    initialize_model_parallel(2, timeout_in_minutes=5.0)
    try:
        results = []
        check = check_lora_bf16 if args.precision == "bfloat16" else check_lora_tp
        if args.checkpoint:
            results.append(check("SmolLM2", device, checkpoint=args.checkpoint))
        else:
            for variant in ["SmolLM2", "E2B", "E4B", "31B"]:
                for recompute in [None, "standard", "optimized"]:
                    for dropout in [0.0, 0.2]:
                        results.append(check(variant, device, dropout=dropout, recompute=recompute))
        if dist.get_rank() == 0 and args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
