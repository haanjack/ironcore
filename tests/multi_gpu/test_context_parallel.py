# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""CP numerical regressions, callable on CPU/Gloo or GPU/NCCL with torchrun.

python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m tests.multi_gpu.test_context_parallel --device cuda --backend ring
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tests.fixtures.lora_tp import shard_like, smollm2_lora_model
from tests.multi_gpu.test_lora_tp import _trainer

from ironcore.layers.context_parallel_attention import ContextParallelAttention
from ironcore.parallel import parallel_states as ps
from ironcore.parallel.context_parallel import synchronize_context_parallel_gradients

pytestmark = [pytest.mark.mp]


def check_attention(device: torch.device, backend: str) -> list[dict]:
    """Compare each owner shard's Q/K/V gradients with full causal attention."""
    rank, size = ps.get_context_parallel_rank(), ps.get_context_parallel_world_size()
    results = []
    dtypes = [torch.bfloat16, torch.float16] if backend == "ring" else [torch.float32]
    for dtype in dtypes:
        for heads, kv_heads in [(4, 4), (4, 2), (4, 1), (9, 3)]:
            for length in [size * 4, size * 32]:
                torch.manual_seed(432)
                q = torch.randn(
                    2, length, heads, 64, device=device, dtype=dtype, requires_grad=True
                )
                k = torch.randn(
                    2, length, kv_heads, 64, device=device, dtype=dtype, requires_grad=True
                )
                v = torch.randn_like(k, requires_grad=True)
                expected = F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=True,
                    enable_gqa=heads != kv_heads,
                ).transpose(1, 2)
                gradient = torch.randn_like(expected)
                expected.backward(gradient)
                leaves = [
                    t.detach().chunk(size, dim=1)[rank].contiguous().requires_grad_()
                    for t in (q, k, v)
                ]
                attention = ContextParallelAttention(backend)
                actual = attention(*leaves).reshape(2, length // size, heads, 64)
                actual.backward(gradient.chunk(size, dim=1)[rank].contiguous())
                errors = []
                tolerance = {torch.float32: 2e-5, torch.float16: 0.008, torch.bfloat16: 0.065}[
                    dtype
                ]
                pairs = [(actual, expected.chunk(size, dim=1)[rank])]
                pairs.extend(
                    (a.grad, b.grad.chunk(size, dim=1)[rank])
                    for a, b in zip(leaves, (q, k, v), strict=True)
                )
                for actual_value, expected_value in pairs:
                    torch.testing.assert_close(
                        actual_value, expected_value, atol=tolerance, rtol=0.015
                    )
                    errors.append((actual_value - expected_value).abs().max().item())
                # Changing later tokens must leave every earlier output intact.
                changed_k, changed_v = leaves[1].detach().clone(), leaves[2].detach().clone()
                if rank == size - 1:
                    changed_k[:, -1].add_(10)
                    changed_v[:, -1].add_(10)
                changed = attention(leaves[0].detach(), changed_k, changed_v).reshape_as(actual)
                before = actual.detach()[:, :-1] if rank == size - 1 else actual.detach()
                after = changed[:, :-1] if rank == size - 1 else changed
                torch.testing.assert_close(after, before, atol=0, rtol=0)
                maximum = torch.tensor(errors, device=device)
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
                results.append(
                    {
                        "dtype": str(dtype),
                        "heads": heads,
                        "kv_heads": kv_heads,
                        "length": length,
                        "max_errors_output_q_k_v": maximum.tolist(),
                    }
                )
    return results


def check_unused_gradients(device: torch.device) -> None:
    """An unused shard contributes zero, but an unused parameter stays None."""
    module = torch.nn.Module()
    module.register_parameter("active", torch.nn.Parameter(torch.ones(3, device=device)))
    module.register_parameter("unused", torch.nn.Parameter(torch.ones(3, device=device)))
    if ps.get_context_parallel_rank() == ps.get_context_parallel_world_size() - 1:
        module.active.grad = torch.ones_like(module.active)
    synchronize_context_parallel_gradients(module)
    torch.testing.assert_close(module.active.grad, torch.ones_like(module.active))
    assert module.unused.grad is None
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.1, weight_decay=0.5)
    optimizer.step()
    torch.testing.assert_close(module.unused, torch.ones_like(module.unused), atol=0, rtol=0)


def _autocast(device: torch.device, backend: str, dtype: torch.dtype = torch.bfloat16):
    return torch.autocast("cuda", dtype=dtype) if backend == "ring" else nullcontext()


def _parameters(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def check_model(
    device: torch.device,
    backend: str,
    *,
    lora: bool,
    recompute: str | None = None,
    checkpoint: Path | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
    moe: bool = False,
    mlp_chunk_size: int | None = None,
    expert_backend: str = "loop",
    blockwise_backend: str = "torch",
) -> dict:
    """Compare full decoder loss/gradients and three accumulated AdamW updates."""
    tp_size, cp_size = (
        ps.get_tensor_model_parallel_world_size(),
        ps.get_context_parallel_world_size(),
    )
    tokens = torch.tensor(
        [[2, 3, 4, 5, 6, 7, 8, 9, 10], [18, 19, 20, 21, 22, 23, 24, 25, 26]], device=device
    )
    if checkpoint and backend == "ring":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        text = (
            "Context parallelism distributes the tokens of a long sequence across several GPUs. "
            "Each device computes attention for its local queries while receiving keys and values "
            "from other devices. Training must return gradients to the original token owners "
            "and combine parameter gradients before updating the optimizer. "
        ) * 2
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) % 2 == 0:
            token_ids.append(token_ids[-1])
        tokens = torch.tensor([token_ids, token_ids], device=device)
    learning_rate = 1e-5 if checkpoint and backend == "ring" else 1e-4
    labels = tokens.roll(-1, dims=1)
    # An entire CP shard can have no valid labels; it still supplies remote KV gradients.
    labels[:, : (tokens.size(1) + cp_size - 1) // cp_size] = -100
    with pytest.MonkeyPatch.context() as patch:
        torch.manual_seed(42)
        single = (
            smollm2_lora_model(patch, checkpoint=checkpoint, lora=lora, moe=moe).to(device).train()
        )
        from ironcore.training_utils import loss_func

        single.loss_fn = loss_func
        with torch.no_grad():
            for name, p in single.named_parameters():
                if name.endswith("lora_B") and (not checkpoint or backend == "sdpa"):
                    p.copy_(torch.linspace(-0.01, 0.01, p.numel(), device=device).reshape_as(p))
        initial = {
            name: value.detach().cpu().clone() for name, value in single.state_dict().items()
        }
        with _autocast(device, backend, compute_dtype):
            expected_logits, _ = single(tokens)
            expected_logits = expected_logits.detach()
            expected_loss = single(tokens, labels=labels)
            if moe:
                from ironcore.training_utils import clear_moe_aux_loss, get_moe_aux_loss

                expected_loss = expected_loss + get_moe_aux_loss(single)
                clear_moe_aux_loss(single)
        expected_loss.backward()
        gradients = {
            name: p.grad.detach().cpu().clone()
            for name, p in single.named_parameters()
            if p.requires_grad
        }
        single.zero_grad(set_to_none=True)
        reference_trainer = _trainer(single, tokens, learning_rate=learning_rate)
        reference_trainer.context["autocast"] = _autocast(device, backend, compute_dtype)
        # BF16 shard reductions perturb tiny gradients. The same supported
        # Adam epsilon bounds their effect on the independent update comparison.
        reference_trainer.optimizer.param_groups[0]["eps"] = 1e-3
        records = []
        for step in range(3):
            loss, norm, _ = reference_trainer.train_step(step)
            records.append((loss, norm, _parameters(single)))
        if checkpoint:
            single.cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    with pytest.MonkeyPatch.context() as patch:
        native = (
            smollm2_lora_model(
                patch,
                tp_size=tp_size,
                cp_size=cp_size,
                cp_backend=backend,
                lora=lora,
                checkpoint=checkpoint,
                moe=moe,
                mlp_chunk_size=mlp_chunk_size,
                expert_backend=expert_backend,
                blockwise_backend=blockwise_backend,
            )
            .to(device)
            .train()
        )
        native.loss_fn = loss_func
        native.load_state_dict(
            {name: shard_like(native, name, full) for name, full in initial.items()}, strict=True
        )
        if recompute:
            native.config.operation.activation_recompute = True
            native.config.operation.recompute_strategy = recompute
            native.model.activation_recompute = True
            native.model.use_reentrant = recompute == "optimized"
            native.config.trainer.loss_chunk_size = 3
            native.config.trainer.recompute_linear_ce = recompute == "optimized"
        with _autocast(device, backend, compute_dtype):
            actual_logits, _ = native(tokens)
            actual_logits = actual_logits.detach()
            actual_loss = native(tokens, labels=labels)
            if moe:
                actual_loss = actual_loss + get_moe_aux_loss(native)
                clear_moe_aux_loss(native)
        logit_error = (
            actual_logits.float() - expected_logits.float()
        ).norm() / expected_logits.float().norm()
        probability_kl = 0.0
        if checkpoint and backend == "ring":
            # Changing GEMM shapes alters BF16 rounding before attention; deep
            # pretrained decoders amplify it, especially for negligible logits.
            # Compare normalized output error and objective-relevant probability
            # distributions, while FP32 checks retain strict elementwise parity.
            assert logit_error.item() < 0.035
            valid = labels != -100
            probability_kl = F.kl_div(
                actual_logits[valid].float().log_softmax(-1),
                expected_logits[valid].float().softmax(-1),
                reduction="batchmean",
            ).item()
            assert probability_kl < 0.005, probability_kl
        else:
            tolerance = 0.035 if backend == "ring" else (2e-4 if checkpoint else 3e-5)
            torch.testing.assert_close(
                actual_logits,
                expected_logits,
                atol=tolerance,
                rtol=0.025 if backend == "ring" else 3e-5,
            )
        torch.testing.assert_close(
            actual_loss,
            expected_loss,
            atol=0.02 if checkpoint and backend == "ring" else 0.005 if backend == "ring" else 3e-6,
            rtol=0.001,
        )
        counts_before = [
            module._total_selections
            for module in native.modules()
            if hasattr(module, "_total_selections")
        ]
        actual_loss.backward()
        assert counts_before == [
            module._total_selections
            for module in native.modules()
            if hasattr(module, "_total_selections")
        ]
        synchronize_context_parallel_gradients(native)
        gradient_error = 0.0
        gradient_difference_sq = torch.zeros((), device=device)
        gradient_reference_sq = torch.zeros((), device=device)
        for name, p in native.named_parameters():
            if p.requires_grad:
                expected_gradient = shard_like(native, name, gradients[name]).to(device)
                gradient_difference_sq += (p.grad - expected_gradient).square().sum()
                gradient_reference_sq += expected_gradient.square().sum()
                if not (checkpoint and backend == "ring"):
                    torch.testing.assert_close(
                        p.grad,
                        expected_gradient,
                        atol=0.001 if backend == "ring" else (1e-4 if checkpoint else 1e-5),
                        rtol=0.035 if backend == "ring" else (1e-3 if checkpoint else 3e-5),
                    )
                gradient_error = max(
                    gradient_error, (p.grad - expected_gradient).abs().max().item()
                )
        gradient_relative_error = (
            (gradient_difference_sq / gradient_reference_sq.clamp_min(1e-30)).sqrt().item()
        )
        gradient_tolerance = (
            0.08
            if checkpoint and backend == "ring" and compute_dtype == torch.bfloat16
            else 0.03
            if backend == "ring"
            else 0.001
        )
        assert gradient_relative_error < gradient_tolerance, gradient_relative_error
        native.zero_grad(set_to_none=True)
        trainer = _trainer(native, tokens, learning_rate=learning_rate)
        trainer.context["autocast"] = _autocast(device, backend, compute_dtype)
        trainer.optimizer.param_groups[0]["eps"] = 1e-3
        updates = []
        for step, (expected, expected_norm, parameters) in enumerate(records):
            loss, norm, _ = trainer.train_step(step)
            if checkpoint and dist.get_rank() == 0:
                print(
                    {
                        "step": step,
                        "loss": loss,
                        "reference_loss": expected,
                        "norm": norm,
                        "reference_norm": expected_norm,
                    },
                    flush=True,
                )
            assert loss == pytest.approx(
                expected,
                abs=0.02
                if checkpoint and backend == "ring"
                else 0.004
                if backend == "ring"
                else 3e-5,
            )
            assert norm == pytest.approx(
                expected_norm,
                rel=0.03
                if checkpoint and backend == "ring" and compute_dtype == torch.bfloat16
                else 0.02
                if backend == "ring"
                else (2e-4 if checkpoint else 5e-5),
                abs=2e-6,
            )
            parameter_error = 0.0
            parameter_difference_sq = torch.zeros((), device=device)
            parameter_update_sq = torch.zeros((), device=device)
            for name, p in native.named_parameters():
                if p.requires_grad:
                    target = shard_like(native, name, parameters[name]).to(device)
                    parameter_difference_sq += (p - target).square().sum()
                    parameter_update_sq += (
                        (target - shard_like(native, name, initial[name]).to(device)).square().sum()
                    )
                    if not (checkpoint and backend == "ring"):
                        torch.testing.assert_close(
                            p,
                            target,
                            atol=3e-5
                            if checkpoint and backend == "ring"
                            else 2e-5
                            if backend == "ring"
                            else 2e-7,
                            rtol=0.001 if backend == "ring" else 3e-5,
                        )
                    parameter_error = max(parameter_error, (p - target).abs().max().item())
            update_relative_error = (
                (parameter_difference_sq / parameter_update_sq.clamp_min(1e-30)).sqrt().item()
            )
            assert update_relative_error < (0.08 if backend == "ring" else 0.001), (
                update_relative_error
            )
            updates.append(
                {
                    "loss": loss,
                    "reference_loss": expected,
                    "norm": norm,
                    "max_parameter_error": parameter_error,
                    "update_relative_l2_error": update_relative_error,
                }
            )
        # Default config enables KV caches, but CP full-sequence evaluation
        # must never enter a stateful cache path.
        assert native.kv_cache_manager is None
        with torch.no_grad(), _autocast(device, backend, compute_dtype):
            train_scores, _ = native(tokens)
            native.eval()
            eval_scores, _ = native(tokens)
        torch.testing.assert_close(eval_scores, train_scores, atol=0, rtol=0)
        assert eval_scores.shape[:2] == tokens.shape
        native.train()
        with pytest.raises(ValueError, match="generation"):
            native.generate(tokens, max_new_tokens=1)
        with pytest.raises(ValueError, match="masks/caches"):
            native(tokens, use_cache=True)
        with _autocast(device, backend, compute_dtype):
            empty_loss = native(tokens, labels=torch.full_like(tokens, -100))
        assert empty_loss.item() == 0
        empty_loss.backward()
        synchronize_context_parallel_gradients(native)
        for p in native.parameters():
            if p.grad is not None:
                torch.testing.assert_close(p.grad, torch.zeros_like(p.grad), atol=0, rtol=0)
        native.zero_grad(set_to_none=True)
        trainer.data_iterator["train"] = iter(
            [
                {"input_ids": tokens, "labels": torch.full_like(tokens, -100)},
            ]
            * 2
        )
        with pytest.raises(ValueError, match="no valid objective"):
            trainer.train_step(3)
        if not checkpoint and not recompute:
            _checkpoint_roundtrip(native, trainer, single, reference_trainer)
        return {
            "lora": lora,
            "moe": moe,
            "mlp_chunk_size": mlp_chunk_size,
            "expert_backend": expert_backend,
            "backend": backend,
            "compute_dtype": str(compute_dtype if backend == "ring" else torch.float32),
            "tp": tp_size,
            "cp": cp_size,
            "recompute": recompute,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "learning_rate": learning_rate,
            "adam_epsilon": 1e-3,
            "max_gradient_error": gradient_error,
            "gradient_relative_l2_error": gradient_relative_error,
            "logit_relative_l2_error": logit_error.item(),
            "valid_token_probability_kl": probability_kl,
            "updates": updates,
        }


def _checkpoint_roundtrip(native, trainer, single, reference_trainer) -> None:
    """Only CP rank zero writes; CP size one can read the same native weights."""
    from ironcore.checkpointing.native import load_checkpoint, save_checkpoint
    from ironcore.trainers.checkpoint_state import load_trainer_state, save_trainer_state

    paths = [tempfile.mkdtemp(prefix="ironcore-cp-") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(paths)
    native.config.trainer.model_path = paths[0]
    native.config.operation.save_full_model = True
    native.config.trainer.gradient_accumulation_steps = 2
    trainer.config = native.config
    trainer.config.optim.load_checkpoint_optim_state = True
    expected = _parameters(native)
    moments = {
        name: trainer.optimizer.state[p]["exp_avg"].clone()
        for name, p in native.named_parameters()
        if p.requires_grad
    }
    for distributed in (False, True):
        native.config.operation.save_dist_ckpt = distributed
        save_trainer_state(trainer, 3)
        save_checkpoint(native.config, native, trainer.optimizer, trainer.lr_scheduler, step=3)
        if not distributed:
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(ps, "_CONTEXT_PARALLEL_WORLD_SIZE", 1)
                patch.setattr(ps, "_TENSOR_MODEL_PARALLEL_WORLD_SIZE", 1)
                patch.setattr(ps, "_DATA_PARALLEL_WORLD_SIZE", 1)
                patch.setattr(ps, "get_tensor_model_parallel_rank", lambda: 0)
                single.config.trainer.model_path = paths[0]
                single.config.optim.load_checkpoint_optim_state = True
                assert (
                    load_checkpoint(
                        single.config,
                        single,
                        reference_trainer.optimizer,
                        reference_trainer.lr_scheduler,
                        step=3,
                    )
                    == 3
                )
        with torch.no_grad():
            for p in native.parameters():
                if p.requires_grad:
                    p.fill_(1)
        trainer.optimizer.state.clear()
        assert (
            load_checkpoint(native.config, native, trainer.optimizer, trainer.lr_scheduler, step=3)
            == 3
        )
        load_trainer_state(trainer, 3)
        for name, p in native.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(p.cpu(), expected[name], atol=0, rtol=0)
                torch.testing.assert_close(
                    trainer.optimizer.state[p]["exp_avg"], moments[name], atol=0, rtol=0
                )
    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(paths[0])


@pytest.mark.parametrize(
    "lora,recompute", [(False, None), (True, None), (True, "standard"), (True, "optimized")]
)
def test_context_parallel_model(lora, recompute):
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    ps.initialize_model_parallel(1, 2, context_parallel_size=2)
    check_model(device, "ring", lora=lora, recompute=recompute)


def test_context_parallel_attention():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    ps.initialize_model_parallel(1, 2, context_parallel_size=2)
    check_attention(device, "ring")
    check_unused_gradients(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--backend", choices=("sdpa", "ring"), default="sdpa")
    parser.add_argument("--compute-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    device = (
        torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        if args.device == "cuda"
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if device.type == "cuda" else "gloo",
        **({"device_id": device} if device.type == "cuda" else {}),
    )
    ps.initialize_model_parallel(args.tp, 2, context_parallel_size=args.cp)
    assert ps.get_data_parallel_world_size() == 1, "Use the DP-specific test for CP+DP"
    results = {"attention": check_attention(device, args.backend), "models": []}
    check_unused_gradients(device)
    cases = (
        [(True, None)]
        if args.checkpoint
        else [(False, None), (True, None), (True, "standard"), (True, "optimized")]
    )
    for lora, recompute in cases:
        results["models"].append(
            check_model(
                device,
                args.backend,
                lora=lora,
                recompute=recompute,
                checkpoint=args.checkpoint,
                compute_dtype=getattr(torch, args.compute_dtype),
            )
        )
    if dist.get_rank() == 0:
        print(json.dumps(results, indent=2), flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(results, indent=2) + "\n")
    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
