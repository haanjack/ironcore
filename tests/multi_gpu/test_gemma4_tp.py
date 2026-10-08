# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 TP=2 parity; standalone mode also supports real CPU/Gloo ranks.

GPU: torchrun --standalone --nproc_per_node=2 -m pytest -o addopts='' tests/multi_gpu/test_gemma4_tp.py
CPU: torchrun --standalone --nproc_per_node=2 -m tests.multi_gpu.test_gemma4_tp --device cpu
"""

from __future__ import annotations

import argparse
import os

import pytest
import torch
import torch.distributed as dist
from tests.fixtures.gemma4 import gemma4_pair

from ironcore.checkpointing.weight_mapping import Architecture, WeightMapper
from ironcore.parallel.parallel_states import destroy_model_parallel, initialize_model_parallel

pytestmark = [pytest.mark.mp, pytest.mark.skipif("RANK" not in os.environ, reason="Needs torchrun")]


def check_tp_parity(variant: str, device: torch.device, recompute: str | None = None) -> None:
    """Compare real TP ranks with independently computed reference logits/gradients."""
    torch.manual_seed(42)
    with pytest.MonkeyPatch.context() as patch:
        native, reference, config = gemma4_pair(patch, variant, tp_size=2)
        native.to(device)
        reference.to(device)
        if recompute:
            config.operation.activation_recompute = True
            native.model.activation_recompute = True
            native.model.use_reentrant = recompute == "optimized"
        tokens = torch.tensor([[2, 3, 4, 5, 6, 7], [2, 18, 19, 20, 21, 22]], device=device)
        actual, _ = native(tokens)
        expected = reference(tokens, use_cache=False).logits
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        coefficients = torch.linspace(-0.7, 0.9, actual.numel(), device=device).reshape_as(actual)
        (actual * coefficients).mean().backward()
        (expected * coefficients).mean().backward()
        gradients = {name: param.grad for name, param in reference.named_parameters()}
        mapped = WeightMapper(Architecture.GEMMA4, 4).hf_to_ironcore(gradients)
        rank = dist.get_rank()
        for name, param in native.named_parameters():
            assert param.grad is not None, name
            full = mapped[name]
            dim = getattr(param, "tp_shard_dim", None)
            expected_grad = full.chunk(2, dim=dim)[rank] if dim is not None else full
            torch.testing.assert_close(
                param.grad,
                expected_grad,
                atol=5e-5,
                rtol=5e-4,
                msg=lambda info, key=name: f"rank={rank}, {key}: {info}",
            )
        native.eval()
        reference.eval()
        actual_tokens = native.generate(tokens[:1, :3], max_new_tokens=4, do_sample=False)
        expected_tokens = reference.generate(
            tokens[:1, :3],
            max_new_tokens=4,
            do_sample=False,
            eos_token_id=None,
        )
        torch.testing.assert_close(actual_tokens, expected_tokens, atol=0, rtol=0)
        _, cache = native(tokens[:, :4], use_cache=True)
        cached, _ = native(tokens[:, 4:], use_cache=True, past_key_values=cache)
        torch.testing.assert_close(cached, expected[:, 4:], atol=2e-5, rtol=2e-5)
        # Drive the production training step, including TP cross-entropy and clipping.
        from tests.unit.trainers.test_training_correctness import make_trainer

        from ironcore.training_utils import forward_step, loss_func

        native.zero_grad(set_to_none=True)
        native.train()
        native.loss_fn = loss_func
        batch = {"input_ids": tokens[:, :-1], "labels": tokens[:, 1:]}
        # The TP=1 production trainer is an independent update reference with
        # the same initial checkpoint, accumulation and gradient clipping.
        torch.manual_seed(42)
        with pytest.MonkeyPatch.context() as single_patch:
            single, _, _ = gemma4_pair(single_patch, variant)
            single.to(device).train()
            single.loss_fn = loss_func
            single_trainer = make_trainer(single, [batch, batch], loss_func, accumulation=2)
            # Avoid amplifying harmless FP32 reduction noise in almost-zero
            # gradients; this is a supported AdamW epsilon, identical in both runs.
            single_trainer.optimizer.param_groups[0]["eps"] = 1e-6
            single_trainer.forward_step_func = forward_step
            single_loss, single_norm, _ = single_trainer.train_step(0)
            single_weights = {name: p.detach().clone() for name, p in single.named_parameters()}
        trainer = make_trainer(native, [batch, batch], loss_func, accumulation=2)
        trainer.optimizer.param_groups[0]["eps"] = 1e-6
        trainer.forward_step_func = forward_step
        before = native.embedding.word_embeddings.weight.detach().clone()
        loss, norm, _ = trainer.train_step(0)
        assert 0 < loss < 10 and norm > 0
        assert not torch.equal(before, native.embedding.word_embeddings.weight)
        assert loss == pytest.approx(single_loss, abs=2e-6)
        assert norm == pytest.approx(single_norm, rel=2e-5)
        for name, param in native.named_parameters():
            full = single_weights[name]
            dim = getattr(param, "tp_shard_dim", None)
            updated = full.chunk(2, dim=dim)[rank] if dim is not None else full
            torch.testing.assert_close(
                param,
                updated,
                atol=2e-5,
                rtol=2e-4,
                msg=lambda info, key=name: f"rank={rank}, updated {key}: {info}",
            )
        print(
            f"rank={rank} {variant} TP=2 {recompute or 'no-recompute'}: logits, all gradients, cache, generation, trainer passed",
            flush=True,
        )


@pytest.mark.parametrize("variant", ["E2B", "E4B", "31B"])
@pytest.mark.parametrize("recompute", [None, "standard", "optimized"])
def test_gemma4_tp2(variant: str, recompute: str | None) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    initialize_model_parallel(2, timeout_in_minutes=5.0)
    try:
        check_tp_parity(variant, torch.device("cuda", local_rank), recompute)
    finally:
        destroy_model_parallel()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.device == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = (
        torch.device(args.device, int(os.environ["LOCAL_RANK"]))
        if args.device == "cuda"
        else torch.device("cpu")
    )
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo")
    initialize_model_parallel(2, timeout_in_minutes=5.0)
    try:
        for variant in ["E2B", "E4B", "31B"]:
            for recompute in [None, "standard", "optimized"]:
                check_tp_parity(variant, device, recompute)
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
