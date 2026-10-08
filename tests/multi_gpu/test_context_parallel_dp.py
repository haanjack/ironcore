# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""CP2+DP2 weighted objectives and replica initialization on four CPU ranks.

python -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m tests.multi_gpu.test_context_parallel_dp
"""

from __future__ import annotations

import json
import os

import pytest
import torch
import torch.distributed as dist
from tests.fixtures.lora_tp import smollm2_lora_model
from tests.unit.trainers.test_training_correctness import make_trainer

from ironcore.parallel import parallel_states as ps
from ironcore.parallel.parallel import initialize_parallelism
from ironcore.training_utils import forward_step, loss_func


def _trainer(model, batches):
    getattr(model, "module", model).loss_fn = loss_func
    trainer = make_trainer(model, batches * 3, loss_func, accumulation=len(batches))
    trainer.forward_step_func = forward_step
    trainer.optimizer.param_groups[0]["eps"] = 1e-3
    return trainer


def check_dp(
    lora: bool, *, moe: bool = False, recompute: str | None = None, expert_backend: str = "loop"
) -> dict:
    """Token weighting includes DP exactly once and CP exactly once."""
    tokens = torch.arange(40).reshape(4, 10) % 32
    labels = tokens[:, 1:].clone()
    # One entire DP worker has no valid tokens in either microbatch.
    # It must participate in backward rather than stranding its DDP peers.
    labels[:2] = -100
    labels[2, :2] = -100
    batches = [{"input_ids": tokens[i : i + 1, :-1], "labels": labels[i : i + 1]} for i in range(4)]
    with pytest.MonkeyPatch.context() as patch:
        torch.manual_seed(43)
        reference = smollm2_lora_model(patch, lora=lora, moe=moe).train()
        initial = {name: value.clone() for name, value in reference.state_dict().items()}
        reference_trainer = _trainer(reference, batches)
        expected = []
        for step in range(3):
            loss, norm, _ = reference_trainer.train_step(step)
            expected.append(
                (loss, norm, {name: p.detach().clone() for name, p in reference.named_parameters()})
            )
    with pytest.MonkeyPatch.context() as patch:
        native = smollm2_lora_model(
            patch,
            cp_size=2,
            lora=lora,
            moe=moe,
            mlp_chunk_size=3 if moe else None,
            expert_backend=expert_backend,
        ).train()
        if recompute:
            native.config.operation.activation_recompute = True
            native.model.activation_recompute = True
            native.model.use_reentrant = recompute == "optimized"
        native.load_state_dict(initial)
        native.config.parallel.rank = dist.get_rank()
        if ps.get_context_parallel_rank() == 1:
            with torch.no_grad():
                for p in native.parameters():
                    p.add_(1)
        wrapped = initialize_parallelism(native.config, native)
        for name, value in native.state_dict().items():
            torch.testing.assert_close(value, initial[name], atol=0, rtol=0)
        dp_rank = ps.get_data_parallel_group_rank()
        trainer = _trainer(wrapped, batches[dp_rank * 2 : dp_rank * 2 + 2])
        records = []
        for step, (loss, norm, parameters) in enumerate(expected):
            actual_loss, actual_norm, _ = trainer.train_step(step)
            assert actual_loss == pytest.approx(loss, abs=1e-6)
            assert actual_norm == pytest.approx(norm, rel=2e-5)
            for name, p in native.named_parameters():
                torch.testing.assert_close(p, parameters[name], atol=2e-7, rtol=3e-5)
            records.append({"loss": actual_loss, "reference_loss": loss, "norm": actual_norm})
        return {"lora": lora, "updates": records}


def main() -> None:
    dist.init_process_group("gloo")
    ps.initialize_model_parallel(1, 2, context_parallel_size=2)
    assert dist.get_world_size() == 4
    results = [
        check_dp(False),
        check_dp(True),
        check_dp(False, moe=True, recompute="standard"),
        check_dp(False, moe=True, recompute="optimized"),
        check_dp(False, moe=True, recompute="standard", expert_backend="grouped"),
        check_dp(False, moe=True, recompute="optimized", expert_backend="grouped"),
    ]
    if int(os.environ["RANK"]) == 0:
        print(json.dumps(results, indent=2), flush=True)
    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
