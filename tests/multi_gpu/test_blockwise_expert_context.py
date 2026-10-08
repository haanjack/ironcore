# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0

"""CP2+EP2 owned experts match a full decoder across trainer updates and resume.

Run four CPU processes with torchrun. CP1+EP2 also runs on two CUDA GPUs.
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
from tests.fixtures.lora_tp import smollm2_lora_model
from tests.multi_gpu.test_context_parallel_dp import _trainer

from ironcore.checkpointing.expert import global_parameter_names
from ironcore.parallel import parallel_states as ps
from ironcore.parallel.expert_parallel import parallel_states as eps
from ironcore.parallel.parallel import initialize_parallelism

pytestmark = [
    pytest.mark.mp,
    pytest.mark.skipif("RANK" not in os.environ, reason="torchrun required"),
]


def check_expert_context(
    device: torch.device, cp: int, recompute: str | None, *, idle: bool = False
) -> dict:
    tokens = torch.arange(40, device=device).reshape(4, 10) % 32
    labels = tokens[:, 1:].clone()
    labels[:2] = -100  # Empty objective on one EP/DP worker still receives remote tokens.
    labels[2, :2] = -100
    batches = [{"input_ids": tokens[i : i + 1, :-1], "labels": labels[i : i + 1]} for i in range(4)]

    def autocast():
        return (
            torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(eps, "_EXPERT_MODEL_PARALLEL_WORLD_SIZE", 1)
        torch.manual_seed(43)
        reference = smollm2_lora_model(patch, lora=False, moe=True).to(device).train()
        if idle:
            for layer in reference.model.layers:
                with torch.no_grad():
                    layer.mlp.router.weight.zero_()
                layer.mlp.router.weight.requires_grad_(False)
        initial = {name: value.clone() for name, value in reference.state_dict().items()}
        ref_trainer = _trainer(reference, batches)
        ref_trainer.optimizer.param_groups[0]["lr"] = 1e-3
        ref_trainer.context["autocast"] = autocast()
        expected = []
        for step in range(3):
            loss, norm, _ = ref_trainer.train_step(step)
            expected.append(
                (loss, norm, {n: p.detach().clone() for n, p in reference.named_parameters()})
            )
    with pytest.MonkeyPatch.context() as patch:
        native = (
            smollm2_lora_model(
                patch,
                cp_size=cp,
                cp_backend="ring" if device.type == "cuda" else "sdpa",
                lora=False,
                moe=True,
                mlp_chunk_size=3,
                ep_size=2,
            )
            .to(device)
            .train()
        )
        if cp == 1:
            patch.setattr(ps, "_DATA_PARALLEL_WORLD_SIZE", 2)
        mapping = global_parameter_names(native)
        native.load_state_dict(
            {name: initial[global_name] for name, global_name in mapping.items()}
        )
        native.config.parallel.rank = dist.get_rank()
        if idle:
            for layer in native.model.layers:
                layer.mlp.router.weight.requires_grad_(False)
        if recompute:
            native.config.operation.activation_recompute = True
            native.model.activation_recompute = True
            native.model.use_reentrant = recompute == "optimized"
        wrapped = initialize_parallelism(native.config, native)
        dp_rank = ps.get_data_parallel_group_rank()
        trainer = _trainer(wrapped, batches[dp_rank * 2 : dp_rank * 2 + 2])
        trainer.optimizer.param_groups[0]["lr"] = 1e-3
        trainer.context["autocast"] = autocast()
        records = []
        for step, (loss, norm, parameters) in enumerate(expected):
            actual_loss, actual_norm, _ = trainer.train_step(step)
            if dist.get_rank() == 0:
                print(
                    {
                        "cp": cp,
                        "step": step,
                        "loss": actual_loss,
                        "reference_loss": loss,
                        "norm": actual_norm,
                        "reference_norm": norm,
                    },
                    flush=True,
                )
            assert actual_loss == pytest.approx(loss, abs=0.002 if device.type == "cuda" else 2e-6)
            assert actual_norm == pytest.approx(norm, rel=0.01 if device.type == "cuda" else 3e-5)
            for name, p in native.named_parameters():
                torch.testing.assert_close(
                    p,
                    parameters[mapping[name]],
                    atol=2e-5 if device.type == "cuda" else 3e-7,
                    rtol=0.001 if device.type == "cuda" else 3e-5,
                    msg=name,
                )
            records.append({"loss": actual_loss, "reference_loss": loss, "norm": actual_norm})
        _checkpoint_roundtrip(native, trainer)
        return {
            "idle": idle,
            "cp": cp,
            "ep": 2,
            "recompute": recompute,
            "updates": records,
            "checkpoint": "passed",
        }


def _checkpoint_roundtrip(native, trainer) -> None:
    from ironcore.checkpointing.native import load_checkpoint, save_checkpoint
    from ironcore.trainers.checkpoint_state import load_trainer_state, save_trainer_state

    paths = [tempfile.mkdtemp(prefix="ironcore-blockwise-ep-") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(paths)
    native.config.trainer.model_path = paths[0]
    native.config.trainer.gradient_accumulation_steps = 2
    native.config.optim.load_checkpoint_optim_state = True
    trainer.config = native.config
    parameters = {n: p.detach().clone() for n, p in native.named_parameters()}
    moments = {
        n: trainer.optimizer.state[p]["exp_avg"].clone()
        for n, p in native.named_parameters()
        if p in trainer.optimizer.state
    }
    save_trainer_state(trainer, 3)
    save_checkpoint(native.config, trainer.model, trainer.optimizer, trainer.lr_scheduler, step=3)
    with torch.no_grad():
        for p in native.parameters():
            p.zero_()
    trainer.optimizer.state.clear()
    assert (
        load_checkpoint(
            native.config, trainer.model, trainer.optimizer, trainer.lr_scheduler, step=3
        )
        == 3
    )
    load_trainer_state(trainer, 3)
    for name, p in native.named_parameters():
        torch.testing.assert_close(p, parameters[name], atol=0, rtol=0)
        if name in moments:
            torch.testing.assert_close(
                trainer.optimizer.state[p]["exp_avg"], moments[name], atol=0, rtol=0
            )
    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(paths[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    device = (
        torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        if args.device == "cuda"
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    assert dist.get_world_size() == 2 * args.cp
    ps.initialize_model_parallel(1, 2, context_parallel_size=args.cp)
    eps.initialize_expert_parallel(2, 1, context_parallel_size=args.cp)
    rows = [check_expert_context(device, args.cp, mode) for mode in (None, "standard", "optimized")]
    rows.append(check_expert_context(device, args.cp, "optimized", idle=True))
    if dist.get_rank() == 0:
        if args.report:
            args.report.write_text(json.dumps(rows, indent=2) + "\n")
        print("EP2 block-wise trainer and checkpoint comparisons passed", flush=True)
    eps.destroy_expert_parallel()
    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()


@pytest.mark.parametrize(
    "recompute,idle",
    [(None, False), ("standard", False), ("optimized", False), ("optimized", True)],
)
def test_blockwise_expert_parallel(recompute: str | None, idle: bool) -> None:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    ps.initialize_model_parallel(1, 2)
    eps.initialize_expert_parallel(2, 1)
    check_expert_context(device, 1, recompute, idle=idle)
