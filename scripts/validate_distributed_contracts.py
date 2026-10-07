# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Independent unequal-batch and one-rank fault oracles (torchrun, DP=2)."""

import argparse
import copy
import json
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ironcore.config import TrainerConfig
from ironcore.controller import TrainingControl
from ironcore.parallel import parallel_states
from ironcore.trainers import LanguageModelTrainer
from ironcore.training_utils import loss_func, loss_func_sft
from ironcore.utils import Timer


def trainer_for(model, batches, loss_fn):
    trainer = object.__new__(LanguageModelTrainer)
    trainer.config = SimpleNamespace(
        trainer=TrainerConfig(gradient_accumulation_steps=2),
        optim=SimpleNamespace(clip_grad=0.5),
        operation=SimpleNamespace(),
        utils=SimpleNamespace(),
    )
    trainer.model = model
    trainer.loss_fn = loss_fn
    trainer.context = {"autocast": nullcontext()}
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        trainer.optimizer, step_size=1, gamma=0.9
    )
    trainer.data_iterator = {"train": iter(batches)}
    trainer.timer = Timer()
    trainer.logger = logging.getLogger(__name__)
    trainer.control = TrainingControl(trainer.config)
    trainer._offload_scheduler = None

    def forward(module, iterator):
        batch = next(iterator)
        labels = batch["labels"].cuda()
        logits = module(batch["features"].cuda())
        per_token = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), reduction="none"
        ).view_as(labels)
        return loss_fn(per_token, labels != -100)

    trainer.forward_step_func = forward
    return trainer


def main(output):
    torch.set_num_threads(2)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    parallel_states.initialize_model_parallel(1, 2)
    rank = dist.get_rank()
    checks = []
    for loss_fn in [loss_func, loss_func_sft]:
        torch.manual_seed(57)
        full = torch.nn.Linear(3, 7).cuda()
        actual = copy.deepcopy(full)
        features = torch.randn(8, 5, 3, device="cuda")
        labels = torch.randint(0, 7, (8, 5), device="cuda")
        for row in range(8):
            labels[row, : row % 5] = -100
        labels[2].fill_(-100)
        spans = [(0, 1), (1, 3)] if rank == 0 else [(3, 5), (5, 8)]
        batches = [{"features": features[a:b].cpu(), "labels": labels[a:b].cpu()} for a, b in spans]
        trainer = trainer_for(DistributedDataParallel(actual, device_ids=[rank]), batches, loss_fn)
        optimizer = torch.optim.AdamW(full.parameters(), lr=0.01)
        losses = torch.nn.functional.cross_entropy(
            full(features).flatten(0, 1), labels.flatten(), reduction="none"
        ).view_as(labels)
        expected = (
            torch.stack(
                [losses[i][labels[i] != -100].mean() for i in range(8) if (labels[i] != -100).any()]
            ).mean()
            if loss_fn is loss_func_sft
            else losses[labels != -100].mean()
        )
        expected.backward()
        torch.nn.utils.clip_grad_norm_(full.parameters(), 0.5)
        optimizer.step()
        actual_loss, *_ = trainer.train_step(0)
        error = max(
            (a - b).abs().max().item()
            for a, b in zip(actual.parameters(), full.parameters(), strict=True)
        )
        for a, b in zip(actual.parameters(), full.parameters(), strict=True):
            torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)
        if abs(actual_loss - expected.item()) > 1e-6:
            raise AssertionError("Global loss normalization mismatch")
        checks.append(
            {
                "check": loss_fn.__name__ + "_unequal_dp_update",
                "max_weight_error": error,
                "loss_error": abs(actual_loss - expected.item()),
            }
        )
    held_out = [10, 20] if rank == 0 else [10]
    trainer._get_data_iterator = lambda: {"eval": iter(held_out)}
    for _ in range(2):
        assert list(trainer._evaluation_batches(5)) == [10]
    checks.append(
        {"check": "repeated_unequal_finite_evaluation", "status": "restarted_and_stopped_together"}
    )
    trainer._get_data_iterator = lambda: {"eval": iter([10] if rank == 0 else [])}
    assert list(trainer._evaluation_batches(5)) == []
    checks.append({"check": "one_rank_empty_evaluation", "status": "all_stopped"})
    for bad in [float("nan"), float("inf")]:
        module = torch.nn.Linear(3, 7).cuda()
        trainer = trainer_for(module, [], loss_func)
        before = copy.deepcopy(module.state_dict())
        scheduler = copy.deepcopy(trainer.lr_scheduler.state_dict())
        for p in module.parameters():
            p.grad = torch.ones_like(p)
        if rank == 1:
            next(module.parameters()).grad.flatten()[0] = bad
        try:
            trainer._prepare_gradients()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Rank-local invalid gradient did not stop all ranks")
        assert not trainer.optimizer.state and scheduler == trainer.lr_scheduler.state_dict()
        assert all(p.grad is None for p in module.parameters())
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        checks.append({"check": "one_rank_nonfinite_gradient_" + str(bad), "status": "all_stopped"})
    trainer._validate_distributed_loss(1.0, 0)
    try:
        trainer._validate_distributed_loss(float("nan") if rank == 1 else 1.0, 0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Rank-local invalid reward did not stop all ranks")
    checks.append({"check": "one_rank_nonfinite_reward", "status": "all_stopped"})
    from ironcore.checkpointing.integrity import (
        collective_checkpoint_action,
        digest_file,
        verify_manifest,
    )

    output.mkdir(parents=True, exist_ok=True)
    folder = output / f"rank{rank}"
    folder.mkdir(exist_ok=True)
    data = folder / "weights"
    data.write_bytes(b"valid checkpoint")
    (folder / "manifest.json").write_text(json.dumps({"files": {"weights": digest_file(data)}}))
    if rank == 1:
        data.write_bytes(b"corrupted checkpoint")
    try:
        verify_manifest(folder, "manifest.json")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Rank-local checksum failure did not stop all ranks")
    checks.append({"check": "one_rank_checksum_failure", "status": "all_stopped"})

    def fail_save():
        if rank == 1:
            raise OSError("Injected disk failure")

    try:
        collective_checkpoint_action(fail_save)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Rank-local save failure did not stop all ranks")
    checks.append({"check": "one_rank_save_failure", "status": "all_stopped"})
    from ironcore.optimizer.adamw import AdamWOptimizer

    for amsgrad in [False, True]:
        for scale in [1.0, 1e-8, 1e-10]:
            p = torch.nn.Parameter(torch.zeros(9, device="cuda"))
            q = torch.nn.Parameter(p.detach().clone())
            kwargs = dict(lr=1e-3, amsgrad=amsgrad, weight_decay=0.03)
            optimizer = AdamWOptimizer(
                [p], **kwargs, offload_enabled=True, offload_min_param_elements=0
            )
            reference = torch.optim.AdamW([q], **kwargs)
            for step in range(7):
                p.grad = torch.full_like(p, scale * (step + 1))
                q.grad = p.grad.clone()
                optimizer.step()
                reference.step()
                if step == 2:
                    optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                    reference.load_state_dict(copy.deepcopy(reference.state_dict()))
                torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-6)
            assert optimizer.state[p]["exp_avg"].device.type == "cpu"
            checks.append(
                {
                    "check": f"gpu_params_cpu_offload_amsgrad_{amsgrad}_scale_{scale}",
                    "max_weight_error": (p - q).abs().max().item(),
                }
            )
    (output / f"result_rank{rank}.json").write_text(
        json.dumps({"status": "passed", "checks": checks}, indent=2) + "\n"
    )
    parallel_states.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
