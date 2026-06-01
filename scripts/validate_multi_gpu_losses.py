#!/usr/bin/env python
"""Quick multi-GPU loss validation script."""

import os
import sys

os.environ["WORLD_SIZE"] = "2"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

# Must set RANK and LOCAL_RANK before importing ironcore
if "RANK" not in os.environ:
    os.environ["RANK"] = os.environ.get("RANK", "0")
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")

from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ironcore.config import OffloadConfig  # isort: skip
from ironcore.global_vars import reset_global_states  # isort: skip
from ironcore.trainers import LanguageModelTrainer  # isort: skip
from tests.fixtures.config_fixtures import create_test_config  # isort: skip
from tests.integration.offload.conftest import (  # isort: skip
    create_mock_data_iterator,
    create_mock_evaluators,
)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

NUM_STEPS = 50
BATCH_SIZE = 2
SEQ_LEN = 256


def create_forward_step():
    step_counter = [0]

    def forward_step(model, _data_iterator):
        device = next(model.parameters()).device
        torch.manual_seed(42 + step_counter[0])
        step_counter[0] += 1
        input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN))
        labels = input_ids.clone()
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits = model(input_ids, labels=None)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss

    return forward_step


def run_training(name, config_kwargs=None):
    reset_global_states()

    config = create_test_config(
        d_model=768,
        d_ffn=3072,
        num_layers=2,
        num_attention_heads=12,
        num_attention_groups=12,
        head_dim=64,
        max_seq_len=SEQ_LEN,
        dropout_attn=0.0,
        dropout_mlp=0.0,
        dropout_embd=0.0,
        precision="bfloat16",
        seed=42,
    )
    config.operation.train_steps = NUM_STEPS + 10
    config.trainer.micro_batch_size = BATCH_SIZE
    config.trainer.train_batch_size = BATCH_SIZE * 2
    config.trainer.gradient_accumulation_steps = 1
    config.parallel.world_size = 2
    config.parallel.rank = int(os.getenv("RANK", "0"))
    config.parallel.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if config_kwargs:
        for key, value in config_kwargs.items():
            if "." in key:
                parts = key.split(".")
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(config, key, value)

    init_loss, final_loss = None, 0.0
    with (
        patch(
            "ironcore.trainers.base_trainer.get_data_iterator",
            return_value=create_mock_data_iterator(),
        ),
        patch(
            "ironcore.trainers.base_trainer.get_evaluators", return_value=create_mock_evaluators()
        ),
    ):
        trainer = LanguageModelTrainer(config, create_forward_step(), F.cross_entropy)
        trainer._initialize()
        for step in range(NUM_STEPS):
            loss, _, _ = trainer.train_step(step=step)
            if step == 0:
                init_loss = loss
            final_loss = loss
        trainer._finalize_process()

    return init_loss, final_loss


if __name__ == "__main__":
    rank = int(os.getenv("RANK", "0"))
    if rank == 0:
        print("=" * 70)
        print("MULTI-GPU LOSS VALIDATION (50 steps)")
        print("=" * 70)

    # DDP + optimizer_offload+activation_spill
    init_ddp, final_ddp = run_training(
        "DDP+optimizer_offload+activation_spill",
        {
            "offload": OffloadConfig(
                enabled=True,
                optimizer_offload=True,
                optimizer_state_precision="bf16",
                activation_spill=True,
                activation_spill_granularity="sub_layer",
                pinned_memory_pool_gb=2.0,
            )
        },
    )

    if rank == 0:
        print(
            f"DDP + optimizer_offload+activation_spill:       Init={init_ddp:.4f}, Final={final_ddp:.4f}"
        )

    # DistOpt + optimizer_offload+activation_spill
    init_dist, final_dist = run_training(
        "DistOpt+optimizer_offload+activation_spill",
        {
            "parallel.use_distributed_optimizer": True,
            "offload": OffloadConfig(
                enabled=True,
                optimizer_offload=True,
                optimizer_state_precision="bf16",
                activation_spill=True,
                activation_spill_granularity="sub_layer",
                pinned_memory_pool_gb=2.0,
            ),
        },
    )

    if rank == 0:
        print(
            f"DistOpt + optimizer_offload+activation_spill:   Init={init_dist:.4f}, Final={final_dist:.4f}"
        )

    # FSDP SHARD_GRAD_OP + optimizer_offload+activation_spill
    init_fsdp, final_fsdp = run_training(
        "FSDP+optimizer_offload+activation_spill",
        {
            "parallel.use_fsdp": True,
            "parallel.fsdp_sharding_strategy": "shard_grad_op",
            "parallel.fsdp_use_orig_params": True,
            "offload": OffloadConfig(
                enabled=True,
                optimizer_offload=True,
                optimizer_state_precision="bf16",
                activation_spill=True,
                activation_spill_granularity="sub_layer",
                pinned_memory_pool_gb=2.0,
            ),
        },
    )

    if rank == 0:
        print(
            f"FSDP SHARD_GRAD_OP+optimizer_offload+activation_spill: Init={init_fsdp:.4f}, Final={final_fsdp:.4f}"
        )

    # FSDP FULL_SHARD + activation_spill
    init_fsdp_full, final_fsdp_full = run_training(
        "FSDP-FULL+activation_spill",
        {
            "parallel.use_fsdp": True,
            "parallel.fsdp_sharding_strategy": "full",
            "parallel.fsdp_use_orig_params": True,
            "offload": OffloadConfig(
                enabled=True,
                activation_spill=True,
                activation_spill_granularity="sub_layer",
                pinned_memory_pool_gb=2.0,
            ),
        },
    )

    if rank == 0:
        print(
            f"FSDP FULL_SHARD+activation_spill:        Init={init_fsdp_full:.4f}, Final={final_fsdp_full:.4f}"
        )
        print("=" * 70)
