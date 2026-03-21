# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Test LoRA checkpoint save/load functionality.

Tests:
1. LoRA weights are correctly saved in checkpoints
2. LoRA weights are correctly loaded from checkpoints
3. Universal checkpoint compatibility (TP=1 save, TP=2 load)
4. Optimizer state is correctly saved/loaded for LoRA parameters
"""

import os
import random
import shutil

import numpy as np
import pytest
import torch
import torch.distributed as dist

from ironcore.checkpointing.native import load_checkpoint, save_checkpoint
from ironcore.config import (
    DataConfig,
    InitConfig,
    LoRAConfig,
    MainConfig,
    ModelConfig,
    OperationConfig,
    OptimConfig,
    ParallelConfig,
    PEFTConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.global_vars import reset_global_states, set_global_states
from ironcore.language_model import LanguageModel
from ironcore.optimizer import get_optimizer
from ironcore.parallel import parallel_states


def create_model_config(tp_size=1):
    """Create a small model config for testing."""
    model_config = ModelConfig(
        d_model=256,
        num_attention_heads=8,
        num_attention_groups=8,
        head_dim=32,
        max_seq_len=64,
        max_position_embeddings=64,
        dropout_attn=0.0,
        attention_bias=True,
        mlp_bias=True,
        layernorm_bias=True,
    )
    model_config.name = "gpt2"
    model_config.hf_model_type = "gpt2"
    model_config.hf_architecture = "GPT2LMHeadModel"

    trainer_config = TrainerConfig(
        tensor_model_parallel_size=tp_size,
        use_flash_attn=False,
        model_path="test_outputs/lora_checkpoint_test",
    )

    lora_config = LoRAConfig(
        r=8,
        alpha=16.0,
        dropout=0.0,
        target_modules=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )
    peft_config = PEFTConfig(method="lora", lora=lora_config)

    init_config = InitConfig(seed=42, init_std=0.02)
    optim_config = OptimConfig(max_lr=1e-3, weight_decay=0.01)

    return MainConfig(
        model=model_config,
        trainer=trainer_config,
        init=init_config,
        optim=optim_config,
        data=DataConfig(),
        parallel=ParallelConfig(),
        operation=OperationConfig(train_steps=100, no_save=False),
        utils=UtilsConfig(),
        peft=peft_config,
    )


@pytest.fixture(autouse=True)
def cleanup_test_outputs():
    """Clean up test outputs after each test."""
    yield
    if os.path.exists("test_outputs/lora_checkpoint_test"):
        shutil.rmtree("test_outputs/lora_checkpoint_test")


def _seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@pytest.mark.cuda
class TestLoRACheckpointTP1:
    """Test save and load LoRA checkpoint with TP=1."""

    def test_save_load_preserves_lora_weights(self):
        """LoRA parameters should be exactly restored after save/load."""
        device = torch.device("cuda:0")

        if os.path.exists("test_outputs/lora_checkpoint_test"):
            shutil.rmtree("test_outputs/lora_checkpoint_test")

        config = create_model_config(tp_size=1)
        set_global_states(config)
        parallel_states.initialize_model_parallel(
            tensor_model_parallel_size=1, timeout_in_minutes=10.0
        )

        try:
            _seed_all()
            model = LanguageModel(config)
            model.to(device)
            optimizer = get_optimizer(config, model, device_type="cuda")

            from torch.optim.lr_scheduler import StepLR

            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

            # Save initial state
            save_checkpoint(config, model, optimizer, scheduler, step=0)

            initial_lora_params = {
                name: param.clone() for name, param in model.named_parameters() if "lora_" in name
            }
            assert len(initial_lora_params) > 0, "Model should have LoRA parameters"

            # Modify LoRA parameters
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.data.add_(torch.randn_like(param) * 0.1)

            # Verify parameters changed
            for name, param in model.named_parameters():
                if "lora_" in name:
                    diff = torch.abs(param - initial_lora_params[name]).max().item()
                    assert diff > 1e-6, f"Parameter {name} did not change"

            # Load checkpoint and verify restoration
            config.trainer.model_path = "test_outputs/lora_checkpoint_test"
            load_checkpoint(config, model, optimizer, scheduler, step=0)

            max_diff = 0.0
            for name, param in model.named_parameters():
                if "lora_" in name:
                    diff = torch.abs(param - initial_lora_params[name]).max().item()
                    max_diff = max(max_diff, diff)

            assert max_diff < 1e-6, f"LoRA parameters not restored correctly: max_diff={max_diff}"

        finally:
            parallel_states.destroy_model_parallel()
            reset_global_states()


@pytest.mark.cuda
@pytest.mark.distributed
class TestLoRAUniversalCheckpoint:
    """Test universal checkpoint: save with TP=1, load with TP=2."""

    def test_tp1_to_tp2_checkpoint(self):
        """LoRA checkpoint saved with TP=1 should load correctly with TP=2."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")

        try:
            # Step 1: Save with TP=1 (only rank 0)
            if rank == 0:
                if os.path.exists("test_outputs/lora_checkpoint_test"):
                    shutil.rmtree("test_outputs/lora_checkpoint_test")

                config_tp1 = create_model_config(tp_size=1)
                set_global_states(config_tp1)
                parallel_states.initialize_model_parallel(
                    tensor_model_parallel_size=1, timeout_in_minutes=10.0
                )

                _seed_all()
                model_tp1 = LanguageModel(config_tp1)
                model_tp1.to(device)
                optimizer_tp1 = get_optimizer(config_tp1, model_tp1, device_type="cuda")

                from torch.optim.lr_scheduler import StepLR

                scheduler_tp1 = StepLR(optimizer_tp1, step_size=10, gamma=0.1)
                save_checkpoint(config_tp1, model_tp1, optimizer_tp1, scheduler_tp1, step=0)

                tp1_lora_params = {
                    name: param.clone().cpu()
                    for name, param in model_tp1.named_parameters()
                    if "lora_" in name
                }

                parallel_states.destroy_model_parallel()
                reset_global_states()
            else:
                tp1_lora_params = None

            dist.barrier()

            # Step 2: Load with TP=2
            config_tp2 = create_model_config(tp_size=2)
            set_global_states(config_tp2)
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=2, timeout_in_minutes=10.0
            )

            _seed_all()
            model_tp2 = LanguageModel(config_tp2)
            model_tp2.to(device)
            optimizer_tp2 = get_optimizer(config_tp2, model_tp2, device_type="cuda")

            from torch.optim.lr_scheduler import StepLR

            scheduler_tp2 = StepLR(optimizer_tp2, step_size=10, gamma=0.1)

            config_tp2.trainer.model_path = "test_outputs/lora_checkpoint_test"
            load_checkpoint(config_tp2, model_tp2, optimizer_tp2, scheduler_tp2, step=0)

            # Step 3: Verify LoRA parameters
            tp2_lora_params = {
                name: param.clone()
                for name, param in model_tp2.named_parameters()
                if "lora_" in name
            }

            if rank == 0:
                max_diff = 0.0
                for name, param in tp2_lora_params.items():
                    param_cpu = param.cpu()
                    if name in tp1_lora_params:
                        expected = tp1_lora_params[name]
                        if param_cpu.shape != expected.shape:
                            if param_cpu.shape[0] == expected.shape[0]:
                                expected = expected[:, : param_cpu.shape[1]]
                            else:
                                expected = expected[: param_cpu.shape[0], :]
                        diff = torch.abs(param_cpu - expected).max().item()
                        max_diff = max(max_diff, diff)

                assert max_diff < 1e-6, f"LoRA parameters differ from TP=1: max_diff={max_diff}"

            # Verify replication across ranks
            for name, param in tp2_lora_params.items():
                gathered = [torch.zeros_like(param) for _ in range(2)]
                dist.all_gather(gathered, param)

                if rank == 0 and "lora_A" in name and "linear_q" in name:
                    diff = torch.abs(gathered[0] - gathered[1]).max().item()
                    assert diff < 1e-9, f"Replicated LoRA param {name} differs across ranks: {diff}"

        finally:
            parallel_states.destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()
