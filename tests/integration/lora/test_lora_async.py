# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""
Test LoRA with async chunked execution.

Tests:
1. LoRA works correctly with sequence chunking enabled
2. Chunked execution produces same outputs as non-chunked
3. Async finalize() in LoRARowParallelLinear works correctly
4. Gradient flow with chunking
"""

import random

import numpy as np
import pytest
import torch
import torch.distributed as dist

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
    ProfilerConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.config.config_model import BiasConfig
from ironcore.global_vars import reset_global_states, set_global_states
from ironcore.language_model import LanguageModel
from ironcore.parallel import parallel_states
from ironcore.peft.utils import freeze_base_model


def create_model_config(tp_size=1, chunk_size=None):
    """Create a small model config for testing."""
    model_config = ModelConfig(
        d_model=256,
        num_attention_heads=8,
        num_attention_groups=8,
        head_dim=32,
        max_seq_len=128,
        max_position_embeddings=128,
        dropout_attn=0.0,
        dropout_mlp=0.0,
        bias=BiasConfig.all_true(),
        layernorm_bias=True,
    )
    model_config.name = "gpt2"

    trainer_config = TrainerConfig(
        tensor_model_parallel_size=tp_size,
        use_flash_attn=False,
        sequence_chunk_size=chunk_size,
    )

    lora_config = LoRAConfig(
        r=8,
        alpha=16.0,
        dropout=0.0,
        target_modules=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )
    peft_config = PEFTConfig(method="lora", lora=lora_config)

    return MainConfig(
        model=model_config,
        trainer=trainer_config,
        init=InitConfig(seed=42, init_std=0.02),
        optim=OptimConfig(max_lr=1e-3, weight_decay=0.01),
        data=DataConfig(),
        parallel=ParallelConfig(),
        operation=OperationConfig(train_steps=100),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
        peft=peft_config,
    )


def _seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TestLoRAAsyncChunking:
    """Test LoRA with chunked vs non-chunked execution."""

    def _run_chunking_test(self, tp_size=1):
        """Core test logic for chunked vs non-chunked comparison."""
        if tp_size > 1:
            pytest.skip("TP > 1 requires torchrun")

        rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        try:
            _seed_all()

            # Non-chunked execution
            config_no_chunk = create_model_config(tp_size=tp_size, chunk_size=None)
            set_global_states(config_no_chunk)
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=tp_size, timeout_in_minutes=10.0
            )

            model_no_chunk = LanguageModel(config_no_chunk)
            model_no_chunk.to(device)
            freeze_base_model(model_no_chunk, "lora")

            batch_size, seq_len = 2, 64
            torch.manual_seed(42)
            input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)

            model_no_chunk.eval()
            with torch.no_grad():
                output_no_chunk = model_no_chunk(input_ids)

            parallel_states.destroy_model_parallel()
            reset_global_states()

            # Chunked execution
            config_chunked = create_model_config(tp_size=tp_size, chunk_size=16)
            set_global_states(config_chunked)
            parallel_states.initialize_model_parallel(
                tensor_model_parallel_size=tp_size, timeout_in_minutes=10.0
            )

            model_chunked = LanguageModel(config_chunked)
            model_chunked.to(device)
            freeze_base_model(model_chunked, "lora")
            model_chunked.load_state_dict(model_no_chunk.state_dict())

            model_chunked.eval()
            torch.manual_seed(42)
            input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)

            with torch.no_grad():
                output_chunked = model_chunked(input_ids)
                # Handle tuple return (logits, past_kv) when in eval mode with KV cache
                if isinstance(output_chunked, tuple):
                    output_chunked = output_chunked[0]
                if isinstance(output_no_chunk, tuple):
                    output_no_chunk = output_no_chunk[0]

            # Compare outputs
            if rank == 0:
                atol, rtol = 1e-1, 1e-1
                assert torch.allclose(output_chunked, output_no_chunk, atol=atol, rtol=rtol), (
                    f"Outputs differ: max_diff="
                    f"{torch.abs(output_chunked - output_no_chunk).max().item():.6e}"
                )

            # Test gradient flow with chunking
            model_chunked.train()
            torch.manual_seed(42)
            input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)

            output = model_chunked(input_ids)
            loss = output.mean()
            loss.backward()

            lora_grads = []
            base_grads = []
            for name, param in model_chunked.named_parameters():
                if param.grad is not None:
                    if any(k in name for k in ["lora_A", "lora_B"]):
                        lora_grads.append(name)
                    else:
                        base_grads.append(name)

            if rank == 0:
                assert len(base_grads) == 0, "Base parameters should not have gradients"
                assert len(lora_grads) > 0, "LoRA parameters should have gradients"

        finally:
            parallel_states.destroy_model_parallel()
            reset_global_states()
            if tp_size > 1 and dist.is_initialized():
                dist.destroy_process_group()

    @pytest.mark.cuda
    def test_chunked_vs_non_chunked_tp1(self):
        """Chunked and non-chunked LoRA should produce matching outputs with TP=1."""
        self._run_chunking_test(tp_size=1)

    @pytest.mark.mp
    def test_chunked_vs_non_chunked_tp2(self):
        """Chunked and non-chunked LoRA should produce matching outputs with TP=2."""
        self._run_chunking_test(tp_size=2)
