#!/usr/bin/env python
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Short training test to validate MFU calculation on RTX 3090."""

import time

import torch

from ironcore.config import (
    DataConfig,
    InitConfig,
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
from ironcore.mfu import MFUCalculator
from ironcore.models.transformer import TransformerModel

# GPT-2 Small config
NUM_LAYERS = 12
D_MODEL = 768
D_FFN = 3072
NUM_HEADS = 12
VOCAB_SIZE = 50257
SEQ_LEN = 1024
BATCH_SIZE = 4

# Test config
WARMUP_STEPS = 3
MEASURE_STEPS = 10


def create_config():
    """Create config for GPT-2 small."""
    return MainConfig(
        model=ModelConfig(
            d_model=D_MODEL,
            d_ffn=D_FFN,
            num_layers=NUM_LAYERS,
            num_attention_heads=NUM_HEADS,
            num_attention_groups=NUM_HEADS,
            head_dim=D_MODEL // NUM_HEADS,
            max_seq_len=SEQ_LEN,
            max_position_embeddings=SEQ_LEN,
            dropout_attn=0.0,
            dropout_mlp=0.0,
            dropout_embd=0.0,
            attention_bias=True, mlp_bias=True, layernorm_bias=True,
            precision="bfloat16",
        ),
        trainer=TrainerConfig(
            tensor_model_parallel_size=1,
            use_flash_attn=True,
            micro_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,
        ),
        init=InitConfig(seed=42, init_std=0.02),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        operation=OperationConfig(),
        utils=UtilsConfig(),
        profiler=ProfilerConfig(),
        peft=PEFTConfig(),
    )


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Print GPU info
    gpu_name = torch.cuda.get_device_properties(0).name
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Model: GPT-2 Small (L={NUM_LAYERS}, H={NUM_HEADS}, D={D_MODEL}, FFN={D_FFN})")
    print(f"Batch: {BATCH_SIZE}, Seq: {SEQ_LEN}")
    print()

    # Create model
    config = create_config()

    # Init distributed (required by model)
    import os

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")

    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

    from ironcore.parallel.parallel_states import initialize_model_parallel

    try:
        initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=1.0)
    except Exception:
        pass

    model = TransformerModel(config).to(device=device, dtype=torch.bfloat16)
    model.init_weights()
    model.train()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Create MFU calculator
    mfu_calc = MFUCalculator(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        d_ffn=D_FFN,
        vocab_size=VOCAB_SIZE,
        num_attention_heads=NUM_HEADS,
    )

    calc_params = mfu_calc.get_num_parameters()
    print(f"MFU calc params: {calc_params:,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create dummy input
    hidden = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=device, dtype=torch.bfloat16)

    # Warmup
    print(f"Warming up ({WARMUP_STEPS} steps)...")
    for _ in range(WARMUP_STEPS):
        optimizer.zero_grad()
        output = model(hidden, attention_mask=None, rotary_pos_emb=None)
        loss = output.pow(2).mean()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    # Measure
    print(f"Measuring ({MEASURE_STEPS} steps)...")
    times = []

    for i in range(MEASURE_STEPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        output = model(hidden, attention_mask=None, rotary_pos_emb=None)
        loss = output.pow(2).mean()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"  Step {i + 1}: {(t1 - t0) * 1000:.2f} ms")

    # Calculate metrics
    avg_time = sum(times) / len(times)
    tokens_per_step = BATCH_SIZE * SEQ_LEN
    throughput = tokens_per_step / avg_time

    # Calculate TFLOPS
    tflops = mfu_calc.compute_tflops(
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        step_time_seconds=avg_time,
        num_gpus=1,
    )

    # RTX 3090 peak: ~142 TFLOPS (BF16 tensor core)
    rtx3090_peak_tflops = 142.0
    mfu_percent = (tflops / rtx3090_peak_tflops) * 100

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Average step time:    {avg_time * 1000:.2f} ms")
    print(f"Throughput:           {throughput:,.0f} tokens/sec")
    print(f"TFLOPS/s/GPU:         {tflops:.2f}")
    print(f"RTX 3090 Peak:        {rtx3090_peak_tflops:.0f} TFLOPS")
    print(f"MFU (vs RTX 3090):    {mfu_percent:.1f}%")
    print("=" * 60)

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
