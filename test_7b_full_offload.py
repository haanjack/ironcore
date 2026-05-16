#!/usr/bin/env python3
"""Test 7B model with FULL OFFLOAD (optimizer_offload+weight_offload+activation_spill) on 24GB GPU."""
import os, sys, time, torch
from pathlib import Path

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29503")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

sys.path.insert(0, str(Path(__file__).parent.parent))

from ironcore.config import *
from ironcore.global_vars import reset_global_states
from ironcore.trainers import LanguageModelTrainer
from unittest.mock import patch
import torch.nn.functional as F
from tests.integration.offload.conftest import create_mock_data_iterator, create_mock_evaluators

# 7B model config
d_model = 4096
num_heads = 32
head_dim = d_model // num_heads

model_cfg = ModelConfig(
    name="llama-7b",
    d_model=d_model,
    d_ffn=11008,
    num_layers=32,
    num_attention_heads=num_heads,
    num_attention_groups=8,  # GQA
    head_dim=head_dim,
    max_seq_len=1024,
    precision="bfloat16",
    ln_type="rmsnorm",
    activation_type="swiglu",
    vocab_name_or_path="gpt2",
)
model_cfg.positional_embedding.type = "rope"
model_cfg.positional_embedding.base = 10000

# FULL OFFLOAD: optimizer_offload+weight_offload+activation_spill with larger pinned pool
offload_cfg = OffloadConfig(
    enabled=True,
    optimizer_offload=True,
    weight_offload=True,
    weight_prefetch_layers=2,
    weight_storage_precision="bf16",
    activation_spill=True,
    activation_spill_granularity="sub_layer",
    pinned_memory_pool_gb=32.0,        # Increased from 16GB
    gpu_staging_pool_mb=0.0,
)

main_config = MainConfig(
    model=model_cfg,
    init=InitConfig(seed=42),
    optim=OptimConfig(optimizer="adamw", max_lr=3e-4, min_lr=3e-5, warmup_steps=1, annealing_steps=10, weight_decay=0.1, clip_grad=1.0),
    data=DataConfig(seq_length=1024),
    parallel=ParallelConfig(world_size=1, rank=0, local_rank=0),
    trainer=TrainerConfig(micro_batch_size=1, train_batch_size=1, gradient_accumulation_steps=1, model_path="/tmp/validate_7b_full", log_interval=1, use_flash_attn=True),
    operation=OperationConfig(train_steps=12, activation_recompute=False, no_save=True),
    utils=UtilsConfig(log_level="WARNING"),
    profiler=ProfilerConfig(),
    peft=PEFTConfig(method="none"),
    alignment=AlignmentConfig(),
    offload=offload_cfg,
)

def forward_step(model, _data_iterator):
    device = next(model.parameters()).device
    input_ids = torch.randint(0, 32000, (1, 1024), device=device)
    labels = input_ids.clone()
    logits = model(input_ids, labels=None)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

print()
print("=" * 70)
print("  7B MODEL FULL OFFLOAD VALIDATION")
print("=" * 70)
print(f"  GPU:     {torch.cuda.get_device_name()}")
print(f"  GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"  Config:  d_model={d_model}, d_ffn=11008, layers=32")
print(f"  Offload:  optimizer_offload + weight_offload + activation_spill (pinned_pool=32GB)")
print("=" * 70)
print()

try:
    with patch('ironcore.trainers.base_trainer.get_data_iterator', return_value=create_mock_data_iterator()), \
         patch('ironcore.trainers.base_trainer.get_evaluators', return_value=create_mock_evaluators()):
        reset_global_states()
        trainer = LanguageModelTrainer(main_config, forward_step, F.cross_entropy)
        trainer._initialize()

        torch.cuda.reset_peak_memory_stats()
        losses = []
        step_times = []

        start = time.time()

        for step in range(10):
            step_start = time.time()
            loss, _, _ = trainer.train_step(step=step)
            step_time = (time.time() - step_start) * 1000
            losses.append(loss)
            step_times.append(step_time)
            vram_mb = torch.cuda.memory_allocated() / 1024**2
            print(f"  Step {step}: loss={loss:.4f}, time={step_time:.1f}ms, VRAM={vram_mb:.0f}MB")

        total_time = time.time() - start
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
        allocated_vram_mb = torch.cuda.memory_allocated() / 1024**2

        trainer._finalize_process()

        print()
        print("=" * 70)
        print("  RESULTS (MEASURED)")
        print("=" * 70)
        print(f"  Model:        ~7B parameters (13.2GB BF16 weights)")
        print(f"  Offload Mode: optimizer_offload + weight_offload + activation_spill (FULL)")
        print(f"  Steps:        {len(losses)}")
        print(f"  Final Loss:   {losses[-1]:.4f}")
        print(f"  Min Loss:     {min(losses):.4f}")
        print(f"  Peak VRAM:    {peak_vram_mb:.1f} MB ({peak_vram_mb / 1024:.2f} GB)")
        print(f"  Allocated:    {allocated_vram_mb:.1f} MB ({allocated_vram_mb / 1024:.2f} GB)")
        print(f"  Total Time:   {total_time:.1f}s")
        print(f"  Avg/step:     {sum(step_times) / len(step_times):.1f}ms")
        print(f"  Throughput:   {10 / total_time * 60:.1f} steps/min")
        print("=" * 70)

except Exception as e:
    print(f"\n  ERROR: {type(e).__name__}: {str(e)[:200]}")
    import traceback
    traceback.print_exc()
