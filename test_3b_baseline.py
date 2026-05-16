#!/usr/bin/env python3
import os, sys, time, torch
from pathlib import Path

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29501")
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

# Same 3B config as optimizer_offload test
d_model = 3072
num_heads = 32
head_dim = d_model // num_heads

model_cfg = ModelConfig(
    name="llama-3b",
    d_model=d_model,
    d_ffn=8192,
    num_layers=26,
    num_attention_heads=num_heads,
    num_attention_groups=8,
    head_dim=head_dim,
    max_seq_len=1024,
    precision="bfloat16",
    ln_type="rmsnorm",
    activation_type="swiglu",
    vocab_name_or_path="gpt2",
)
model_cfg.positional_embedding.type = "rope"
model_cfg.positional_embedding.base = 10000

# NO OFFLOAD (baseline)
offload_cfg = OffloadConfig(enabled=False)

main_config = MainConfig(
    model=model_cfg,
    init=InitConfig(seed=42),
    optim=OptimConfig(optimizer="adamw", max_lr=3e-4, min_lr=3e-5, warmup_steps=1, annealing_steps=10, weight_decay=0.1, clip_grad=1.0),
    data=DataConfig(seq_length=1024),
    parallel=ParallelConfig(world_size=1, rank=0, local_rank=0),
    trainer=TrainerConfig(micro_batch_size=1, train_batch_size=1, gradient_accumulation_steps=1, model_path="/tmp/validate_3b_base", log_interval=1, use_flash_attn=True),
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

with patch('ironcore.trainers.base_trainer.get_data_iterator', return_value=create_mock_data_iterator()), \
     patch('ironcore.trainers.base_trainer.get_evaluators', return_value=create_mock_evaluators()):
    reset_global_states()
    trainer = LanguageModelTrainer(main_config, forward_step, F.cross_entropy)
    trainer._initialize()

    torch.cuda.reset_peak_memory_stats()
    losses = []
    step_times = []

    print()
    print("=" * 70)
    print("  3B MODEL BASELINE (NO OFFLOAD) VALIDATION")
    print("=" * 70)
    print(f"  GPU:     {torch.cuda.get_device_name()}")
    print(f"  GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Config:  d_model={d_model}, d_ffn=8192, layers=26, head_dim={head_dim}")
    print("=" * 70)
    print()

    start = time.time()

    for step in range(10):
        step_start = time.time()
        loss, _, _ = trainer.train_step(step=step)
        step_time = (time.time() - step_start) * 1000
        losses.append(loss)
        step_times.append(step_time)
        print(f"  Step {step}: loss={loss:.4f}, time={step_time:.1f}ms, VRAM={torch.cuda.memory_allocated()/1024**2:.0f}MB")

    total_time = time.time() - start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    allocated_vram_mb = torch.cuda.memory_allocated() / 1024**2

    trainer._finalize_process()

    print()
    print("=" * 70)
    print("  RESULTS (MEASURED)")
    print("=" * 70)
    print(f"  Model:        ~3B parameters (6.0GB BF16 weights)")
    print(f"  Steps:        {len(losses)}")
    print(f"  Final Loss:   {losses[-1]:.4f}")
    print(f"  Min Loss:     {min(losses):.4f}")
    print(f"  Peak VRAM:    {peak_vram_mb:.1f} MB ({peak_vram_mb / 1024:.2f} GB)")
    print(f"  Allocated:    {allocated_vram_mb:.1f} MB ({allocated_vram_mb / 1024:.2f} GB)")
    print(f"  Total Time:   {total_time:.1f}s")
    print(f"  Avg/step:     {sum(step_times) / len(step_times):.1f}ms")
    print(f"  Throughput:   {10 / total_time * 60:.1f} steps/min")
    print("=" * 70)
