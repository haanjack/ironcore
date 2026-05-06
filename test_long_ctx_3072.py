#!/usr/bin/env python3
"""13B full offload — seq_len 3072 test (long context gap fill)."""
import os, sys, time, torch
from pathlib import Path

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29509")
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

SEQ_LEN = 3072
NUM_STEPS = 3

reset_global_states()

model_cfg = ModelConfig(
    name="llama-13b", d_model=5120, d_ffn=13824, num_layers=40,
    num_attention_heads=40, num_attention_groups=8, head_dim=128,
    max_seq_len=SEQ_LEN, precision="bfloat16", ln_type="rmsnorm",
    activation_type="swiglu", vocab_name_or_path="gpt2",
)
model_cfg.positional_embedding.type = "rope"
model_cfg.positional_embedding.base = 10000

offload_cfg = OffloadConfig(
    enabled=True, optimizer_offload=True, optimizer_state_precision="bf16",
    weight_offload=True, weight_prefetch_layers=2, weight_storage_precision="bf16",
    activation_spill=True, activation_spill_granularity="sub_layer",
    pinned_memory_pool_gb=80.0, gpu_staging_pool_mb=0.0,
)

main_config = MainConfig(
    model=model_cfg,
    init=InitConfig(seed=42),
    optim=OptimConfig(optimizer="adamw", max_lr=3e-4, min_lr=3e-5,
                     warmup_steps=1, annealing_steps=NUM_STEPS+1,
                     weight_decay=0.1, clip_grad=1.0),
    data=DataConfig(seq_length=SEQ_LEN),
    parallel=ParallelConfig(world_size=1, rank=0, local_rank=0),
    trainer=TrainerConfig(micro_batch_size=1, train_batch_size=1,
                           gradient_accumulation_steps=1,
                           model_path="/tmp/validate_3072", log_interval=1,
                           use_flash_attn=True),
    operation=OperationConfig(train_steps=NUM_STEPS+2,
                               activation_recompute=False, no_save=True),
    utils=UtilsConfig(log_level="WARNING"), profiler=ProfilerConfig(),
    peft=PEFTConfig(method="none"), alignment=AlignmentConfig(),
    offload=offload_cfg,
)

def forward_step(model, _data_iterator):
    device = next(model.parameters()).device
    input_ids = torch.randint(0, 32000, (1, SEQ_LEN), device=device)
    labels = input_ids.clone()
    logits = model(input_ids, labels=None)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

print()
print("=" * 70)
print(f"  13B FULL OFFLOAD — seq_len={SEQ_LEN}")
print("=" * 70)
print(f"  GPU: {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
print("=" * 70)

try:
    with patch('ironcore.trainers.base_trainer.get_data_iterator',
               return_value=create_mock_data_iterator()), \
         patch('ironcore.trainers.base_trainer.get_evaluators',
               return_value=create_mock_evaluators()):
        trainer = LanguageModelTrainer(main_config, forward_step, F.cross_entropy)
        trainer._initialize()

        torch.cuda.reset_peak_memory_stats()

        # warmup
        trainer.train_step(step=0)

        step_times = []
        for step in range(1, NUM_STEPS + 1):
            t0 = time.time()
            loss, _, _ = trainer.train_step(step=step)
            step_time = (time.time() - t0) * 1000
            step_times.append(step_time)
            peak = torch.cuda.max_memory_allocated() / 1024**2
            alloc = torch.cuda.memory_allocated() / 1024**2
            print(f"  Step {step}: loss={loss:.4f}, time={step_time:.0f}ms, "
                  f"peak={peak:.0f}MB, alloc={alloc:.0f}MB")

        trainer._finalize_process()

        peak_vram = torch.cuda.max_memory_allocated() / 1024**2
        alloc_vram = torch.cuda.memory_allocated() / 1024**2
        avg_step = sum(step_times) / len(step_times)

        print()
        print(f"  RESULT: {'OK' if peak_vram < 24000 else 'NEAR_OOM'}")
        print(f"  Peak VRAM:  {peak_vram:.0f} MB ({peak_vram/1024:.1f} GB)")
        print(f"  Alloc VRAM: {alloc_vram:.0f} MB ({alloc_vram/1024:.1f} GB)")
        print(f"  Avg step:   {avg_step:.0f} ms")
        print(f"  Throughput: {60000/avg_step:.1f} steps/min")
        print("=" * 70)

except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print(f"  RESULT: OOM")
    else:
        print(f"  RESULT: FAIL - {e}")
    print("=" * 70)
