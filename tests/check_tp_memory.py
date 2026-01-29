#!/usr/bin/env python3
import torch
import torch.distributed as dist
import os
import sys
sys.path.insert(0, '/mnt/wsl/PHYSICALDRIVE4p1/hanjack/workspace/ironcore')

from ironcore.config import (
    MainConfig, ModelConfig, TrainerConfig, InitConfig, OptimConfig,
    DataConfig, ParallelConfig, OperationConfig, UtilsConfig
)
from ironcore.layers.attention import Attention
from ironcore.parallel import parallel_states

def create_config(tp_size):
    model_config = ModelConfig(
        d_model=512, num_attention_heads=8, num_attention_groups=8,
        head_dim=64, max_seq_len=128, max_position_embeddings=128,
        dropout_attn=0.0, no_bias=False,
    )
    trainer_config = TrainerConfig(tensor_model_parallel_size=tp_size, use_flash_attn=False)
    init_config = InitConfig(seed=42, init_std=0.02)
    optim_config = OptimConfig(max_lr=1e-3, weight_decay=0.01)
    data_config = DataConfig()
    parallel_config = ParallelConfig()
    operation_config = OperationConfig(train_steps=100)
    utils_config = UtilsConfig()
    return MainConfig(
        model=model_config, trainer=trainer_config, init=init_config,
        optim=optim_config, data=data_config, parallel=parallel_config,
        operation=operation_config, utils=utils_config,
    )

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
device = torch.device(f'cuda:{local_rank}')

dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)

tp_size = 2
parallel_states.initialize_model_parallel(tensor_model_parallel_size=tp_size, timeout_in_minutes=10.0)
config = create_config(tp_size)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

attention = Attention(config).to(device)
attention.init_weights()

local_params = sum(p.numel() for p in attention.parameters())
mem_init = torch.cuda.memory_allocated() / 1024**2

hidden_states = torch.randn(2, 64, 512, device=device, requires_grad=True)
attn_mask = torch.tril(torch.ones(64, 64, device=device)).unsqueeze(0).unsqueeze(0).expand(2, -1, -1, -1)

output = attention(hidden_states, attn_mask)
mem_fwd = torch.cuda.memory_allocated() / 1024**2

output.sum().backward()
mem_peak = torch.cuda.max_memory_allocated() / 1024**2

print(f"[Rank {rank}] TP=2 Memory:")
print(f"  Local params:     {local_params:,} ({local_params * 4 / 1024**2:.2f} MB)")
print(f"  After init:       {mem_init:.2f} MB")
print(f"  After forward:    {mem_fwd:.2f} MB")
print(f"  Peak (backward):  {mem_peak:.2f} MB")

dist.barrier()
dist.destroy_process_group()
