import os
import sys

import torch

# Add project root to path
sys.path.append(os.getcwd())

from ironcore.config import (
    DataConfig,
    InitConfig,
    MainConfig,
    ModelConfig,
    OperationConfig,
    OptimConfig,
    ParallelConfig,
    TrainerConfig,
    UtilsConfig,
)
from ironcore.models.transformer import TransformerLayer
from ironcore.parallel.parallel_states import initialize_model_parallel


def test_chunking():
    print("Testing TransformerLayer with Chunking...")

    # Initialize parallel state (tp_size=1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

    try:
        initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=1.0)
    except Exception as e:
        # Might already be initialized
        print(f"Parallel init check: {e}")

    # Mock Config
    config = MainConfig(
        model=ModelConfig(
            d_model=64,
            num_layers=1,
            num_attention_heads=4,
            num_attention_groups=4,
            head_dim=16,
            d_ffn=128,
            max_seq_len=128,
        ),
        trainer=TrainerConfig(tensor_model_parallel_size=1),
        init=InitConfig(),
        optim=OptimConfig(),
        data=DataConfig(),
        parallel=ParallelConfig(),
        operation=OperationConfig(),
        utils=UtilsConfig(),
    )

    layer = TransformerLayer(config).cuda()
    layer.init_weights()

    batch_size = 2
    seq_len = 32
    hidden_states = torch.randn(batch_size, seq_len, 64).cuda()
    # Create a basic causal mask or none
    attention_mask = None
    rotary_pos_emb = None

    # 1. Run Standard Forward (Chunks = 1)
    config.trainer.sequence_chunk_size = None
    config.trainer.use_flash_attn = False
    config.model.dropout_attn = 0.0
    config.model.dropout_mlp = 0.0
    config.model.dropout_embd = 0.0

    # Create causal mask: [b, 1, s, s]
    # 1.0 means attend, 0.0 means mask
    mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), device=hidden_states.device))
    attention_mask = mask

    layer.eval()

    with torch.no_grad():
        output_std = layer(hidden_states, attention_mask, rotary_pos_emb)

    print("Standard Output shape:", output_std.shape)
    if torch.isnan(output_std).any():
        print("Standard Output contains NaNs!")
        print(output_std)

    # 2. Run Chunked Forward (Chunk Size = 16)
    # seq_len=32. chunk_size=16 => 2 chunks
    config.trainer.sequence_chunk_size = 16

    with torch.no_grad():
        output_chunked = layer(hidden_states, attention_mask, rotary_pos_emb)

    print("Chunked Output shape:", output_chunked.shape)
    if torch.isnan(output_chunked).any():
        print("Chunked Output contains NaNs!")

    # 3. Compare
    diff = torch.abs(output_std - output_chunked).max()
    print(f"Max difference between Standard and Chunked: {diff}")

    # Allow small floating point error
    assert diff < 1e-5, f"Difference too high: {diff}"

    # 4. Run with Chunk Size = 8 (4 chunks)
    config.trainer.sequence_chunk_size = 8
    with torch.no_grad():
        output_chunked_4 = layer(hidden_states, attention_mask, rotary_pos_emb)

    diff_4 = torch.abs(output_std - output_chunked_4).max()
    print(f"Max difference between Standard and Chunked(4): {diff_4}")
    assert diff_4 < 1e-5, f"Difference too high for chunks=4: {diff_4}"

    # 5. Run with Target Chunk Size (Uneven)
    # seq_len=32. chunk_size=12 => 3 chunks [12, 12, 8]
    config.trainer.sequence_chunk_size = 12

    with torch.no_grad():
        output_dynamic = layer(hidden_states, attention_mask, rotary_pos_emb)

    print("Dynamic Chunked Output shape:", output_dynamic.shape)
    diff_dynamic = torch.abs(output_std - output_dynamic).max()
    print(f"Max difference between Standard and Dynamic(Size=12): {diff_dynamic}")
    assert diff_dynamic < 1e-5, f"Difference too high for dynamic chunk size: {diff_dynamic}"

    print("Test passed!")


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_chunking()
    else:
        print("No CUDA, skipping.")
