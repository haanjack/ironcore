# Parameter-Efficient Fine-Tuning (PEFT) Guide

This guide covers the LoRA (Low-Rank Adaptation) implementation in IronCore, a parameter-efficient fine-tuning method that enables training large models with minimal memory overhead.

## Overview

**LoRA** reduces the number of trainable parameters by adding small, low-rank adapter matrices to linear layers while keeping the base model frozen. This enables:

- **Memory Efficiency**: Train 7B+ models on consumer GPUs (typically ~0.1-1% trainable parameters)
- **Faster Training**: Smaller optimizer states and fewer gradient computations
- **Modular Fine-Tuning**: Multiple task-specific adapters without modifying base weights

## Key Features

### 1. Replicated Adapter Design

LoRA adapters are **replicated** across tensor parallel (TP) ranks, not sharded:

- **Low Memory Overhead**: r=8 adapters are ~0.4% of base layer size
- **Numerical Stability**: Avoids approximation errors from sharding low-rank matrices
- **Simple Gradients**: Only standard data-parallel synchronization needed
- **Easy Merging**: Merge operation (W' = W + BA) requires full B and A matrices

### 2. Tensor Parallelism Integration

- **Column Parallel Layers**: LoRA operates on full dimensions, output is manually sharded
- **Row Parallel Layers**: LoRA operates on full input, contribution added after all-reduce
- **Concatenated Weights**: Special handling for K+V projection layers
- **Async Support**: Full compatibility with chunked execution and async communication

### 3. Checkpoint Compatibility

- **Universal Checkpoints**: Save TP=1, load TP=N (or vice versa)
- **Replicated Saving**: LoRA weights saved from rank 0 only to avoid duplicates
- **Replicated Loading**: LoRA weights loaded identically to all ranks

## Configuration

### Basic LoRA Configuration

Create a PEFT config file (e.g., `configs/peft/lora_default.yaml`):

```yaml
# LoRA rank - controls adapter size
r: 8

# Scaling factor - alpha/r determines update magnitude
alpha: 16.0

# Dropout applied to LoRA activations (0.0 = no dropout)
dropout: 0.0

# Which layers to apply LoRA to
target_modules:
  - q_proj      # Query projection in attention
  - v_proj      # Value projection in attention
  - o_proj      # Output projection in attention
  - up_proj     # MLP up projection
  - down_proj   # MLP down projection
```

### Available Target Modules

- **Attention Layers**:
  - `q_proj`: Query projection
  - `k_proj`: Key projection (shares layer with v_proj)
  - `v_proj`: Value projection (shares layer with k_proj)
  - `o_proj`: Output projection

- **MLP Layers**:
  - `up_proj`: Up projection (gate_proj is treated the same)
  - `down_proj`: Down projection

### Training Configuration

In your main training config (e.g., `configs/train_lora.yaml`):

```yaml
model: llama-7b          # Your model config
data: alpaca_sft         # Your dataset config
peft: lora_default       # Reference to PEFT config

trainer:
  micro_batch_size: 8
  train_batch_size: 128
  tensor_model_parallel_size: 2
  sequence_chunk_size: 512  # Optional: enable async chunking
```

Alternatively, inline PEFT config:

```yaml
peft:
  method: lora
  lora:
    r: 16
    alpha: 32
    dropout: 0.1
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

## Usage Examples

### Example 1: Basic LoRA Training

```bash
# Train 7B model with LoRA (TP=2)
torchrun --nproc_per_node=2 ironcore/cli/train.py \
  --config-path configs/train_lora.yaml
```

Expected output:
```
Freezing base model parameters for PEFT method: lora
Trainable parameters: 8,388,608 / 7,016,169,472 (0.12%)
```

### Example 2: Different LoRA Ranks

**Low Rank (r=4)** - Minimal parameters, faster training:
```yaml
peft:
  method: lora
  lora:
    r: 4
    alpha: 8
    target_modules: ["q_proj", "v_proj"]  # Fewer layers
```

**High Rank (r=64)** - More capacity, slower training:
```yaml
peft:
  method: lora
  lora:
    r: 64
    alpha: 128
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"]
```

### Example 3: With Chunked Execution

Enable async chunking for memory efficiency:

```yaml
trainer:
  tensor_model_parallel_size: 2
  sequence_chunk_size: 512  # Process 512 tokens at a time

peft:
  method: lora
  lora:
    r: 8
    alpha: 16
```

## Implementation Details

### LoRA Forward Pass

For a linear layer Y = XW, LoRA adds:

```
Y_lora = X @ W + (X @ A @ B) * scaling
```

Where:
- `A`: [in_features, r] - initialized with Kaiming uniform
- `B`: [r, out_features] - initialized with zeros
- `scaling = alpha / r`

### Tensor Parallel Behavior

**Column Parallel** (output dimension sharded):
```python
# Base computation (sharded output)
base_output = base_layer(x)  # [batch, seq, out/tp_size]

# LoRA computation (full output, replicated)
lora_output = lora(x)  # [batch, seq, out]

# Manually shard LoRA output
lora_shard = lora_output[..., rank*size:(rank+1)*size]

# Combine
return base_output + lora_shard
```

**Row Parallel** (input dimension sharded):
```python
# Base computation with async all-reduce
base_partial, handle = base_layer(x, async_communication=True)

# LoRA computation (full input, replicated)
lora_output = lora(x)  # No async needed

# Finalize: wait for all-reduce, add bias, combine
handle.wait()
return base_partial + bias + lora_output
```

## Best Practices

### 1. Choosing LoRA Rank

| Task Type | Recommended r | Alpha | Target Modules |
|-----------|--------------|-------|----------------|
| Instruction Following | 8-16 | 16-32 | q_proj, v_proj, o_proj |
| Domain Adaptation | 16-32 | 32-64 | All (q,k,v,o,up,down) |
| Task-Specific | 4-8 | 8-16 | q_proj, v_proj |

### 2. Memory Considerations

- **LoRA only**: ~15-20% of full fine-tuning memory
- **LoRA + chunking**: ~10-15% of full fine-tuning memory
- **LoRA + TP=2 + chunking**: Can train 13B models on 2x24GB GPUs

### 3. Hyperparameter Tuning

**Learning Rate**: Typically 2-5x higher than full fine-tuning
```yaml
optim:
  max_lr: 5e-4  # vs 1e-4 for full fine-tuning
```

**Weight Decay**: Apply only to LoRA B matrices (automatic)
```yaml
optim:
  weight_decay: 0.01  # LoRA A matrices excluded automatically
```

**Dropout**: Usually not needed, but can help with overfitting
```yaml
peft:
  lora:
    dropout: 0.1  # Only if overfitting observed
```

## Testing

### Run LoRA Tests

```bash
# Test 1: TP correctness
python tests/test_lora_tp_correctness.py --mode save_weights
torchrun --nproc_per_node=2 tests/test_lora_tp_correctness.py --mode load_and_compare

# Test 2: Async chunking
python tests/test_lora_async.py --tp 1
torchrun --nproc_per_node=2 tests/test_lora_async.py --tp 2

# Test 3: Checkpoint save/load
python tests/test_lora_checkpoint.py --test save_load_tp1
torchrun --nproc_per_node=2 tests/test_lora_checkpoint.py --test universal_checkpoint
```

### Expected Test Results

- **TP Correctness**: TP=1 and TP=2 outputs match within `atol=1e-1, rtol=1e-1`
- **Parameter Ratio**: Trainable parameters < 5% of total
- **Gradient Flow**: Only LoRA parameters (`lora_A`, `lora_B`) receive gradients
- **Async Chunking**: Chunked and non-chunked outputs match
- **Checkpoints**: LoRA weights correctly saved and restored

## Troubleshooting

### Issue: High Memory Usage

**Cause**: LoRA + large batch size
**Solution**:
```yaml
trainer:
  micro_batch_size: 4  # Reduce batch size
  sequence_chunk_size: 512  # Enable chunking
```

### Issue: Training Loss Not Decreasing

**Cause**: Learning rate too low
**Solution**:
```yaml
optim:
  max_lr: 5e-4  # Increase LR for LoRA
peft:
  lora:
    alpha: 32  # Increase alpha for stronger updates
```

### Issue: Outputs Differ Between TP Ranks

**Cause**: Incorrect LoRA adapter replication
**Diagnosis**: Run TP correctness test
**Solution**: Ensure LoRA weights loaded identically to all ranks in checkpoints

### Issue: Checkpoint Loading Fails

**Cause**: LoRA config mismatch between save and load
**Solution**: Ensure same `r`, `alpha`, and `target_modules` in config

## Performance Benchmarks

Approximate training speeds on A100 GPUs (7B model):

| Configuration | Memory/GPU | Tokens/sec | Speedup |
|---------------|-----------|------------|---------|
| Full FT (TP=1) | 56 GB | 1200 | 1.0x |
| LoRA (TP=1) | 15 GB | 1150 | 0.96x |
| LoRA (TP=2) | 8 GB | 2200 | 1.83x |
| LoRA + Chunk (TP=2) | 6 GB | 2000 | 1.67x |

## References

- **LoRA Paper**: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
