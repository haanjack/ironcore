# PEFT guide: LoRA

IronCore's LoRA adds low-rank adapter matrices (`lora_A`, `lora_B`) to attention and MLP layers while keeping the base model frozen. Adapters are **replicated** across TP ranks rather than sharded. Replication keeps the math correct without approximation errors from splitting low-rank matrices, and it means checkpointing and gradient sync don't need any special TP-awareness.

## Configuration

### Basic config

Create a PEFT config (e.g., `configs/peft/lora_default.yaml`):

```yaml
# LoRA rank — controls adapter size
r: 8

# Scaling factor — alpha/r determines update magnitude
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

### Available target modules

Attention: `q_proj`, `k_proj`, `v_proj` (k and v share a layer), `o_proj`.

MLP: `up_proj` (gate_proj is treated the same), `down_proj`.

### Training config

```yaml
model: llama-7b          # your model config
data: alpaca_sft         # your dataset config
peft: lora_default       # reference to PEFT config

trainer:
  micro_batch_size: 8
  train_batch_size: 128
  tensor_model_parallel_size: 2
  sequence_chunk_size: 512  # optional: async chunking
```

Or inline:

```yaml
peft:
  method: lora
  lora:
    r: 16
    alpha: 32
    dropout: 0.1
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

## Usage examples

### Basic LoRA training

```bash
torchrun --nproc_per_node=2 ironcore/cli/train.py --config-path configs/train_lora.yaml
```

Expected output:
```
Freezing base model parameters for PEFT method: lora
Trainable parameters: 8,388,608 / 7,016,169,472 (0.12%)
```

### Different ranks

Low rank (`r=4`): minimal parameters, faster training:
```yaml
peft:
  method: lora
  lora:
    r: 4
    alpha: 8
    target_modules: ["q_proj", "v_proj"]
```

High rank (`r=64`): more capacity, slower training:
```yaml
peft:
  method: lora
  lora:
    r: 64
    alpha: 128
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"]
```

### With chunked execution

```yaml
trainer:
  tensor_model_parallel_size: 2
  sequence_chunk_size: 512

peft:
  method: lora
  lora:
    r: 8
    alpha: 16
```

## Implementation details

### Forward pass

For a linear layer `Y = XW`, LoRA adds:

```
Y_lora = X @ W + (X @ A @ B) * scaling
```

Where `A` is `[in_features, r]` (Kaiming uniform init), `B` is `[r, out_features]` (zero init), and `scaling = alpha / r`.

### Tensor parallel behavior

**Column parallel** (output dimension sharded):
```python
base_output = base_layer(x)           # [batch, seq, out/tp_size]
lora_output = lora(x)                 # [batch, seq, out] — replicated
lora_shard = lora_output[..., rank*size:(rank+1)*size]
return base_output + lora_shard
```

**Row parallel** (input dimension sharded):
```python
base_partial, handle = base_layer(x, async_communication=True)
lora_output = lora(x)                 # full input, replicated
handle.wait()
return base_partial + bias + lora_output
```

## Choosing rank and targets

| Task | r | alpha | Modules |
|---|---|---|---|
| Instruction following | 8–16 | 16–32 | q, v, o |
| Domain adaptation | 16–32 | 32–64 | q, k, v, o, up, down |
| Task-specific | 4–8 | 8–16 | q, v |

Memory rough estimates (7B model):
- LoRA only: ~15–20% of full fine-tuning memory
- LoRA + chunking: ~10–15%
- LoRA + TP=2 + chunking: fits 13B on 2x24GB GPUs

LoRA typically needs a 2–5x higher learning rate than full fine-tuning:

```yaml
optim:
  max_lr: 5e-4   # vs ~1e-4 for full fine-tuning
```

## Testing

```bash
# TP correctness
python tests/test_lora_tp_correctness.py --mode save_weights
torchrun --nproc_per_node=2 tests/test_lora_tp_correctness.py --mode load_and_compare

# Async chunking
python tests/test_lora_async.py --tp 1
torchrun --nproc_per_node=2 tests/test_lora_async.py --tp 2

# Checkpoint save/load
python tests/test_lora_checkpoint.py --test save_load_tp1
torchrun --nproc_per_node=2 tests/test_lora_checkpoint.py --test universal_checkpoint
```

Expected: TP=1 and TP=2 outputs match within `atol=1e-1`; only `lora_A`/`lora_B` parameters receive gradients; trainable parameters below 5% of total.

## Troubleshooting

**High memory usage with large batches.** Reduce `micro_batch_size` or enable `sequence_chunk_size: 512`.

**Loss not decreasing.** LoRA often needs a higher learning rate than full fine-tuning. Try `max_lr: 5e-4` and increase `alpha` (e.g., 32) for stronger updates.

**Outputs differ between TP ranks.** Run the TP correctness test — the usual cause is LoRA weights not being loaded identically to all ranks.

**Checkpoint loading fails.** The `r`, `alpha`, and `target_modules` in the load config must match exactly what was used during training.

## Performance benchmarks

Approximate speeds on A100 (7B model):

| Configuration | Memory/GPU | Tokens/sec |
|---|---|---|
| Full fine-tuning (TP=1) | 56 GB | 1200 |
| LoRA (TP=1) | 15 GB | 1150 |
| LoRA (TP=2) | 8 GB | 2200 |
| LoRA + chunking (TP=2) | 6 GB | 2000 |

## References

- LoRA paper: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
