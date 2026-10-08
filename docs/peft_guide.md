# PEFT guide: LoRA

IronCore's LoRA adds low-rank adapter matrices (`lora_A`, `lora_B`) to attention and MLP layers while keeping the base model frozen. Both adapter matrices are **replicated** across TP ranks. Their computation uses temporary TP views and explicit gradient communication; simply copying the parameters would not synchronize gradients.

| Projection | Parallel computation | Replicated gradient handling |
| --- | --- | --- |
| Column (Q/K/V, gate/up) | Compute `x @ A`, use the local B output columns | Sum low-rank gradients before A; gather column gradients into full B |
| Row (attention output, down) | Use local A input rows, sum partial low-rank activations, then multiply full B | Gather row gradients into full A; B receives the complete output gradient |

This also applies to fused KV and gate/up projections. Adapter initialization
uses stable per-module seeds and restores zero B after model initialization.
Dropout uses replicated TP RNG streams; checkpoint recomputation replays the
original masks and preserves the stream for the next microbatch. Reentrant
checkpointing receives a differentiable embedding output when embeddings are
frozen, so first-layer adapters still receive gradients.

New checkpoints store full replicated adapters and optimizer moments at either
TP degree. Old distributed checkpoints containing sharded adapter matrices need
conversion before loading into this replicated layout.

## TP validation

```bash
# Real CPU/Gloo ranks; no model downloads.
torchrun --standalone --nproc_per_node=2 -m tests.multi_gpu.test_lora_tp --device cpu
# Corresponding GPU/NCCL test items.
torchrun --standalone --nproc_per_node=2 -m pytest -o addopts='' tests/multi_gpu/test_lora_tp.py
# BF16 autocast with FP32 master weights (native trainer, finite gradients,
# actual updates and exact TP rank replicas; separate from FP32 parity).
torchrun --standalone --nproc_per_node=2 -m tests.multi_gpu.test_lora_tp \
  --device cuda --precision bfloat16 --output .local/lora-tp2-gpu-bf16-tiny-results.json
```

The test uses a scaled SmolLM2 decoder with TP-compatible head counts, and
Gemma 4 E2B/E4B/31B layouts. It checks nonzero-adapter gradients, three native
trainer steps starting from zero B, exact equality between rank replicas,
checkpoint/optimizer restoration, dropout/recompute parity and adapter merging.
The AdamW comparison uses LR 1e-3 and epsilon 1e-4 to avoid amplifying tiny FP32
reduction differences in almost-zero gradients; direct gradient tests remain
independent. This is a correctness check, not a training-quality evaluation.
Universal checkpoints also resume at TP=1 with exact adapter weights and
optimizer moments. The full pretrained model check uses LR 1e-4.

SmolLM2-135M has 9 query/3 KV heads and 360M has 15/5; these presets cannot be
evenly partitioned by the current generic TP=2 attention path. The official
[1.7B configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/blob/main/config.json)
has 32/32 heads. A local 1.7B checkpoint can be checked with the same driver:

```bash
torchrun --standalone --nproc_per_node=2 -m tests.multi_gpu.test_lora_tp \
  --device cpu --checkpoint .local/models/SmolLM2-1.7B \
  --output .local/lora-tp2-smollm2-1.7b-results.json
```

CPU/Gloo results below are separate from the GPU/NCCL checks described afterward.

Validation on 2026-10-08 passed all 24 tiny cases and the official SmolLM2-1.7B
checkpoint with PyTorch 2.13.0 / Transformers 5.17.0 on CPU/Gloo. The full-size
three-step check used fixed token batches, LoRA rank 2 and alpha 4. Reference
losses were 12.67387867, 10.30386639 and 8.42096615; the maximum TP loss difference
was **2.00e-5**. Maximum relative gradient L2 error over complete adapter matrices
was **6.93e-5** (bound 1e-4); FP32 absolute gradient differences reached **0.00110**
for larger gradients. Near-zero entries are not claimed to match bitwise.
Updated adapters were compared within FP32 tolerance and were identical between
the two TP ranks. Merge preserved logits within the same FP32 comparison bound.

SmolLM2-1.7B rank-2 adapters contain 2,260,992 parameters (**9.04 MB per rank** in
FP32). The previous partially sharded representation used 1,474,560 local
parameters (**5.90 MB**), so correct replication adds **3.15 MB per rank**. The
partitioned low-rank matmul shapes keep the same FLOP count; new low-rank
reductions and adapter-gradient gathers supply previously missing communication.
No GPU throughput or MFU was measured. Logs and JSON results are under ignored
`.local/`; the CPU unit/regression/property selection passed **655 tests**.

GPU validation on the same date used **two RTX 3090 24 GiB GPUs**, PyTorch
2.14.0+cu130, Transformers 5.17.0 and NCCL 2.30.7 in a CUDA container. All
**24 FP32 parity cases** and **24 BF16 autocast training cases** passed. BF16
uses FP32 master weights and checks finite logits/adapter gradients, three
native trainer steps, falling loss and exact adapter replicas between ranks.
It does **not** assert BF16 TP=1/2 numerical parity, BF16 checkpoint/merge parity,
or long-run training quality. The FP32 cases check checkpoint/optimizer resume,
degree changes and merge as described above.

The actual SmolLM2-1.7B checkpoint also passed on GPU. FP32 TP=1/2 comparison
had maximum loss difference **2.38e-5** and maximum per-adapter relative gradient
L2 error **4.50e-5**. BF16 TP=2 losses were **12.55744 → 10.84642 → 8.72879**;
rank replicas agreed exactly and peak allocated memory was **5,204,510,208 bytes
per GPU**. This uses fixed token batches, rank-2 adapters, accumulation 2,
LR 1e-4 and AdamW epsilon 1e-4. Full reference snapshots are held on CPU so the
FP32 TP=1 reference and TP=2 model can coexist on a 24 GiB GPU.

```bash
torchrun --standalone --nproc_per_node=2 -m tests.multi_gpu.test_lora_tp \
  --device cuda --checkpoint .local/models/SmolLM2-1.7B \
  --output .local/lora-tp2-gpu-smollm2-1.7b-results.json
torchrun --standalone --nproc_per_node=2 -m tests.multi_gpu.test_lora_tp \
  --device cuda --precision bfloat16 --checkpoint .local/models/SmolLM2-1.7B \
  --output .local/lora-tp2-gpu-bf16-smollm2-1.7b-results.json
```

These checks use native SDPA; a separate flash-attn package is not required for
this validation. GPU logs and result JSON files are stored in ignored `.local/`
and `/tmp/ironcore-*-gpu-*.log`. No steady-state throughput or MFU benchmark was
performed.

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
base_output = base_layer(x)  # [batch, seq, out/tp_size]
lora_output = lora(x)  # [batch, seq, out] — replicated
lora_shard = lora_output[..., rank * size : (rank + 1) * size]
return base_output + lora_shard
```

**Row parallel** (input dimension sharded):
```python
base_partial, handle = base_layer(x, async_communication=True)
lora_output = lora(x)  # full input, replicated
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
