# Parallelism in IronCore

IronCore supports four orthogonal parallelism strategies that can be combined: **Data Parallelism (DP)**, **Tensor Parallelism (TP)**, **Expert Parallelism (EP)**, and **Fully Sharded Data Parallel (FSDP)**. This document covers how each is implemented, how they compose, and how to configure them.
This document is written with Claude, but will be kinder with pictures.

---

## Table of Contents

1. [Overview](#overview)
2. [Process Group Layout](#process-group-layout)
3. [Tensor Parallelism (TP)](#tensor-parallelism-tp)
4. [Data Parallelism (DP) and FSDP](#data-parallelism-dp-and-fsdp)
5. [Expert Parallelism (EP)](#expert-parallelism-ep)
6. [Async TP Communication](#async-tp-communication)
7. [Initialization Order](#initialization-order)
8. [Configuration Reference](#configuration-reference)
9. [Usage Examples](#usage-examples)
10. [Known Limitations](#known-limitations)

---

## Overview

| Strategy | Splits | Communication | Use Case |
|----------|--------|---------------|----------|
| Data Parallel (DDP) | Batch | All-reduce gradients | Multi-GPU, same model fits in one GPU |
| FSDP | Batch + params | All-gather params, reduce-scatter grads | Large models, gradient/optimizer state sharding |
| Tensor Parallel | Model weights (layer-wise) | All-gather / All-reduce per layer | Very large layers that don't fit on one GPU |
| Expert Parallel | MoE expert subsets | All-to-all token dispatch | Mixture-of-Experts models |

TP and DP/FSDP are orthogonal and can be combined freely. EP adds a third dimension on top of TP when using MoE layers.

---

## Process Group Layout

### TP + DP

All processes are arranged in a 2D grid: `[DP × TP]`.

```
World size = TP_size × DP_size

Example: TP=2, DP=2 (world=4)

         TP rank 0   TP rank 1
DP rank 0  [Rank 0]   [Rank 1]   ← TP Group 0: [0, 1]
DP rank 1  [Rank 2]   [Rank 3]   ← TP Group 1: [2, 3]
              │           │
           DP Group 0   DP Group 1
            [0, 2]       [1, 3]
```

- **TP group**: Ranks that share weights for the same layer (communicate during forward/backward).
- **DP group**: Ranks that hold independent micro-batches but identical weights (communicate during gradient sync).

Initialized via `initialize_model_parallel(tensor_model_parallel_size)` in `parallel/parallel_states.py`.

### TP + EP + DP

When using MoE, a third EP dimension is added: `[DP × EP × TP]`.

```
World size = DP_size × EP_size × TP_size

Example: DP=2, EP=2, TP=2 (world=8)

Rank layout: rank = dp_idx*(ep*tp) + ep_idx*tp + tp_idx

EP Group within DP rank 0: [0, 2] and [1, 3]   ← different expert subsets
TP Group within EP rank: [0, 1] and [2, 3]      ← TP sharding of same experts
```

Initialized via `initialize_expert_parallel(expert_model_parallel_size, tensor_model_parallel_size)` in `parallel/expert_parallel/parallel_states.py`.

---

## Tensor Parallelism (TP)

IronCore implements Megatron-style column/row parallel linear layers. The key insight is that an MLP or attention projection can be split across GPUs such that each requires only one communication operation.

### Column Parallel Linear

Splits the **output dimension** across TP ranks. Each rank holds `output_size / tp_size` columns.

```
Input  [B, S, H]  →  replicated on all ranks
Weight [H, F/tp]  →  each rank has F/tp columns
Output [B, S, F/tp] →  all-gather → [B, S, F]  (if gather_output=True)
```

Used for: FFN up-projection, QKV projections, embedding input.

```python
# ironcore/parallel/tensor_parallel/layers.py
self.up_proj = ColumnParallelLinear(config, hidden_size, d_ffn, gather_output=False)
```

### Row Parallel Linear

Splits the **input dimension** across TP ranks. Each rank holds `input_size / tp_size` rows.

```
Input  [B, S, F/tp] →  scattered across ranks (or already parallel from prior Col layer)
Weight [F/tp, H]    →  each rank has F/tp rows
Output [B, S, H]    →  all-reduce → [B, S, H]  (sums partial results)
```

Used for: FFN down-projection, attention output projection.

```python
self.down_proj = RowParallelLinear(config, d_ffn, hidden_size, input_is_parallel=True)
```

### Attention with TP

Attention heads are partitioned across TP ranks:

```python
# ironcore/layers/attention.py
self.num_local_attention_heads  = num_attention_heads  // tp_size
self.num_local_attention_groups = num_attention_groups // tp_size  # for GQA/MQA
```

Constraint: `num_attention_heads` and `num_attention_groups` must both be divisible by `tp_size`.

### Vocab Parallel Embedding and Cross-Entropy

The vocabulary is partitioned across TP ranks. Each rank holds `vocab_size / tp_size` embeddings.

During the forward pass, tokens outside a rank's range produce zero embeddings; an all-reduce sums across ranks to recover the full embedding. Cross-entropy loss is computed in a numerically stable way by:

1. All-reducing the max logit across ranks for stabilization.
2. Each rank contributes its partial `exp(logit)` sum; ranks reduce to get the full denominator.
3. Only the rank holding the correct token's logit contributes that value; reduces to get the numerator.

```python
# ironcore/parallel/tensor_parallel/cross_entropy.py
loss = vocab_parallel_cross_entropy(vocab_parallel_logits, labels)
```

### TP Communication Primitives

All communication is handled via autograd-compatible functions in `parallel/tensor_parallel/comm.py`.

| Operation | Used by | PyTorch primitive |
|-----------|---------|-------------------|
| `copy_inputs_to_model_parallel_workers` | ColumnParallel forward | broadcast (no-op, inputs already replicated) |
| `gather_from_model_parallel_workers` | ColumnParallel (gather_output=True) | `dist.all_gather` |
| `scatter_input_to_model_parallel_workers` | RowParallel (input not yet parallel) | slice by rank |
| `reduce_inputs_from_model_parallel_workers` | RowParallel forward | `dist.all_reduce` |

**Buffer pooling**: A singleton `BufferPool` caches pre-allocated tensors keyed by `(shape, dtype, device)` to eliminate repeated allocation overhead during all-gather.

Backward passes use the inverse operations (registered as `torch.autograd.Function`), so gradients propagate correctly through TP layers without any manual handling.

---

## Data Parallelism (DP) and FSDP

### Standard DDP

When `use_fsdp=False`, standard `torch.nn.parallel.DistributedDataParallel` is used, scoped to the DP process group:

```python
model = DDP(model, process_group=get_data_parallel_group(), broadcast_buffers=False)
```

Each DP rank holds a full copy of the model. Gradients are all-reduced across the DP group after each backward pass.

### FSDP

When `use_fsdp=True`, FSDP shards parameters, gradients, and optionally optimizer states across the DP group:

```python
model = FSDP(
    model,
    process_group=get_data_parallel_group(),
    auto_wrap_policy=transformer_auto_wrap_policy({TransformerLayer}),
    sharding_strategy=ShardingStrategy[config.parallel.fsdp_sharding_strategy],
    mixed_precision=MixedPrecision(...),
    cpu_offload=CPUOffload(offload_params=config.parallel.fsdp_offload_params),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    forward_prefetch=(tp_size == 1),  # disabled when TP>1 to avoid contention
    use_orig_params=config.parallel.fsdp_use_orig_params,
)
```

**Sharding strategies:**

| Strategy | What is sharded | Notes |
|----------|----------------|-------|
| `FULL_SHARD` | Params + grads + optimizer states | Maximum memory savings |
| `SHARD_GRAD_OP` | Grads + optimizer states only | Faster, good with CPU offload |
| `HYBRID_SHARD` | Full shard within node, replicated across nodes | Multi-node large models |
| `NO_SHARD` | Nothing (equivalent to DDP) | Debugging |

**State dict types** (`fsdp_state_dict_type`): `full` (gather to rank 0), `local` (each rank saves its shard), `sharded` (distributed checkpoint format).

### Distributed Optimizer

An alternative to FSDP for optimizer state sharding: each DP rank holds `1/N` of the optimizer states while keeping full parameter copies. Enabled with `use_distributed_optimizer=True`.

```python
optimizer = DistributedOptimizer(
    optimizer,
    process_group=get_data_parallel_group(),
    bucket_cap_mb=config.parallel.dist_opt_bucket_cap_mb,
)
```

Note: incompatible with FSDP — use one or the other.

---

## Expert Parallelism (EP)

Expert Parallelism distributes MoE expert subsets across EP ranks. Tokens are dispatched to the rank holding the selected expert, processed, then gathered back.

### Token Dispatch: All-to-All

IronCore uses a single `all_to_all_single` call per dispatch/gather by packing all metadata (hidden states, expert indices, routing weights, token indices) into one buffer:

```
packed buffer layout: [hidden_states | expert_idx | routing_weight | token_idx]
                       (hidden_size cols) (1 col)    (1 col)          (1 col)
```

This reduces all_to_all calls from 4 to 1 at the cost of ~3/hidden_size bandwidth overhead (negligible for typical hidden sizes of 1024+).

### Dispatch Flow

```
1. Compute top-k routing (router scores → expert indices)
2. Sort tokens by destination EP rank
3. all_to_all_single: send tokens to their expert ranks
4. Local expert computation on received tokens
5. all_to_all_single: send results back to origin ranks
6. Weighted combine: output = sum(routing_weight_k * expert_output_k)
```

### Expert Index Mapping

Each EP rank holds a contiguous slice of routed experts:

```python
# ironcore/parallel/expert_parallel/parallel_states.py
def get_local_expert_indices(num_routed_experts, ep_size):
    ep_rank = get_expert_model_parallel_rank()
    experts_per_rank = num_routed_experts // ep_size
    start = ep_rank * experts_per_rank
    end = start + experts_per_rank
    return start, end
```

Shared experts (always-active) are replicated on all ranks and do not participate in all-to-all.

---

## Async TP Communication

`RowParallelLinear` and `ParallelMLP` support an async communication mode that overlaps the all-reduce with subsequent computation:

```python
# Standard (blocking):
output = down_proj(x)

# Async (overlapped):
partial_output, handle = down_proj(x, async_communication=True)
# ... do other work (e.g., compute next MoE expert) ...
output = mlp.finalize(partial_output, handle)  # waits for handle, adds bias
```

The `finalize()` method waits on the async handle, adds bias, and applies dropout. This is particularly useful in MoE layers where expert computation can overlap with the TP all-reduce of the previous layer.

**Note**: FSDP forward prefetch is automatically disabled when TP > 1 to prevent contention between FSDP parameter prefetching and TP async communication.

---

## Initialization Order

The correct initialization order matters. The trainer (`trainers/base_trainer.py`) follows this sequence:

```
1. initialize_process(config)
       └─ dist.init_process_group(backend="nccl")
       └─ torch.cuda.set_device(local_rank)

2. initialize_model_parallel(tensor_model_parallel_size)
       └─ Creates TP and DP process groups

3. initialize_expert_parallel(ep_size, tp_size)   [only if MoE + EP > 1]
       └─ Creates EP process groups within DP groups

4. Build model, cast to dtype

5. (Optional) Load HF checkpoint

6. Build optimizer

7. (Optional) torch.compile(model)   ← before DDP/FSDP wrapping

8. initialize_parallelism(config, model)
       └─ Wraps with DDP or FSDP using the DP process group
```

`torch.compile` must be applied **before** DDP/FSDP wrapping. FSDP auto-wrap policy targets `TransformerLayer` boundaries for optimal sharding granularity.

---

## Configuration Reference

All parallelism options live in the `parallel` config group (see `ironcore/config/config_parallel.py`) and `trainer.tensor_model_parallel_size`.

```yaml
trainer:
  tensor_model_parallel_size: 2   # TP degree (default: 1)

parallel:
  # FSDP
  use_fsdp: false
  fsdp_sharding_strategy: "full"         # full | shard_grad_op | hybrid | no_shard
  fsdp_mixed_precision: "mixed"          # mixed | fp16 | bf16 | fp32
  fsdp_state_dict_type: "full"           # full | local | sharded
  fsdp_offload_params: false             # CPU offload parameters
  fsdp_use_orig_params: false            # better compat with compile/optimizers

  # Distributed optimizer (alternative to FSDP for optimizer state sharding)
  use_distributed_optimizer: false
  dist_opt_bucket_cap_mb: 25.0

  # Process group
  dist_backend: "nccl"
  timeout_minute: 10.0
```

For MoE expert parallelism, set in the model config:

```yaml
model:
  moe:
    use_moe: true
    expert_model_parallel_size: 2        # EP degree (default: 1)
    num_routed_experts: 64
    num_shared_experts: 2
    top_k: 2
```

---

## Usage Examples

### Single GPU

```bash
ironcore train --config configs/example.yaml
```

No distributed setup needed.

### 2-GPU Tensor Parallel

```bash
torchrun --nproc_per_node 2 -m ironcore train \
  --config configs/example.yaml \
  --tensor-model-parallel-size 2
```

Model weights are split across both GPUs. Each GPU processes the full batch.

### 2-GPU Data Parallel

```bash
torchrun --nproc_per_node 2 -m ironcore train \
  --config configs/example.yaml \
  --tensor-model-parallel-size 1
```

Each GPU processes half the global batch. Gradients are all-reduced after each step.

### 4-GPU: TP=2, DP=2

```bash
torchrun --nproc_per_node 4 -m ironcore train \
  --config configs/example.yaml \
  --tensor-model-parallel-size 2
```

Process grid: 2 TP groups × 2 DP ranks. Each TP group shares a model copy, the two DP ranks hold independent batches.

### 4-GPU with FSDP

```yaml
# config.yaml
trainer:
  tensor_model_parallel_size: 1

parallel:
  use_fsdp: true
  fsdp_sharding_strategy: "full"
  fsdp_mixed_precision: "mixed"
```

```bash
torchrun --nproc_per_node 4 -m ironcore train --config config.yaml
```

### Multi-node (2 nodes × 8 GPUs)

```bash
# Node 0
torchrun --nproc_per_node 8 --nnodes 2 --node_rank 0 \
    --master_addr <MASTER_IP> --master_port 29500 \
    -m ironcore train --config configs/example.yaml \
    --tensor-model-parallel-size 2

# Node 1
torchrun --nproc_per_node 8 --nnodes 2 --node_rank 1 \
    --master_addr <MASTER_IP> --master_port 29500 \
    -m ironcore train --config configs/example.yaml \
    --tensor-model-parallel-size 2
```

---

## Known Limitations

- **No Pipeline Parallelism**: PP is not implemented. All transformer layers run on the same TP group.
- **No Context Parallelism**: Sequence length is not distributed across ranks.
- **TP divisibility constraints**: `num_attention_heads`, `num_attention_groups`, and `vocab_size` (after padding) must all be divisible by `tensor_model_parallel_size`.
- **FSDP + TP**: Combining FSDP and TP works but FSDP forward prefetch is automatically disabled when TP > 1.
- **Distributed optimizer incompatible with FSDP**: Use one or the other for optimizer state sharding.
- **EP requires EP × TP ≤ world_size / DP**: The product of all parallel degrees must equal world size.
