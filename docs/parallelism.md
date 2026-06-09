# Parallelism

> This guide covers how to configure and combine parallelism strategies. For process group
> layout, TP communication primitives, and EP dispatch internals, see the
> [Parallelism system design](design/parallelism.md).

## Strategies at a glance

| Strategy | Splits | Communication | Use case |
|---|---|---|---|
| Data Parallel (DDP) | Batch | All-reduce gradients | Multi-GPU, model fits on one GPU |
| FSDP | Batch + params | All-gather params, reduce-scatter grads | Large models, full state sharding |
| Tensor Parallel (TP) | Model weights (per layer) | All-gather / all-reduce per layer | Layers too large for one GPU |
| Expert Parallel (EP) | MoE expert subsets | All-to-all token dispatch | Mixture-of-Experts models |
| Distributed Optimizer | Optimizer states only | Broadcast updated params | ZeRO-1 without full FSDP overhead |

TP and DP/FSDP are orthogonal and freely combinable. EP adds a third axis on top of TP for MoE models.

---

## Process group layout

Ranks are arranged in a 2D grid `[DP × TP]`:

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

- **TP group**: ranks sharing the same data shard; hold different weight shards.
- **DP group**: ranks holding independent data shards; hold identical weight shards (same TP position).

---

## Tensor Parallelism (TP)

TP splits layer weights across GPUs using Megatron-style column/row parallel linear layers.
Each layer requires exactly one collective operation (all-gather or all-reduce).

**Constraints:** `num_attention_heads`, `num_attention_groups`, and `vocab_size` must all be
divisible by `tensor_model_parallel_size`.

Enable with:

```yaml
trainer:
  tensor_model_parallel_size: 2   # number of TP ranks
```

---

## Data Parallelism (DP) and FSDP

### Standard DDP

Default when `use_fsdp: false`. Each DP rank holds a full model copy; gradients are all-reduced
after each backward pass. Use when the model fits in one GPU's VRAM.

### FSDP

Shards parameters, gradients, and optionally optimizer states across the DP group.

```yaml
parallel:
  use_fsdp: true
  fsdp_sharding_strategy: full     # full | shard_grad_op | hybrid | no_shard
  fsdp_mixed_precision: mixed      # mixed | fp16 | bf16 | fp32
```

**Sharding strategies:**

| Strategy | What is sharded | Notes |
|---|---|---|
| `full` | Params + grads + optimizer states | Maximum memory savings |
| `shard_grad_op` | Grads + optimizer states only | Faster; good with CPU offload |
| `hybrid` | Full shard within node, replicated across nodes | Multi-node large models |
| `no_shard` | Nothing (equivalent to DDP) | Debugging |

**Checkpoint format** (`fsdp_state_dict_type`): `full` gathers to rank 0; `local` saves each
rank's shard; `sharded` produces a distributed checkpoint.

**Note:** FSDP forward prefetch is automatically disabled when TP > 1 to avoid contention with
TP async communication.

### Distributed Optimizer (ZeRO-1)

An alternative to FSDP for optimizer state sharding. Parameters and gradients remain fully
replicated; only optimizer states (moments) are partitioned across DP ranks, saving `(N-1)/N`
of optimizer state memory at DP size N.

```yaml
parallel:
  use_distributed_optimizer: true
  dist_opt_bucket_cap_mb: 25.0    # broadcast bucket size
```

**Incompatible with FSDP** — use one or the other.

---

## Expert Parallelism (EP)

For MoE models, EP distributes expert subsets across EP ranks. Tokens are dispatched to the
rank holding their selected expert via all-to-all, computed locally, then gathered back.

Shared experts (always-active) are replicated on all ranks and do not participate in dispatch.

Configure in the model config:

```yaml
model:
  moe:
    use_moe: true
    expert_model_parallel_size: 2
    num_routed_experts: 64
    num_shared_experts: 2
    top_k: 2
```

EP can be combined with TP. World size must equal `DP × EP × TP`.

---

## Initialization order

The trainer enforces this fixed initialization order — do not rearrange:

```
1. initialize_process()             # dist.init_process_group + cuda.set_device
2. initialize_model_parallel(tp)    # create TP/DP process groups
3. initialize_expert_parallel(ep)   # only when MoE + EP > 1
4. Build model and cast to dtype
5. (Optional) Load HF checkpoint
6. Build optimizer
7. torch.compile(model)             # must be BEFORE parallelism wrapping
8. initialize_parallelism()         # wrap with DDP or FSDP
```

`torch.compile` must precede DDP/FSDP wrapping — compiling after wrapping produces incorrect results.

---

## Configuration reference

```yaml
trainer:
  tensor_model_parallel_size: 2

parallel:
  # FSDP
  use_fsdp: false
  fsdp_sharding_strategy: full        # full | shard_grad_op | hybrid | no_shard
  fsdp_mixed_precision: mixed         # mixed | fp16 | bf16 | fp32
  fsdp_state_dict_type: full          # full | local | sharded
  fsdp_offload_params: false
  fsdp_use_orig_params: false

  # Distributed optimizer
  use_distributed_optimizer: false
  dist_opt_bucket_cap_mb: 25.0

  # Process group
  dist_backend: nccl
  timeout_minute: 10.0
```

---

## Usage examples

### Single GPU

```bash
ironcore train --config configs/example.yaml
```

### 2-GPU Tensor Parallel

```bash
torchrun --nproc_per_node 2 -m ironcore train --config configs/example.yaml \
  --tensor-model-parallel-size 2
```

### 4-GPU: TP=2, DP=2

```bash
torchrun --nproc_per_node 4 -m ironcore train --config configs/example.yaml \
  --tensor-model-parallel-size 2
```

### 4-GPU with FSDP

```yaml
trainer:
  tensor_model_parallel_size: 1
parallel:
  use_fsdp: true
  fsdp_sharding_strategy: full
```

```bash
torchrun --nproc_per_node 4 -m ironcore train --config config.yaml
```

### Multi-node (2 nodes × 8 GPUs)

```bash
# Node 0
torchrun --nproc_per_node 8 --nnodes 2 --node_rank 0 \
  --master_addr <MASTER_IP> --master_port 29500 \
  -m ironcore train --config configs/example.yaml --tensor-model-parallel-size 2

# Node 1
torchrun --nproc_per_node 8 --nnodes 2 --node_rank 1 \
  --master_addr <MASTER_IP> --master_port 29500 \
  -m ironcore train --config configs/example.yaml --tensor-model-parallel-size 2
```

---

## Known limitations

- **No pipeline parallelism.** All transformer layers run on the same TP group.
- **No context parallelism.** Sequence length is not distributed across ranks.
- **TP divisibility:** `num_attention_heads`, `num_attention_groups`, and `vocab_size` must all be divisible by `tensor_model_parallel_size`.
- **Distributed optimizer is incompatible with FSDP.** Use one or the other.
- **EP requires `EP × TP ≤ world_size / DP`.**
