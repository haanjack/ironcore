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
| Context Parallel (CP) | Sequence tokens | KV ring; sum replica gradients | Dense/MoE causal long-context training |
| Expert Parallel (EP) | MoE expert subsets | All-to-all token dispatch | Mixture-of-Experts models |
| Distributed Optimizer | Optimizer states only | Broadcast updated params | ZeRO-1 without full FSDP overhead |

Dense CP is orthogonal to TP and DDP. CP currently rejects FSDP, distributed optimizers,
offload. MoE EP1 and bounded EP2 composition are supported; see the contract below.

---

## Process group layout

Ranks are arranged in a `[DP][CP][TP]` mesh. The global rank is
`(dp_rank * CP_size + cp_rank) * TP_size + tp_rank`. CP defaults to one,
which preserves the previous TP/DP group layout:

```
World size = TP_size × CP_size × DP_size

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

## Context Parallelism (CP)

CP partitions the sequence across ranks while keeping each TP weight shard replicated
across CP. Attention rotates K/V blocks between CP peers; embeddings, projections,
norms and dense MLPs operate on local tokens. CP does not partition model weights or
optimizer states, so the model must still fit on each CP rank.

```yaml
trainer:
  tensor_model_parallel_size: 1
  context_parallel_size: 2
  context_parallel_backend: ring
model:
  precision: bfloat16
  dropout_attn: 0.0
  dropout_mlp: 0.0
  dropout_embd: 0.0
  reset_attention_mask: false
  reset_position_ids: false
```

`ring` wraps PyTorch's native FlashAttention forward/LSE/backward operators. It requires
CUDA FP16 or BF16 compute, including BF16 autocast with FP32 stored weights. It does not
require Transformer Engine or the separate `flash-attn` package. Operator details are
isolated in `FlashAttentionKernel`; rerun the distributed regressions when upgrading
PyTorch because these are internal ATen operators. CP initialization disables reduced-precision
GEMM reductions so FP16/BF16 matrix products keep FP32 accumulation; the original
reduced-precision flags are restored when CP groups are reset. BF16 outputs still have normal
rounding differences across sequence partitions. The GPU implementation was exercised
on PyTorch 2.14/CUDA 13.0 with two RTX 3090s.

`sdpa` is a differentiable full-KV all-gather reference for FP32/CPU debugging. It saves
the full sequence's K/V per rank and is not the memory-saving training backend.

### Data, attention and loss semantics

Each CP peer receives the **same full batch** from the DP sampler. Pass already-shifted
labels into `LanguageModel.forward`; the model partitions inputs, labels and global
position IDs together. Non-divisible sequences are right-padded, and padding labels are
ignored. A shard with no valid labels still computes attention and supplies remote K/V
gradients. An entirely masked DP batch returns zero loss/gradients and still participates
in DDP; the trainer rejects an update only when all DP batches contain no valid tokens.

For contiguous causal shards, attention computes its own block with a causal mask,
all earlier blocks without a mask, and skips later blocks. Partial outputs are merged
with FP32 log-sum-exp normalization. Backward uses the output/LSE normalized over global K/V for local queries and
returns accumulated K/V gradients to their original owners. Communication buffers remain
bounded by the local sequence length; full K/V is not retained by `ring`.

Loss is the global valid-token mean over CP. Its backward contributes the local numerator
only. The trainer sums parameter gradients over CP after DP synchronization and AMP
unscale, before finite checks, clipping and optimizer updates. DP still averages distinct
batches. Gradient norms count a CP replica once. LoRA follows the same CP synchronization.

When labels are omitted, scoring reconstructs full sequence logits to preserve the public
forward shape. This reconstruction allocates full logits and should be used for debugging
or evaluation, rather than memory-sensitive training. Autoregressive generation and KV
caches are not supported with CP.

### Supported combinations and checkpoints

The contract supports generic dense/MoE causal `pretrain`, including SmolLM2, with
zero attention/MLP/embedding/LoRA dropout. SFT, document packing/reset masks, arbitrary
attention masks, alignment, Gemma 4, FSDP, distributed optimizers and offload are
rejected. `torch.compile` is skipped for CP communication. Dense activation recomputation,
chunked vocabulary loss and recomputed linear cross-entropy can be used with CP.

MoE routes local real tokens once per layer. CP divisibility padding is excluded from
expert dispatch and load-balancing statistics. Small per-expert counts/probability sums
are reduced over CP to compute the original full-sequence auxiliary loss; backward
keeps local router derivatives, followed by the normal CP parameter-gradient sum.
Router jitter must be zero. Both loop and batched expert backends support EP1.
EP2 supports TP1 with `world_size = 2 * CP`; additional expert replicas and EP+TP
training remain rejected. EP exchanges connect distinct DP batches at the same CP
position, while CP peers own identical expert subsets. EP ranks therefore remain part
of the trainer's DP batch-count axis.

`world_size` must be divisible by `TP * CP`, and `DP = world_size / (TP * CP)`.
Global batch size is `micro_batch_size * accumulation * DP`; CP does not multiply it.
For example, TP2+CP2 needs four ranks. SmolLM2-135M has 9 query heads and 3 KV heads, so
it supports CP2 but cannot use TP2; SmolLM2-1.7B has 32 heads and can use TP2.

Only CP rank zero in each weight-shard group writes model/optimizer files, preventing
replica write races. Native weights can be loaded without CP. Exact trainer resume
requires the same DP/TP/CP topology, parameter precision and data-loader configuration;
old trainer checkpoints without `cp_size` are interpreted as CP1. EP checkpoints use
separate rank files with global expert identities and require the same world/EP/TP/CP
topology; they do not use the dense CP writer gate.

### Run and verify

The example imports SmolLM2-135M and uses random tokens to test system behavior:

```bash
torchrun --standalone --nproc_per_node=2 -m ironcore train \
  --config configs/experiments/smollm2_135m_cp2.yaml

# Native GPU ring: output/Q/K/V gradients, causality, full/LoRA training,
# activation/loss recomputation and checkpoint roundtrips.
torchrun --standalone --nproc_per_node=2 \
  -m tests.multi_gpu.test_context_parallel --device cuda --backend ring

# Four CPU ranks: TP2+CP2 correctness against an independent full decoder.
torchrun --standalone --nproc_per_node=4 \
  -m tests.multi_gpu.test_context_parallel --tp 2 --cp 2

# Four CPU ranks: CP2+DP2 with unequal valid-token counts.
torchrun --standalone --nproc_per_node=4 -m tests.multi_gpu.test_context_parallel_dp
```

GPU coverage is CP2 on one node. TP2+CP2 and CP2+DP2 composition is additionally checked
on CPU/Gloo; four-GPU NCCL and multi-node performance are not yet established. Contiguous
causal shards have unequal attention work, so CP2 does not imply a 2x speedup. Zigzag
load balancing remains follow-up work. Block-wise MLP is described below.

## Block-wise MLP

```yaml
trainer:
  mlp_chunk_size: 512
operation:
  activation_recompute: false
```

`mlp_chunk_size` counts flattened local tokens for dense/shared experts and routed
tokens within each loop expert. Training checkpoints every block independently and
recomputes expanded FFN activations in backward, including when inputs are frozen.
Evaluation also computes in blocks without checkpointing. Full hidden-size inputs,
outputs, weights, gradients and optimizer state remain resident. The default `null`
preserves unchunked execution. This setting is separate from the unimplemented
`sequence_chunk_size` async TP scheduler; block-wise MLP uses synchronous TP.

MoE routing and auxiliary loss are computed once over the original token set, not
once per block. The batched backend builds bounded `[experts, block_tokens, hidden]`
padding inside checkpoints and recomputes input gather and mixture weighting.
Checkpoints share the original hidden and router-weight storage. Weighted block
outputs accumulate into one token-output buffer through a custom autograd operation
that saves token indices, avoiding retained routed-input and unweighted-output
copies. Expert weight stacks and routing metadata remain; smaller blocks also
increase gather-backward and checkpoint overhead.
Globally idle experts keep `grad=None`. Layer checkpointing returns auxiliary loss as
a differentiable output, including reentrant mode; recomputation does not recount
router diagnostics or retain another auxiliary-loss graph.
MoE with ordinary DDP uses non-reentrant layer checkpointing even when
`recompute_strategy: optimized` is selected, because dynamic idle experts require
DDP unused-parameter traversal. EP2's explicit gradient wrapper supports both engines.

Zero MLP/LoRA dropout is required. Gemma 4, FSDP, offload and async MLP calls are
rejected for this initial block-wise contract. Smaller GEMMs, repeated TP collectives,
padding and checkpoint overhead can reduce throughput. Whole-layer checkpointing
already removes most retained FFN activations, so adding block-wise MLP can increase
time and even peak allocation. Benchmark both settings for the intended model.

Examples: `configs/experiments/smollm2_135m_cp2_blockwise.yaml` and
`configs/experiments/cs336_55m_moe_cp2_blockwise.yaml`.
Measured memory and validation scope: [block-wise validation](experiments/blockwise_mlp_validation.md).
The batched checkpoint storage improvement and long-context measurements are in
[streaming MoE validation](experiments/streaming_moe_validation.md).

### Budgeted grouped GEMM

```yaml
model:
  moe:
    expert_backend: grouped
    grouped_token_budget: 4096
    blockwise_backend: torch
trainer:
  mlp_chunk_size: 4096  # Shared experts; grouped routed execution has its own budget.
```

The grouped backend sorts routed assignments once and fills execution groups
directly from expert counts up to the configured row budget. A group can contain
several experts with different token counts; it calls CUDA
`torch.nn.functional.grouped_mm` for up/gate and down projections without padding
all experts to a common capacity. Empty experts receive no GEMM or parameter gradient.
An expert can span multiple groups, and a group can span multiple experts.
Every group except the final tail fills its row budget. No logical token-tile
list or virtual-block setting is maintained.

`grouped_token_budget` caps total valid routed rows in one execution group, rather
than rows per expert. It must be a positive integer. The default `torch`
path checkpoints groups independently even when `mlp_chunk_size` is null. Shared experts still use the ordinary
MLP setting. A token budget bounds expanded FFN rows; it is not a complete VRAM cap:
weight packing, GEMM workspace, full hidden tensors and optimizer state remain.
Only active weights for the current group are packed inside its checkpoint, avoiding
retained per-layer copies. Routing weights, auxiliary objectives, TP input gradients
and the single final TP output reduction preserve the existing mixture semantics.

This backend currently requires EP1 and zero MLP/LoRA dropout and rejects FSDP/offload.
CUDA requires the public grouped-mm API; missing support raises an error rather than
silently running a Python expert loop. Native CUDA values/gradients were tested in
FP32/FP16/BF16 on PyTorch 2.14 and RTX 3090. CPU runs use an independent dense-GEMM
reference for correctness and do not establish CPU grouped acceleration.
The planner currently copies expert counts to CPU once per routing call; metadata
planning and weight packing can dominate small or balanced workloads. GPU-only
planning, CUDA graph capture, attention/FFN fusion and grouped EP dispatch are future work.

Example: `configs/experiments/cs336_55m_moe_cp2_grouped.yaml`.
The retired `virtual_block_size` YAML field and `--virtual-block-size` benchmark
option must be removed from older configurations and commands. Historical
virtual-tile comparisons remain in the experiment reports.
See [direct row-budget validation](experiments/direct_grouped_validation.md) for
the retirement checks and before/after trainer measurements.
See [grouped validation and measurements](experiments/grouped_moe_validation.md).

`blockwise_backend` controls recomputation and routing for batched/grouped experts:

- `torch` preserves the ordinary autograd/checkpoint path and remains the default.
- `scheduled` wraps all execution blocks in one first-order autograd Function.
  Its backward recomputes a bounded block with detached Parameter views, then
  accumulates into one full input-gradient buffer and returns each Parameter's
  gradient once. Original Parameter hooks are kept outside block recomputation.
- `triton` uses the same scheduler and adds CUDA kernels for input/weight gather,
  mixture-weighted output scatter, upstream-gradient gather and gradient scatter.
  Verified single-expert groups use ordinary scatter stores; mixed-expert groups
  and duplicate expert slots retain atomic addition. Uniqueness is checked with
  the existing expert-count transfer rather than assumed for caller-provided ids.
  Expert GEMMs still use native PyTorch batched or grouped GEMM. Triton is required
  only when explicitly selecting this backend on CUDA; CPU uses the scheduler's
  reference operations and does not establish Triton acceleration.

Scheduled batched experts require a positive `trainer.mlp_chunk_size`; grouped
experts use their grouped row budget. These paths require parameter-free
activations and support first-order training gradients. Use `torch` for higher-order
derivatives. Triton scatter uses FP32 accumulation with an atomic fallback and rejects deterministic
algorithm mode; use `scheduled` for native deterministic operation support.
The existing EP1, synchronous TP, zero-dropout and FSDP/offload constraints apply;
scheduled backends require the default expert dispatch mode.
Routing still sorts on the GPU and copies counts to CPU once per layer; this does
not implement GPU-only planning, expert parallel dispatch or CUDA graph capture.

An opt-in configuration is `configs/experiments/cs336_55m_moe_cp2_triton.yaml`.
See [scheduled MoE validation and temporal activity trace](experiments/scheduled_moe_validation.md)
for before/after memory, throughput and correctness scope.
The [whole-layer checkpoint and block-size sweep](experiments/layer_checkpoint_block_validation.md)
validates 512K training and distinguishes MLP transient budgets from retained
attention/normalization state. The benchmark's `--layer-checkpoint` option controls
whole-layer recomputation independently of MLP token blocks.

The wrapper approach follows the same kernel boundary used by
[PyTorch's context parallel implementation](https://docs.pytorch.org/tutorials/unstable/context_parallel.html).

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
    num_experts_per_token: 2
```

The native trainer supports EP2, TP1 and `world_size = 2 * CP`. EP overlays the DP
batch axis; EP+TP and additional DP replicas of owned experts are not yet supported.
The process-group primitives alone do not establish a supported training combination.

---

## Initialization order

The trainer enforces this fixed initialization order — do not rearrange:

```
1. initialize_process()             # dist.init_process_group + cuda.set_device
2. initialize_model_parallel(tp, context_parallel_size=cp)  # TP/CP/DP groups
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
- **Context parallel scope:** generic dense/MoE causal pretraining with zero dropout; see the support contract above.
- **TP divisibility:** `num_attention_heads`, `num_attention_groups`, and `vocab_size` must all be divisible by `tensor_model_parallel_size`.
- **Distributed optimizer is incompatible with FSDP.** Use one or the other.
- **Native EP training:** EP2, TP1, `world_size = 2 * CP`; EP exchanges overlay the DP batch axis.
