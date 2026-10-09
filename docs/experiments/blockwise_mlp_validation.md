# Block-wise dense and MoE MLP validation

Validation date: 2026-10-08. See [execution contract](../parallelism.md#block-wise-mlp).

## Implementation stages

1. Dense MLP: synchronous token blocks, non-reentrant per-block checkpointing,
   frozen inputs/LoRA and nested layer recomputation.
2. EP1 MoE: shared/routed experts use the same block executor. Routing, top-k weights
   and full-token auxiliary loss stay outside expert blocks.
3. CP2 MoE: exclude divisibility padding from dispatch/counts; reduce small routing
   statistics over CP with local derivatives and sum replica parameter gradients.
4. Batched EP1: compact routed inputs are checkpointed; expert padding and expanded
   GEMM intermediates are rebuilt one bounded block at a time. Idle experts stay unused.
5. EP2: EP groups connect different DP batches at the same CP position, with identical
   expert ownership along CP. Native checkpoints preserve global expert identities
   and world/EP/TP/CP topology. EP2+TP and extra expert replicas remain unsupported.

Layer checkpointing exposes auxiliary loss as a tensor output so reentrant mode
retains its router gradient. Recompute suppresses duplicate selection counts and
restores the prior auxiliary-loss slot rather than retaining another graph.
MoE with ordinary DDP selects non-reentrant layer checkpointing for both configured
strategies, avoiding reentrant hooks conflicting with dynamic unused experts.
The EP2 wrapper and CP2 without DDP retain both checkpoint engines.

## Validation

- RTX 3090 x2, PyTorch 2.14.0+cu130, NCCL 2.30.7, Transformers 5.17.0.
- CPU/Gloo with PyTorch 2.13; actual four-process launches for TP2+CP2 and EP2+CP2.
- CP2+DP2 checks include unequal token counts and an entirely empty DP objective,
  with MoE block execution and both configured layer-recompute strategies.
- Unit comparisons pin dense/MoE GELU/SwiGLU outputs, input and parameter gradients,
  shared/routed/batched experts, frozen inputs, auxiliary gradients and diagnostic
  counts. Saved-tensor checks ensure expanded FFN activations are not retained.
- The default offline CPU suite passes 854 tests (221 skipped, 5 deselected).
- GPU distributed pytest: 16 passing tests per rank, including the original five CP
  tests, seven block-wise CP2 cases and four block-wise EP2 cases. Both whole-layer
  recompute strategies, Dense LoRA, idle destinations and native checkpoints are covered.
- Independent full decoders and the production `LanguageModelTrainer.train_step`
  compare three accumulated AdamW updates. CUDA parity uses FP16, epsilon 1e-3;
  EP tests use learning rate 1e-3. CPU comparisons remain FP32 with tight tolerances.
- A production CPU CP2+EP2 CLI run saved step 3 and resumed to step 6, restoring the
  optimizer/trainer topology and skipping six consumed microbatches per data worker.
- Production BF16 CP2 GPU runs complete three finite updates for loop/batched MoE
  with and without blocks. These random-token runs establish systems behavior,
  not pretrained MoE quality or long-training convergence.

The two available GPUs validate CP2+EP1 and CP1+EP2 separately. CP2+EP2 and TP2+CP2
are CPU evidence; four-GPU NCCL composition and multi-node performance are unverified.

## Memory and time

All rows use CP2, sequence length 8192, global batch 2, microbatch 1, accumulation 2,
FP32 stored weights, BF16 compute, full-parameter Adam and recomputed linear CE.
Peak allocation is the maximum across ranks; time is the mean of updates 2 and 3,
taking the slower CUDA-synchronized rank. The block size is 512 local tokens.

| Model / backend | Whole-layer recompute | MLP block | Peak per GPU (MiB) | Update (s) | Estimated model TFLOPS/s/GPU |
|---|---|---|---|---|---|
| SmolLM2-135M | On | Off | 2876.9 | 0.857 | 7.71 |
| SmolLM2-135M | On | 512 | 2891.0 | 0.942 | 7.02 |
| SmolLM2-135M | Off | Off | 5782.6 | 0.699 | 9.46 |
| SmolLM2-135M | Off | 512 | 4314.8 | 0.781 | 8.47 |
| Small MoE / loop | Off | Off | 3142.6 | 0.328 | 10.61 |
| Small MoE / loop | Off | 512 | 2262.3 | 0.519 | 6.70 |
| Small MoE / batched | Off | Off | 3248.7 | 0.321 | 10.83 |
| Small MoE / batched | Off | 512 | 2422.5 | 0.507 | 6.87 |

The small MoE has width 384, eight layers, six MHA heads, four routed experts,
one shared expert, top-2 routing and SwiGLU expert width 1024. It uses the cached
SmolLM2 tokenizer (49152 tokens), untied embeddings and random initialization.
It differs from the shipped 55M preset, whose expert width is 256 and context is 1K.

Without whole-layer recompute, the measured peak reduction is 25.4% for Dense,
28.0% for loop MoE and 25.4% for batched MoE. Update time increases 11.7%, 58.3%
and 57.6% respectively. With whole-layer recompute already on, Dense peak allocation
increases 0.5% and time increases 9.8%; this combination is not a memory win here.

TFLOPS estimates use `MFUCalculator.from_config` and its approximate `6*N*tokens`
model FLOPs divided by measured time and two GPUs. MoE uses active FFN width
`(shared + top_k) * expert_width` as a dense-equivalent estimate. The estimates
exclude extra checkpoint FLOPs, routing, expert padding and quadratic attention;
they are model-throughput estimates, not measured hardware utilization. More
recomputation and smaller GEMMs reduce useful model throughput in these runs.

The full hidden-size tensors, routing/combination buffers, stacked batched weights
and replicated parameter/gradient/Adam state remain resident. Block-wise MLP is
opt-in and does not imply full block-wise transformer scheduling or communication
overlap. Gemma 4, MLP/LoRA dropout, FSDP/offload and async MLP calls are rejected.

### Follow-up: block granularity

The same 8K CP2 small MoE batched configuration was measured with additional
block sizes. The unchunked and 512-token rows reuse the earlier measurements;
the 128/1024/4096 rows are a subsequent short sweep on the same GPUs.

| MLP block tokens | Peak per GPU (MiB) | Update time (s) |
|---|---|---|
| Disabled | 3248.7 | 0.321 |
| 128 | 2399.9 | 0.900 |
| 512 | 2422.5 | 0.507 |
| 1024 | 2431.5 | 0.380 |
| 4096 | 2408.4 | 0.343 |

4096 tokens already cover the entire local sequence in this microbatch. Each
expert receives at most that many tokens, so this is effectively selective
expert/shared-MLP checkpointing with a single block per expert. The principal
memory benefit here is avoiding retained FFN intermediates; progressively smaller
blocks add little peak reduction and more scheduling/GEMM/checkpoint overhead.
These short runs do not establish the optimal setting for larger experts or contexts.

The follow-up [context doubling experiment](context_block_scaling_validation.md)
compares absent, full-size and 512-token MLP checkpoints independently until
each mode reaches a measured CUDA OOM ceiling.

A possible follow-up is to decouple logical routing tiles from the execution
token budget, grouping ready expert tiles for larger, bounded GEMMs. This needs
an execution backend and measurements, rather than only smaller Python-loop blocks.
Grouped or block-sparse execution could also avoid padding all experts to a common
capacity, as in [MegaBlocks](https://github.com/databricks/megablocks). Integrating
attention-output tiles with immediate FFN/MoE execution is a larger scheduling step,
following the attention/feed-forward fusion studied in
[Blockwise Parallel Transformers](https://arxiv.org/abs/2305.19370).

## Reproduce

```bash
torchrun --standalone --nproc_per_node=2 -m ironcore train \
  --config configs/experiments/smollm2_135m_cp2_blockwise.yaml

torchrun --standalone --nproc_per_node=2 -m ironcore train \
  --config configs/experiments/cs336_55m_moe_cp2_blockwise.yaml

torchrun --standalone --nproc_per_node=2 -m pytest \
  tests/multi_gpu/test_blockwise_context.py \
  tests/multi_gpu/test_blockwise_expert_context.py

torchrun --standalone --nproc_per_node=4 \
  -m tests.multi_gpu.test_blockwise_context --tp 2

torchrun --standalone --nproc_per_node=4 \
  -m tests.multi_gpu.test_blockwise_expert_context
```

The four-rank shipped CP2+EP2 experiment uses CUDA ring attention; the CPU comparison
driver uses the differentiable SDPA reference. Local JSON evidence and the Korean
standalone study HTML under `.local/` are excluded from Git.
