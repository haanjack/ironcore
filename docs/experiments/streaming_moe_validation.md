# Streaming batched MoE checkpoint execution

Validation date: 2026-10-09. Before baseline: `5e5f313`.
This change removes global routing/output storage from the bounded batched-expert
checkpoint path and reduces temporary FP32 CP attention merge allocations.
At 128K context, naturally routed MoE peak memory falls by 40.7%, and 256K now
completes three trainer updates where the baseline failed. Dense's measured
context ceiling remains unchanged. MoE 512K has not been tested in this stage;
256K is a passing measurement, not a measured maximum.

## Problem and behavior

Previously, every expert block produced unweighted outputs which were written
with an out-of-place `index_copy` into a full `[local_tokens * top_k, hidden]`
buffer. Router gradients retained the entire expert-output tensor. Compact
routed-input checkpoint copies also survived backward. Smaller FFN blocks did
not remove these global buffers, and each output insertion copied the whole buffer.

The batched checkpoint now covers input gather, padded expert projections and
mixture weighting. Each checkpoint references the shared original hidden and
mixture-weight tensors, rather than saving expanded routed inputs or unweighted
expert outputs. A private additive autograd Function accumulates each weighted
block into one `[local_tokens, hidden]` buffer. Its VJP returns the destination
gradient unchanged and gathers the source gradient by token ids. It saves only
indices and dtype metadata, and declares the mutation with `mark_dirty`.
Native `index_add` autograd retains a source Tensor; this explicit VJP avoids that
retention. It does not modify expert Parameters or checkpoint keys.

FP16/BF16 weighted values accumulate in FP32 and cast to the original mixture
output dtype at the end. CUDA autocast promotes the original mixture `sum` to
FP32 even when both operands are low precision; the new path preserves this
[documented autocast policy](https://docs.pytorch.org/docs/2.14/amp.html).
The original projection-output cast and master-bias
semantics remain intact. TP copies of hidden/mixture weights precede block
execution and the output TP reduction still occurs once afterward. Globally
idle experts remain detached and keep `grad=None`. All-valid CP masks bypass
unnecessary compaction/scatter copies; partial padding keeps the prior path.

CP ring attention uses in-place FP32 multiply/addcmul instead of materializing
several full-query FP32 merge temporaries. The custom attention Function's forward
runs without autograd recording; saved Q/K/V/O/LSE and backward communication
semantics remain unchanged. Compute remains native PyTorch CUDA GEMM/FlashAttention
and indexed operations. No Triton/CUDA extension has been added in this stage.

The mutation contract follows the official
[PyTorch autograd documentation](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.function.FunctionCtx.mark_dirty.html).

## Correctness

- Default offline CPU suite: 909 passed; unit checks include duplicate token ids,
  chained in-place accumulation, noncontiguous gradients, gradcheck/gradgradcheck,
  saved-storage assertions and independent online-softmax merge equations.
- Single-GPU FP32/FP16/BF16 values and all input/router/expert derivatives: 16 cases,
  including nonzero biases, idle experts, low-precision inputs and top-3.
- Distributed GPU suite: 24 passing tests per rank, including CP2, batched standard
  and optimized outer checkpoints, frozen-base MoE LoRA, and EP2's existing loop.
- Actual four-process CPU TP2+CP2 and CP2+DP2 executions check parallel composition;
  they do not establish four-GPU NCCL performance.
- The optimized batched path continues to require EP1, synchronous execution and
  zero MLP dropout under the existing blockwise contract.

## GPU measurements

RTX 3090 x2, PyTorch 2.14.0+cu130, native ring CP2/TP1/EP1. Same width 384,
eight layers, six MHA heads, SwiGLU expert width 1024, four natural routed experts,
top-2, one shared expert, 49152 vocabulary, BF16 compute, FP32 master weights,
Adam, global batch 2, microbatch 1, accumulation 2, seed 42. Whole-layer checkpoint
is off. Linear CE recomputation uses chunk 128. `full` uses the complete local
sequence as its MLP budget; `blocked` uses 512 tokens. These are random-initialized
systems checks, not long-context language-quality or training-convergence claims.

| Model | Context | MLP mode | Before peak MiB / update s | After peak MiB / update s |
|---|---:|---|---:|---:|
| moe | 65536 | full | 7586 / 4.139 | 5183 / 4.039 |
| moe | 65536 | blocked | 7284 / 5.351 | 4596 / 4.783 |
| moe | 131072 | full | 13785 / 13.004 | 8985 / 12.673 |
| moe | 131072 | blocked | 13100 / 16.473 | 7764 / 14.615 |
| moe | 262144 | blocked | OOM | 14121 / 49.931 |
| dense | 262144 | blocked | 13380 / 44.289 | 13382 / 44.188 |
| dense | 524288 | blocked | OOM | OOM |

Peak is maximum allocated CUDA memory across ranks over three production trainer
updates. Update time averages steps 2 and 3, taking the slower synchronized rank.
Short timing samples do not establish small throughput differences. Combined
changes are measured together; this does not isolate the attention merge's speedup.
An OOM means an actual allocation exception, not a measured successful peak.

[CSV results](streaming_moe_validation.csv) retain before/after model TFLOPS/s/GPU
estimated with `MFUCalculator` and active FFN width. The `6*N*tokens` approximation
excludes quadratic attention, recomputation and routing, and is not hardware utilization.
Raw rank records, config snapshots and OOM diagnostics are ignored in
`.local/streaming-context/`; saved before records remain in `.local/context-block-scaling/`.
Maximum paired loss difference for settings that completed before and after is
2.38e-06.

## Saved-storage diagnosis

At MoE 64K with 512-token blocks, the first microbatch's unique non-master CUDA
storage retained by autograd in the routing/dispatch/combine category falls from
3093.000 to 393.000 MiB (2700 MiB less). Attention/projections (1164.75 MiB),
normalization (818.125 MiB), embedding/head/other (307.375 MiB) and router
(6 MiB) are unchanged. This independently supports the save/recompute-boundary
explanation rather than attributing the gain to faster GEMMs.

The diagnostic deduplicates storage aliases, excludes original master Parameters,
includes cast-weight copies and assigns storage to its first save category.
It describes saved forward storage, not a full training-peak decomposition.
Raw hook records are ignored under `.local/context-memory-diagnosis/streaming/`;
the prior records remain in that directory's original runs.

## Remaining work

The new path still has CPU expert-count metadata planning, Python/checkpoint/kernel
launches and per-block full-hidden input-gradient materialization through gather
backward. Full-length attention/normalization state and parameter/Adam storage
remain resident. Fused routing/scatter/gather kernels or a whole-expert custom
backward scheduler require separate profiling and correctness validation. This
stage moves Dense 512K's failure from an attention merge allocation to native
RMSNorm, without making that context fit. Normalization/attention saved-state
lifetimes and recomputation need examination; a fused normalization kernel alone
has not been shown to resolve the remaining capacity limit. This
stage demonstrates that controlling the autograd save/recompute boundary yields
substantial memory gains before introducing a new GPU kernel dependency.
