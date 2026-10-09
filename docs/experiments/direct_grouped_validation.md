# Direct row-budget grouped MoE planning

Validated 2026-10-09 on the same two RTX 3090 GPUs as the
[context and block-size study](layer_checkpoint_block_validation.md).

`virtual_block_size`, its benchmark CLI option and the virtual-token-tile classes
have been retired. Both ordinary Torch grouped execution and scheduled/Triton
execution call `plan_execution_groups(expert_counts, grouped_token_budget)`.
The planner fills a contiguous sorted assignment range up to the positive row
budget, splitting at expert boundaries as needed. Every group except the final
tail is full; idle experts have no segment. Metadata work scales with expert and
actual execution-group counts. No per-virtual-tile descriptors are constructed.

Remove `virtual_block_size` from older YAML/config updates and
`--virtual-block-size` from benchmark commands. No compatibility field is retained.
Expert weight names and optimizer-state layout are unaffected. The CPU config
contract explicitly rejects updates using the retired field. Shared MLP chunks
remain controlled by `trainer.mlp_chunk_size`.

## Before/after actual trainer comparison

Natural top-2 routing over four experts plus one shared expert; width 384,
eight layers, six MHA heads, FFN width 1024, vocabulary 49152, CP2/TP1/EP1,
BF16 compute, FP32 master weights, Adam, seed 42, 128K context, microbatch 1,
global batch 2, accumulation 2, shared MLP chunk 512, grouped row budget 4096,
Triton routing, whole-layer checkpointing off, recomputed linear CE chunk 128.
Weights are randomly initialized; SmolLM2 supplies only the offline tokenizer.

The before row reuses the immediately preceding 128K reference run with virtual
tile 128. The after row completes three fresh production `LanguageModelTrainer`
updates. Peak is maximum allocated CUDA memory across ranks over all updates;
time averages updates 2 and 3, choosing the slower rank for each update.

| Planner | Peak MiB/GPU | Update seconds | Sample SD seconds | Estimated model TFLOPS/s/GPU |
|---|---:|---:|---:|---:|
| Virtual tiles | 7845.1 | 13.924 | 0.007 | 3.998 |
| Direct row budget | 7851.6 | 13.668 | 0.058 | 4.073 |

Peak is effectively unchanged. Two post-warmup samples do not establish the
small observed timing difference as a general speedup. This change removes an
unhelpful configuration/control layer; expanded intermediates remain bounded
by the same execution budget. `MFUCalculator` retains its `6*N*tokens` estimate,
which excludes quadratic attention, recomputation, routing and padding and is
not measured hardware utilization.

Production `get_detailed_memory_breakdown` snapshots after rank-0 optimizer updates:

| Planner | Parameters MiB | Adam states MiB | Post-update allocated MiB |
|---|---:|---:|---:|
| Virtual tiles | 342 | 684 | 1115 |
| Direct row budget | 342 | 684 | 1113 |

Post-update storage is not a training-peak decomposition. Both runs use all four
experts in every layer, with exact selection-counter totals and finite metrics.
Maximum paired absolute loss difference is 9.5367432e-07; gradient-norm
difference is 1.0579824e-06. Both must remain below 1e-4.

## Validation

- Default CPU suite: 954 passed, 280 skipped, 5 deselected; offline CI-filtered
  CPU suite: 920 passed, 30 skipped, 289 deselected.
- CUDA expert/routing values and derivatives: 59 passed (FP32/FP16/BF16, bias,
  idle experts, strided inputs and duplicate expert slots).
- Actual CP2 GPU trainer/model comparisons: 32 passed per rank, including
  scheduled/Triton, Torch grouped, standard/reentrant checkpoints and LoRA.
- Four-process CPU/Gloo TP2+CP2: 12 trainer comparisons; CP2+DP2: eight comparisons,
  including uneven/empty objectives and scheduled/grouped recomputation.
- The row-budget unit tests pin expert-boundary splitting, complete row coverage,
  full groups except the tail, idle ownership, invalid counts/budgets and the
  retired-config rejection. Ruff check/format checks pass.

Raw records and the study HTML are ignored under `.local/`; the
[CSV](direct_grouped_validation.csv) records the source paths. Historical virtual
comparisons remain unchanged and are marked as pre-retirement measurements.
