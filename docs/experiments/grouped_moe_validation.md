# Virtual-block grouped MoE validation

These measurements predate retirement of `virtual_block_size`. Current grouped
execution plans groups directly from `grouped_token_budget`; historical tile
values below describe the measured implementation and are not current options.

Validation date: 2026-10-08. See [configuration](../parallelism.md#budgeted-grouped-gemm).

## Implementation

`model.moe.expert_backend: grouped` decouples logical expert token tiles from the
total valid-token budget of a grouped execution. The planner covers each routed
assignment exactly once, skips empty experts, coalesces adjacent same-expert tiles,
and can include several jagged expert segments in one execution group. It does
not pad every expert to the largest token count or launch a kernel per virtual tile.

The CUDA backend uses public `torch.nn.functional.grouped_mm`; it selects compute
dtype explicitly because the tested operator did not automatically follow CUDA
autocast. The CPU implementation is an independent dense-GEMM reference. Active
weights are packed inside each group checkpoint and are not retained as full-layer
copies. Original expert Parameters and checkpoint names remain unchanged; unused
experts keep `grad=None`. Bias semantics and TP gradient/reduction boundaries are
preserved. Routing and auxiliary loss remain outside the grouped executor.

The token budget bounds expanded FFN rows rather than all CUDA memory. Planner
metadata currently uses one GPU-to-CPU expert-count copy per routing invocation.
Weight packing, backend workspace, full hidden tensors and optimizer states still
consume memory. This is EP1 support; grouped EP dispatch, GPU-only planning, CUDA
graphs and attention/FFN fusion are separate work.

## Correctness and execution

- RTX 3090 x2, PyTorch 2.14.0+cu130, NCCL 2.30.7; CPU reference on PyTorch 2.13/Gloo.
- Native CUDA FP32/FP16/BF16 output, input-gradient and expert-gradient comparisons
  pass six single-GPU cases, including idle experts and nonzero projection biases.
- Distributed GPU pytest passes 21 tests per rank: the original CP and block-wise
  coverage plus grouped CP2, standard/reentrant layer recomputation and Dense/MoE LoRA.
- Independent full decoders and production `LanguageModelTrainer.train_step` compare
  three accumulated AdamW updates, evaluation and native checkpoint roundtrips.
- Actual four-process CPU launches cover TP2+CP2 and CP2+DP2, including an entirely
  empty DP objective and both configured recompute strategies. Four-GPU NCCL is
  not established by these CPU results.
- The default offline CPU suite passes 896 tests (227 skipped, 5 deselected).
- Production BF16 CP2 GPU benchmarks complete three finite updates per setting.
  Random-token and forced-routing runs establish systems behavior, not MoE quality
  or long-training convergence.

## Memory and throughput

Width 384, eight layers, six MHA heads, one shared expert, top-2 routing, SwiGLU
expert width 1024, untied embeddings and cached SmolLM2 vocabulary (49152 tokens).
CP2, 8192 tokens, global batch 2, microbatch 1, accumulation 2, FP32 master weights,
BF16 compute, Adam, linear-CE recomputation, no whole-layer checkpointing.
Shared MLP block size is 4096. Batched uses a per-expert block size of 4096;
grouped uses virtual tiles of 128 and a total execution budget of 4096.

| Routed experts / routing | Backend | Peak per GPU (MiB) | Update time (s) | Estimated model TFLOPS/s/GPU |
|---|---|---|---|---|
| 4 / natural random-model routing | Batched | 2408.4 | 0.343 | 10.15 |
| 4 / natural random-model routing | Grouped | 2354.6 | 0.366 | 9.51 |
| 16 / natural random-model routing | Batched | 4155.9 | 0.401 | 8.67 |
| 16 / natural random-model routing | Grouped | 4107.9 | 0.408 | 8.53 |
| 16 / forced two-expert routing | Batched | 3643.0 | 0.557 | 6.24 |
| 16 / forced two-expert routing | Grouped | 2496.1 | 0.342 | 10.18 |

Peak is maximum allocated CUDA memory across ranks. Time averages updates 2 and 3
after synchronization, taking the slower rank. Measurements are short local runs;
the 4-expert batched row reuses the preceding block-size sweep.

The forced-routing case sets router weights to zero and biases to 4 and 3 for the
first two experts and -4 for the others before training. It retains the configured
auxiliary objective and optimizer. Its empty experts make capacity-padded GEMMs
perform eight times the valid routed rows: 16*4096 padded rows versus 2*4096 real
rows. Grouped execution skips that work. Memory falls 31.5%, update time falls
38.7%, and measured tokens/s increases about 1.63x. This is the combined planner,
idle-expert and padding-elimination benefit, not an isolated grouped-kernel speedup.

For natural routing, peak falls only 2.2% (4 experts) or 1.2% (16 experts), and
update time increases 6.8% or 1.7%. The grouped backend therefore remains opt-in.
Expert count alone does not establish an advantage; routing skew, FFN dimensions,
token budget and kernel/metadata costs matter.

Model TFLOPS estimates use `MFUCalculator.from_config` with active FFN width
`(shared + top_k) * expert_width`, its `6*N*tokens` approximation, measured time
and two GPUs. They exclude extra recomputation, routing, padding and quadratic
attention work and do not measure hardware utilization.

## Run

```bash
torchrun --standalone --nproc_per_node=2 -m ironcore train \
  --config configs/experiments/cs336_55m_moe_cp2_grouped.yaml

pytest tests/integration/moe/test_grouped_experts.py

torchrun --standalone --nproc_per_node=2 -m pytest \
  tests/multi_gpu/test_blockwise_context.py

torchrun --standalone --nproc_per_node=4 \
  -m tests.multi_gpu.test_blockwise_context --tp 2

torchrun --standalone --nproc_per_node=4 \
  -m tests.multi_gpu.test_context_parallel_dp
```

The shipped 55M preset has expert width 256 and sequence length 1024; it is a
portable smoke example rather than the exact 8K measurement above. The native
API is documented by [PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.grouped_mm.html).
Ignored `.local/` JSON/configs and a standalone Korean study HTML retain local evidence.

The follow-up [expert-count scaling report](moe_scaling_validation.md) compares
4, 8, 16, 32 and 64 experts under natural and forced routing on two GPUs.
