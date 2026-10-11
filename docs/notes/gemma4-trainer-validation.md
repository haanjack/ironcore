# IronCore as a LoRA SFT trainer: learning validation and execution costs

The current implementation produced HF-level short SFT results on the actual
Gemma 4 26B-A4B-IT model, then completed 16 updates at 32K context while
preserving reasoning and long-context retrieval. These results support using
IronCore instead of HF for the tested LoRA SFT recipes on this machine.
The comparison validates learning quality and practical execution; it is not
a benchmark of the fastest achievable HF training backend.

The tested code is `9d8fc9e12938bb34874544654665470e41aa02e5`, after fixing
the native LoRA initialization's input fan-in. The model revision is
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. Hardware is two 24 GiB RTX 3090
GPUs connected by NV4, with a 96 GiB container RAM limit.
A dated, compact snapshot is retained in
[the validation JSON](../assets/gemma4-trainer-validation.json).
Raw local experiments and the live HTML report remain ignored by Git.

## Learning quality

The short comparison uses 128 public, complete conversations, context 512,
batch 4 and 32 AdamW updates. HF and IronCore start from the same canonical
FP32 adapter, use the same data and optimizer settings, and retain BF16 frozen
base weights without training quantization. Rank-8/alpha-16 adapters target
attention, the shared MLP, and all routed experts: 333,696,000 parameters.

| Short SFT result | HF | IronCore |
| --- | ---: | ---: |
| Final heldout test CE, 32 conversations | 1.566229 | 1.569177 |
| Thinking accuracy, same 140 questions | 113/140 | 116/140 |
| Unfinished thoughts | 3 | 2 |
| Repetitive unfinished thoughts | 0 | 0 |

Both reduce heldout loss and retain thinking. The original checkpoint scores
110/140 under the same evaluation protocol. The final test CE difference is
about 0.19% of HF's CE, and the observed thinking scores are close. This is
positive evidence that the production IronCore training path produces useful
adapters despite differences between BF16 execution paths.

A separate SmolLM2-135M-Instruct study trained 4,096 conversations for 256
updates using IronCore, an independent HF/PEFT loop, and the actual
`transformers.Trainer`. Full ARC-Easy, ARC-Challenge, HellaSwag and PIQA splits
provide an additional, smaller-model trainer control. Gemma E2B tests also
exercise HF, Native TP1 and Native TP2 learning and checkpoint evaluation.
The large Gemma control is an HF/PEFT custom CPU-streaming loop, not the
standard Trainer lifecycle.

## Completed Native 32K continuation

Starting from the validated short-run Native adapter, the current production
example completes 16 LongAlign updates with actual 25K–32K conversations
padded to 32,768 input tokens. It uses TP2/CP1, CPU weight streaming,
full-layer activation spill/recompute, attention chunks of 128, MLP chunks
of 512, Triton blockwise grouped MoE with assignment budget 4096, and
recomputed CE chunks of 128. Expert accumulation uses model precision BF16.
AdamW starts fresh at LR 1e-5 with two warmup updates; prior moments are not
resumed. Every attention/shared-MLP/routed-expert adapter is trainable.

| Native 32K evaluation | Before | Saved and reloaded final |
| --- | ---: | ---: |
| Train probe CE, 8 conversations | 1.587681 | 1.518397 |
| Validation CE, 4 conversations | 1.648069 | 1.596004 |
| Test CE, 4 conversations | 2.181955 | 2.097082 |
| Short retention CE, 16 conversations | 1.344030 | 1.308117 |
| Strict long-context retrieval | 20/20 | 20/20 |
| Thinking accuracy, 140 questions | 116/140 | 112/140 |
| Unfinished thoughts | 2 | 5 |
| Repetitive unfinished thoughts | 0 | 0 |

All 140 outputs open the thought channel. The four-question decrease from
the starting adapter and three additional unfinished thoughts are recorded;
this is not the earlier large reasoning collapse. Final thinking remains
near the original checkpoint's 110/140. Category-stratified paired bootstrap
for final minus the starting adapter gives -2.86 percentage points with
95% interval [-6.43, 0.00]. A single seed and 140 questions do not establish
identical performance over arbitrary training histories.

TP adapter replicas have identical canonical hashes. Standalone FP32 adapters
reload exactly, including the heldout validation loss. The Native checkpoint
uses `ironcore_lora_v1`, A[input, rank] and B[rank, output]. A separate converted
BF16 inference copy was evaluated with vLLM; this does not imply every PEFT
loader accepts the Native file directly.

## Speed and memory: different system configurations

| Execution | GPUs | Completed updates | Mean seconds/update | PyTorch peak GiB/GPU |
| --- | ---: | ---: | ---: | ---: |
| HF short, 512 | 1 | 32 | 46.45 | 3.69 |
| IronCore short, 512 | 2 | 32 | 32.25 | 7.04 |
| IronCore long, 32K | 2 | 16 | 75.16 | 6.83 |

The Native 32K step-time range is 73.79–78.65 seconds; its coefficient of
variation is 1.60%. The 16 recorded training steps sum to 20.04 minutes,
excluding model preparation, heldout evaluation, checkpoint work and the
separate capability evaluation. Padded input throughput is approximately
436 tokens/second. This is input throughput, not supervised-answer throughput.

The ongoing HF 32K control has completed optimizer updates using a
broadcast-preserving saved-tensor CPU offload path. Its dated partial timing
and memory measurements are in the JSON snapshot; final HF 32K capability
results are pending. The live ignored HTML report updates as this run and
its evaluation finish.

These are **system execution costs, not equal-resource kernel benchmarks**:

- Native uses two GPUs and tensor parallelism; HF uses one GPU.
- Native bounds attention/MoE intermediates using its query chunks and
  Triton scheduling. HF uses Transformers SDPA and an independent grouped
  expert implementation with CPU-staged layers and saved tensors.
- CPU staging, saved-tensor copies, determinism settings and reduction
  boundaries also differ. The observed time ratio cannot be attributed
  just to TP, one kernel, or GPU count.
- Step time multiplied by GPU count is a useful occupancy-cost estimate,
  not measured kernel utilization, energy, or billing.
- Short-run Native memory is higher than HF memory. The short and long
  Native recipes also change batch size, chunk sizes and expert backend,
  so their peaks are not a sequence-length scaling curve.

Native 32K NVML sampling reports approximately 9.02 GiB per GPU as the
maximum resident device memory during its training phase. NVML includes
reserved pools and CUDA context memory that the PyTorch allocated metric
does not; five-second sampling can miss short peaks. CPU offload also
requires host memory: BF16 text weights alone are theoretically about
47 GiB, plus FP32 adapters, optimizer moments, staging and activation
storage. Whole-run Native CPU peak was not measured and is not inferred
from tensor byte counts.

The observed PCIe rates are bursty, with about 0.91–0.92 million KB/s mean
RX and about 13.9 million KB/s p99 in a three-minute sample. NV4 traffic is
separate: a ten-second counter interval shows approximately 2.64 GB/s
in each direction on GPU0. These measurements describe the transfer pattern;
they do not identify the dominant bottleneck by themselves.

## Why the HF control uses one GPU

One GPU is an implementation choice in this independent control, not a
Hugging Face GPU-count restriction. Its runner explicitly selects CUDA0,
keeps trainable leaves and AdamW moments on CPU, and stages each complete
layer with differentiable copies and `torch.func.functional_call`.
It contains no distributed training integration.

Installed Transformers 5.17.0's `Gemma4TextConfig.base_model_tp_plan` includes
attention projections, shared MLP and routed expert rules. Transformers
[documents tensor parallelism](https://huggingface.co/docs/transformers/v5.15.0/en/tensor_parallelism).
Using that plan together with this control's custom expert LoRA, CPU optimizer
and saved-tensor offloading needs separate integration and validation. That
combination was not tested here, so HF TP2 was not shown to be impossible.

DDP replicates a model and splits samples; it does not shard a single model
or 32K sample as TP does. Likewise, Accelerate's automatic CPU/disk
[Big Model Inference dispatch](https://huggingface.co/docs/accelerate/main/en/concept_guides/big_model_inference)
is documented for inference, not a drop-in distributed training solution.
A future performance comparison should give both implementations the same
GPU budget and separately specify their supported optimized recipes.
The present learning-quality evidence remains useful without turning this
resource mismatch into a speed superiority claim.

## Evaluation and remaining scope

The large-model capability table uses the same balanced 140 MMLU-Pro
questions, prompts, generation settings and two batches of 70. Thinking is
stock five-shot CoT with a 16K generation cap, temperature 1, top-p 0.95,
top-k 64 and seed 3407. Evaluation uses vLLM TP2 with MoE-only FP8 weight
inference and BF16 dense/activation computation. Training remains
unquantized. Completed original/short-run control results are reused,
not silently presented as new runs.

The evidence supports the tested LoRA SFT replacement and the completed
Native 32K execution. Full fine-tuning, long convergence studies, multiple
seeds, arbitrary HF API/checkpoint compatibility and reproduction of official
full benchmark scores remain outside this validation. Historical failing
adapters are retained separately; their results do not override successful
fresh training with the corrected implementation.
