# Gemma 4 text models

IronCore supports the **text decoders of Gemma 4 E2B, E4B, 31B and 26B A4B** through
`LanguageModel` and `TransformerModel`. The implementation includes:

- Alternating causal sliding-window and full attention with per-type head dimensions.
- FP32 RMS normalization of Q/K/V, attention scale 1, and proportional global RoPE.
- Four decoder norms, GELU-tanh gated MLPs, and final logit softcapping.
- Scaled token embeddings and E2B/E4B per-layer embeddings (PLE).
- E2B/E4B cross-layer KV reuse, E2B's double-wide shared-layer MLPs, and 31B's global K=V projection.
- Explicit KV tuples for `LanguageModel.generate()`, including multi-token cached decode.
- LoRA projection wrappers and activation recomputation with gradients through shared KV.
- HF import of either text-only or public multimodal checkpoints; text-only HF export.

Image/audio encoders are outside this implementation. A4B includes its native
normalized router, per-expert routing scales, GELU-tanh experts and separate
normalizations of the shared and routed MLP outputs. Generic DeepSeek routing
is not substituted for this architecture.
Text-only HF exports use `Gemma4ForCausalLM` and `model_type: gemma4_text`.
Merge trained LoRA adapters with `ironcore.peft.merge_lora_weights()` before
exporting a dense HF model; unmerged adapter exports are rejected. For native
LoRA checkpoint resume through the current generic loader, set
`operation.save_full_model: true` so the checkpoint also contains base weights.

## Current learning validation

Matched actual-26B short SFT produces HF-level heldout loss and retains
thinking. The corrected Native trainer also completes 16 updates at 32K,
saves/reloads FP32 attention/shared-MLP/routed-expert adapters, and preserves
reasoning and long-context retrieval. See the
[trainer validation and resource comparison](notes/gemma4-trainer-validation.md)
for measured quality, speed, memory, checkpoint behavior and the supported
replacement scope. Timing compares Native TP2 with a single-GPU HF CPU-streaming
control, so it is not an equal-resource HF speed benchmark.

## Presets

| Preset | Hidden size | Layers | Local heads / KV heads | Global head dimension | Shared KV layers | PLE dimension |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `configs/model/gemma4-e2b.yaml` | 1536 | 35 | 8 / 1 | 512 | 20 | 256 |
| `configs/model/gemma4-e4b.yaml` | 2560 | 42 | 8 / 2 | 512 | 18 | 256 |
| `configs/model/gemma4-31b.yaml` | 5376 | 60 | 32 / 16 | 512 | 0 | 0 |
| `configs/model/gemma4-26b-a4b.yaml` | 2816 | 30 | 16 / 8 | 512 | 0 | 0 |

31B global layers use 4 KV heads; local layers use head dimension 256. E2B uses a
4:1 local/global pattern; E4B and 31B use 5:1. Presets retain official positional
capacity but default training sequences to 2048 tokens. Model sizes include
large token/PLE tables: "E2B" is an effective size, not a 2B-total allocation.

## Using a pretrained decoder

Include a model preset in a complete IronCore training config:

```yaml
model:
  config_path: configs/model/gemma4-e2b.yaml
trainer:
  tensor_model_parallel_size: 1
  load_from_hf: google/gemma-4-E2B-it
  micro_batch_size: 1
  train_batch_size: 1
  gradient_accumulation_steps: 1
  recompute_linear_ce: false
data:
  vocab_name_or_path: google/gemma-4-E2B-it
  tokenizer_type: sentencepiece
  vocab_size: 262144
```

This snippet is an overlay, not a standalone dataset/training recipe. Supply the
usual dataset, optimizer and operation settings. Set data and model tokenizer
paths to the same repository or local tokenizer directory. `sentencepiece`
routes to `AutoTokenizer`, without assuming the tokenizer's underlying algorithm.
The ordinary `ironcore train --config <config.yaml>` command and SFT trainer apply.
Use `gemma4-e4b.yaml` / `google/gemma-4-E4B-it` or
`gemma4-31b.yaml` / `google/gemma-4-31B-it` for the other sizes.

For config conversion from an existing local HF checkpoint:

```python
import json
from ironcore.config.config_gemma4 import model_config_from_gemma4

hf_config = json.load(open("/models/gemma4/config.json"))
config.model = model_config_from_gemma4(hf_config)
config.model.vocab_name_or_path = "/models/gemma4"
config.data.vocab_name_or_path = "/models/gemma4"
config.data.vocab_size = hf_config.get("text_config", hf_config)["vocab_size"]
config.model.max_seq_len = 2048
```

The converter understands public `text_config` nesting, A4B MoE and recent
Transformers `per_layer_config` serialization. Unsupported attention layouts
are rejected. Packed HF expert tensors are split and transposed into native
expert projections. Safetensors imports stream one source tensor at a time
and do not allocate unused image/audio weights.

## Execution limits

Dense models support **TP=1 and TP=2** with existing data parallel training.
Query heads and MLPs are sharded. KV heads are sharded where divisible by TP;
a single KV head is materialized on both ranks by gathering projection channel
shards and summing gradients from their different query heads. Q/K norm scales
also sum gradients where heads are split. PLE tables are vocabulary-sharded;
PLE projection outputs are gathered for normalization, then each decoder rank
receives its corresponding PLE channels. **LoRA also supports TP=1 and TP=2**
with replicated adapter parameters and gradient communication at the TP
boundaries. LoRA dropout and both activation-recompute strategies are covered
by tiny-model CPU/Gloo and GPU/NCCL tests. Paged KV caches remain unsupported.
Recomputed linear CE preserves final logit softcapping, including its derivative;
a frozen output head allocates no weight gradient. CPU weight streaming and
full-layer activation spilling require a layout without PLE or shared KV
(A4B and 31B). E2B/E4B still reject those offload combinations.
Optimizer-state offload remains a separate existing feature.

The tuple cache retains the complete history of producer layers, including local
layers; it reuses producer tensors across shared layers. The sliding mask bounds
attention, but this SDPA path does not implement a sparse sliding-window kernel.
Full-context configurations may therefore require substantial attention memory.
Set `model.gemma4.attention_chunk_size` to bound score memory by query blocks:
local blocks crop K/V to the sliding window, and global blocks crop to the causal
prefix. Each block recomputes attention during backward. This supports 512-wide
global heads through Torch SDPA without constructing a full sequence-square mask.

A4B routed experts support `loop` or `grouped` with EP=1; grouped execution uses
the existing assignment budget and scheduled/Triton routing kernels. A shared
MLP can also use `trainer.mlp_chunk_size`. Virtual blocks are not reintroduced.
The public A4B layout uses 128 routed experts and top-8 natural routing, with no
token dropping or forced concentration. Attention-only LoRA leaves expert and
router weights frozen while preserving their gradients with respect to inputs.

For A4B/31B, CP requires `context_parallel_backend: sdpa` and query-block attention.
This backend gathers K/V and sums backward contributions; it is not ring
FlashAttention. The existing ring kernel does not support Gemma's 512-wide global
heads. CP SFT requires `data.sft_packing: false`; it combines answer token counts
across shards before taking the equal-sample mean, including shards containing
only masked prompt tokens. TP=2 plus CP=2 requires four GPUs. On two GPUs, test
TP=2/CP=1 and TP=1/CP=2 as separate choices.

Gemma 4 norms scale directly by `weight`; they do not use older Gemma's `1 + weight`
parameterization. HF projections are transposed into IronCore's native matrix
layout. The extra PLE and normalization weights must all be present on import;
falling back to LLaMA's weight mapper would silently omit them.
Official E2B checkpoints can retain unused K/V projection and K-norm weights
in sharing consumer layers. The importer discards only those recognized keys,
requires the checkpoint and native KV-sharing layout to match, and still
requires every parameter used by the native decoder.

## Verification

Download-free tests instantiate tiny versions of all three architecture layouts
and load identical Transformers reference weights:

```bash
pytest tests/regression/test_gemma4_values.py tests/unit/models/test_gemma4.py
```

The tests compare logits and every trainable parameter gradient, test generation
and cached/full equivalence beyond the local window, both activation-recompute
strategies, the native trainer's optimizer update, configuration parsing, and
HF import/export round trips. Numerical reference checks require a Transformers
version exposing `Gemma4TextConfig` / `Gemma4ForCausalLM` (verified with 5.17.0);
they skip on older versions. Configuration/mapping checks remain CPU-only.

MFU estimates count local/global projections, skipped shared-layer KV projections,
PLE tables and double-wide MLPs separately. PLE lookup weights contribute to
parameter memory but are excluded from per-token projection FLOPs. Dense SDPA
attention matmuls are included. These are estimates, not measured GPU throughput.
CPU/Gloo and GPU/NCCL evidence are recorded separately below.

Validation on 2026-10-08 used a CPU container with PyTorch 2.13.0 and
Transformers 5.17.0. The unit/regression/property CPU selection completed with
**655 passed, 12 skipped, 164 deselected**, and Ruff check/format passed.
TP=2 was additionally validated with real CPU/Gloo ranks
for all three tiny layouts: logits, every parameter gradient, cached decode,
greedy generation, and the production trainer's loss, global gradient norm and
AdamW updates against TP=1. The distributed tests cover no recomputation and
both standard and optimized activation recomputation, including HF imports.
LoRA parity additionally covers all attention/MLP projections, PLE gate/output
projections, nonzero adapters for gradient comparisons, and three AdamW steps
starting from zero B. The 24 cases combine a scaled SmolLM2 and all three Gemma
layouts, dropout 0/0.2, and no/standard/optimized recomputation. Tests compare
logits, every adapter gradient, loss, global gradient norm and updated weights;
adapter replicas agree exactly between ranks. Universal and distributed native
checkpoints restore adapter weights and optimizer moments, and merging LoRA
preserves logits. The universal checkpoint also resumes at TP=1 with exact
adapter weights and optimizer moments. See the [LoRA guide](peft_guide.md) for
the common TP boundaries and full-size SmolLM2 validation.

Subsequent GPU validation on two RTX 3090 24 GiB cards used PyTorch 2.14.0+cu130,
Transformers 5.17.0 and NCCL 2.30.7. All **9 dense FP32 cases**, **24 LoRA FP32
parity cases**, and **24 LoRA BF16 autocast training cases** passed on real
CUDA/NCCL ranks. Dense tests compare HF logits/gradients, cached decoding,
generation and TP=1/2 native trainer updates. BF16 training keeps FP32 master
weights and checks finite gradients, actual updates, decreasing loss and exact
adapter replicas; it does not assert BF16 TP=1/2 numerical parity. These Gemma
training cases use tiny E2B/E4B/31B layouts; full pretrained SmolLM2-1.7B GPU
LoRA checks are recorded in the [LoRA guide](peft_guide.md).

```bash
# Download-free distributed parity (CPU or CUDA container).
torchrun --standalone --nproc_per_node=2 -m tests.multi_gpu.test_gemma4_tp --device cpu
torchrun --standalone --nproc_per_node=2 -m tests.multi_gpu.test_lora_tp --device cpu
torchrun --standalone --nproc_per_node=2 -m pytest -o addopts='' tests/multi_gpu/test_gemma4_tp.py
```

## E2B pretrained generation

Download the official checkpoint into an ignored local directory:

```bash
hf download google/gemma-4-E2B-it config.json tokenizer.json tokenizer_config.json \
  generation_config.json chat_template.jinja model.safetensors \
  --local-dir .local/models/gemma-4-E2B-it
```

Use the same IronCore generation example at either TP size:

```bash
python examples/gemma4_generate.py --checkpoint .local/models/gemma-4-E2B-it \
  --prompt 'What is 2 + 2? Answer briefly.' --output .local/gemma4-e2b-tp1.json
torchrun --standalone --nproc_per_node=2 examples/gemma4_generate.py \
  --checkpoint .local/models/gemma-4-E2B-it \
  --prompt 'What is 2 + 2? Answer briefly.' --output .local/gemma4-e2b-tp2.json
```

The example loads every dense text parameter through the production HF loader
and uses `LanguageModel.generate()` with the official chat template and greedy
decoding. It defaults to BF16 and CUDA when available. Add `--device cpu` for
CPU/Gloo, or `--precision float32` for FP32 comparisons. JSON output includes
generated token IDs, text, timings and per-rank parameter memory. The standalone
`ironcore generate` CLI still forces TP=1; use this example for TP=2 inference.
The official multiple EOS IDs (including the end-of-turn token) are respected.
Use `--logits-output .local/prompt-logits.pt` to save the full prompt logits for
a numerical TP comparison.

Full-size E2B validation on 2026-10-08 used the official pretrained weights,
the prompt `What is 2 + 2? Answer briefly.`, a 20-token chat-template input,
greedy decoding, and the official EOS IDs:

| Precision / TP | Generated answer | Parameter memory per rank (bytes) |
| --- | --- | ---: |
| BF16 / TP=1 | `4` | 9,257,138,688 |
| BF16 / TP=2 | `4` | 4,628,855,296 |
| FP32 / TP=1 | `4` | 18,514,277,376 |
| FP32 / TP=2 | `4` | 9,257,710,592 |

TP=1 and TP=2 agreed on argmax at all 20 prompt positions for both precisions.
FP32 prompt logits had maximum absolute difference **8.49e-5** and mean absolute
difference **8.57e-6**. BF16 logits were not bitwise equal: maximum absolute
difference **1.125**, mean **0.13665**. FP32 parity and identical answers on this
prompt establish this execution check; other prompts may produce different
greedy tokens across TP sizes when BF16 rounding changes a close prediction.
These are CPU/Gloo results and parameter memory, not measured GPU memory or
throughput. JSON results and saved logits are under ignored `.local/`.

The same official E2B checkpoint and prompt also passed on **two RTX 3090 GPUs**
with the CUDA/NCCL environment described above. BF16 and FP32 both generated
`4` at TP=1 and TP=2, with identical argmax at all 20 prompt positions. GPU
FP32 prompt logits passed `atol=1e-4, rtol=1e-5`: maximum absolute difference
**7.53e-5**, mean **8.65e-6**. GPU BF16 maximum/mean differences were
**1.625 / 0.19377**; equal argmax on this prompt does not establish BF16
logit parity on other inputs.

BF16 peak allocated GPU memory during generation was **9,300,853,760 bytes
(8.66 GiB)** at TP=1 and **4,665,488,384 bytes (4.35 GiB) per rank** at TP=2.
The measurement includes model allocations and short-prompt generation; it
excludes model loading peaks, driver/context memory and allocator reservations.
The example records one invocation's timing; one generated token is not a
steady-state throughput benchmark. JSON outputs and saved prompt logits use
the ignored `.local/gemma4-e2b-gpu-{bf16,fp32}-tp{1,2}*` paths. The consolidated
report is `.local/gpu-validation.json` and its study HTML is
`.local/gpu-validation-study.html`.

The text decoder has 4,628,569,344 parameters. `MFUCalculator` estimates
**32.34 trillion training FLOPs per step** at batch size 1 and sequence length
2048; TP divides this compute across ranks without reducing total model work.
No GPU MFU was measured.

## E2B pretrained SFT smoke test

On 2026-10-09 the full official E2B text checkpoint passed native
`LanguageModelTrainer` SFT on two RTX 3090 GPUs (TP=2, CP=1), with BF16
autocast and FP32 stored parameters. The run used 16 hand-authored instruction
conversations: 12 training examples and 4 held-out examples. Maximum actual
conversation length was 31 tokens; collation padded to 128. This validates
the training pipeline, not long-context training or downstream quality.

LoRA rank 8 / alpha 16 targeted `q_proj`, `v_proj`, `o_proj`, `up_proj`, and
`down_proj`, giving **8,724,480 replicated trainable parameters per rank**.
Microbatch 1 and accumulation 2 produced 12 Adam updates at LR 1e-4, epsilon
1e-4, clipping 1, and zero weight decay. Standard activation recomputation
was enabled, vocabulary CE was chunked at 32 tokens, and
`recompute_linear_ce` remained false to preserve logit softcapping.

| Measurement | Before | After 12 updates |
| --- | ---: | ---: |
| Fixed training-set mean SFT loss | 9.492615 | 0.109005 |
| Four-example held-out mean SFT loss | 11.696751 | 0.120599 |
| Greedy answer to `What is 2 + 2? Answer briefly.` | `4` | `4.` |

Loss averages include assistant content, the template's end-of-turn token and
trailing newline; this tiny formatting-sensitive dataset does not establish
generalization. All 310 adapter tensors changed and TP replicas agreed exactly.
Peak allocated memory after initialization was **10,109,623,808 bytes
(9.42 GiB) per GPU**. Allocator peak reserved memory was **21.87 GiB**, including
cached allocations from initial model loading; allocated memory excludes
loading peaks and CUDA driver/NCCL allocations.

The full unmerged native LoRA checkpoint (`save_full_model: true`,
`save_dist_ckpt: true`) occupied 18,725,882,587 bytes including trainer state.
A fresh torchrun process restored step 12 adapters, Adam state and scheduler
exactly. Its step 13 loss (**0.0271968404**), updated adapters and optimizer
state matched an uninterrupted native trainer update bitwise. Native trainer
state also restored the consumed data cursor.

This check exposed and fixed preprocessing support for `sentencepiece`,
Transformers `BatchEncoding` chat output, conversation-prefix assistant masks,
and read-only NumPy memmap collation. Regenerate old SFT binary files after
updating preprocessing; existing files are otherwise skipped. Training configs
use `data.task_type: sft` and the supported `data.train_datasets` schema.
For a full native resume, clear `trainer.load_from_hf` to avoid reloading the
original base checkpoint before loading the saved model.

Configs, data, observation scripts, checkpoint and JSON evidence are under
ignored `.local/gemma4-e2b-sft/`; the standalone study is
`.local/gemma4-e2b-sft-study.html`. The CPU selection excluding Hub downloads
passed 925 tests; 12 focused preprocessing/collator tests also passed in the
CUDA container. Ruff checks and formatting passed.

## A4B long-context LoRA SFT

`examples/gemma4_sft.py` uses the production trainer, the official pretrained
text weights and the official tokenizer. It builds a real instruction conversation
with exactly `sequence_length + 1` tokens, then shifts it into a fully occupied
input. It repeats one conversation to validate feasibility; it is not a quality
evaluation. Assistant-only masking means only the short final answer is supervised,
while the entire 32K prompt participates in forward and attention backward.

```bash
hf download google/gemma-4-26B-A4B-it --local-dir .local/models/gemma-4-26B-A4B-it
MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 examples/gemma4_sft.py \
  --attention-only --sequence-length 32768 --output .local/gemma4-a4b-study/full-tp2-32768
```

The run combines BF16 frozen weights, FP32 attention LoRA (rank 8, alpha 16),
full-layer activation spilling/recomputation, query blocks of 128, shared MLP
blocks of 512, grouped routed-expert assignment budget 4096 and recomputed
softcapped CE blocks of 128. Adam LR is 1e-4, epsilon 1e-4, clipping 1, microbatch
1 and accumulation 1. Router and all expert weights remain frozen; natural
top-8 routing is preserved. The allocator environment limits retained CPU
arenas when transferring frozen parameters into pinned tiles. The example
also releases unused glibc memory once after initialization, outside the train loop.

The current example defaults to **non-thinking SFT** and follows the checkpoint's
official generation prefix. A4B prefixes plain assistant answers with the empty
`<|channel>thought\n<channel|>` channel and excludes it from loss. E2B ends its
generation prefix at the model header, so its SFT answers do not add that channel.
Training and held-out loss use the same template. Format `gemma4-sft-v3` and
template/mode fingerprints isolate serialized caches. `--enable-thinking`
requires structured `reasoning` or `reasoning_content` and one assistant target
per example; split multi-turn reasoning data before training. Inference keeps
the official template, including stripping past thoughts between ordinary turns.

The historical recorded feasibility runs rendered the official template with
`enable_thinking: true` for complete conversations and assistant prefixes,
through per-dataset `chat_template_kwargs`. This satisfied the preprocessing
prefix checks. However, the answers contained no structured reasoning, and
this does **not** follow Google's recommendation to prepend an empty thought
channel when fine-tuning A4B/31B on non-thinking answers. See the
[official formatting guidance](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4).
Treat these runs as execution and adapter validation; they do not validate
the training format or reproduce official model capabilities. The learning
study records this mismatch separately from its loss measurements. An audit of
all 346 learning-study assistant turns confirmed that their first answer labels
penalized opening the thought channel under a thinking request. Native BF16
probes found channel-opening probability near 100% for the base model but
0.0061% and 0.0159% for the saved adapter on two fixed prompts. Existing adapters
need retraining from the official base with the corrected format; a larger
generation cap cannot repair these learned first-token probabilities.

On two RTX 3090 24 GiB GPUs, the full pretrained checkpoint passed TP=2/CP=1
SFT at **32,768 real input tokens** on 2026-10-09:

| Step | Assistant SFT loss | Seconds |
| --- | ---: | ---: |
| 1 | 9.765177 | 67.79 |
| 2 | 4.299227 | 64.18 |
| 3 | 2.886663 | 63.51 |

Peak allocated GPU memory was **10,149,425,152 bytes (9.45 GiB) per rank**;
peak reserved was 11.96 GiB. These allocator measurements exclude CUDA driver
and NCCL memory. The initial run retained CPU allocation arenas and reached
approximately 121 GiB of system RAM use, so the allocator settings above are
material on a 123 GiB host. This is not a GPU-only recipe.
With the allocator environment, source expert transpose views and
`PYTORCH_ALLOC_CONF=expandable_segments:True`, a fresh TP2 process loaded the
saved adapter and completed another real 32K step (loss **0.442490**).
Its allocated/reserved peaks were **9.45 / 10.31 GiB** per GPU, and per-process
training RSS was approximately **31.5 GiB**, compared with 54.5 GiB in the initial
run. Observed host RAM use during this run was approximately 74 GiB.
An eight-token `LanguageModel.generate()` smoke check also passed; it produced
the beginning of a thought response, not a verified completed instruction answer.
First-forward natural routing used **96–128 of 128 experts per layer** across
the real 32K input. All 230 adapter tensors changed, and the two TP replicas
agreed bitwise. The standalone adapter was zeroed and reloaded exactly.

The adapter contains **5,744,640 FP32 parameters** and its safetensors file
occupies 23,004,712 bytes. It contains no base-model weights. Its format is
`ironcore_lora_v1`, with native `A[in, rank]` and `B[rank, out]` matrices and full
TP replicas; it is not a PEFT-format Hub adapter. Load the same pretrained base
and set `peft.lora.adapter_path` to the directory, or call:

```python
from ironcore.peft import load_lora_adapter
load_lora_adapter(model, ".local/gemma4-a4b-study/full-tp2-32768/adapter")
```

`--adapter-path <directory>` on the example loads these weights before a fresh
trainer run. This restores adapters; it does not restore Adam moments, scheduler
or consumed data. Use native full checkpoints for an exact optimizer resume.
Use `--tp 1` for the separate CP=2 alternative. JSON evidence, runnable config,
tokenized data and adapter files remain in the ignored output directory.

The full pretrained **TP=1/CP=2** alternative also completed three real 32K
steps, with the CPU allocator environment above:

| Step | Assistant SFT loss | Seconds |
| --- | ---: | ---: |
| 1 | 10.3216 | 63.00 |
| 2 | 10.2312 | 58.43 |
| 3 | 5.3305 | 59.15 |

Allocated peaks were **8.75 / 8.93 GiB** on ranks 0/1. Reserved peaks were
**11.48 / 22.86 GiB**; growing causal K/V prefix allocations caused allocator
retries on rank 1, which recovered and completed training. Sampled system RAM
use reached approximately **115.2 GiB**. Every adapter changed, replicas matched
bitwise and standalone reload was exact. CP has comparable allocated GPU memory
here but doubles frozen CPU weight storage relative to TP2; TP2 leaves more host
RAM available. These BF16 runs do not establish numerical parity between TP and CP.
With expandable CUDA segments, a fresh CP2 process also loaded the **same TP2
adapter** and completed a real 32K update (loss **0.690831**, 62.22 seconds).
Allocated peaks stayed at **8.75 / 8.93 GiB**, while reserved peaks dropped to
**9.43 / 9.61 GiB**. This checks adapter portability across TP2 and CP2 and
addresses the allocation-growth issue without changing attention semantics.

Download-free verification additionally compares tiny A4B logits and every
parameter gradient with Transformers, loop/grouped routing, bounded attention,
shared MLP chunks, masked softcapped CE, standalone adapter round trips and CP2
sample weighting/adapter gradients. Tiny TP2 offload and resident runs had
identical three-step BF16 losses; tiny CP2 plus offload also updated adapters
and decreased loss. Tiny-model results are separate from the full 32K result.

On CUDA/NCCL, 18 A4B TP cases passed: six FP32, six BF16 training and six FP64
checks across dropout 0/0.2 and no/standard/optimized recomputation. FP64 adapter
gradients agreed within 4.45e-16. The tiny A4B's separate shared/routed RMSNorms
amplify FP32 reduction rounding; FP32 adapter gradients use a per-tensor relative
L2 bound of 5e-4, while FP64 retains `atol=1e-10, rtol=1e-8`. Updated adapters,
losses, clipping, native checkpoint resume and TP replica equality are checked
separately. CPU tests excluding CUDA, MP, expensive E2E and Hub downloads passed
**952 cases**, with 30 skipped and 289 deselected. Sixteen existing CUDA
weight-tile/scheduler regression cases also passed.

For the fixed tiny A4B example (TP2, batch 1, 128 real tokens, three updates),
resident and offload modes had bitwise-identical losses. The production
`get_detailed_memory_breakdown()` measurements, taken after the optimizer step,
show why offloading a small model has overhead without useful VRAM savings:

| Measurement | Resident | Offload |
| --- | ---: | ---: |
| Parameter bytes, CPU + GPU | 17,205,792 | 17,205,792 |
| Optimizer-state bytes | 122,880 | 122,880 |
| Gradient bytes after step | 0 | 0 |
| Peak CUDA allocated, bytes | 269,437,440 | 269,867,520 |
| Mean time of steps 2–3, seconds | 0.0586 | 0.2754 |
| Nominal model TFLOPS/s/GPU | 0.1123 | 0.0239 |

These tiny timings are smoke measurements, not a throughput benchmark. Transfer,
CPU optimizer and scheduling overhead dominate the tiny layer weights; the large
resident vocabulary table is identical in both runs. The formal memory helper
counts CPU parameters too, so its activation estimate is not a trustworthy GPU
activation breakdown for offload. Raw CUDA peaks are reported separately.
`MFUCalculator` now counts all stored Gemma experts but only top-k routed GEMMs,
and accounts for cropped query-block attention. Its full 32K TP2 nominal model
estimate is **1.062e15 FLOPs/step**, or **8.32 nominal TFLOPS/s/GPU** using the
mean time of steps 2–3. This does not include extra recomputation, CPU work,
transfer traffic or a precise frozen-LoRA backward correction; it is not measured
hardware MFU. Trainer MoE logging continues to report measured tokens/s.

## A4B attention, shared MLP and routed-expert LoRA

Public-data learning validation, with fixed held-out FineTome and LongAlign
conversations, is documented in the
[learning validation study](notes/gemma4-learning-validation.md). That study
measures loss before/after training and after standalone-adapter reload;
the repeated-instruction feasibility runs below have a different purpose.

The SFT example now defaults to attention plus **both** MLP paths. Set
`--attention-only` to reproduce the earlier attention-only validation and load
its adapters. Explicit configuration is:

```yaml
peft:
  method: lora
  lora:
    r: 8
    alpha: 16
    dropout: 0
    parameter_precision: float32
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

`gate_proj`, `up_proj` and `down_proj` select the shared MLP and every routed
expert. Each expert's fused gate/up base projection has separate gate and up
adapters; down has its own adapter. The router and all pretrained weights stay
frozen. Natural top-8 routing decides which expert adapters receive gradients;
globally idle experts keep `grad=None` and do not acquire Adam moments.

| Adapter component | Rank-8 trainable parameters |
| --- | ---: |
| Attention | 5,744,640 |
| Shared MLP | 3,548,160 |
| Routed experts | 324,403,200 |
| Total | 333,696,000 |

The adapter is approximately 1.24 GiB in FP32. Expert adapters dominate its
size even though each token activates only eight experts. Standalone native
adapter save/load includes all three components without pretrained weights.
`merge_lora_weights()` folds expert deltas into the frozen fused gate/up and
separate down weights; native adapters are still not HF PEFT-format exports.

Grouped execution uses two small GEMMs for each low-rank update rather than
materializing a dense weight delta. These operations remain inside the bounded
execution group and are recomputed in backward. Replicated column-A/row-B
adapter gradients are summed across TP; column-B/row-A compute views gather
their gradients. Communication boundaries are built on packed active adapters
outside tile recomputation, avoiding a collective for each individual expert
in each tile. Small ranks are padded only in CUDA grouped-GEMM operands to
satisfy 16-byte stride alignment; saved rank and parameter shapes stay unchanged.
Dropout streams are replayed in bounded backward. A fixed seed does not promise
identical dropout masks between loop and grouped execution orders.

For this example, `offload.optimizer_cpu_threads=1` is intentional: thousands
of small expert adapter tensors cannot amortize CPU thread-pool overhead. A
concurrent diagnostic on 22,528-element FP32 tensors measured 200 `sqrt` calls
at 0.00176 seconds with one thread versus 2.78 seconds with 19 threads. This is
a microbenchmark, not an end-to-end training speedup. The initial 19-thread
full-model attempt was stopped during its first optimizer update, without a
completed-step claim. The general offload default is unchanged.

Download-free validation covers expert output, input/router-weight derivatives,
all active adapter gradients against an independent loop reference, idle
experts, partial target selection, dropout replay, adapter reload and merged
inference. CUDA/NCCL checks cover grouped `torch`, `scheduled` and `triton`
backends, nonzero-adapter FP32 TP1/TP2 gradient parity, BF16 training, exact
replicas and native checkpoint/optimizer-state round trips. CP2 checks compare
masked sample loss and every adapter gradient to CP1 on CPU and CUDA, including
Triton routing on CUDA. These checks do not establish BF16 TP/CP bitwise parity.

On two RTX 3090s, the full A4B checkpoint completed three real 32K
TP2/CP1 updates with FP32 attention/shared/expert adapters and FP32 Adam
moments. The remaining execution settings match the earlier attention-only
run. The command is:

```bash
MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 examples/gemma4_sft.py \
  --sequence-length 32768 --steps 3 \
  --output .local/gemma4-a4b-mlp-study/full-tp2-32768-t1
```

| Step | Assistant loss | Pre-clip gradient norm | Seconds |
| --- | ---: | ---: | ---: |
| 1 | 10.035909 | 1869.22 | 79.60 |
| 2 | 4.067889 | 49830.63 | 74.74 |
| 3 | 1.604439 | 268.53 | 75.73 |

Allocated/reserved GPU peaks were **10.50 / 12.00 GiB per rank**, excluding
CUDA driver and NCCL memory. Mean time of steps 2–3 was **75.23 seconds**.
The earlier attention-only run used 9.45 GiB allocated and averaged 63.85
seconds for those steps. These are separate smoke runs rather than a controlled
throughput benchmark. Gradient clipping was 1; the large pre-clip spike is
recorded rather than interpreted as established stable long-run training.

All 230 attention and 180 shared-MLP adapter tensors changed. Routed experts
had **16,890 of 23,040 adapter tensors** change; adapters were not constrained
to a fixed small set of experts. First-forward natural routing used **96–127
of 128 experts per layer** across the 32K input. Forward routing coverage does
not guarantee every adapter has a nonzero supervised gradient on a short answer.
The complete adapter contains **333,696,000 FP32 parameters**, with a
1,337,595,536-byte safetensors file. Cross-rank fingerprints agreed and zeroing
then reloading every adapter reproduced each tensor exactly. The repeated
one-conversation loss decrease does not establish held-out instruction quality.

Raw metrics are summarized in
[`gemma4-a4b-32k-mlp-lora.json`](assets/gemma4-a4b-32k-mlp-lora.json).
The checkpoint and raw configs/results remain under the ignored output path.

The subsequent full-model CP2 attempt **failed from host memory pressure**.
It loaded the same TP2 adapter with fresh Adam state. Kernel logs at
2026-10-09 12:25:08 Asia/Seoul report global OOM and the OOM-killer terminating
Traefik. Rank 1 reported one finite step after 2,563 seconds, but rank 0 did not
finish and the following collective timed out. No complete CP2 run or final
CP2 adapter is claimed. GPU OOM was not reported; the rank-1 allocated peak
was 9.83 GiB. The CPU weight pool alone used 46.9 GiB **per rank**, followed
by adapter gradients, moments, spilled activations and runtime overhead.
The external sampler stalled with the host and did not prevent this incident.
Traefik subsequently recovered and the node's MemoryPressure condition cleared.

The example now screens RAM **before model allocation**, with estimated
requirements of 91.26 GiB available for TP2 and 133.10 GiB for CP2 at this
configuration, including scratch space and 16 GiB headroom. It also respects
cgroup v2 remaining capacity while accounting for reclaimable inactive file
cache. The 123 GiB host fails the CP2 check even when otherwise idle. This
conservative estimate is not a memory reservation or a measured peak.

The dedicated GPU test container was subsequently capped at **96 GiB RAM with
swap disabled**, bounding test processes independently of the preflight check.
The successful full TP2 result above predates that cap; it was not rerun under
the new cap. Full CP2 was not retried. After restarting the dedicated test container to
restore its GPU access, 30 TP CUDA cases and separate CPU/CUDA CP gradient
checks passed again. Host/cgroup budget screening has five passing tests.
Tiny CP2 numerical tests remain a correctness check, not evidence that full CP2 fits this host. Use the validated
TP2 configuration here; CP2 requires more host RAM or a different weight-storage
strategy. TP2+CP2 together requires four GPUs in the current parallel layout.

The public [Axolotl A4B recipe](https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/gemma4/26b-a4b-moe-qlora.yaml)
likewise adapts attention, shared MLP and routed experts. Its base checkpoint,
QLoRA quantization, rank and dataset differ from this BF16 feasibility run.

## BF16 rounding and numerical validation

Gemma's HF configuration selects `moe.expert_accumulation_precision: model`.
Grouped execution rounds each weighted expert contribution to the model dtype
and adds contributions in expert order. The generic MoE default remains
`float32`; this option changes the rounding policy, not the expert selection.
Frozen CUDA RMSNorm computes statistics in FP32 and writes its output and
input gradient directly in the model dtype, without full-size FP32 activation
buffers. Trainable norm weights and unsupported layouts use the ordinary path.

Same-weight, same-input standalone A4B expert comparisons on two RTX 3090s
measured maximum BF16 TP2 relative L2 differences of 0.53% for outputs,
0.84% for input gradients and 0.63% for LoRA gradients. These local checks
do not certify complete BF16 training parity. In a four-layer random scaled
A4B model, a same-state, fixed-HF-top8 comparison measured a 27.12% full LoRA
gradient difference. A separate HF-only experiment that split base GEMMs
into two shards and added BF16 row-partial outputs reproduced 27.34%; its
FP32 counterpart differed by 0.000828%. Matching each layer's input values
reduced the native/HF gradient difference to 2.77%. Small forward rounding
differences can therefore accumulate and amplify through attention even
when each module's common-input comparison is close.

For the scaled model's narrow K/V projection, the following process-level
backend option removed the isolated column-GEMM discrepancy while preserving
BF16 output tensors:

```python
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
```

The probe rejected Python-visible FP32 CUDA tensor results during those
operations; opaque cuBLAS workspace was not inspected. The option does not
remove differences from summing already-rounded BF16 row-partial outputs and
is not enabled automatically. The real pretrained A4B has wider K/V projections
and different learned norm scales than the random fixture. These experiments
explain the scaled numerical discrepancy, not the previously observed
full-size SFT capability regression. Raw traces, controls, BF16 storage notes
and the standalone HTML report are retained under the ignored
`.local/gemma4-boundary-study/` directory.

## Sources

- [Google model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Transformers Gemma 4 documentation](https://huggingface.co/docs/transformers/model_doc/gemma4)
- [Transformers reference decoder](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py)
- [E2B config](https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json),
  [E4B config](https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json),
  [31B config](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json)
- [A4B config](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/config.json)
