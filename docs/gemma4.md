# Gemma 4 dense text models

IronCore supports the **text decoders of Gemma 4 E2B, E4B and 31B** through
`LanguageModel` and `TransformerModel`. The implementation includes:

- Alternating causal sliding-window and full attention with per-type head dimensions.
- FP32 RMS normalization of Q/K/V, attention scale 1, and proportional global RoPE.
- Four decoder norms, GELU-tanh gated MLPs, and final logit softcapping.
- Scaled token embeddings and E2B/E4B per-layer embeddings (PLE).
- E2B/E4B cross-layer KV reuse, E2B's double-wide shared-layer MLPs, and 31B's global K=V projection.
- Explicit KV tuples for `LanguageModel.generate()`, including multi-token cached decode.
- LoRA projection wrappers and activation recomputation with gradients through shared KV.
- HF import of either text-only or public multimodal checkpoints; text-only HF export.

Image/audio encoders and the 26B A4B MoE model are outside this implementation.
Text-only HF exports use `Gemma4ForCausalLM` and `model_type: gemma4_text`.
Merge trained LoRA adapters with `ironcore.peft.merge_lora_weights()` before
exporting a dense HF model; unmerged adapter exports are rejected. For native
LoRA checkpoint resume through the current generic loader, set
`operation.save_full_model: true` so the checkpoint also contains base weights.

## Presets

| Preset | Hidden size | Layers | Local heads / KV heads | Global head dimension | Shared KV layers | PLE dimension |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `configs/model/gemma4-e2b.yaml` | 1536 | 35 | 8 / 1 | 512 | 20 | 256 |
| `configs/model/gemma4-e4b.yaml` | 2560 | 42 | 8 / 2 | 512 | 18 | 256 |
| `configs/model/gemma4-31b.yaml` | 5376 | 60 | 32 / 16 | 512 | 0 | 0 |

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

The converter understands public `text_config` nesting and recent Transformers
`per_layer_config` serialization. It rejects MoE and unsupported attention layouts.

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
by tiny-model CPU/Gloo and GPU/NCCL tests. Paged
KV caches, activation spilling, weight streaming and recomputed linear CE are
rejected before model construction. Logit softcapping must remain in the loss
path, so bypassing it with the existing linear CE kernel would change training.
Optimizer-state offload remains a separate existing feature.

The tuple cache retains the complete history of producer layers, including local
layers; it reuses producer tensors across shared layers. The sliding mask bounds
attention, but this SDPA path does not implement a sparse sliding-window kernel.
Full-context configurations may therefore require substantial attention memory.

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

## Sources

- [Google model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Transformers Gemma 4 documentation](https://huggingface.co/docs/transformers/model_doc/gemma4)
- [Transformers reference decoder](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py)
- [E2B config](https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json),
  [E4B config](https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json),
  [31B config](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json)
