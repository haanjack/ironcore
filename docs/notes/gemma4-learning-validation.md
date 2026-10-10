# Gemma 4 A4B public-data learning validation

This study starts from the official BF16 `google/gemma-4-26B-A4B-it` checkpoint
(`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`). It does not start from the
earlier repeated-COPPER adapter. Evidence is written under
`.local/gemma4-learning-study/`; that directory is ignored by Git.

## Protocol

The short experiment uses 240 FineTome conversations, 32 validation
conversations and 16 test conversations. The subset is shuffled across the
entire source training split, with seed 3407, and filtered to 128–2049 tokens
including the official chat template. Conversation and normalized first-user
prompt hashes are disjoint across splits. No answer is truncated. The pinned
dataset revision is `c2343c1372ff31f51aa21248db18bffa3193efdb`.

The long experiment uses LongAlign conversations re-tokenized with Gemma's
tokenizer, with 24576–32769 tokens per conversation. It has 16 training
conversations, 4 validation conversations and 4 test conversations. Source
pages are shuffled, then selected conversations are shuffled across splits.
The first 2000 normalized characters of the user prompt are also deduplicated
to avoid placing different questions about the same document in different
splits. The Hub revision observed before and after Viewer selection is
`12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc`.
The Viewer does not accept a revision argument. The exact downloaded subset
is therefore retained locally with checksums, rather than assuming its cached
Parquet representation is revision-pinned.

The general SFT configuration is TP2/CP1, microbatch 1, accumulation 4,
60 optimizer updates, LR 2e-5 with 5 warmup updates, LoRA rank 8 / alpha 16,
BF16 base weights and FP32 adapters. Attention, shared MLP and routed experts
are targeted: 333696000 trainable parameters. The frozen parameters must not
acquire gradients. CPU weight streaming, full-layer activation spilling,
MLP chunks of 512, attention query chunks of 128, Triton blockwise grouped
MoE with a 4096-token budget, and recomputed CE chunks of 128 are enabled.
The container has a 96 GiB RAM limit with swap disabled. Large-model CP2 is
excluded because its replicated CPU weights exceed this host's budget.
The preallocation guard discounts regular checkpoint file cache on both
active/inactive reclaim lists; it does not discount shared pinned offload
storage. This follows the memory-list distinctions in the
[Linux cgroup v2 documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html).

The 60-step and accumulation-4 structure is informed by
[Unsloth's public Gemma 4 training guide](https://unsloth.ai/docs/models/gemma-4/train).
This is a smaller local validation with different precision, adapter scaling,
optimizer and learning rate, rather than a numerical reproduction of that
recipe. The data sources are [FineTome](https://huggingface.co/datasets/mlabonne/FineTome-100k)
and [LongAlign](https://huggingface.co/datasets/zai-org/LongAlign-10k).

## Evaluation

Before training, at intermediate validation checkpoints, and after saving,
zeroing and reloading the standalone adapter, compute assistant-only
teacher-forced cross entropy. `mean_sample_loss` weights every conversation
equally, matching the SFT objective. `mean_token_loss` weights by the number
of supervised assistant tokens. Both are saved with individual sample losses
and target counts. Evaluation requests the labeled, chunked CE model path;
it does not construct an entire 32K-by-262144 logit tensor.

The eight-conversation `train_probe` belongs to the training set and measures
fitting, not generalization. Validation and test conversations never enter
the training iterator. A varied one-epoch training stream need not have
monotonically decreasing per-step loss: compare the same fixed evaluation
conversations before and after training.

The historical loss evaluation used the unchanged official template with
`enable_thinking=True`. The training-format audit found 330 short-run and
16 long-run assistant turns with plain answers, no structured reasoning, and
no empty thought channel. Google's
[A4B/31B fine-tuning guidance](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)
recommends an empty thought channel before non-thinking training answers.
The recorded recipe therefore differs from that recommendation. Its loss
reduction measures fit to this particular format, not preservation of official
thinking capability.

The subsequent root-cause audit confirmed **346/346** assistant prefixes
requested thinking, while **0/346** first assistant labels targeted
`<|channel>`. Shift and prompt masking were correct. At each such first-token
position, CE's derivative with respect to the channel-opening logit is
`p(channel)` rather than `p(channel)-1`, directly penalizing channel opening.
Native IronCore BF16 TP2 probes on two fixed MMLU-Pro prompts found the base
model's channel-opening probability near 100%, versus **0.00614% / 0.01585%**
with the trained adapter. Both trained first tokens were `The`. This reproduces
the defect without vLLM, FP8, output parsing, or a generation-token limit.
The completed 8K lm-eval subset also contained thought openings in 140/140 base
outputs and 0/140 adapter outputs; the 16K rerun was stopped for diagnosis.
These checks identify a concrete suppression mechanism, without establishing
that it explains every change in benchmark accuracy.

The corrected example uses non-thinking mode and the official empty thought
prefix for plain answers. Its format is `gemma4-sft-v2`; training and held-out
loss share the template, and template/mode hashes isolate preprocessing caches.
An audit of the same 256 training conversations confirmed all 346 channel-open
positions were masked, every original answer target was unchanged, and all
conversations still fit their respective 2K/32K windows. Structured reasoning
requires `--enable-thinking`, `reasoning`/`reasoning_content`, and one assistant
target per example. Multi-turn reasoning must be split; inference continues to
strip past thoughts according to the original template.

Verification: 37 targeted CPU tests passed, including a CE-gradient check of
the suppression mechanism. A tiny Gemma TP2 GPU run with CPU weight offloading,
activation spilling, grouped MoE and attention/shared/routed LoRA completed
three updates (loss 12.4972 → 12.1290); both rank replicas and saved/reloaded
adapters matched exactly. These are format and execution checks, **not a
successful retraining or a thinking-preservation benchmark**. Existing adapters
remain unchanged and should be replaced by fresh training from the official
base before repeating capability evaluation. Raw audits and native probes are
saved in `.local/gemma4-thinking-rootcause` and
`.local/gemma4-capability-eval/native-thinking-mode.json`.
The short generation probes use
`enable_thinking=False`, deterministic greedy decoding and stop at EOS or
`<turn|>`. Raw output and whether the token limit was reached are saved.
The two probes check arithmetic and simple Python instruction following;
they are a generation sanity check, not a comprehensive capability benchmark.
Short probes also do not establish long-document answer quality.
The long continuation also re-evaluates the short experiment's test set as
`retention`, so loss regression on the earlier task remains visible.

## Reproduction

The commands below now use the corrected format and will not reproduce the
historical loss numbers. The old runners are preserved as
`.local/gemma4-learning-study/runner-finetome.py` and `runner-longalign.py`.
Prepare data, then launch through the supplied examples. Keep each run's
output directory fresh. The artifact manifests preserve exact selected row
IDs, token counts, revisions and file checksums.

```bash
python examples/gemma4_sft_data.py --kind short \
  --output .local/gemma4-learning-study/short-data
python examples/gemma4_sft_data.py --kind long \
  --output .local/gemma4-learning-study/long-data \
  --train-samples 16 --eval-samples 4 --test-samples 4 \
  --min-tokens 24576 --max-tokens 32769

MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 examples/gemma4_sft.py \
  --tp 2 --sequence-length 2048 --steps 60 --gradient-accumulation 4 \
  --learning-rate 0.00002 --warmup-steps 5 \
  --training-data .local/gemma4-learning-study/short-data/train.jsonl \
  --validation-data .local/gemma4-learning-study/short-data/validation.jsonl \
  --test-data .local/gemma4-learning-study/short-data/test.jsonl \
  --train-probe-data .local/gemma4-learning-study/short-data/train_probe.jsonl \
  --generation-prompts .local/gemma4-learning-study/generation-prompts.json \
  --output .local/gemma4-learning-study/finetome-tp2

MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=131072 MALLOC_ARENA_MAX=2 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 examples/gemma4_sft.py \
  --tp 2 --sequence-length 32768 --steps 16 --gradient-accumulation 1 \
  --learning-rate 0.00001 --warmup-steps 2 --eval-interval 8 \
  --training-data .local/gemma4-learning-study/long-data/train.jsonl \
  --validation-data .local/gemma4-learning-study/long-data/validation.jsonl \
  --test-data .local/gemma4-learning-study/long-data/test.jsonl \
  --train-probe-data .local/gemma4-learning-study/long-data/train_probe.jsonl \
  --retention-data .local/gemma4-learning-study/short-data/test.jsonl \
  --generation-prompts .local/gemma4-learning-study/generation-prompts.json \
  --adapter-path .local/gemma4-learning-study/finetome-tp2/adapter \
  --output .local/gemma4-learning-study/longalign-tp2-32k
```

`generation-prompts.json` is a list of objects with `name`, `messages`,
`expected` (informational) and `max_new_tokens`. This study asks “What is 12
times 13? Reply with only the number.” and “Write a Python function add(a, b)
that returns their sum. Output only the function, without markdown or
explanation.”, with limits 48 and 96 tokens, respectively.

## Results (2026-10-09)

Both experiments finished on two RTX 3090 24 GiB GPUs, using Torch
2.14.0+cu130 and Transformers 5.17.0. All final values below were measured
after standalone adapter save, zeroing and reload. The long baseline starts
from the FineTome adapter, not the untouched base model.

| Experiment / evaluation | Conversations | Before loss | Reloaded final loss | Samples with lower loss |
|---|---:|---:|---:|---:|
| FineTome / validation | 32 | 5.633159 | 0.683711 | 32/32 |
| FineTome / test | 16 | 5.574651 | 0.730736 | 16/16 |
| LongAlign / validation | 4 | 1.039885 | 0.981469 | 4/4 |
| LongAlign / test | 4 | 1.536492 | 1.420836 | 4/4 |
| FineTome test retained during long continuation | 16 | 0.732341 | 0.730176 | 9/16 |

The small retention difference is comparable to BF16 rerun variation and
should be treated as preservation, not a meaningful quality improvement.
Likewise, a four-conversation long-context test does not establish broad
long-document answer quality. The long training subset contained 452552
actual conversation tokens (3978 supervised assistant tokens), with complete
conversations ranging from 24837 to 32389 tokens, right-padded to a 32768-token
input window. The short training subset contained 147157 conversation tokens
(101730 supervised assistant tokens); all 240 conversations were consumed
once across 60 updates with accumulation 4.

| Execution | Updates | Median seconds/update | GPU peak allocated | GPU peak reserved |
|---|---:|---:|---:|---:|
| FineTome, 2048 input slots, accumulation 4 | 60 | 34.493 | 6.855 GiB/rank | 7.086 GiB/rank |
| LongAlign, 32768 input slots, accumulation 1 | 16 | 77.496 | 10.602 GiB/rank | 12.045 GiB/rank |

No NaN/Inf loss or gradient and no cgroup OOM/kill occurred. Gradient clipping
was active at norm 1: the maximum **preclip** norms were 1416.121 in the short
run and 176.886 in the long run. The finite spikes are retained in the raw
records; the study does not claim uniform gradient behavior. Container peak
usage reached the 96 GiB limit while reclaiming checkpoint/dataset page
cache, so it is not a model working-set measurement.

Both short deterministic generation probes passed before and after both
experiments: `156` and `def add(a, b):\n    return a + b`. These stopped at the
turn terminator before the token limit. They check generation sanity and
instruction following, not long-context generation.

All 230 attention and 180 shared-MLP adapter tensors changed in both runs.
The short run changed 22052/23040 routed-expert adapter tensors; the long
continuation changed 21484/23040 relative to its starting FineTome adapter.
Router counts in these runs describe only the first unpadded evaluation
forward, not full-run expert usage. Frozen parameters acquired no gradients.
Full FP32 adapter fingerprints were identical across the TP ranks and after
zeroing/reload. Each native `ironcore_lora_v1` adapter weight file is
1337595536 bytes, with no base weights or optimizer states; this is an
IronCore adapter, not a converted Hugging Face PEFT adapter.

| Adapter | SHA256 |
|---|---|
| FineTome | `7bcca02c3a18be181a7982af7f982ef9a9f65566e0956e9deb207f56d710432a` |
| LongAlign continuation | `47ddc1f2328e29be8b7679d12bebed6fe495abc83c60bbaa8d3614476d316753` |

An orchestration mistake caused a second long launch while the intended
launch was initializing. The duplicate was rejected by the RAM guard before
model allocation (91.3 GiB required, 43.4 GiB available). One accepted job
continued and completed. The original mixed log and a separate blocked-launch
record are retained, so this rejected attempt is not confused with a training
OOM or a second successful run. Future supervisors use an exclusive file
lock to prevent duplicate scheduling.

Validation also includes 28 passing CPU unit cases (including an independent
HF/full-logit oracle for the chunked eval SFT loss) and a tiny-model TP2 CUDA
offload run with accumulation 2, disjoint evaluation data and adapter reload.
Ruff and `git diff --check` passed.

The compact record is
[gemma4-a4b-learning-validation.json](../assets/gemma4-a4b-learning-validation.json).
Local complete artifacts are `.local/gemma4-learning-study/report.html`,
`summary.json`, exact runner snapshots, data manifests, `memory-watch.jsonl`,
and each run's `results.json`, `evaluations.json`, `generations.json`,
`config.yaml`, `execution.log` and `adapter/` directory. The `.local` artifacts
and model weights are ignored by Git.
