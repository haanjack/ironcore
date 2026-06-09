# Dataloader

## Task types and collator modes

`data.task_type` determines the collator mode and what each batch contains:

| `task_type` | Batch keys |
|---|---|
| `pretrain` | `input_ids`, `labels` |
| `sft` | `input_ids`, `labels`, `position_ids`, optionally `cu_seqlens` (FlashAttention) |
| `dpo` | `chosen_*` and `rejected_*` prefixed SFT keys |
| `grpo` | `input_ids`, `attention_mask`, `metadata` list |

## Dataset formats

**`StreamingDataset`** — block-based shuffling for large text corpora. Supports weighted mixing of multiple datasets via the `ratio` field.

**`StreamingBinaryDataset`** — memory-mapped access to preprocessed `.bin`/`.idx` files produced by `ironcore preprocess`. Use for large pretrain corpora.

## SFT bin-packing

The SFT collator packs multiple short samples into each sequence up to `seq_length` to maximize GPU utilization. Each packed sample gets independent `position_ids` (reset to `[0, 1, 2, …]`). When `use_flash_attn: true`, `cu_seqlens` boundaries are passed to FlashAttention so each sample attends only to itself.

## Column mapping

Configure which dataset columns map to which roles:

| Config field | Default | Used by |
|---|---|---|
| `text_column` | `"text"` | Pretrain |
| `messages_column` | `"messages"` | SFT |
| `chosen_column` | `"chosen"` | DPO |
| `rejected_column` | `"rejected"` | DPO |
| `prompt_column` | `"prompt"` | GRPO |
| `answer_column` | `"answer"` | GRPO |

## Multi-dataset mixing

Multiple datasets can be specified in `data.datasets`. Each entry has a `ratio` field;
sampling is proportional to `dataset_size × ratio`. See [multi_dataset_sft.md](multi_dataset_sft.md)
for mixing semantics and best practices.

## FIM preprocessing

Fill-in-the-Middle is applied at **preprocessing time** (`ironcore preprocess`), not during
training. At training time the collator treats FIM-transformed sequences as ordinary pretrain
sequences.

| Config field | Default | Description |
|---|---|---|
| `data.fim_rate` | `0.0` | Fraction of pretrain documents to transform (0 = disabled) |
| `data.fim_split_type` | `"random"` | `"random"` or `"line_aware"` split strategy |
| `data.fim_prefix_token` | `"<fim_prefix>"` | PSM format prefix sentinel |
| `data.fim_suffix_token` | `"<fim_suffix>"` | PSM format suffix sentinel |
| `data.fim_middle_token` | `"<fim_middle>"` | PSM format middle sentinel |

## Configuration reference

| Field | Default | Description |
|---|---|---|
| `data.task_type` | `"pretrain"` | Determines collator mode |
| `data.seq_length` | `1024` | Sequence length |
| `data.tokenizer_type` | `"bbpe"` | `"bbpe"`, `"tiktoken"`, or `"sentencepiece"` |
| `data.pad_token_id` | `null` | Padding token (null = use EOS) |
| `data.vocab_name_or_path` | `"gpt2"` | Tokenizer vocab name or path |
| Dataset `ratio` | `1.0` | Sampling weight for this dataset |
