# Training Configuration

IronCore uses a hierarchical YAML configuration system. Configuration is organized into groups that can be defined inline or reference external files.

## Config Structure

```yaml
# Reference external config file
model: gpt2-small  # loads configs/model/gpt2-small.yaml

# Or define inline
trainer:
  micro_batch_size: 4
  train_batch_size: 480
```

## Configuration Groups

### trainer

Training process settings including batch sizes, parallelism, and checkpointing.

| Name | Default | Description |
|------|---------|-------------|
| `model_name` | `"model"` | Model name |
| `model_path` | `""` | Model save/load directory |
| `micro_batch_size` | `2` | Micro batch size per GPU |
| `train_batch_size` | `None` | Global training batch size |
| `gradient_accumulation_steps` | `None` | Gradient accumulation steps |
| `tensor_model_parallel_size` | `1` | Tensor parallelism size |
| `pipeline_model_parallel_size` | `1` | Pipeline parallelism size |
| `save_checkpoint_steps` | `1000` | Checkpoint save interval |
| `log_interval` | `20` | Progress print interval |
| `grad_norm_log_interval` | `None` | Gradient norm logging cadence (`log`, `checkpoint`, or `None`) |
| `param_norm_log_interval` | `None` | Parameter norm logging cadence (`log`, `checkpoint`, or `None`) |
| `use_flash_attn` | `true` | Use Flash Attention |
| `vocab_padding_unit` | `128` | Vocab padding for tensor core optimization |
| `num_workers` | `8` | DataLoader workers |
| `do_eval` | `false` | Enable evaluation |
| `eval_batch_size` | `None` | Evaluation batch size |

### operation

Training operation controls.

| Name | Default | Description |
|------|---------|-------------|
| `train_steps` | `1000` | Total training steps |
| `eval_interval` | `100` | Evaluation interval |
| `eval_samples` | `100` | Number of evaluation samples |
| `test_samples` | `100` | Number of test samples |
| `activation_recompute` | `false` | Enable activation checkpointing |
| `recompute_strategy` | `"default"` | Recompute strategy (`default`, `optimized`) |
| `no_save` | `false` | Disable checkpoint saving |
| `save_dist_ckpt` | `false` | Use distributed checkpoint saving |
| `exit_interval` | `None` | Exit after N steps (for debugging) |

### model

Model architecture configuration. Can reference a file (e.g., `model: gpt2-small`) or define inline.

| Name | Default | Description |
|------|---------|-------------|
| `d_model` | `512` | Hidden dimension size |
| `d_ffn` | `2048` | Feed-forward dimension size |
| `num_layers` | `2` | Number of transformer layers |
| `num_attention_heads` | `8` | Number of attention heads |
| `num_attention_groups` | `None` | KV groups for GQA (None = MHA) |
| `head_dim` | `128` | Attention head dimension |
| `max_seq_len` | `512` | Maximum sequence length |
| `precision` | `"bfloat16"` | Model precision (`bfloat16`, `float16`, `float32`) |
| `ln_type` | `"layernorm"` | Layer norm type (`layernorm`, `rmsnorm`) |
| `ln_eps` | `1e-5` | Layer norm epsilon |
| `post_ln` | `false` | Use post-LN (vs pre-LN) |
| `activation_type` | `"gelu"` | Activation function |
| `dropout_embd` | `0.1` | Embedding dropout |
| `dropout_attn` | `0.1` | Attention dropout |
| `dropout_mlp` | `0.1` | MLP dropout |
| `no_bias` | `false` | Disable bias in layers |
| `untie_embed` | `false` | Untie input/output embeddings |
| `tokenizer_type` | `"gpt2"` | Tokenizer type |
| `vocab_name_or_path` | `"gpt2"` | Vocab name or path |

#### positional_embedding (nested in model)

| Name | Default | Description |
|------|---------|-------------|
| `type` | `"absolute"` | PE type (`absolute`, `rope`, `none`) |
| `base` | `10000` | RoPE base frequency |
| `scaling_factor` | `1.0` | RoPE scaling factor |
| `offset` | `0` | RoPE position offset |

### data

Data loading configuration. Can reference a file via `config_path`.

| Name | Default | Description |
|------|---------|-------------|
| `config_path` | `None` | Path to data config YAML |
| `task_type` | `None` | Task type (`pretrain`, `sft`, `dpo`) |
| `splits` | `[0.97, 0.029, 0.001]` | Train/eval/test split ratios |
| `train_datasets` | `None` | Training dataset configs |
| `eval_datasets` | `None` | Evaluation dataset configs |
| `test_datasets` | `None` | Test dataset configs |
| `vocab_size` | `51200` | Vocabulary size |
| `pad_to_max_length` | `false` | Pad sequences to max length |
| `pad_token_id` | `-1` | Pad token ID (default: EOS) |

### optim

Optimizer and learning rate schedule.

| Name | Default | Description |
|------|---------|-------------|
| `optimizer` | `"adam"` | Optimizer type |
| `lr_scheduler` | `"cosine"` | LR scheduler type |
| `max_lr` | `5e-4` | Maximum learning rate |
| `min_lr` | `0.0` | Minimum learning rate |
| `warmup_steps` | `0` | LR warmup steps |
| `annealing_steps` | `0` | LR annealing steps |
| `weight_decay` | `0.01` | Weight decay |
| `no_decay_on_embedding` | `true` | Skip weight decay on embeddings |
| `adam_beta1` | `0.9` | Adam beta1 |
| `adam_beta2` | `0.95` | Adam beta2 |
| `adam_eps` | `1e-8` | Adam epsilon |
| `clip_grad` | `1.0` | Gradient clipping value |
| `load_checkpoint_optim_state` | `true` | Load optimizer state from checkpoint |
| `load_checkpoint_lr_scheduler` | `true` | Load LR scheduler from checkpoint |

### init

Weight initialization settings.

| Name | Default | Description |
|------|---------|-------------|
| `seed` | `1337` | Random seed |
| `init_std` | `0.006` | Initialization std dev |
| `xavier_init` | `false` | Use Xavier initialization |
| `data_parallel_random_init` | `false` | Enable DP random init |

### parallel

Distributed training configuration (mostly auto-configured from environment).

| Name | Default | Description |
|------|---------|-------------|
| `rank` | `-1` | Global rank (auto) |
| `local_rank` | `0` | Local rank (auto) |
| `world_size` | `1` | World size (auto) |
| `dist_backend` | `"nccl"` | Distributed backend |
| `timeout_minute` | `10.0` | Distributed timeout |
| `use_fsdp` | `false` | Enable FSDP |
| `fsdp_offload_params` | `false` | FSDP CPU offload |
| `fsdp_mixed_precision` | `"mixed"` | FSDP precision (`mixed`, `fp16`, `bf16`, `fp32`) |
| `fsdp_sharding_strategy` | `"full"` | Sharding strategy (`full`, `hybrid`, `no_shard`) |
| `fsdp_state_dict_type` | `"full"` | State dict type (`full`, `local`, `sharded`) |

### utils

Logging and profiling utilities.

| Name | Default | Description |
|------|---------|-------------|
| `log_level` | `"INFO"` | Log level |
| `tensorboard_dir` | `None` | TensorBoard log directory |
| `mlflow_tracking_uri` | `None` | MLflow tracking URI |
| `mlflow_experiment_name` | `None` | MLflow experiment name |
| `profile_nsys` | `false` | Enable Nsight Systems profiling |
| `profile_torch` | `false` | Enable PyTorch profiler |
| `profile_step_start` | `10` | Profile start step |
| `profile_step_end` | `12` | Profile end step |
| `profile_ranks` | `[0]` | Ranks to profile |
| `deterministic` | `false` | Enable deterministic mode |
| `report_memory_usage` | `true` | Report memory usage |

## Example Configuration

See `example.yaml` for a complete training configuration:

```yaml
trainer:
  micro_batch_size: 4
  train_batch_size: 480
  gradient_accumulation_steps: 60
  tensor_model_parallel_size: 1
  save_checkpoint_steps: 1000
  model_path: models/my_model
  use_flash_attn: true

operation:
  train_steps: 2000
  eval_interval: 1000
  activation_recompute: false

model: gpt2-small  # References configs/model/gpt2-small.yaml

data:
  config_path: configs/data/full_owt_pretrain.yaml
  task_type: pretrain

optim:
  optimizer: adam
  lr_scheduler: cosine
  max_lr: 6.0e-4
  min_lr: 6.0e-5
  warmup_steps: 100
  annealing_steps: 2000
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  clip_grad: 1.0

init:
  seed: 42
  init_std: 0.02
```

## Available Presets

### Model Configs (`configs/model/`)
- `gpt2-small.yaml` - GPT-2 Small (124M)
- `gpt2-medium.yaml` - GPT-2 Medium (355M)
- `gpt2-large.yaml` - GPT-2 Large (774M)
- `gpt2-xl.yaml` - GPT-2 XL (1.5B)
- `gpt3.yaml` - GPT-3 style config
- `llama.yaml` - LLaMA style config

### Data Configs (`configs/data/`)
- `pretrain_example.yaml` - Basic pretraining setup
- `full_owt_pretrain.yaml` - OpenWebText pretraining
- `sft_example.yaml` - Supervised fine-tuning example
- `dpo_example.yaml` - DPO training example
