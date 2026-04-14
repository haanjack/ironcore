# CLI Reference

IronCore provides 10 subcommands. Run `ironcore --help` for the full list.

```
ironcore <command> [options]
```

| Command | Description |
|---------|-------------|
| [`train`](#train--run-training) | Run training (pretrain, SFT, FIM, DPO, GRPO) |
| [`preprocess`](#preprocess--preprocess--inspect-datasets) | Tokenize and serialize datasets; inspect integrity |
| [`track`](#track--configure-logging-backends) | Patch YAML config with logging backend settings |
| [`evaluate`](#evaluate--run-evaluation-benchmarks) | Run eval benchmarks against a checkpoint |
| [`verify-step`](#verify-step--single-step-loss-verification) | Run 1 training step, report loss |
| [`verify-parity`](#verify-parity--parallelism-correctness-verification) | Compare loss curves across TP/DP/FSDP configs |
| [`profile`](#profile--profile-training-runs) | Profile training with mode presets |
| [`profile-mfu`](#profile-mfu--mfu-profiling) | Measure Model FLOP Utilization |
| [`analyze-scaling`](#analyze-scaling--scaling-analysis) | Run multi-scale training, fit scaling laws |
| [`gen-report`](#gen-report--generate-experiment-reports) | Generate markdown experiment reports |

## Core Commands

### `train` — Run Training

Starts a training run from a YAML config. Supports pretraining, SFT, FIM, DPO, and GRPO.

```bash
# Single GPU
ironcore train --config configs/example.yaml

# Tensor Parallel (2 GPUs)
torchrun --nproc_per_node 2 -m ironcore train --config configs/example.yaml

# Multi-node
torchrun --nproc_per_node 8 --nnodes 2 --node_rank 0 \
    --master_addr <IP> --master_port 29500 \
    -m ironcore train --config configs/example.yaml
```

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes | Path to training config YAML |

### `preprocess` — Preprocess & Inspect Datasets

Tokenizes and serializes datasets for training. Optional inspection mode checks integrity and prints statistics.

```bash
# Preprocess only
ironcore preprocess --config configs/data/pretrain_example.yaml

# Preprocess then inspect
ironcore preprocess --config configs/data/pretrain_example.yaml --inspect

# Inspect existing preprocessed files
ironcore preprocess --config configs/data/pretrain_example.yaml --only-inspect

# Inspect with sample preview
ironcore preprocess --config configs/data/pretrain_example.yaml --only-inspect --preview 5
```

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes | Path to data config YAML |
| `--inspect` | No | Run inspection after preprocessing |
| `--only-inspect` | No | Skip preprocessing, inspect only |
| `--preview` | No | Number of samples to preview (implies `--inspect`) |

## Experiment Tools

### `track` — Configure Logging Backends

Patches a training config YAML with logging backend settings. Supports TensorBoard, MLflow, and WandB. This only modifies the config file; the actual backend initialization happens when training starts.

```bash
# Interactive mode (prompts for each backend)
ironcore track --config configs/example.yaml

# Non-interactive: enable specific backends
ironcore track --config configs/example.yaml --backends wandb,tensorboard

# With backend-specific options
ironcore track --config configs/example.yaml \
    --backends wandb \
    --wandb-project my-project \
    --wandb-entity my-team

# Write patched config to file
ironcore track --config configs/example.yaml \
    --backends wandb --wandb-project my-project \
    --output configs/example_tracked.yaml
```

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes | Path to training config YAML |
| `--backends` | No | Comma-separated: `tensorboard`, `mlflow`, `wandb`. Interactive if omitted |
| `--wandb-project` | No | WandB project name |
| `--wandb-entity` | No | WandB entity/username |
| `--wandb-name` | No | WandB run name |
| `--mlflow-uri` | No | MLflow tracking URI |
| `--mlflow-experiment` | No | MLflow experiment name |
| `--tensorboard-dir` | No | TensorBoard log directory |
| `--output` | No | Write patched config to file (default: print snippet to stdout) |

### `evaluate` — Run Evaluation Benchmarks

Runs evaluation tasks against a trained checkpoint. Launches a training subprocess with `train_steps=0` and eval enabled.

```bash
# Default: HellaSwag
ironcore evaluate --config configs/example.yaml --checkpoint models/my_run

# Custom task and sample count
ironcore evaluate --config configs/example.yaml \
    --task hellaswag --num-samples 500

# Save results to JSON
ironcore evaluate --config configs/example.yaml \
    --checkpoint models/my_run --output eval_results.json
```

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes | Path to training config YAML |
| `--checkpoint` | No | Checkpoint path (overrides `trainer.model_path`) |
| `--task` | No | Eval task name (default: `hellaswag`) |
| `--num-samples` | No | Number of evaluation samples |
| `--batch-size` | No | Evaluation batch size |
| `--output` | No | Output file for results JSON |

### `verify-step` — Single Step Loss Verification

Runs exactly 1 training step and reports loss, grad norm, and timing. Useful for debugging and regression testing.

```bash
# Basic 1-step verification
ironcore verify-step --config configs/example.yaml

# With reference loss comparison
ironcore verify-step --config configs/example.yaml \
    --reference-loss 10.5432 --tolerance 0.01

# Verbose output (timing, memory, throughput)
ironcore verify-step --config configs/example.yaml --verbose

# Save results to JSON
ironcore verify-step --config configs/example.yaml \
    --verbose --output step_result.json
```

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes | Path to training config YAML |
| `--reference-loss` | No | Expected loss for comparison |
| `--tolerance` | No | Acceptable difference (default: `0.01`) |
| `--output` | No | Output file for results JSON |
| `--verbose` | No | Print grad norm, timing, throughput |

### `verify-parity` — Parallelism Correctness Verification

Compares loss curves across different parallelism configurations using the same seed. Verifies that TP, DP, and FSDP produce numerically equivalent results.

```bash
# Verify TP=1 matches TP=2 (default)
ironcore verify-parity --config configs/example.yaml --num-steps 10

# Verify FSDP on vs off
ironcore verify-parity --config configs/example.yaml --mode fsdp

# Custom TP sizes and tolerance
ironcore verify-parity --config configs/example.yaml \
    --mode tp --tp-sizes 1,2 --tolerance 1e-5 --num-steps 20

# Save results
ironcore verify-parity --config configs/example.yaml --output parity_results.json
```

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes | Base training config YAML |
| `--mode` | No | `tp`, `dp`, or `fsdp` (default: `tp`) |
| `--tp-sizes` | No | Comma-separated TP sizes for `tp` mode (default: `1,2`) |
| `--num-steps` | No | Steps per run (default: `10`) |
| `--tolerance` | No | Max acceptable loss difference (default: `1e-5`) |
| `--seed` | No | Random seed (default: `42`) |
| `--output` | No | Output file for results JSON |

### `profile` — Profile Training Runs

Wrapper around IronCore's built-in profiler with four mode presets.

```bash
# Quick: layer timing only
ironcore profile --config configs/example.yaml --mode quick

# Full: all profilers + traces
ironcore profile --config configs/example.yaml --mode full

# Communication profiling only
ironcore profile --config configs/example.yaml --mode comm

# Memory profiling
ironcore profile --config configs/example.yaml --mode memory

# Custom window
ironcore profile --config configs/example.yaml \
    --start-step 10 --end-step 20 --mode full
```

Mode presets:

| Mode | Features Enabled |
|------|-----------------|
| `quick` | Layer timing |
| `full` | Layer timing, torch profiler, GPU profiler, comm profiler, memory snapshot, Chrome trace, CSV export |
| `comm` | Communication profiler only |
| `memory` | Memory snapshot + OOM monitor |

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes | Path to training config YAML |
| `--mode` | No | `quick`, `full`, `comm`, `memory` (default: `quick`) |
| `--start-step` | No | Step to start profiling (default: `5`) |
| `--end-step` | No | Step to end profiling (default: `7`) |
| `--output-dir` | No | Output directory (default: `./logs/profile/`) |
| `--ranks` | No | Comma-separated ranks to profile (default: `0`) |
| `--train-steps` | No | Override total training steps (default: `end_step + 2`) |

### `profile-mfu` — MFU Profiling

Measures Model FLOP Utilization: achieved TFLOPS/s divided by hardware peak. Runs a warmup window followed by a measurement window.

```bash
# Default: 3 warmup + 5 measure steps
ironcore profile-mfu --config configs/example.yaml

# Custom windows and hardware peak
ironcore profile-mfu --config configs/example.yaml \
    --warmup-steps 5 --measure-steps 10 --hardware-peak 35.6

# Compare against previous run
ironcore profile-mfu --config configs/example.yaml \
    --compare previous_mfu.json --output current_mfu.json
```

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes | Path to training config YAML |
| `--warmup-steps` | No | Warmup steps before measurement (default: `3`) |
| `--measure-steps` | No | Steps to measure (default: `5`) |
| `--hardware-peak` | No | Hardware peak TFLOPS/s (default: `35.6` for RTX 3090 bf16) |
| `--output` | No | Output file for MFU results JSON |
| `--compare` | No | Previous MFU results JSON for comparison |

Output example:
```
MFU Profile Results
=======================================================
  Model:          gpt2-small (~124,438,272 params)
  Config:         TP=1, batch=128, seq=1024
  Hardware Peak:  35.6 TFLOPS/s

  Avg step time:  0.2340s (5 steps)
  Tokens/step:    131,072
  Throughput:     560,000 tokens/s

  Achieved:       18.50 TFLOPS/s/GPU
  MFU:            52.0%
```

### `analyze-scaling` — Scaling Analysis

Runs training at multiple model or batch sizes, collects final losses, and fits a Chinchilla-style power law. Optional matplotlib plots.

```bash
# Model scaling across mini configs
ironcore analyze-scaling --config configs/pretrain_micro.yaml \
    --scale-dimension model --model-sizes gpt2-micro,gpt2-tiny,gpt2-small-test \
    --num-steps 100

# Batch scaling
ironcore analyze-scaling --config configs/example.yaml \
    --scale-dimension batch --batch-sizes 32,64,128,256 \
    --num-steps 50

# With scaling law fit and plot
ironcore analyze-scaling --config configs/pretrain_micro.yaml \
    --model-sizes gpt2-micro,gpt2-tiny,gpt2-small-test --num-steps 100 --fit-law --plot
```

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes | Base training config YAML |
| `--scale-dimension` | No | `model`, `batch`, or `compute` (default: `model`) |
| `--model-sizes` | No | Comma-separated model names for model scaling |
| `--batch-sizes` | No | Comma-separated batch sizes for batch scaling |
| `--num-steps` | No | Steps per scale point (default: `100`) |
| `--output-dir` | No | Output directory (default: `experiments/scaling/`) |
| `--fit-law` | No | Fit power law `L(N) = aN^b + c` (default: true) |
| `--plot` | No | Generate scaling plot (requires matplotlib) |

### `gen-report` — Generate Experiment Reports

Generates a markdown report in `experiments/<category>/` from a template. Auto-fills metadata (git hash, date, config info).

```bash
# Basic report
ironcore gen-report --name "pretrain_convergence" --category pretrain \
    --config configs/pretrain_micro.yaml

# With analysis and conclusion
ironcore gen-report --name "tp_parity" --category parallelism \
    --config configs/example.yaml \
    --objective "Verify TP=2 matches TP=1" \
    --status PASS

# Interactive mode (prompts for fields)
ironcore gen-report --name "sft_eval" --category sft \
    --config configs/sft_small.yaml \
    --interactive
```

Categories: `pretrain`, `sft`, `dpo`, `grpo`, `scaling`, `parallelism`, `mfu`, `profile`

| Flag | Required | Description |
|------|----------|-------------|
| `--name` | Yes | Experiment name |
| `--category` | Yes | Experiment category |
| `--config` | No | Training config for metadata extraction |
| `--checkpoint-dir` | No | Checkpoint directory path |
| `--log-dir` | No | Log directory path |
| `--output-dir` | No | Output directory (default: `experiments/`) |
| `--status` | No | `PASS`, `FAIL`, `PARTIAL`, `PENDING` (default: `PENDING`) |
| `--objective` | No | Experiment objective |
| `--analysis` | No | Analysis text |
| `--conclusion` | No | Conclusion text |
| `--interactive` | No | Prompt for all fields |

Report template:

Reports are written to `experiments/<category>/<name>.md` and include:

- Metadata (category, date, git commit, config path)
- Objective
- Methodology (model, hardware, software, parallelism, hyperparameters)
- Results (training curves, key metrics, comparisons)
- Analysis
- Conclusion (status, criteria, next steps)
- Artifacts (paths to configs, checkpoints, logs, profiles)

## Mini Model Configs

Three small model configs for quick iteration:

| Config | Layers | d_model | d_ffn | Heads | ~Params | Use Case |
|--------|--------|---------|-------|-------|---------|----------|
| `gpt2-micro` | 2 | 256 | 1024 | 4 | ~2M | Debugging, 1-step verification |
| `gpt2-tiny` | 4 | 512 | 2048 | 8 | ~10M | Quick validation runs |
| `gpt2-small-test` | 8 | 768 | 3072 | 12 | ~40M | Short training experiments |

Usage: set `model: gpt2-micro` (or `gpt2-tiny`, `gpt2-small-test`) in your training config.

## Experiment Configs

Ready-to-use configs in `configs/`:

| Config | Task | Model | Dataset | Steps |
|--------|------|-------|---------|-------|
| `pretrain_micro.yaml` | Pretrain | gpt2-micro | OpenWebText 10K | 1000 |
| `pretrain_tiny.yaml` | Pretrain | gpt2-tiny | OpenWebText 10K | 2000 |
| `pretrain_small.yaml` | Pretrain | gpt2-small-test | OpenWebText 10K | 3000 |
| `sft_small.yaml` | SFT | gpt2-small-test | UltraChat 5K | 1000 |
| `dpo_small.yaml` | DPO | gpt2-small-test | HH-RLHF 5K | 500 |
| `grpo_small.yaml` | GRPO | gpt2-small-test | GSM8K 1K | 200 |
| `lora_sft_small.yaml` | SFT + LoRA | gpt2-small-test | UltraChat 5K | 1000 |
