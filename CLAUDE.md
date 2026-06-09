# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

For contribution guidelines (container setup, coding standards, testing policy, performance
guide, PR workflow), see [CONTRIBUTING.md](CONTRIBUTING.md). Full documentation lives under
[`docs/`](docs/) and builds into an MkDocs site (`mkdocs serve`).

## Commands

```bash
# Install (inside the NGC container — see CONTRIBUTING.md for container setup)
# Flash attention is only available inside the container; integration tests require it.
pip install -e .[dev]
pip install -e .[docs]   # MkDocs (mkdocs serve / mkdocs build)

# Lint and format
ruff check ironcore/          # check errors
ruff format ironcore/         # auto-format
ruff check --fix ironcore/    # auto-fix fixable errors

# Run tests (e2e excluded by default; profiler/multi_gpu via --ignore in pyproject.toml)
pytest tests/                                                     # default: unit + regression (CPU OK)
pytest tests/unit/attention/test_attention.py -v                 # single file
pytest -m "not cuda and not mp and not e2e and not hf_hub" tests/ # CPU-only (GitHub Actions)
pytest -m "cuda and not mp and not e2e" tests/                    # 1-GPU tests
./scripts/run_tests.sh                                            # full suite incl. multi-GPU (torchrun)

# Train
ironcore train --config configs/<name>.yaml
torchrun --nproc_per_node 4 -m ironcore train --config configs/<name>.yaml
```

Testing details (markers, fixtures): [tests/test_suite.md](tests/test_suite.md).
CI/CD setup & GPU runner registration: [docs/ci_cd_guide.md](docs/ci_cd_guide.md).

## Architecture

### Package layout

```
ironcore/
├── config/          # All dataclass configs (ModelConfig, TrainerConfig, etc.)
├── models/          # TransformerModel — the single model class for all architectures
├── layers/          # Attention, MLP, embedding, MoE, KV cache primitives
├── trainers/        # BaseTrainer → LanguageModelTrainer / DPOTrainer / GRPOTrainer
├── alignment/       # GRPO rollout, DPO/GRPO loss, reward system, buffer, dataset
├── optimizer/       # Muon + AdamW hybrid, LR scheduler, DistributedOptimizer (ZeRO-1)
├── parallel/        # TP/DP/EP process group init + grad_norm utilities
├── peft/            # LoRA adapters (TP-correct, replicated not sharded)
├── checkpointing/   # Native and HuggingFace-interop save/load
├── dataloader/      # Pretrain/SFT/FIM/DPO/GRPO dataset and collator
├── eval/            # Perplexity and HellaSwag evaluators
├── offload/         # ExecutionScheduler — optimizer/weight/activation offload
├── preprocessing/   # Data serialization (bin packing, tokenization pipelines)
├── tokenizer/       # BBPE and tiktoken tokenizer implementations
├── utils/           # mfu.py, memory.py, timer.py, device.py, config.py, profiling.py
└── cli/             # 15 CLI subcommands (registry.py); see docs/cli_reference.md
```

### Invariants that aren't obvious from the code

**Config-driven everything.** Architecture, parallelism, and training mode are selected via
YAML parsed into the dataclasses in `ironcore/config/` — not code changes.

**Single model class.** `TransformerModel` (`ironcore/models/transformer.py`) covers GPT-2/3,
LLaMA, Gemma, Qwen, Phi via config flags (`attention_type` MHA/GQA/MQA, `norm_type`, positional
embedding style, etc.).

**Parallelism initialization order — must be followed exactly:**
1. `initialize_process()` — distributed setup
2. `initialize_model_parallel()` — TP/DP groups
3. `initialize_expert_parallel()` — EP groups (MoE only)
4. Build model and optimizer
5. `initialize_parallelism()` — DDP/FSDP wrapping

**LoRA is replicated, not sharded.** Adapter weights are identical across TP ranks; TP-correctness
is handled at the column/row parallel layer boundaries. See [docs/peft_guide.md](docs/peft_guide.md).

### Subsystem → doc

| Subsystem | One-liner | Doc |
| --- | --- | --- |
| Getting started | Install, CLI, first run | [getting_started](docs/getting_started.md) |
| CLI | 15 subcommands; add one via `ironcore/cli/registry.py` + `cli/<name>.py` | [cli_guide](docs/cli_guide.md) · [cli_reference](docs/cli_reference.md) |
| Parallelism | TP/EP/DP, multi-node, FSDP, `DistributedOptimizer` vs FSDP | [parallelism](docs/parallelism.md) |
| Trainers | `BaseTrainer` lifecycle, grad accumulation, mixed precision | [trainers](docs/trainers.md) |
| Optimizer | Muon (Newton-Schulz) + AdamW hybrid, ZeRO-1 sharding | [optimizer](docs/optimizer.md) |
| Dataloader | Streaming datasets, bin-packing SFT collator, FIM | [dataloader](docs/dataloader.md) · [multi_dataset_sft](docs/multi_dataset_sft.md) |
| Checkpointing | Native (universal/distributed TP) + HuggingFace interop | [checkpointing](docs/checkpointing.md) |
| Inference & KV cache | Prefill/decode, `KVCacheManager`, paged + prefix caching | [inference](docs/inference.md) |
| Eval | HellaSwag + perplexity, pluggable evaluators | [eval](docs/eval.md) |
| PEFT / LoRA | TP-correct replicated adapters | [peft_guide](docs/peft_guide.md) |
| Alignment | Offline DPO + online GRPO rollout training | [alignment](docs/alignment.md) |
| Rewards (GRPO) | `RewardManager` registry; rule/template + model backends | [reward_manager](docs/reward_manager.md) |
| Offload | Optimizer/weight/activation offload via `ExecutionScheduler` | [offload](docs/offload.md) · [design](docs/design/offload.md) |

### Test structure

By execution requirements: `unit/` (CPU), `integration/` (single-GPU, torchrun),
`multi_gpu/` (2+ GPU, torchrun), `regression/`, `property/`. Fixtures and smoke-test configs in
`tests/fixtures/`. See [tests/test_suite.md](tests/test_suite.md) for markers and adding tests.
