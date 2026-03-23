# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
pip install -e .

# Lint and format
ruff check ironcore/          # check errors
ruff format ironcore/         # auto-format
ruff check --fix ironcore/    # auto-fix fixable errors

# Run tests (e2e excluded by default via pyproject.toml)
pytest tests/unit/                                    # unit tests only (no GPU)
pytest tests/                                         # all non-e2e tests
pytest tests/unit/attention/test_attention.py -v      # single file
pytest tests/unit/attention/test_attention.py::TestAttention::test_forward -v  # single test
pytest -m "not cuda and not distributed" tests/unit/  # skip GPU/distributed tests
pytest -m integration tests/                          # integration tests only

# Train
ironcore train --config configs/<name>.yaml
torchrun --nproc_per_node 4 -m ironcore train --config configs/<name>.yaml
```

## Architecture

### Package layout

```
ironcore/
├── config/          # All dataclass configs (ModelConfig, TrainerConfig, etc.)
├── models/          # TransformerModel — the single model class for all architectures
├── layers/          # Attention, MLP, embedding, MoE, KV cache primitives
├── trainers/        # BaseTrainer → LMTrainer / DPOTrainer / GRPOTrainer
├── alignment/       # GRPO rollout, DPO/GRPO loss, reward system, buffer, dataset
├── optimizer/       # Muon + AdamW hybrid, LR scheduler, distributed optimizer
├── parallel/        # TP/DP/EP process group init + grad_norm utilities
├── peft/            # LoRA adapters (TP-correct, replicated not sharded)
├── checkpointing/   # Native and HuggingFace-interop save/load
├── dataloader/      # Pretrain/SFT/FIM/DPO/GRPO dataset and collator
├── eval/            # Perplexity and HellaSwag evaluators
├── utils/           # mfu.py, memory.py, timer.py, device.py, config.py, profiling.py
└── cli/             # train and preprocess CLI entry points
```

### Key patterns

**Config-driven everything.** All subsystems are configured through YAML files parsed into dataclasses in `ironcore/config/`. Changing model architecture, parallelism strategy, or training mode is done via config, not code.

**Single model class.** `TransformerModel` in `ironcore/models/` covers GPT-2/3, LLaMA, Gemma, Qwen, Phi via config flags (`attention_type`: MHA/GQA/MQA, `norm_type`, positional embedding style, etc.).

**Parallelism initialization order.** Must be followed exactly:
1. `initialize_process()` — distributed setup
2. `initialize_model_parallel()` — TP/DP groups
3. `initialize_expert_parallel()` — EP groups (MoE only)
4. Build model and optimizer
5. `initialize_parallelism()` — DDP/FSDP wrapping

**LoRA is replicated, not sharded.** LoRA adapter weights are identical across all TP ranks. TP-correct integration is handled at the column/row parallel layer boundaries. See `docs/peft_guide.md`.

**Reward system (GRPO).** `RewardManager` is a weighted registry of `RewardFunction` implementations. Rule-based rewards use `TemplateRuleReward` with YAML-defined templates (modes: `answer_match`, `tag_check`, `regex_match`). Model-based rewards use `RewardModelFunction` with pluggable backends. See `docs/reward_manager.md`.

**Distributed optimizer vs FSDP.** The `DistributedOptimizer` in `ironcore/optimizer/` is an alternative to FSDP for optimizer state sharding — shards optimizer states across DP ranks without changing model weight layout. Use when you need ZeRO-1 style sharding without full FSDP wrapping.

### Test structure

```
tests/
├── fixtures/        # Shared fixtures: config_fixtures.py, model_fixtures.py, utils.py, mocks.py
├── unit/            # Fast, no-GPU tests organized by feature (alignment, attention, moe, optimizer, …)
├── integration/     # Multi-component tests organized by feature
├── multi_gpu/       # Tests requiring multiple GPUs (torchrun)
├── regression/      # Tests pinning specific bug fixes
├── property/        # Property-based tests (FIM invariants)
└── benchmarks/      # Performance benchmarks (not collected by pytest by default)
```

Config helpers in `tests/fixtures/config_fixtures.py` (`create_small_test_config`, `create_moe_test_config`, `create_lora_test_config`, `create_grpo_test_config`, etc.) are the standard way to build test configs — prefer them over constructing `ModelConfig` directly.

### CI

GitHub Actions runs `ruff check` and `ruff format --check` on push to `main` (Python 3.10 and 3.12).
