# Test Suite

## Quick Start for Developers

### Directory Structure

```
tests/
├── fixtures/                    # Reusable test configs and helpers
│   ├── config_fixtures.py       # Config builders: create_small_test_config(), create_moe_test_config(), etc.
│   ├── model_fixtures.py        # Model fixtures
│   ├── configs/                 # Test YAML configurations
│   │   ├── model/               # Test models (e.g., qwen2.5-0.5B.yaml)
│   │   ├── data/                # Dataset configs (e.g., grpo_gsm8k.yaml)
│   │   └── *.yaml               # Smoke test configs (grpo_gsm8k_smoke_*.yaml)
│   └── utils.py, mocks.py       # Utility functions
├── unit/                        # Fast logic tests (no GPU, ~5 min)
│   ├── alignment/               # GRPO, DPO loss tests
│   ├── attention/               # Attention mechanism tests
│   ├── checkpointing/           # Save/load functionality
│   ├── dataloader/              # Dataset & preprocessing
│   ├── layers/                  # Component layers (MLP, embedding, etc.)
│   ├── models/                  # Model architecture
│   ├── moe/                     # Mixture of Experts
│   ├── optimizer/               # Optimizer & scheduler
│   ├── parallel/                # TP/DP/EP initialization
│   ├── peft/                    # LoRA adapter tests
│   ├── reward/                  # Reward system (unit + E2E GRPO)
│   ├── sequence/                # Sequence generation
│   ├── tokenizer/               # Tokenization
│   ├── trainer/                 # Training loop logic
│   └── utils/                   # Utility functions
├── integration/                 # Multi-component tests (requires GPU)
│   ├── alignment/               # GRPO/DPO training
│   ├── attention/               # Attention variants
│   ├── dataloader/              # Dataset integration
│   ├── eval/                    # Evaluator integration
│   ├── kvcache/                 # KV cache with training
│   ├── lora/                    # LoRA training
│   ├── memory/                  # Memory optimization
│   ├── moe/                     # MoE routing
│   ├── optimizer/               # Optimizer TP/FSDP integration
│   └── parallelism/             # Parallel state correctness
├── multi_gpu/                   # Multi-GPU tests (2+ GPUs with torchrun)
│   ├── test_distributed_optimizer.py
│   ├── test_expert_parallel.py
│   └── ...
├── regression/                  # Bug fix validation tests
├── property/                    # Property-based tests (invariant validation)
└── benchmarks/                  # Performance benchmarks (not auto-collected)
```

### Pytest Markers System

All 10 markers are the single source of truth — registered in `pyproject.toml` with `--strict-markers` enforced.

**Execution-resource markers** (control CI scheduling):

| Marker | Meaning |
|--------|---------|
| `@pytest.mark.cuda` | Requires a CUDA/GPU device; conftest auto-skips if unavailable |
| `@pytest.mark.mp` | Requires 2+ GPUs and `torchrun`; conftest auto-skips if `<2` GPUs |
| `@pytest.mark.hf_hub` | Downloads from HuggingFace Hub (network required); excluded from CPU CI |
| `@pytest.mark.e2e` | Expensive end-to-end test (~10 min, spawns torchrun internally); opt-in only |

**Character / pipeline markers** (describe what is being tested):

| Marker | Meaning |
|--------|---------|
| `@pytest.mark.smoke` | Pipeline liveness check — just enough steps to confirm nothing crashes |
| `@pytest.mark.pretrain` | Pre-training LM pipeline |
| `@pytest.mark.sft` | Supervised fine-tuning pipeline |
| `@pytest.mark.dpo` | DPO alignment pipeline |
| `@pytest.mark.grpo` | GRPO online RL pipeline (rollout, reward, advantage, policy update) |
| `@pytest.mark.checkpointing` | Checkpoint save/load — native, HF interop, distributed |

**Key design points:**
- `grpo` marks any test touching the GRPO pipeline (cheap math tests **and** expensive training tests).
- `e2e` is a separate gate for the expensive ones. Default `addopts` excludes only `e2e`, so cheap `grpo` math tests run in every CI run.
- `mp` tests are auto-skipped by `tests/conftest.py` unless launched under a real `torchrun` (checked via the `TORCHELASTIC_RUN_ID` env var torchrun's elastic agent sets — not `RANK`, which other tests can leave set in the process even outside torchrun), so they safely skip under plain `pytest` and are exercised by the `distributed-tests` CI job via per-file `torchrun`. No per-file skip guard is required, though a couple of files keep a redundant local one.

**Marker selection by example:**

```python
import pytest

# CPU-only logic test — no marker needed (runs by default)
def test_advantage_normalization(): ...

# Single-GPU test
@pytest.mark.cuda
def test_attention_forward_cuda(): ...

# 2-GPU test (conftest.py auto-skips unless launched via torchrun; no
# per-file skipif needed)
@pytest.mark.mp
def test_tensor_parallel(): ...

# Expensive E2E test (spawns torchrun internally)
@pytest.mark.grpo
@pytest.mark.e2e
@pytest.mark.mp
def test_grpo_full_training(): ...

# HuggingFace-dependent test (excluded from CPU CI)
@pytest.mark.hf_hub
def test_hf_weight_roundtrip(): ...

# Checkpoint test found across multiple dirs
@pytest.mark.cuda
@pytest.mark.checkpointing
def test_checkpoint_roundtrip(): ...
```

**Common filter commands:**
```bash
# CPU-only (no GPU, no network, no e2e)
pytest tests/ -m "not cuda and not mp and not e2e and not hf_hub"

# Single-GPU tests
pytest tests/ -m "cuda and not mp and not e2e"

# Run all DPO tests
pytest tests/ -m dpo

# Run all checkpointing tests
pytest tests/ -m checkpointing

# E2E tests (opt-in, expensive)
pytest tests/ -m e2e
```

### Adding New Tests

1. **Choose location** based on what you're testing:
   - Logic test → `tests/unit/[feature]/test_*.py`
   - Multi-component → `tests/integration/[feature]/test_*.py`
   - Multi-GPU → `tests/multi_gpu/test_*.py`

2. **Add appropriate markers** (see table above for all 10):
   - CPU logic test → no marker (runs in default `pytest tests/`)
   - Single GPU → `@pytest.mark.cuda`
   - 2+ GPU with torchrun → `@pytest.mark.mp` (auto-skipped by `tests/conftest.py` unless run under `torchrun`)
   - Expensive E2E → `@pytest.mark.e2e` (+ pipeline marker: `grpo`, `dpo`, etc.)
   - HuggingFace download → `@pytest.mark.hf_hub`
   - Pipeline feature → `@pytest.mark.pretrain / sft / dpo / grpo`
   - Checkpoint path → `@pytest.mark.checkpointing`

3. **Use shared fixtures:**
   ```python
   from tests.fixtures.config_fixtures import create_small_test_config

   def test_my_feature(create_small_test_config):
       config = create_small_test_config()
       # ... test code
   ```

4. **Run and verify:**
   ```bash
   # Locally (CPU only)
   pytest tests/unit/path/test_file.py::TestClass::test_method -v

   # With GPU (if available)
   pytest tests/ -m cuda -v -k test_method
   ```

### CI/CD Integration

Four jobs in `.github/workflows/test.yml`:

| Job | Trigger | Runner | Filter | Duration |
|-----|---------|--------|--------|----------|
| `logic-tests` | Every PR + push to main | ubuntu-latest | `-m "not cuda and not mp and not e2e and not hf_hub"` | ~5 min |
| `gpu-tests` | Every PR + push to main | self-hosted GPU | `-m "cuda and not mp and not e2e"` | ~15 min |
| `distributed-tests` | Push to main + manual dispatch | self-hosted GPU | per-file `torchrun` on `mp` files | ~10 min |
| `e2e-tests` | Manual dispatch only | self-hosted GPU | `-m "e2e"` | ~10 min |

See [docs/ci_cd_guide.md](../docs/ci_cd_guide.md) for workflow configuration and runner setup.

---

## Overview

| Tier | Runner | GPU | Runtime |
|------|--------|-----|---------|
| Unit + Regression (default) | `pytest tests/` | No | ~5 min |
| Single-GPU integration | `pytest -m "cuda and not mp and not e2e"` | 1 | ~15 min |
| Distributed (`mp` tests) | per-file `torchrun --nproc_per_node=2` | 2 | ~10 min |
| E2E (opt-in) | `pytest -m "e2e"` | 2 | ~10 min |

### Execution

```bash
# Default: unit + regression (CPU OK)
pytest tests/

# Single-GPU integration tests
pytest tests/ -m "cuda and not mp and not e2e"

# Specific pipeline filter
pytest tests/ -m dpo       # all DPO tests
pytest tests/ -m grpo      # GRPO pipeline (cheap math + training)
pytest tests/ -m e2e       # expensive E2E only (opt-in)

# Full suite including multi-GPU (run inside container)
./scripts/run_tests.sh              # full suite
./scripts/run_tests.sh --quick      # skip multi-GPU
./scripts/run_tests.sh --e2e        # add E2E smoke tests
```

### pytest Configuration

Default `addopts` in `pyproject.toml`:

```
-v --tb=short -m 'not e2e' --strict-markers --ignore=tests/unit/profiler/ --ignore=tests/multi_gpu/
```

`--strict-markers` prevents ghost markers — any unregistered marker usage is an immediate error. `mp` tests in `tests/multi_gpu/` are ignored by default and exercised via per-file `torchrun` in the `distributed-tests` CI job.

---

## Unit Tests (405 passed, 0 skipped)

Run with `pytest tests/`. No GPU required.

### Alignment (42 tests)

| File | Class | Tests | Description |
|------|-------|-------|-------------|
| `test_dpo_loss.py` | TestComputeLogSoftmaxTPSafe | 3 | TP-safe log softmax shape, normalization, CPU error |
| `test_dpo_loss.py` | TestExtractLogpsFromLogProbs | 3 | Log prob extraction, ignore index, mask handling |
| `test_dpo_loss.py` | TestComputeLogps | 2 | Basic computation, loss mask |
| `test_dpo_loss.py` | TestDpoLoss | 11 | Scalar output, metrics, beta, label smoothing, gradient flow |
| `test_dpo_loss.py` | TestDpoLossEdgeCases | 3 | Single batch, all ignore, extreme beta |
| `test_grpo_math.py` | TestAdvantageNormalization | 5 | Sum-to-zero, std=1, identical rewards, single element, formula |
| `test_grpo_math.py` | TestKLDivergence | 5 | Zero KL, positive KL, masked KL, numerical stability, logits |
| `test_grpo_math.py` | TestGRPOLoss | 3 | Loss components, positive advantage, metrics |
| `test_grpo_math.py` | TestGrpoLossEntropy | 6 | Entropy args, coef=0, raw mean, bonus reduces loss, None skips |
| `test_grpo_math.py` | TestTPSafeSoftmax | 1 | TP-safe log softmax single rank |

### Attention (49 tests)

| File | Class | Tests | Description |
|------|-------|-------|-------------|
| `test_attention.py` | TestAttentionBasics | 4 | Forward shape, init, causal mask, no mask |
| `test_attention.py` | TestAttentionWithDifferentConfigs | 3 | GQA, MHA, MQA configs |
| `test_attention.py` | TestAttentionVaryingDimensions | 3 | Batch size, head dim, sequence length |
| `test_attention.py` | TestAttentionGradients | 1 | Gradient flow |
| `test_attention.py` | TestTransformerLayerBasics | 4 | Forward shape, gradient, RoPE, init |
| `test_attention.py` | TestTransformerLayerWithDifferentConfigs | 3 | GQA, MHA, MQA layers |
| `test_attention.py` | TestAttentionStandardVsFlash | 6 | Flash gradient flow (GQA, MQA), availability, forward comparison |
| `test_attention.py` | TestAttentionMemory | 1 | Memory usage |
| `test_rotary.py` | TestRoPEBasics | 4 | Init, sin/cos cache, values, theta frequencies |
| `test_rotary.py` | TestRoPERotation | 5 | Reference comparison, positions, formula, shape preservation |
| `test_rotary.py` | TestRoPEEdgeCases | 5 | Cache extension, custom positions, odd dim, offset, scale |
| `test_rotary.py` | TestRoPENumericalPrecision | 5 | bf16, fp16, fp32, large values, norm preservation |
| `test_rotary.py` | TestRoPEGradients | 3 | Correctness, existence, non-zero |
| `test_rotary.py` | TestRoPEHuggingFaceComparison | 1 | vs HuggingFace LLaMA RoPE |
| `test_rotary.py` | TestRoPEIntegration | 1 | RoPE with attention QK only |

### Checkpointing (20 tests)

| File | Class | Tests | Description |
|------|-------|-------|-------------|
| `test_hf_interop.py` | TestArchitectureDetection | 4 | GPT-2, LLaMA, LLaMA family, unknown default |
| `test_hf_interop.py` | TestGPT2WeightMapping | 4 | HF-to-ironcore mapping, QKV split, transpose, roundtrip |
| `test_hf_interop.py` | TestLLaMAWeightMapping | 4 | HF-to-ironcore mapping, KV fusion, gate/up fusion, roundtrip |
| `test_hf_interop.py` | TestGPT2Integration | 3 | Load state dict, weight conversion, roundtrip |
| `test_hf_interop.py` | TestLLaMAIntegration | 3 | Load state dict, weight conversion, roundtrip |
| `test_hf_interop.py` | TestExportFunctionality | 2 | Export creates config and files |

### Dataloader (53 tests)

| File | Class | Tests | Description |
|------|-------|-------|-------------|
| `test_fim_config.py` | TestFIMConfigFields | 4 | Fields present, defaults |
| `test_fim_config.py` | TestFIMConfigValidation | 2 | Rate range, split type values |
| `test_fim_config.py` | TestFIMYAMLParsing | 3 | YAML parsing, defaults, custom |
| `test_fim_config.py` | TestFIMConfigConsistency | 1 | Dataclass vs YAML consistency |
| `test_fim_edge_cases.py` | TestFIMShortSequences | 4 | Single token, below threshold, boundary, above threshold |
| `test_fim_edge_cases.py` | TestFIMEmptySequences | 2 | Empty, zero length |
| `test_fim_edge_cases.py` | TestFIMRepeatedTokens | 2 | Repeated, alternating |
| `test_fim_edge_cases.py` | TestFIMEdgeSplits | 2 | Boundary splits, near end |
| `test_fim_edge_cases.py` | TestFIMEmptyMiddlePrevention | 2 | Empty middle, consecutive splits |
| `test_fim_edge_cases.py` | TestFIMUnicodeHandling | 2 | Unicode, emoji |
| `test_fim_edge_cases.py` | TestFIMLargeSequences | 2 | Large, very large |
| `test_fim_token_lookup.py` | TestGetTokenIDSuccess | 2 | Success, not UNK |
| `test_fim_token_lookup.py` | TestGetTokenIDErrors | 3 | Missing, all missing, error message |
| `test_fim_token_lookup.py` | TestGetTokenIDCustomTokens | 1 | Custom tokens |
| `test_fim_token_lookup.py` | TestGetTokenIDEdgeCases | 3 | Empty, whitespace, nonexistent |
| `test_fim_token_lookup.py` | TestTiktokenLimitation | 2 | Unsupported documented, HF tokenizer required |
| `test_fim_token_lookup.py` | TestGetTokenIDConsistency | 2 | Consistent calls, matches tokenizer |
| `test_fim_transformation.py` | TestFIMPSMFormat | 3 | PSM format, token order, count |
| `test_fim_transformation.py` | TestFIMTokenConservation | 4 | Conservation, no duplication, reconstruction, length |
| `test_fim_transformation.py` | TestFIMLengthInvariant | 2 | Length invariant, various sizes |
| `test_fim_transformation.py` | TestFIMSplitRandomness | 2 | Uniform, different seeds |
| `test_fim_transformation.py` | TestFIMDeterminism | 2 | Deterministic seed, multiple calls |
| `test_fim_transformation.py` | TestFIMNonEmptySections | 1 | All sections nonempty |

### KV Cache (10 tests)

| File | Tests | Description |
|------|-------|-------------|
| `test_kv_cache_basic.py` | 10 | Single token caching, multi-token, batch independence, reset, statistics, per-sequence positions, selective reset, position divergence |

### MoE (50 tests)

| File | Class | Tests | Description |
|------|-------|-------|-------------|
| `test_expert.py` | TestExpertMLP | 9 | Shape, forward/backward, varying tokens, activations, bias, dropout, ID tracking |
| `test_load_balance_loss.py` | TestComputeLoadBalanceLoss | 5 | Basic loss, perfect balance, alpha scaling, gradient flow, single token |
| `test_load_balance_loss.py` | TestComputeRouterZLoss | 4 | Basic Z loss, large logits, weight scaling, gradient flow |
| `test_load_balance_loss.py` | TestLoadBalanceLoss | 3 | Forward without/with Z loss, backward |
| `test_load_balance_loss.py` | TestGetExpertUtilization | 4 | Basic, perfect balance, all-to-one, single token |
| `test_load_balance_loss.py` | TestEdgeCases | 5 | Empty batch, large experts, high top-k, zero alpha, deterministic |
| `test_moe_utils.py` | TestFlattenMoeInputs | 4 | Basic flatten, batch/seq variations, value preservation |
| `test_moe_utils.py` | TestValidateMoeInput | 8 | Valid input, wrong ndim/hidden/empty/NaN/Inf, custom name |
| `test_router.py` | TestTopKRouter | 8 | Shapes, weight sum, valid range, top-k, jitter, determinism, gradient flow |

### Optimizer (39 tests)

| File | Class | Tests | Description |
|------|-------|-------|-------------|
| `test_distributed_optimizer.py` | TestDistributedOptimizerSingleGPU | 10 | Init, step, param groups, delegation, zero_grad, state dict, load, isinstance, repr |
| `test_muon_optimizer.py` | TestNewtonSchulz | 7 | Identity, random, tall/wide matrices, numerical stability, bf16 |
| `test_muon_optimizer.py` | TestMuonParameterClassification | 9 | Attention (muon), MLP (muon), embedding/output/bias/LN/pos (adamw), non-2D |
| `test_muon_optimizer.py` | TestMuonOptimizerStep | 6 | Momentum init, AdamW init, step, weight decay, state dict, zero_grad |
| `test_muon_optimizer.py` | TestMuonOptimizerEdgeCases | 5 | Empty params, muon-only, adamw-only, sparse gradient, repr |
| `test_muon_optimizer.py` | TestMuonVsAdamBehavior | 2 | Nesterov momentum, orthogonalization effect |

### Parallel (24 tests)

| File | Class | Tests | Description |
|------|-------|-------|-------------|
| `test_comm_grad_correctness.py` | TestAllReduceEPGradient | 4 | Forward single, EP size 1, backward, grad function |
| `test_comm_grad_correctness.py` | TestDispatchGatherGradient | 2 | Dispatch structure, combine outputs |
| `test_comm_grad_correctness.py` | TestAutogradFunctions | 2 | Save for backward, no grad for EP size |
| `test_comm_grad_correctness.py` | TestNumericalStability | 3 | Small, large, mixed values |
| `test_expert_parallel.py` | TestExpertParallelStates | 3 | Single GPU, rank calc, destroy/reinit |
| `test_expert_parallel.py` | TestExpertParallelCommunication | 3 | All-reduce single GPU, with grad, gradient correctness |
| `test_expert_parallel.py` | TestMoEWithEP | 2 | Config, gradient with EP |
| `test_grad_norm.py` | (module) | 5 | Basic clip [2.0, inf], no gradients, single tensor, clipping effect |

### Reward (84 tests)

| File | Class | Tests | Description |
|------|-------|-------|-------------|
| `test_reward_manager.py` | TestTemplateRuleRewardAnswerMatch | 11 | Hash/boxed/answer patterns, wrong answer, no answer, normalization |
| `test_reward_manager.py` | TestTemplateRuleRewardTagCheck | 4 | All tags, missing, partial, custom scoring |
| `test_reward_manager.py` | TestTemplateRuleRewardRegexMatch | 4 | Pattern match, no match, DOTALL, custom scoring |
| `test_reward_manager.py` | TestTemplateRuleRewardEdgeCases | 4 | Missing mode, unknown mode, empty, nonexistent path |
| `test_reward_manager.py` | TestRewardModelFunction | 7 | Init backends, local inference, endpoint retry/score, API mocked |
| `test_reward_manager.py` | TestRewardManager | 11 | Register, weighted sum, error, from_config (rule, model, math, composite, keyword), unknown type |
| `test_reward_manager.py` | TestConfigDataclasses | 6 | Dict conversion, entry list, alignment config, YAML |
| `test_reward_manager.py` | TestYAMLTemplates | 4 | Math, COT, DeepSeek YAML loading, validation |
| `test_reward_manager.py` | TestBuiltinRewardFunctions | 8 | Math (strict/normal), keyword, soft keyword, format, strict format, code NotImplementedError |
| `test_reward_manager.py` | TestErrorHandling | 5 | Nonexistent path, invalid mode, empty functions, unreachable endpoint, malformed regex |
| `test_reward_manager.py` | TestRewardWeightEdgeCases | 3 | Equal weights sum, zero weight ValueError, code without test_cases |
| `test_reward_manager.py` | TestDeepSeekFormatToken | 3 | Rejects currwork, YAML uses think tag, YAML rejects currwork |
| `test_reward_manager.py` | TestLocalEndpointRewardCache | 5 | Cache hit, LRU eviction, size zero/negative clamped, OrderedDict |
| `test_reward_manager.py` | TestLocalInferenceRewardCache | 4 | Size zero clamped, OrderedDict, LRU overflow, absent vocab tokens |
| `test_reward_manager.py` | TestIntegration | 3 | Weighted sum, custom YAML, worker pool batch |
| `test_reward_manager.py` | TestImports | 2 | Reward package exports, alignment exports |

### Trainers (9 tests)

| File | Class | Tests | Description |
|------|-------|-------|-------------|
| `test_trainer_norm.py` | TestBaseTrainerNormComputation | 4 | Returns float, no clipping, param norm, mock |
| `test_trainer_norm.py` | TestDPOTrainerNormComputation | 2 | DPO clip, no clip |
| `test_trainer_norm.py` | TestGRPOTrainerNormComputation | 2 | GRPO clip, no clip |
| `test_trainer_norm.py` | TestNormComputationIntegration | 1 | All trainers use same clip_grad_norm |

### Cross-cutting (16 tests)

| File | Class | Tests | Description |
|------|-------|-------|-------------|
| `test_imports.py` | TestMFUImports | 3 | Utils submodule, re-exported from init, public API |
| `test_imports.py` | TestParallelImports | 2 | Deferred import, TP package re-export |
| `test_imports.py` | TestTrainerModuleLoad | 3 | Base/GRPO trainer load, canonical import path |
| `test_mfu.py` | TestMFUCalculator | 6 | Init basic/GQA, parameter count, TFLOPS basic/multi-GPU, from config |
| `test_mfu.py` | TestConvenienceFunction | 1 | compute_tflops function |
| `test_mfu.py` | TestMFUResult | 1 | String representation |

---

## Regression Tests (10 tests)

Run with `pytest tests/` (included by default). Pins for known bugs.

| File | Class | Tests | Description |
|------|-------|-------|-------------|
| `test_fim_known_issues.py` | TestDefaultInconsistencyRegression | 1 | FIM rate consistency across config formats |
| `test_fim_known_issues.py` | TestMissingTokenErrorRegression | 2 | Missing FIM tokens error, message quality |
| `test_fim_known_issues.py` | TestShortSequenceRegression | 2 | Short sequence untransformed, boundary transformed |
| `test_fim_known_issues.py` | TestEmptyMiddleRegression | 1 | Empty middle handled |
| `test_fim_known_issues.py` | TestMetadataTypeRegression | 1 | No type tracking |
| `test_fim_known_issues.py` | TestTiktokenRegression | 2 | Tiktoken unsupported, HF tokenizer required |
| `test_fim_known_issues.py` | TestRandomSampleRegression | 1 | Sample guarantees different values |

---

## Integration Tests (158 tests)

Run via `scripts/run_tests.sh` with `torchrun`.

### Single-GPU Integration (torchrun --nproc_per_node=1)

| File | Class | Tests | Description |
|------|-------|-------|-------------|
| `test_dpo_integration.py` | TestDpoLossIntegration | 3 | Model forward, backward flow, bin-packed position IDs |
| `test_dpo_integration.py` | TestReferenceModel | 2 | Creation, not updated during training |
| `test_dpo_integration.py` | TestGradientAccumulation | 1 | Correctness across accumulation steps |
| `test_dpo_integration.py` | TestDPOCheckpoint | 2 | Save/load weight preservation, reference model recreation |
| `test_dpo_integration.py` | TestDPOTrainingLoop | 2 | Multi-step, different betas |
| `test_dpo_integration.py` | TestConcatForwardPassOptimization | 1 | Concat vs separate equivalence |
| `test_dpo_integration.py` | TestNaNHandling | 1 | NaN loss detection |
| `test_dpo_integration.py` | TestDPOMetrics | 1 | Metrics interval skip |
| `test_dpo_integration.py` | TestCollatorTruncation | 2 | SFT/DPO collator truncation |
| `test_chunked_parallel.py` | TestChunkedValidation | 8 | Standard/flash attention, short/medium/long/uneven sequences |
| `test_chunked_parallel.py` | TestChunkedValidationTP2 | 3 | Flash attention TP=2, short/medium/long sequences |
| `test_flash_attention_cache.py` | TestFlashAttentionWithCache | 3 | Cache concat, equivalence vs standard, GQA |
| `test_flash_attention_cache.py` | TestFlashAttentionEdgeCases | 3 | Empty cache, single token gen, multi-turn |
| `test_flash_attention_cache.py` | TestAttentionWithoutFlashAttn | 2 | Standard cache, GQA cache |
| `test_eval_integration.py` | (module) | 5 | KV cache config, forward, cache manager, eval accuracy, statistics |
| `test_fim_integration.py` | TestFIMDatasetFlow | 3 | Enabled/disabled flow, partial rate |
| `test_fim_integration.py` | TestFIMSerialization | 2 | Config serialization, unique tokens |
| `test_fim_integration.py` | TestFIMEndToEnd | 1 | MainConfig integration |
| `test_kv_cache.py` | TestKVCacheTP1 | 4 | End-to-end gen, GQA shapes, cached equivalence, cache reuse |
| `test_kv_cache_stateful.py` | TestStatefulKVCache | 2 | Stateful vs stateless parity, multi-step |
| `test_lora_async.py` | TestLoRAAsyncChunking | 2 | Chunked equivalence TP=1, TP=2 |
| `test_lora_checkpoint.py` | TestLoRACheckpointTP1 | 1 | Save/load preserves weights |
| `test_lora_checkpoint.py` | TestLoRAUniversalCheckpoint | 1 | TP=1 save, TP=2 load |
| `test_lora_correctness.py` | TestLoRATP2Correctness | 4 | Trainable ratio, requires_grad, param replication, forward output |
| `test_memory_optimization.py` | TestRolloutBufferCat | 5 | Basic cat, group IDs, metadata, mismatch, multiple chunks |
| `test_memory_optimization.py` | TestConfigValidation | 3 | Valid config, zero/indivisible micro group |
| `test_memory_optimization.py` | TestTransformerModelActivationCheckpointing | 5 | Disabled default, enable, strategy, config roundtrip |
| `test_memory_optimization.py` | TestMemoryCleanupBehavior | 1 | del and empty_cache pattern |
| `test_moe_correctness.py` | TestMoECorrectness | 6 | Dense equivalence, gradient, shared experts, shape, load balance, NaN |
| `test_moe_functional.py` | TestDeterministicRouting | 4 | Top-k, weight sum, valid range, deterministic |
| `test_moe_functional.py` | TestGradientFlowAndSparsity | 3 | Selected expert gradients, unselected zero, shared gradient |
| `test_moe_functional.py` | TestSharedVsRoutedExpertGradients | 2 | Shared larger norm, shared from all tokens |
| `test_moe_functional.py` | TestInputValidation | 4 | NaN, Inf, wrong hidden size, wrong ndim |
| `test_moe_layer.py` | TestMoEMLP | 9 | Shape, forward/backward, shared experts, aux loss, top-k, NaN |
| `test_moe_layer.py` | TestMoEMLPAllToAll | 5 | All-to-all shape, forward/backward, NaN, deterministic, aux loss |
| `test_moe_layer.py` | TestMoEConfigValidation | 3 | Invalid top-k, zero experts, negative aux loss |
| `test_optimizer.py` | TestOptimizerTrainerIntegration | 4 | AdamW, Muon, multi-step, state dict |
| `test_optimizer.py` | TestOptimizerTPIntegration | 3 | Muon TP=1, TP=2, multi-step |
| `test_optimizer.py` | TestOptimizerFSDPIntegration | 3 | Muon FSDP, multi-step, state dict |
| `test_optimizer.py` | TestDistributedOptimizerIntegration | 2 | With trainer, state dict |
| `test_integration.py` | TestConfigurationValidation | 5 | Config creation (3 variants), serializable, validation |
| `test_integration.py` | TestModelInitialization | 4 | Forward pass (3 variants), parameter count |
| `test_integration.py` | TestTrainingStep | 3 | Single step, gradient flow, loss decrease |
| `test_integration.py` | TestCheckpointing | 1 | Checkpoint roundtrip |
| `test_integration.py` | TestDataPipeline | 1 | Dataloader exists |
| `test_integration.py` | TestGRPOIntegration | 1 | GRPO loss computation |
| `test_integration.py` | TestMemoryEfficiency | 1 | No memory leak |

### TP=2 Integration (torchrun --nproc_per_node=2)

| File | Class | Tests | Description |
|------|-------|-------|-------------|
| `test_attention_multi_gpu.py` | TestTP2Attention | 4 | Forward shape, finiteness, norm consistency, causal mask |
| `test_kv_cache.py` | TestKVCacheTP2 | 4 | Cache sharding, GQA shape, lifecycle, selective reset |
| `test_lora_correctness.py` | TestLoRATP2Correctness | 4 | Trainable ratio, requires_grad, param replication, forward output |

---

## Multi-GPU Tests (27 tests)

Run via `scripts/run_tests.sh` with `torchrun --nproc_per_node=2`.

| File | Tests | Description |
|------|-------|-------------|
| `test_expert_parallel.py` | 7 | EP init, MoE with EP, all-reduce, gradient sync, all-to-all, load balance, EP+TP combined |
| `test_grad_norm.py` | 8 | DP2, TP2, FSDP gradient norms (clipping, parameter norms, EP2 MoE) |
| `test_all_to_all_ep.py` | 2 | All-to-all mode, compare dispatch/gather modes |
| `test_distributed_optimizer.py` | 3 | Parameter partitioning, step with DDP, cross-rank consistency |
| `test_distributed_optimizer_checkpoint.py` | 7 | Universal/distributed checkpoint switching, loss trajectory, edge cases |

---

## E2E Tests (2 tests, opt-in)

Run via `./scripts/run_tests.sh --e2e` or `pytest tests/ -m e2e`.

Excluded from default run by `-m 'not e2e'` in `pyproject.toml`. Requires 2 GPUs, HuggingFace model cache (~1 GB), and ~10 min. These tests self-spawn `torchrun` internally so they run under plain `pytest`.

| File | Class | Tests | Markers | Description |
|------|-------|-------|---------|-------------|
| `test_reward_manager.py` | TestRLVRTraining | 2 | `grpo e2e mp smoke` | GRPO training with reward_manager config, composite math reward |

Note: `grpo` also appears on cheap unit tests in `test_grpo_math.py` which run in the default CPU tier. Only the `e2e`-marked subset requires GPUs and opt-in.

---

## Profiler Tests (58 tests, opt-in)

Run via `./scripts/run_tests.sh --profiler` or `pytest tests/unit/profiler/`.

Excluded from default run by `--ignore=tests/unit/profiler/` in `pyproject.toml`.

| File | Class | Tests | Description |
|------|-------|-------|-------------|
| `test_profiler.py` | TestCommProfiler | 7 | Singleton, disable, record, enable/reset, accumulate, get/reset |
| `test_profiler.py` | TestTimedComm | 3 | Duration recording, no overhead disabled, exception handling |
| `test_profiler.py` | TestLayerTimingCollector | 9 | Singleton, disable, no-data summary, reset, CUDA, overwrite, summary |
| `test_profiler.py` | TestTimedDataIterator | 7 | Iteration, elapsed time, get/reset, len, iter |
| `test_profiler.py` | TestProfileManager | 26 | Active state, step triggers, CUDA sync, rank filtering, comm stats, memory snapshots, layer timing, data iterator, chrome trace, versioning |
| `test_profiler.py` | TestBaseModuleHooks | 6 | No stdout, unknown kwarg, valid kwargs, nested reach, non-base skip, idempotent |
