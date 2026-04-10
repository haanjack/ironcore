# Test Suite

## Overview

| Tier | Runner | Tests | GPU | Runtime |
|------|--------|-------|-----|---------|
| Unit + Regression | `pytest tests/` | 405 | No | ~5 min |
| Integration (single GPU) | `torchrun --nproc_per_node=1` | 146 | 1 | ~15 min |
| Integration (TP=2) | `torchrun --nproc_per_node=2` | 12 | 2 | ~10 min |
| Multi-GPU | `torchrun --nproc_per_node=2` | 27 | 2 | ~10 min |
| RLVR Smoke | `torchrun --nproc_per_node=2` | 2 | 2 | ~10 min |

**Total: 592 tests across all tiers.**

### Execution

```bash
# Unit + regression (default)
pytest tests/

# Integration (single GPU)
torchrun --nproc_per_node=1 -m pytest tests/integration/kvcache/test_kv_cache.py -v

# Integration (TP=2)
torchrun --nproc_per_node=2 -m pytest tests/integration/attention/test_attention_multi_gpu.py -v

# All tiers
./scripts/run_tests.sh              # full suite
./scripts/run_tests.sh --quick      # skip multi-GPU
./scripts/run_tests.sh --rlvr       # add RLVR smoke tests
```

### pytest Configuration

Default `addopts` in `pyproject.toml`:

```
-v --tb=short -m 'not rlvr' --ignore=tests/unit/profiler/ --ignore=tests/multi_gpu/ --ignore=tests/integration/
```

Integration and multi-GPU tests require `torchrun` and are excluded from `pytest` by default. They are run via `scripts/run_tests.sh` which groups them by GPU requirement.

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

## RLVR Smoke Tests (2 tests, opt-in)

Run via `./scripts/run_tests.sh --rlvr` or `pytest tests/ -m rlvr`.

Excluded from default run by `-m 'not rlvr'` in `pyproject.toml`. Requires 2 GPUs, HuggingFace model cache (~1 GB), and ~10 min.

| File | Class | Tests | Description |
|------|-------|-------|-------------|
| `test_reward_manager.py` | TestRLVRTraining | 2 | GRPO training with reward_manager config, composite math reward |

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
