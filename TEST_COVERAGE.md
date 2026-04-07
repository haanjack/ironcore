# Test Coverage Report

## Summary
- **Total Test Files**: 50
- **Total Tests Run**: 593 passed, 24 skipped, 0 failed
- **Duration**: ~7 minutes (413s)

## Test Categories

| Category | Files | Status |
|----------|-------|--------|
| Unit Tests | 24 | ✅ Covered by pytest |
| Integration Tests | 20 | ✅ Covered by pytest |
| Multi-GPU Tests | 4 | ✅ Covered by torchrun |
| Regression Tests | 2 | ✅ Covered by pytest |

## Tests NOT Covered by Default `pytest tests/`

### 1. Excluded by `pyproject.toml` (addopts --ignore)
These tests are intentionally excluded from the default pytest run:

| File | Reason |
|------|--------|
| `tests/unit/profiler/` | Profiler tests excluded (entire directory) |
| `tests/integration/kvcache/test_kv_cache_tp2.py` | Requires TP=2, needs torchrun |
| `tests/integration/attention/test_tp_comparison_simple.py` | Requires TP comparison |
| `tests/integration/lora/test_lora_tp_correctness.py` | Requires LoRA TP correctness |

### 2. Excluded by `-m 'not rlvr'` marker
GRPO smoke tests that run actual distributed training (rule-based rewards, no API keys needed):

| File | Test | Requirement |
|------|------|-------------|
| `tests/unit/reward/test_reward_manager.py` | `test_reward_manager_config_trains` | 2 GPUs + HF model cache |
| `tests/unit/reward/test_reward_manager.py` | `test_reward_manager_composite_math_trains` | 2 GPUs + HF model cache |

### 3. Excluded by module-level `pytestmark` (requires distributed)
These tests have `pytestmark = pytest.mark.skipif(...)` and only run with torchrun:

| File | Condition | Covered by torchrun |
|------|-----------|---------------------|
| `tests/integration/kvcache/test_kv_cache_tp2.py` | `CAN_RUN_TP2` (2+ GPUs) | ✅ Yes |
| `tests/integration/kvcache/test_kv_cache_tp_comparison.py` | `CAN_RUN_MULTI_GPU` | ✅ Yes |
| `tests/multi_gpu/test_ep_multi_gpu.py` | `dist.is_initialized()` + 2+ GPUs | ✅ Yes |

### 4. Excluded by individual `@pytest.mark.skipif`
These tests skip conditionally based on hardware/requirements:

| File | Condition | Notes |
|------|-----------|-------|
| `tests/unit/alignment/test_grpo_math.py::TestTPSafeSoftmax` | CUDA not available | Runs on GPU systems |
| `tests/unit/parallel/test_comm_grad_correctness.py::test_grad_norm_distributed` | CUDA not available | Runs on GPU systems |
| `tests/integration/attention/test_flash_attention_cache.py` | CUDA not available | Flash Attn needs CUDA |
| `tests/multi_gpu/test_distributed_optimizer_checkpoint.py` | Various | Needs 4+ GPUs for some tests |

## How to Run All Tests

### Standard pytest (Unit + Integration + Regression)
```bash
pytest tests/ -v
```

### Multi-GPU tests (requires 2+ GPUs)
```bash
# Run specific multi-GPU test
torchrun --nproc_per_node=2 -m pytest tests/integration/kvcache/test_kv_cache_tp2.py -v

# Run all multi-GPU tests
torchrun --nproc_per_node=2 -m pytest tests/multi_gpu/ -v
```

### RLVR tests (GRPO smoke, 2 GPUs + HF model cache)
```bash
pytest tests/ -v -m rlvr
```

### Profiler tests (excluded by default)
```bash
pytest tests/unit/profiler/ -v
```

### Complete test suite
```bash
./run_all_tests.sh
```

## Test Files by Directory

### `tests/unit/` (24 files)
- `alignment/` - DPO loss, GRPO math (4 files)
- `attention/` - Attention, rotary, TP validation (3 files)
- `checkpointing/` - HF interop (1 file)
- `dataloader/` - FIM config, edge cases, token lookup, transformation (4 files)
- `kvcache/` - KV cache basic (1 file)
- `moe/` - Expert, load balance, utils, router (4 files)
- `optimizer/` - Distributed optimizer, Muon (2 files)
- `parallel/` - Comm grad correctness, expert parallel, grad norm (3 files)
- `profiler/` - Profiler tests (1 file, excluded by default)
- `reward/` - Reward manager (1 file)
- `trainers/` - Trainer norm (1 file)
- `test_mfu.py` - MFU calculator (1 file)

### `tests/integration/` (20 files)
- `alignment/` - DPO e2e, GRPO math (2 files)
- `attention/` - TP simple, chunked TP, flash attention cache (3 files)
- `dataloader/` - Eval integration, FIM integration (2 files)
- `kvcache/` - Stateful, TP1, TP2, TP comparison (4 files)
- `lora/` - Async, checkpoint, TP correctness (3 files)
- `memory/` - Memory optimization (1 file)
- `moe/` - Correctness, functional, layer (3 files)
- `optimizer/` - Optimizer (1 file)
- `test_integration.py` - General integration (1 file)

### `tests/multi_gpu/` (4 files)
- `test_all_to_all_ep.py` - All-to-all EP communication
- `test_distributed_optimizer_checkpoint.py` - Distributed optimizer checkpointing
- `test_ep_multi_gpu.py` - Expert parallel multi-GPU
- `test_grad_norm_multi_gpu.py` - Gradient norm multi-GPU

### `tests/regression/` (2 files)
- `test_fim_known_issues.py` - FIM known issues
- `test_fix_review_0316.py` - Fix review

## Recommendations

1. **Add to run_all_tests.sh**: The following tests could be added to the multi-GPU test suite:
   - `tests/integration/attention/test_chunked_tp.py`
   - `tests/integration/attention/test_flash_attention_cache.py`

2. **RLVR tests**: Run GRPO smoke tests manually (2 GPUs, ~10 min):
   ```bash
   pytest tests/ -v -m rlvr --timeout=600
   ```

3. **Profiler tests**: Enable profiler tests by removing `--ignore=tests/unit/profiler/` from pyproject.toml if profiler functionality is critical.
