# GRPO Testing Suite

This directory contains comprehensive tests for the GRPO (Group Relative Policy Optimization) implementation.

## Test Structure

```
tests/alignment/
├── __init__.py
├── test_grpo_math.py        # Mathematical correctness tests
├── test_grpo_rollout.py     # KV-cache and rollout tests
├── test_grpo_distributed.py # Distributed advantage computation tests
└── test_grpo_hardware.py    # Memory and throughput tests
```

## Running Tests

### Unit Tests (Mathematical Correctness)

```bash
# Run all unit tests
pytest tests/alignment/test_grpo_math.py -v

# Run specific test class
pytest tests/alignment/test_grpo_math.py::TestAdvantageNormalization -v

# Run with coverage
pytest tests/alignment/test_grpo_math.py --cov=ironcore.alignment.loss.grpo
```

### Rollout Tests

```bash
pytest tests/alignment/test_grpo_rollout.py -v
```

### Distributed Tests

Requires multiple GPUs:

```bash
# Run with 4 GPUs
torchrun --nproc_per_node=4 tests/alignment/test_grpo_distributed.py
```

### Hardware Tests

```bash
# Requires CUDA
pytest tests/alignment/test_grpo_hardware.py -v -s
```

## Smoke Test (Behavioral Convergence)

Run the 100-step smoke test:

```bash
python scripts/run_grpo_smoke_test.py --config configs/alignment/grpo_smoke.yaml
```

### Expected Results

| Metric | Target | Indicator |
|--------|--------|-----------|
| Mean Reward | Increasing | Policy learning correctly |
| KL Divergence | Bounded (<10) | Policy staying close to reference |
| Advantage Std | ~1.0 | Normalization correct |
| Policy Loss | Oscillating | Gradient flowing correctly |

## Test Categories

### 1. Mathematical Correctness

Tests the underlying calculus of policy gradient:

- **Advantage Normalization**: `sum(A) = 0`, `std(A) = 1`
- **Identical Rewards**: Produce exactly 0 advantage
- **KL Divergence**: `KL(P||P) = 0`, positive otherwise
- **Numerical Stability**: Handles extreme log probabilities

### 2. Rollout Correctness

Tests KV-cache expansion and generation:

- **Expansion Shape**: `[B] → [B×G]`
- **Content Correctness**: Each prompt replicated G times
- **Order Correctness**: `[p0_g0, p0_g1, ..., p1_g0, ...]`
- **Sampling**: Greedy, temperature, top-k, top-p

### 3. Distributed Correctness

Tests multi-GPU advantage computation:

- **All-Gather**: Rewards synchronized across DP ranks
- **Group Normalization**: Correct even when groups split across ranks

### 4. Hardware Integrity

Tests memory and performance:

- **VRAM Leaks**: No memory growth across iterations
- **KV-Cache Cleanup**: Proper cleanup after generation
- **Worker Pool Latency**: Reward computation overhead
- **Throughput**: Tokens/second generation rate

## Adding New Tests

1. Create test file in `tests/alignment/`
2. Follow existing naming convention: `test_grpo_*.py`
3. Import from `ironcore.alignment.*`
4. Use pytest fixtures for setup/teardown
5. Add to this README
