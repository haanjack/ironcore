# Multi-Dataset SFT Training

IronCore supports training with multiple SFT datasets using weighted mixing.

## Configuration

Define multiple datasets in your data config YAML:

```yaml
train_datasets:
  - name: gsm8k
    dataset_path: openai/gsm8k
    task_type: sft
    ratio: 0.3      # 30% weight
    split: train

  - name: alpaca
    dataset_path: tatsu-lab/alpaca
    task_type: sft
    ratio: 0.5      # 50% weight
    split: train

  - name: code_data
    dataset_path: my-org/code-dataset
    task_type: sft
    ratio: 0.2      # 20% weight
    split: train
```

## How Weighted Mixing Works

### Current Implementation: Epoch-Based Sampling

The current implementation uses **weighted sampling without replacement** per epoch:

1. **Within Each Epoch**:
   - All samples from all datasets are seen exactly once
   - Weights control the **sampling priority/order**
   - Higher-weight datasets tend to appear earlier in the epoch

2. **Across Multiple Epochs**:
   - Each dataset contributes proportionally to its **size**, not its ratio weight
   - Example: If dataset A has 1000 samples and dataset B has 500 samples:
     - Over multiple epochs: ~67% samples from A, ~33% from B
     - Regardless of configured ratios

### Effective Contribution Formula

```
Effective contribution = (dataset_size * ratio) / sum(all dataset_size * ratio)
```

**Example:**
```
Dataset A: 1000 samples, ratio=0.2 → weighted contribution = 1000 * 0.2 = 200
Dataset B:  500 samples, ratio=0.6 → weighted contribution =  500 * 0.6 = 300
Dataset C:  200 samples, ratio=0.2 → weighted contribution =  200 * 0.2 =  40

Total weighted = 540

Actual proportions over training:
  A: 200/540 = 37.0%
  B: 300/540 = 55.6%
  C:  40/540 =  7.4%
```

## Best Practices

### 1. Balancing Dataset Sizes

If you want **equal representation** regardless of dataset size:

```python
# Calculate inverse size weights
size_a = 10000
size_b = 2000

# To get 50/50 split, use inverse ratios
ratio_a = 1 / size_a  # = 0.0001
ratio_b = 1 / size_b  # = 0.0005

# Normalize
total = ratio_a + ratio_b
ratio_a_normalized = ratio_a / total  # ~0.17
ratio_b_normalized = ratio_b / total  # ~0.83
```

### 2. Controlling Dataset Importance

Use ratios to **prioritize** certain datasets:

```yaml
# Emphasize reasoning data over general chat
train_datasets:
  - name: math_reasoning
    ratio: 2.0    # Higher priority

  - name: general_chat
    ratio: 0.5    # Lower priority
```

The high-ratio dataset will:
- Appear earlier in each epoch
- Have more effective contribution if dataset sizes differ

### 3. Subsampling Large Datasets

If one dataset is very large, subsample it:

```yaml
train_datasets:
  - name: huge_dataset
    max_samples: 10000  # Limit to 10K samples
    ratio: 1.0

  - name: small_dataset
    ratio: 1.0
```

## Alternative: Frequency-Based Mixing (Future)

For true frequency-based mixing where ratios directly control the proportion of training batches:

```python
# Proposed implementation: sampling WITH replacement
sampled_indices = rng.choice(
    num_samples,
    size=num_steps,      # Sample for fixed number of steps
    replace=True,        # Allow replacement
    p=weights_array
)
```

This would give:
- `ratio: 0.3` → exactly 30% of batches from that dataset
- Independent of dataset size
- May oversample small datasets or undersample large datasets

**Tradeoffs:**
- ✓ Intuitive ratio semantics
- ✓ Direct control over batch proportions
- ✗ May not see all data (small ratios)
- ✗ May repeat data many times (large ratios on small datasets)

## Verification

Test your mixing ratios:

```python
from ironcore.config.config_data import UniversalDataConfig
from ironcore.dataloader.universal_dataset import WeightedMixingDataset

config = UniversalDataConfig.from_yaml("your_config.yaml")
dataset = WeightedMixingDataset(
    data_config=config,
    mode="sft",
    split="train"
)

# Check loaded datasets and weights
print(f"Datasets: {len(dataset.datasets)}")
print(f"Weights: {dataset.weights}")
print(f"Sizes: {[len(ds) for ds in dataset.datasets]}")

# Sample and verify
counts = {}
for i, sample in enumerate(iter(dataset)):
    if i >= 1000:
        break
    # Track dataset origin and analyze distribution
```

## Summary

✓ **Multiple SFT datasets are fully supported**
✓ **Weights control sampling priority and effective contribution**
⚠ **Actual proportions depend on both weights AND dataset sizes**
📝 **Use `max_samples` to balance very large datasets**

For questions or feature requests about alternative mixing strategies, please open an issue.
