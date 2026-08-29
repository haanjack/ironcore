# Multi-Dataset SFT Training

IronCore supports training with multiple SFT datasets using weighted mixing.

## Configuration

Define multiple datasets in your data config YAML:

```yaml
data:
  task_type: sft
  datasets:
    - source: openai/gsm8k
      task_type: sft
      ratio: 0.3
      split: train

    - source: tatsu-lab/alpaca
      task_type: sft
      ratio: 0.5
      split: train

    - source: my-org/code-dataset
      task_type: sft
      ratio: 0.2
      split: train
```

## How weighted mixing works

The mixer samples proportionally to `dataset_size × ratio`. A higher `ratio` increases a
dataset's effective contribution, but actual proportions depend on both the ratio **and**
the dataset size:

```
Effective contribution = (dataset_size × ratio) / sum(all dataset_size × ratio)
```

**Example:**
```
Dataset A: 1000 samples, ratio=0.2  →  weighted = 1000 × 0.2 = 200
Dataset B:  500 samples, ratio=0.6  →  weighted =  500 × 0.6 = 300
Dataset C:  200 samples, ratio=0.2  →  weighted =  200 × 0.2 =  40

Actual proportions: A=37%, B=56%, C=7%
```

## Best practices

### Balancing unequal dataset sizes

To get equal representation regardless of dataset size, use inverse-size weights:

```python
# For 50/50 split between size-10000 and size-2000 datasets:
ratio_a = 1 / 10000  # 0.0001
ratio_b = 1 / 2000  # 0.0005
# normalize: ratio_a=0.17, ratio_b=0.83
```

### Emphasizing certain datasets

```yaml
data:
  datasets:
    - source: math_reasoning
      ratio: 2.0    # higher effective contribution
    - source: general_chat
      ratio: 0.5
```

### Subsampling large datasets

```yaml
data:
  datasets:
    - source: huge_dataset
      max_samples: 10000   # cap at 10K samples
      ratio: 1.0
    - source: small_dataset
      ratio: 1.0
```

## Summary

- Multiple SFT datasets are fully supported via `data.datasets`
- `ratio` controls sampling weight relative to dataset size
- Actual proportions depend on both `ratio` and dataset size — use `max_samples` to cap large datasets
