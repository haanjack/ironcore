# Data Directory

This directory contains all data artifacts for the IronCore platform. **Contents are gitignored** due to large file sizes.

## Directory Structure

```
data/
├── cache/              # HuggingFace datasets cache (auto-created)
├── preprocessed/       # Serialized binary datasets (.bin + .idx files)
│   ├── <dataset_name>.bin  # Token ID arrays (uint16/uint32)
│   └── <dataset_name>.idx  # Metadata (NumPy structured array)
├── raw/                # (Optional) Raw text files
└── downloads/          # (Optional) Downloaded datasets
```

## File Formats

### `.bin` files
- Flattened arrays of token IDs
- Data type: `uint16` or `uint32` (auto-detected based on vocab size)
- Memory-mapped for efficient access

### `.idx` files
- NumPy structured arrays (`.npy` format)
- Schema:
  ```python
  dtype=[
      ('offset', np.uint64),      # Start position in .bin file
      ('length', np.uint32),      # Number of tokens
      ('type', 'U20'),            # Sample type (e.g., "pretrain", "sft_sample")
      ('group_id', np.int64),     # Group ID (for DPO pairs, -1 otherwise)
      ('mask_ranges', 'U500'),    # JSON string: [[start, end], ...] for masking
  ]
  ```

## Usage

### 1. Preprocess Datasets

```bash
# Serialize datasets into .bin/.idx format
ironcore preprocess --config configs/data/pretrain_example.yaml
```

This creates:
- `data/preprocessed/openwebtext.bin` (~2GB for 10k samples)
- `data/preprocessed/openwebtext.idx` (~500KB metadata)

### 2. Inspect Preprocessed Data

```bash
# Check integrity and statistics
ironcore preprocess --config configs/data/sft_example.yaml --only-inspect

# Preview with decoded samples (color-coded masking)
ironcore preprocess --config configs/data/sft_example.yaml --only-inspect --preview 5
```

### 3. Train

```bash
# Use preprocessed data for training
ironcore train --config configs/example.yaml
```

## Configuration

Data preprocessing is controlled by YAML configs in `configs/data/`:

- `pretrain_example.yaml` - Pretraining on raw text
- `sft_example.yaml` - Supervised fine-tuning with chat templates
- `dpo_example.yaml` - Direct preference optimization

See individual config files for dataset mixing ratios, sequence lengths, and other parameters.

## Storage Requirements

Approximate storage per dataset type:

| Dataset Type | Tokens | Samples | `.bin` Size | `.idx` Size |
|--------------|--------|---------|-------------|-------------|
| Pretrain     | 100M   | N/A     | ~200 MB     | ~1 KB       |
| SFT          | 50M    | 100K    | ~100 MB     | ~50 MB      |
| DPO          | 50M    | 50K pairs | ~100 MB   | ~50 MB      |

**Note**: Cache directory (`data/cache/`) can grow large with HuggingFace datasets. Clear periodically if disk space is limited.

## Cleaning Up

```bash
# Remove all preprocessed files
rm -rf data/preprocessed/*

# Clear HuggingFace cache
rm -rf data/cache/*

# Remove everything (start fresh)
rm -rf data/
```

The directory will be recreated automatically on next preprocessing run.
