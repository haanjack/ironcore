#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0

"""Download GSM8K dataset and convert to GRPO-compatible format.

This script downloads the GSM8K dataset and renames fields to match
our GRPO dataset expectations:
- "question" → "prompt"
- "answer" → "answer" (already correct)

No tokenization or preprocessing is performed - just field renaming.
"""

import json
from pathlib import Path

from datasets import load_dataset


def main():
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    print("Downloading GSM8K dataset...")

    # Download train split
    print("  Loading train split...")
    train = load_dataset("openai/gsm8k", "main", split="train")
    train_path = data_dir / "gsm8k_train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train:
            record = {"prompt": sample["question"], "answer": sample["answer"], "type": "math"}
            f.write(json.dumps(record) + "\n")
    print(f"  Saved {len(train)} samples to {train_path}")

    # Download test split
    print("  Loading test split...")
    test = load_dataset("openai/gsm8k", "main", split="test")
    test_path = data_dir / "gsm8k_test.jsonl"
    with open(test_path, "w", encoding="utf-8") as f:
        for sample in test:
            record = {"prompt": sample["question"], "answer": sample["answer"], "type": "math"}
            f.write(json.dumps(record) + "\n")
    print(f"  Saved {len(test)} samples to {test_path}")

    print(f"\nDone! Train: {len(train)}, Test: {len(test)}")


if __name__ == "__main__":
    main()
