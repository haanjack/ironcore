#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Preprocess GSM8K for GRPO training.

Downloads GSM8K dataset and converts to JSONL format expected by ironcore:
{
    "prompt": "<question>",
    "answer": "#### <number>",  # Preserves GSM8K format
    "type": "math"
}
"""

import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # Download and process training split
    print("Downloading GSM8K train split...")
    train_dataset = load_dataset("openai/gsm8k", "main", split="train")
    print(f"  Found {len(train_dataset)} training samples")

    train_path = output_dir / "gsm8k_train.jsonl"
    with open(train_path, "w") as f:
        for sample in tqdm(train_dataset, desc="Processing train"):
            f.write(
                json.dumps(
                    {
                        "prompt": sample["question"],
                        "answer": sample["answer"],  # Contains "#### <number>"
                        "type": "math",
                    }
                )
                + "\n"
            )
    print(f"  Saved to {train_path}")

    # Download and process test split
    print("Downloading GSM8K test split...")
    test_dataset = load_dataset("openai/gsm8k", "main", split="test")
    print(f"  Found {len(test_dataset)} test samples")

    test_path = output_dir / "gsm8k_test.jsonl"
    with open(test_path, "w") as f:
        for sample in tqdm(test_dataset, desc="Processing test"):
            f.write(
                json.dumps(
                    {
                        "prompt": sample["question"],
                        "answer": sample["answer"],
                        "type": "math",
                    }
                )
                + "\n"
            )
    print(f"  Saved to {test_path}")

    # Print sample
    print("\nSample from training data:")
    with open(train_path) as f:
        sample = json.loads(f.readline())
        print(f"  Prompt: {sample['prompt'][:100]}...")
        print(f"  Answer: {sample['answer']}")


if __name__ == "__main__":
    main()
