#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Evaluate model on GSM8K test set using IronCore inference.

Usage:
    # Evaluate with IC inference (default)
    WORLD_SIZE=1 python scripts/eval_gsm8k_baseline.py

    # Quick test with fewer samples
    WORLD_SIZE=1 python scripts/eval_gsm8k_baseline.py --max_samples 10
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm

# Set up paths for IronCore imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
os.chdir(project_dir)

from ironcore import get_tokenizer
from ironcore.checkpointing import load_from_huggingface
from ironcore.config import load_trainer_config
from ironcore.global_vars import set_global_states
from ironcore.language_model import LanguageModel
from ironcore.parallel.parallel_states import initialize_model_parallel


def extract_answer(text: str) -> str | None:
    """Extract final numerical answer from completion."""
    # GSM8K format: #### <number>
    match = re.search(r"####\s*(-?\d[\d,]*)", text)
    if match:
        return match.group(1).replace(",", "")
    # Fallback: last number in text
    numbers = re.findall(r"-?\d[\d,]*", text)
    return numbers[-1].replace(",", "") if numbers else None


def extract_gold_answer(answer_text: str) -> str:
    """Extract gold answer from GSM8K format."""
    match = re.search(r"####\s*(-?\d[\d,]*)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d[\d,]*", answer_text)
    return numbers[-1].replace(",", "") if numbers else ""


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K using IronCore")
    parser.add_argument("--config_path", type=str, default="configs/grpo_gsm8k_smoke.yaml")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    # Set up sys.argv for IronCore config loader
    sys.argv = ["eval", "--config-path", args.config_path]

    # Load config
    config = load_trainer_config()

    # Initialize model parallel (no tensor parallel for inference)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=10)
    set_global_states(config)

    # Create and load model
    print(f"Loading IronCore model from {config.trainer.load_from_hf}")
    model = LanguageModel(config).to("cuda")
    cache_dir = snapshot_download(config.trainer.load_from_hf)
    load_from_huggingface(cache_dir, model, "qwen2", strict=False)
    model.eval()

    # Get tokenizer
    tokenizer = get_tokenizer()

    print("Loading GSM8K test split...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    print(f"  Evaluating {len(dataset)} samples")

    system_prompt = "Solve the math problem step by step. Put your final numerical answer after ####."

    correct = 0
    total = 0
    results = []

    for sample in tqdm(dataset, desc="Evaluating"):
        question = sample["question"]
        gold_answer = extract_gold_answer(sample["answer"])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        # Handle BatchEncoding (dict-like but not dict)
        if hasattr(enc, "input_ids"):
            input_ids = enc["input_ids"]
        elif isinstance(enc, dict):
            input_ids = enc["input_ids"]
        else:
            input_ids = enc
        input_ids = input_ids.to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part (skip prompt)
        generated_ids = output[0][input_ids.shape[1]:].tolist()
        response = tokenizer.batch_decode([generated_ids], skip_special_tokens=True)[0]
        pred = extract_answer(response)
        is_correct = pred == gold_answer

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question[:100] + "..." if len(question) > 100 else question,
            "gold": gold_answer,
            "pred": pred,
            "correct": is_correct,
            "response": response[:200] + "..." if len(response) > 200 else response,
        })

        if total % 10 == 0:
            print(f"  Accuracy: {correct}/{total} = {100*correct/total:.2f}%")

    accuracy = 100 * correct / total
    print(f"\n{'='*60}")
    print(f"Final Accuracy: {correct}/{total} = {accuracy:.2f}%")
    print(f"{'='*60}")

    output_path = Path("outputs/gsm8k_ic_eval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": config.trainer.load_from_hf,
            "inference": "IronCore",
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
