#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Compare GSM8K evaluation between HuggingFace and IronCore inference.

Usage:
    # Compare with 20 samples
    python scripts/compare_gsm8k_hf_ic.py --max_samples 20

    # Compare with 100 samples
    python scripts/compare_gsm8k_hf_ic.py --max_samples 100

    # Full test set
    python scripts/compare_gsm8k_hf_ic.py
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up paths for IronCore imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
os.chdir(project_dir)


def extract_answer(text: str, strict: bool = False) -> str | None:
    """Extract final numerical answer from completion."""
    match = re.search(r"####\s*(-?\d[\d,]*)", text)
    if match:
        return match.group(1).replace(",", "")
    
    if strict:
        return None
        
    numbers = re.findall(r"-?\d[\d,]*", text)
    return numbers[-1].replace(",", "") if numbers else None


def extract_gold_answer(answer_text: str) -> str:
    """Extract gold answer from GSM8K format."""
    match = re.search(r"####\s*(-?\d[\d,]*)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d[\d,]*", answer_text)
    return numbers[-1].replace(",", "") if numbers else ""


FEW_SHOT_PROMPT = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
#### 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $10.
#### 10

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: In the beginning, Betty has only 100 / 2 = $50.
Betty's grandparents gave her 15 * 2 = $30.
This means, Betty needs 100 - 50 - 30 - 15 = $5 more.
#### 5

Question: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
Answer: Julie read 12 x 2 = 24 pages today.
So she was able to read a total of 12 + 24 = 36 pages since yesterday.
There are 120 - 36 = 84 pages left to be read.
Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = 42 pages.
#### 42

Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
Answer: He writes each friend 3*2=6 pages a week
So he writes 6*2=12 pages every week
That means he writes 12*52=624 pages a year
#### 624

"""

def evaluate_huggingface(
    model_name: str,
    dataset,
    max_new_tokens: int = 512,
    device: str = "cuda",
    few_shot: bool = False,
    strict: bool = False,
) -> dict:
    """Evaluate using HuggingFace transformers."""
    print(f"\n{'='*60}")
    print(f"Evaluating with HuggingFace (few_shot={few_shot}, strict={strict})")
    print(f"{'='*60}")

    # Load model and tokenizer
    print(f"Loading {model_name} with HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    system_prompt = "Solve the math problem step by step. Put your final numerical answer after ####."

    correct = 0
    total = 0
    results = []
    start_time = time.time()

    for sample in tqdm(dataset, desc="HF Evaluation"):
        question = sample["question"]
        gold_answer = extract_gold_answer(sample["answer"])

        if few_shot:
            prompt = FEW_SHOT_PROMPT + f"Question: {question}\nAnswer:"
            result = tokenizer(prompt, return_tensors="pt")
            if hasattr(result, "input_ids"):
                input_ids = result["input_ids"]
            else:
                input_ids = result
            input_ids = input_ids.to(model.device)
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            result = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            # Handle result which can be a tensor, list, or BatchEncoding
            if hasattr(result, "input_ids"):
                input_ids = result["input_ids"]
            elif isinstance(result, dict):
                input_ids = result["input_ids"]
            elif isinstance(result, list):
                input_ids = torch.tensor(result).unsqueeze(0)
            else:
                input_ids = result
            input_ids = input_ids.to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.0,
            )

        generated_ids = output[0][input_ids.shape[1]:].tolist()
        if total == 0:
            print(f"  First 20 tokens: {generated_ids[:20]}")
            print(f"  First 20 decoded: '{tokenizer.decode(generated_ids[:20])}'")
        response = tokenizer.batch_decode([generated_ids], skip_special_tokens=True)[0]
        pred = extract_answer(response, strict=strict)
        is_correct = pred == gold_answer

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question[:100] + "..." if len(question) > 100 else question,
            "gold": gold_answer,
            "pred": pred,
            "correct": is_correct,
            "response": response,  # Store full response
        })

        if total % 10 == 0:
            print(f"  HF Accuracy: {correct}/{total} = {100*correct/total:.2f}%")

    elapsed = time.time() - start_time
    accuracy = 100 * correct / total

    print(f"\nHF Final: {correct}/{total} = {accuracy:.2f}%")
    print(f"HF Time: {elapsed:.2f}s ({elapsed/total:.3f}s per sample)")

    # Free memory
    del model
    torch.cuda.empty_cache()

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "time": elapsed,
        "time_per_sample": elapsed / total,
        "results": results,
    }


def evaluate_ironcore(
    model_name: str,
    dataset,
    max_new_tokens: int = 512,
    config_path: str = "configs/eval_qwen_gsm8k.yaml",
    compile_model: bool = False,
    compile_mode: str = "reduce-overhead",
    few_shot: bool = False,
    strict: bool = False,
) -> dict:
    """Evaluate using IronCore inference."""
    print(f"\n{'='*60}")
    print(f"Evaluating with IronCore (few_shot={few_shot}, strict={strict})")
    print(f"{'='*60}")

    from ironcore import get_tokenizer
    from ironcore.checkpointing import load_from_huggingface
    from ironcore.config import load_trainer_config
    from ironcore.global_vars import set_global_states
    from ironcore.language_model import LanguageModel
    from ironcore.parallel.parallel_states import initialize_model_parallel

    # Initialize model parallel (no tensor parallel for inference)
    initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=10)

    # Load config from YAML file
    sys.argv = ["eval", "--config-path", config_path]
    config = load_trainer_config()
    set_global_states(config)

    # Determine architecture
    if "qwen" in model_name.lower():
        hf_arch = "qwen2"
    elif "gpt2" in model_name.lower():
        hf_arch = "gpt2"
    else:
        hf_arch = "llama"

    # Download checkpoint
    print(f"Downloading checkpoint from {model_name}...")
    cache_dir = snapshot_download(model_name)

    # Create and load model
    print(f"Creating IronCore model...")
    from ironcore.utils import get_model_dtype
    model = LanguageModel(config).to("cuda")

    # Convert to proper dtype (bf16) for faster inference
    dtype = get_model_dtype(config)
    print(f"Converting model to dtype: {dtype}")
    model = model.to(dtype=dtype)

    print(f"Loading weights into IronCore model...")
    load_result = load_from_huggingface(cache_dir, model, hf_arch, strict=False)
    print(f"  Loaded keys: {len(load_result['loaded_keys'])}")
    if load_result["missing_keys"]:
        print(f"  Missing keys ({len(load_result['missing_keys'])}): {load_result['missing_keys'][:5]}...")
    if load_result["unexpected_keys"]:
        print(f"  Unexpected keys ({len(load_result['unexpected_keys'])}): {load_result['unexpected_keys'][:5]}...")

    model.eval()

    # Enable torch.compile for faster inference
    # Compile just the transformer backbone, not the whole model (generate loop can't be compiled)
    if compile_model:
        print(f"Compiling transformer backbone with torch.compile (mode={compile_mode})...")
        compile_options = {
            "mode": compile_mode,
            "fullgraph": False,
        }
        try:
            model.model = torch.compile(model.model, **compile_options)
            print(f"  Transformer backbone compiled successfully")
        except Exception as e:
            print(f"  torch.compile failed: {e}. Running without compilation.")

    # Get tokenizer
    tokenizer = get_tokenizer()

    system_prompt = "Solve the math problem step by step. Put your final numerical answer after ####."

    correct = 0
    total = 0
    results = []
    start_time = time.time()

    for sample in tqdm(dataset, desc="IC Evaluation"):
        question = sample["question"]
        gold_answer = extract_gold_answer(sample["answer"])

        if few_shot:
            prompt = FEW_SHOT_PROMPT + f"Question: {question}\nAnswer:"
            result = tokenizer(prompt, return_tensors="pt")
            if hasattr(result, "input_ids"):
                input_ids = result["input_ids"]
            else:
                input_ids = result
            input_ids = input_ids.to(model.device)
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            enc = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
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
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output[0][input_ids.shape[1]:].tolist()
        if total == 0:
            print(f"  First 20 tokens: {generated_ids[:20]}")
            print(f"  First 20 decoded: '{tokenizer.decode(generated_ids[:20])}'")
        response = tokenizer.batch_decode([generated_ids], skip_special_tokens=True)[0]
        pred = extract_answer(response, strict=strict)
        is_correct = pred == gold_answer

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question[:100] + "..." if len(question) > 100 else question,
            "gold": gold_answer,
            "pred": pred,
            "correct": is_correct,
            "response": response,  # Store full response
        })

        if total % 10 == 0:
            print(f"  IC Accuracy: {correct}/{total} = {100*correct/total:.2f}%")

    elapsed = time.time() - start_time
    accuracy = 100 * correct / total

    print(f"\nIC Final: {correct}/{total} = {accuracy:.2f}%")
    print(f"IC Time: {elapsed:.2f}s ({elapsed/total:.3f}s per sample)")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "time": elapsed,
        "time_per_sample": elapsed / total,
        "results": results,
    }


def compare_results(hf_results: dict, ic_results: dict) -> dict:
    """Compare results from HF and IC."""
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")

    # Accuracy comparison
    hf_acc = hf_results["accuracy"]
    ic_acc = ic_results["accuracy"]
    acc_diff = ic_acc - hf_acc

    print(f"\nAccuracy:")
    print(f"  HuggingFace: {hf_acc:.2f}% ({hf_results['correct']}/{hf_results['total']})")
    print(f"  IronCore:    {ic_acc:.2f}% ({ic_results['correct']}/{ic_results['total']})")
    print(f"  Difference:  {acc_diff:+.2f}%")

    # Time comparison
    print(f"\nTime:")
    print(f"  HuggingFace: {hf_results['time']:.2f}s ({hf_results['time_per_sample']:.3f}s/sample)")
    print(f"  IronCore:    {ic_results['time']:.2f}s ({ic_results['time_per_sample']:.3f}s/sample)")

    # Agreement analysis
    hf_correct_set = set(i for i, r in enumerate(hf_results["results"]) if r["correct"])
    ic_correct_set = set(i for i, r in enumerate(ic_results["results"]) if r["correct"])

    both_correct = len(hf_correct_set & ic_correct_set)
    hf_only = len(hf_correct_set - ic_correct_set)
    ic_only = len(ic_correct_set - hf_correct_set)

    print(f"\nAgreement Analysis:")
    print(f"  Both correct: {both_correct}")
    print(f"  Both wrong:   {len(hf_results['results']) - len(hf_correct_set | ic_correct_set)}")
    print(f"  HF only:      {hf_only}")
    print(f"  IC only:      {ic_only}")

    # Sample-level comparison for disagreements
    disagreements = []
    for i, (hf_r, ic_r) in enumerate(zip(hf_results["results"], ic_results["results"])):
        if hf_r["correct"] != ic_r["correct"]:
            disagreements.append({
                "idx": i,
                "question": hf_r["question"],
                "gold": hf_r["gold"],
                "hf_pred": hf_r["pred"],
                "ic_pred": ic_r["pred"],
                "hf_correct": hf_r["correct"],
                "ic_correct": ic_r["correct"],
            })

    if disagreements:
        print(f"\nDisagreements (first 5):")
        for d in disagreements[:5]:
            print(f"  [{d['idx']}] Gold: {d['gold']}")
            print(f"       HF: {d['hf_pred']} ({'✓' if d['hf_correct'] else '✗'})")
            print(f"       IC: {d['ic_pred']} ({'✓' if d['ic_correct'] else '✗'})")

    return {
        "hf_accuracy": hf_acc,
        "ic_accuracy": ic_acc,
        "accuracy_diff": acc_diff,
        "both_correct": both_correct,
        "hf_only": hf_only,
        "ic_only": ic_only,
        "disagreements": disagreements,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare GSM8K evaluation: HF vs IronCore")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--hf_only", action="store_true", help="Run only HuggingFace evaluation")
    parser.add_argument("--ic_only", action="store_true", help="Run only IronCore evaluation")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for IC inference")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")
    parser.add_argument("--few_shot", action="store_true", help="Use 5-shot few-shot prompt")
    parser.add_argument("--strict", action="store_true", help="Use strict answer extraction (must have ####)")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Max samples: {args.max_samples or 'full test set'}")
    print(f"Few-shot: {args.few_shot}")
    print(f"Strict extraction: {args.strict}")
    print(f"torch.compile: {args.compile} (mode: {args.compile_mode if args.compile else 'N/A'})")

    # Load dataset
    print("\nLoading GSM8K test split...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    print(f"Evaluating {len(dataset)} samples")

    # Run evaluations
    hf_results = None
    ic_results = None

    if not args.ic_only:
        hf_results = evaluate_huggingface(
            args.model, dataset, args.max_new_tokens,
            few_shot=args.few_shot, strict=args.strict
        )

    if not args.hf_only:
        ic_results = evaluate_ironcore(
            args.model, dataset, args.max_new_tokens,
            few_shot=args.few_shot, strict=args.strict
        )

    # Compare if both ran
    comparison = None
    if hf_results and ic_results:
        comparison = compare_results(hf_results, ic_results)

    # Save results
    output_path = Path("outputs/gsm8k_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "model": args.model,
        "max_samples": args.max_samples,
        "total_samples": len(dataset),
        "max_new_tokens": args.max_new_tokens,
    }

    if hf_results:
        output_data["huggingface"] = {
            "accuracy": hf_results["accuracy"],
            "correct": hf_results["correct"],
            "total": hf_results["total"],
            "time": hf_results["time"],
            "time_per_sample": hf_results["time_per_sample"],
        }

    if ic_results:
        output_data["ironcore"] = {
            "accuracy": ic_results["accuracy"],
            "correct": ic_results["correct"],
            "total": ic_results["total"],
            "time": ic_results["time"],
            "time_per_sample": ic_results["time_per_sample"],
        }

    if comparison:
        output_data["comparison"] = comparison

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save detailed results separately
    if hf_results:
        with open(output_path.parent / "gsm8k_hf_details.json", "w") as f:
            json.dump(hf_results["results"], f, indent=2)
    if ic_results:
        with open(output_path.parent / "gsm8k_ic_details.json", "w") as f:
            json.dump(ic_results["results"], f, indent=2)


if __name__ == "__main__":
    main()
