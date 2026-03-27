#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: Apache-2.0
"""Compare GSM8K evaluation between HuggingFace and IronCore inference."""

import argparse
import os
import re
import sys
import time

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
    print(f"\n{'=' * 60}")
    print(f"Evaluating with HuggingFace (few_shot={few_shot}, strict={strict})")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    system_prompt = (
        "Solve the math problem step by step. Put your final numerical answer after ####."
    )

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
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            result = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            if hasattr(result, "input_ids"):
                input_ids = result["input_ids"]
            elif isinstance(result, list):
                input_ids = torch.tensor(result, dtype=torch.long).unsqueeze(0)
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

        generated_ids = output[0][input_ids.shape[1] :].tolist()
        response = tokenizer.batch_decode([generated_ids], skip_special_tokens=True)[0]
        pred = extract_answer(response, strict=strict)
        is_correct = pred == gold_answer

        if is_correct:
            correct += 1
        total += 1

        results.append(
            {
                "gold": gold_answer,
                "pred": pred,
                "correct": is_correct,
                "response": response,
            }
        )

    time.time() - start_time
    accuracy = 100 * correct / total
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def evaluate_ironcore(
    model_name: str,
    dataset,
    max_new_tokens: int = 512,
    config_path: str = "configs/eval_qwen_gsm8k.yaml",
    few_shot: bool = False,
    strict: bool = False,
) -> dict:
    """Evaluate using IronCore inference."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating with IronCore (few_shot={few_shot}, strict={strict})")
    print(f"{'=' * 60}")

    from ironcore import get_tokenizer
    from ironcore.checkpointing import load_from_huggingface
    from ironcore.config import load_trainer_config
    from ironcore.global_vars import set_global_states
    from ironcore.language_model import LanguageModel
    from ironcore.parallel.parallel_states import initialize_model_parallel

    try:
        initialize_model_parallel(tensor_model_parallel_size=1, timeout_in_minutes=10)
    except Exception as e:
        print(f"Warning: Failed to initialize model parallel: {e}")
        pass

    sys.argv = ["eval", "--config-path", config_path]
    config = load_trainer_config()
    try:
        set_global_states(config)
    except Exception as e:
        print(f"Warning: Failed to set global states: {e}")
        pass

    cache_dir = snapshot_download(model_name)
    model = LanguageModel(config).to("cuda").to(dtype=torch.bfloat16)
    load_from_huggingface(cache_dir, model, "qwen2", strict=False)
    model.eval()

    tokenizer = get_tokenizer()
    system_prompt = (
        "Solve the math problem step by step. Put your final numerical answer after ####."
    )

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
            else:
                input_ids = enc

        input_ids = input_ids.to(device=model.device, dtype=torch.long)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output[0][input_ids.shape[1] :].tolist()
        response = tokenizer.batch_decode([generated_ids], skip_special_tokens=True)[0]
        pred = extract_answer(response, strict=strict)
        is_correct = pred == gold_answer

        if is_correct:
            correct += 1
        total += 1

        results.append(
            {
                "gold": gold_answer,
                "pred": pred,
                "correct": is_correct,
                "response": response,
            }
        )

    time.time() - start_time
    accuracy = 100 * correct / total
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset("openai/gsm8k", "main", split="test").select(range(args.max_samples))

    hf_res = evaluate_huggingface(args.model, dataset, few_shot=args.few_shot, strict=args.strict)
    print(f"HF Accuracy: {hf_res['accuracy']:.2f}%")

    ic_res = evaluate_ironcore(args.model, dataset, few_shot=args.few_shot, strict=args.strict)
    print(f"IC Accuracy: {ic_res['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
