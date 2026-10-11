# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Prepare fixed, disjoint public-data subsets for Gemma 4 learning validation.

Short: FineTome shuffled across its entire training split. Long: random Dataset
Viewer pages of LongAlign, remeasured with the actual Gemma tokenizer. Neither
conversation nor assistant answer is truncated to fit the requested window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import requests
from datasets import load_dataset
from transformers import AutoTokenizer

from ironcore.config import DataConfig
from ironcore.preprocessing.gemma4_chat import (
    gemma4_sft_chat_template,
    gemma4_sft_format_metadata,
)
from ironcore.preprocessing.serializer import DataSerializer


def hub_revision(dataset):
    response = requests.get(f"https://huggingface.co/api/datasets/{dataset}", timeout=60)
    response.raise_for_status()
    return response.json()["sha"]


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["short", "long"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", default=".local/models/gemma-4-26B-A4B-it")
    parser.add_argument("--train-samples", type=int, default=240)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--min-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=2049)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = "mlabonne/FineTome-100k" if args.kind == "short" else "zai-org/LongAlign-10k"
    revision = hub_revision(dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    template = gemma4_sft_chat_template(tokenizer.chat_template)
    serializer = DataSerializer(
        DataConfig(
            task_type="sft",
            preprocessed_dir=args.output / "scratch",
            cache_dir=args.output / "cache",
        ),
        tokenizer,
        verbose=False,
    )
    needed = args.train_samples + args.eval_samples + args.test_samples
    selected, seen_conversations, seen_prompts = [], set(), set()
    inspected = 0
    if args.kind == "short":
        raw = load_dataset(
            dataset,
            revision=revision,
            split="train",
            cache_dir=str(args.output / "hf-cache"),
        )
        candidates = (
            (int(index), raw[int(index)])
            for index in random.Random(args.seed).sample(range(len(raw)), len(raw))
        )
    else:
        # Viewer is read-only. Verify the Hub revision again at completion so a
        # concurrent dataset update cannot silently change the selected subset.
        def long_rows():
            offsets = list(range(0, 9888, 100))
            random.Random(args.seed).shuffle(offsets)
            for offset in offsets:
                response = requests.get(
                    "https://datasets-server.huggingface.co/rows",
                    params={
                        "dataset": dataset,
                        "config": "default",
                        "split": "train",
                        "offset": offset,
                        "length": 100,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                for item in response.json()["rows"]:
                    if item.get("truncated_cells"):
                        raise ValueError("Dataset Viewer truncated a conversation")
                    yield item["row_idx"], item["row"]

        candidates = long_rows()
    for index, row in candidates:
        inspected += 1
        if args.kind == "short":
            roles = {"human": "user", "gpt": "assistant", "system": "system"}
            if any(turn["from"] not in roles for turn in row["conversations"]):
                continue
            messages = [
                {"role": roles[turn["from"]], "content": turn["value"]}
                for turn in row["conversations"]
            ]
        else:
            messages = row["messages"]
        if not messages or messages[-1]["role"] != "assistant":
            continue
        conversation_hash = digest(json.dumps(messages, ensure_ascii=False, sort_keys=True))
        first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
        normalized = re.sub(r"\s+", " ", first_user).strip().lower()
        # LongAlign can ask multiple questions about a shared source document.
        prompt_hash = digest(normalized if args.kind == "short" else normalized[:2000])
        if conversation_hash in seen_conversations or prompt_hash in seen_prompts:
            continue
        try:
            tokens, masks = serializer._apply_chat_template_and_get_masks(
                messages, chat_template=template, chat_template_kwargs={"enable_thinking": False}
            )
        except ValueError:
            continue
        if not args.min_tokens <= len(tokens) <= args.max_tokens:
            continue
        supervised = (
            len(tokens)
            - 1
            - sum(max(0, min(len(tokens), end) - max(1, start)) for start, end in masks)
        )
        if supervised < 16:
            continue
        seen_conversations.add(conversation_hash)
        seen_prompts.add(prompt_hash)
        selected.append(
            {
                "messages": messages,
                "source_row": index,
                "source": row.get("source", dataset),
                "token_count": len(tokens),
                "assistant_token_count": supervised,
                "conversation_sha256": conversation_hash,
                "prompt_sha256": prompt_hash,
            }
        )
        print(f"SELECTED {len(selected)}/{needed} row={index} tokens={len(tokens)}", flush=True)
        if len(selected) == needed:
            break
    if len(selected) != needed:
        raise RuntimeError(f"Only {len(selected)} eligible conversations, needed {needed}")
    if hub_revision(dataset) != revision:
        raise RuntimeError("Dataset revision changed during preparation")
    if args.kind == "long":
        # Spread conversations from different Viewer pages across the splits.
        random.Random(args.seed).shuffle(selected)
    splits = {
        "train": selected[: args.train_samples],
        "validation": selected[args.train_samples : args.train_samples + args.eval_samples],
        "test": selected[args.train_samples + args.eval_samples :],
    }
    manifest = {
        "dataset": dataset,
        "revision": revision,
        "seed": args.seed,
        "selection": "whole-split shuffle" if args.kind == "short" else "shuffled Viewer pages",
        "inspected": inspected,
        **gemma4_sft_format_metadata(template, enable_thinking=False),
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "no_truncation": True,
        "dedup": "conversation and normalized first-user prompt",
        "splits": {},
    }
    for name, rows in splits.items():
        path = args.output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        manifest["splits"][name] = {
            "samples": len(rows),
            "file_sha256": digest(path.read_text()),
            "source_rows": [r["source_row"] for r in rows],
            "total_tokens": sum(r["token_count"] for r in rows),
            "assistant_tokens": sum(r["assistant_token_count"] for r in rows),
            "length_range": [
                min(r["token_count"] for r in rows),
                max(r["token_count"] for r in rows),
            ],
        }
    (args.output / "train_probe.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in splits["train"][:8])
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
