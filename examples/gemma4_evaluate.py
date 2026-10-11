# Copyright (c) 2025-2026 Jaegeun Han
# SPDX-License-Identifier: Apache-2.0
"""Compare native Gemma 4 A4B LoRA with its BF16 base on fixed MMLU-Pro questions.

Requires vLLM with mixed 2D MoE LoRA support (validated on 0.30.0). The adapter
export is for vLLM, not a claim that Transformers PEFT can attach adapters to
Gemma's batched expert Parameters. This is a bounded regression evaluation;
Google's unpublished exact evaluation protocol is not reproduced by this script.

Dataset preparation:
  --prepare downloads a pinned MMLU-Pro revision and selects 10 per subject.
Evaluation:
  --checkpoint PATH --adapter PATH --output PATH --phases direct,thinking
Fast regression on two 24 GiB GPUs:
  --phases direct --quantization fp8-moe --offload-mode none
This quantizes only routed-expert weights; it does not reproduce BF16 scores.

Reference prompt: https://github.com/TIGER-AI-Lab/MMLU-Pro
"""

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path


def answer_text(text):
    # Gemma's channel END token closes thought; the answer has no "answer" label.
    if "<channel|>" in text:
        return text.rsplit("<channel|>", 1)[1].replace("<turn|>", "").strip()
    if "<|channel>thought" in text:
        return ""  # A truncated reasoning trace is not a final answer.
    return text.replace("<turn|>", "").strip()


def extract(text, n):
    text = answer_text(text)
    letters = "ABCDEFGHIJ"[:n]
    patterns = [
        r"(?:the\s+)?answer\s+is\s*\(?\s*([A-J])\s*\)?",
        r"\banswer\s*:\s*\(?\s*([A-J])",
        r"(?m)^\s*\(?([A-J])\)?\s*(?:[.:]|$)",
        r"\b(?:option|choice)\s+\(?([A-J])\)?",
        r"\\boxed\{\s*([A-J])\s*\}",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.I))
        if matches:
            p = matches[-1].group(1).upper()
            return p if p in letters else None
    return None


def example(row, answer):
    p = "Question:\n" + row["question"] + "\nOptions:\n"
    p += "".join(f"{chr(65 + i)}. {v}\n" for i, v in enumerate(row["options"]) if v != "N/A")
    if answer:
        return (
            p
            + row["cot_content"].replace(
                "A: Let's think step by step.", "Answer: Let's think step by step."
            )
            + "\n\n"
        )
    return p + "Answer: Let's think step by step."


def make_prompt(tok, row, vals, thinking):
    if thinking:
        p = (
            "The following are multiple choice questions (with answers) about "
            + row["category"]
            + '. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.\n'
        )
        for v in vals:
            if v["category"] == row["category"]:
                p += example(v, True)
        p += example(row, False)
    else:
        p = (
            "Choose the correct option. Reply with only the single option letter.\n"
            + example(row, False).split("Answer: Let's think step by step.")[0]
        )
    return tok.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def summary(rows):
    correct = sum(r["correct"] for r in rows)
    n = len(rows)
    p = correct / n
    z = 1.959964
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    cats = {}
    for row in rows:
        c = cats.setdefault(row["category"], {"correct": 0, "total": 0})
        c["correct"] += int(row["correct"])
        c["total"] += 1
    return {
        "correct": correct,
        "total": n,
        "accuracy": p,
        "wilson95": [center - rad, center + rad],
        "unparsed": sum(r["prediction"] is None for r in rows),
        "length_limited": sum(r["finish_reason"] == "length" for r in rows),
        "generated_tokens": sum(r["output_tokens"] for r in rows),
        "by_category": cats,
    }


def convert_adapter(source, target, checkpoint):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, target = Path(source), Path(target)
    meta = json.loads((source / "adapter_config.json").read_text())
    if meta.get("format") != "ironcore_lora_v1" or not meta.get("layout", "").startswith(
        "A[in_features"
    ):
        raise ValueError("Unsupported native adapter format")
    cfg = json.loads(Path(checkpoint, "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    if cfg.get("model_type") != "gemma4_text" or not cfg.get("enable_moe_block"):
        raise ValueError("This export supports Gemma 4 MoE text adapters")
    tensors = {}
    mapping = []
    native = 0
    counts = {"attention": 0, "shared_mlp": 0, "experts": 0, "shared_kv_copies": 0}
    with safe_open(str(source / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
        for key in f.keys():
            match = re.fullmatch(
                r"(model.layers.\d+).experts.(\d+).lora_(gate_proj|up_proj|down_proj).(lora_[AB])",
                key,
            )
            if match:
                parent, e, proj, ab = match.groups()
                name = f"{parent}.moe.experts.{e}.{proj}"
                group = "experts"
            else:
                match = re.fullmatch(
                    r"(model.layers.\d+.(?:self_attn|mlp).\w+).lora.(lora_[AB])", key
                )
                if not match:
                    raise ValueError(f"Unsupported adapter key: {key}")
                name, ab = match.groups()
                group = "shared_mlp" if ".mlp." in name else "attention"
            value = f.get_tensor(key)
            if (
                value.ndim != 2
                or value.dtype != torch.float32
                or value.shape[1 if ab == "lora_A" else 0] != meta["r"]
            ):
                raise ValueError(f"Invalid native adapter shape/dtype: {key}")
            native += value.numel()
            counts[group] += 1
            dest = "base_model.model." + name + "." + ab + ".weight"
            if dest in tensors:
                raise ValueError(f"Duplicate exported key: {dest}")
            tensors[dest] = value.T.contiguous()
            assert torch.equal(tensors[dest].T, value)
            mapping.append({"source": key, "target": dest, "shape": list(value.shape)})
            if ".self_attn.k_proj" in name:
                layer = int(name.split(".")[2])
                if cfg["attention_k_eq_v"] and cfg["layer_types"][layer] == "full_attention":
                    vk = dest.replace(".k_proj.", ".v_proj.")
                    if vk in tensors:
                        raise ValueError(f"Shared KV export collision: {vk}")
                    tensors[vk] = tensors[dest].clone()
                    counts["shared_kv_copies"] += 1
                    mapping.append(
                        {
                            "source": key,
                            "target": vk,
                            "shape": list(value.shape),
                            "shared_kv_copy": True,
                        }
                    )
    validate_adapter_coverage(mapping, cfg, meta)
    target.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(target / "adapter_model.safetensors"))
    out = {
        "base_model_name_or_path": str(checkpoint),
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": meta["r"],
        "lora_alpha": meta["alpha"],
        "lora_dropout": meta["dropout"],
        "target_modules": meta["target_modules"],
        "bias": "none",
        "inference_mode": True,
        "fan_in_fan_out": False,
        "use_rslora": False,
    }
    (target / "adapter_config.json").write_text(json.dumps(out, indent=2) + "\n")
    (target / "conversion.json").write_text(
        json.dumps(
            {
                "format": "vllm_gemma4_mixed_2d_moe_lora",
                "native_parameters": native,
                "tensor_counts": counts,
                "converted_parameters": sum(t.numel() for t in tensors.values()),
                "all_tensors_lossless_transpose": True,
                "mapping": mapping,
            },
            indent=2,
        )
        + "\n"
    )
    print(counts, flush=True)


def validate_adapter_coverage(mapping, cfg, meta):
    """Reject missing projection/expert tensors instead of dropping adapters."""
    expected = set()
    for layer in range(cfg["num_hidden_layers"]):
        for proj in meta["target_modules"]:
            if proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if (
                    proj == "v_proj"
                    and cfg["attention_k_eq_v"]
                    and cfg["layer_types"][layer] == "full_attention"
                ):
                    continue
                names = [f"model.layers.{layer}.self_attn.{proj}.lora"]
            elif proj in ("gate_proj", "up_proj", "down_proj"):
                names = [f"model.layers.{layer}.mlp.{proj}.lora"]
                names += [
                    f"model.layers.{layer}.experts.{e}.lora_{proj}"
                    for e in range(cfg["num_experts"])
                ]
            else:
                raise ValueError(f"Unsupported projection {proj}")
            expected.update(n + "." + ab for n in names for ab in ("lora_A", "lora_B"))
    observed = {r["source"] for r in mapping}
    if expected != observed:
        raise ValueError(
            f"Incomplete adapter coverage: {len(expected - observed)} missing, {len(observed - expected)} unexpected"
        )


def prepare_dataset(root, revision, per_category, seed):
    import random

    from datasets import load_dataset
    from huggingface_hub import HfApi

    name = "TIGER-Lab/MMLU-Pro"
    revision = revision or HfApi().dataset_info(name).sha
    data = load_dataset(name, revision=revision, cache_dir=str(root / "hf-cache"))
    rows, vals = list(data["test"]), list(data["validation"])
    selected = []
    categories = sorted({r["category"] for r in rows})
    for cat in categories:
        pool = [r for r in rows if r["category"] == cat]
        random.Random(seed).shuffle(pool)
        selected.extend(pool[:per_category])
    for row in selected:
        row["options"] = [x for x in row["options"] if x != "N/A"]
    for name, subset in [("mmlu-pro-test-subset", selected), ("mmlu-pro-validation", vals)]:
        (root / (name + ".jsonl")).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset)
        )
    info = {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "revision": revision,
        "test_count": len(rows),
        "selected_count": len(selected),
        "selection_seed": seed,
        "per_category": per_category,
        "categories": categories,
        "categories_test_counts": {c: sum(r["category"] == c for r in rows) for c in categories},
        "subset_sha256": hashlib.sha256(
            (root / "mmlu-pro-test-subset.jsonl").read_bytes()
        ).hexdigest(),
    }
    (root / "dataset-manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def generate_observed(engine, prompts, params, adapter, root, on_finished=None, seeds=None):
    """Persist live decode diagnostics even when one request is very long."""
    pending = {}
    completed = set()
    llm = engine.llm_engine
    start = time.time()
    last = 0
    for i, prompt in enumerate(prompts):
        request_params = copy.deepcopy(params)
        if seeds is not None:
            request_params.seed = seeds[i]
        llm.add_request(str(i), prompt, request_params, lora_request=adapter)
    while llm.has_unfinished_requests():
        for output in llm.step():
            pending[output.request_id] = output
            if output.finished and output.request_id not in completed:
                completed.add(output.request_id)
                if on_finished is not None:
                    on_finished(int(output.request_id), output)
        if time.time() - last >= 20:
            partial = [
                {
                    "id": k,
                    "tokens": len(v.outputs[0].token_ids),
                    "finished": v.finished,
                    "tail": v.outputs[0].text[-800:],
                }
                for k, v in pending.items()
            ]
            (root / "decode-progress.json").write_text(
                json.dumps({"elapsed": time.time() - start, "requests": partial}, indent=2) + "\n"
            )
            last = time.time()
    return [pending[str(i)] for i in range(len(prompts))]


def main():
    import argparse

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    a = argparse.ArgumentParser()
    a.add_argument("--checkpoint", type=Path, required=True)
    a.add_argument("--adapter", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--prepare", action="store_true")
    a.add_argument("--dataset-revision")
    a.add_argument("--per-category", type=int, default=10)
    a.add_argument("--limit", type=int, default=140)
    a.add_argument("--max-new-tokens", type=int, default=8192)
    a.add_argument("--temperature", type=float, default=1.0)
    a.add_argument("--top-p", type=float, default=0.95)
    a.add_argument("--top-k", type=int, default=64)
    a.add_argument("--phases", default="direct,thinking")
    a.add_argument("--batch-size", type=int, default=16)
    a.add_argument("--cpu-offload-gb", type=float, default=6)
    a.add_argument("--offload-mode", choices=["none", "uva", "prefetch"], default="prefetch")
    a.add_argument("--quantization", choices=["bf16", "fp8-moe"], default="bf16")
    a.add_argument("--offload-group-size", type=int, default=2)
    a.add_argument("--max-batched-tokens", type=int, default=4096)
    args = a.parse_args()
    ROOT = args.output.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    BASE = str(args.checkpoint.resolve())
    lock = open(ROOT / "engine.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.prepare:
        print(json.dumps(prepare_dataset(ROOT, args.dataset_revision, args.per_category, 3407)))
        return
    if args.limit < 1 or args.batch_size < 1 or args.max_new_tokens < 1:
        a.error("Limits and batch size must be positive")
    if any(p not in ("direct", "thinking") for p in args.phases.split(",")):
        a.error("Allowed phases: direct,thinking")
    convert_adapter(args.adapter.resolve(), ROOT / "vllm-adapter", args.checkpoint.resolve())
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(BASE)
    rs = [
        json.loads(line) for line in (ROOT / "mmlu-pro-test-subset.jsonl").read_text().splitlines()
    ]
    # Interleave subjects, so pilots cover subjects rather than only biology.
    rs = [
        rs[c * args.per_category + i]
        for i in range(args.per_category)
        for c in range(len(set(r["category"] for r in rs)))
    ][: args.limit]
    vals = [
        json.loads(line) for line in (ROOT / "mmlu-pro-validation.jsonl").read_text().splitlines()
    ]
    with (args.adapter / "adapter_model.safetensors").open("rb") as source:
        adapter_sha = hashlib.file_digest(source, "sha256").hexdigest()
    protocol = {
        "adapter_sha256": adapter_sha,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(
            (ROOT / "mmlu-pro-validation.jsonl").read_bytes()
        ).hexdigest(),
        "limit": args.limit,
        "phases": args.phases,
        "checkpoint": BASE,
        "adapter": str(args.adapter.resolve()),
        "dataset": json.loads((ROOT / "dataset-manifest.json").read_text()),
        "thinking_temperature": args.temperature,
        "thinking_top_p": args.top_p,
        "thinking_top_k": args.top_k,
        "direct_temperature": 0,
        "sampling_source": "https://huggingface.co/google/gemma-4-26B-A4B-it",
        "sampling_note": "Recommended generation defaults; not a published benchmark-specific protocol",
        "thinking_fewshot": 5,
        "direct_fewshot": 0,
        "thinking_max_new_tokens": args.max_new_tokens,
        "direct_max_new_tokens": 32,
        "seed": 3407,
        "per_question_seed": "3407 + question_id, identical for base and trained",
        "tensor_parallel_size": 2,
        "cpu_offload_gb": args.cpu_offload_gb if args.offload_mode == "uva" else 0,
        "cpu_offload_params": ["moe.experts"],
        "offload_mode": args.offload_mode,
        "worker_compatibility_shim": "examples.gemma4_eval_worker.PrefetchLoRAWorker"
        if args.offload_mode == "prefetch"
        else None,
        "offload_group_size": args.offload_group_size if args.offload_mode == "prefetch" else 0,
        "batch_size": args.batch_size,
        "base_dtype": "bfloat16",
        "moe_weight_quantization": "fp8_per_tensor_static"
        if args.quantization == "fp8-moe"
        else None,
        "inference_lora_dtype": "bfloat16",
        "original_lora_dtype": "float32",
        "official_reproduction": False,
        "reason": "Google exact prompt, shots, generation budget and sampling protocol not published in the technical report; fixed stratified subset instead of full benchmark.",
    }
    protocol_path = ROOT / f"protocol-{args.limit}.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Evaluation protocol changed; use a fresh output directory")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    (ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    if args.offload_mode == "none":
        offload_kwargs = {"cpu_offload_gb": 0, "offload_group_size": 0}
    elif args.offload_mode == "uva":
        offload_kwargs = {
            "cpu_offload_gb": args.cpu_offload_gb,
            "cpu_offload_params": {"moe.experts"},
        }
    else:
        offload_kwargs = {
            "offload_group_size": args.offload_group_size,
            "offload_num_in_group": 1,
            "offload_prefetch_step": 1,
            "worker_cls": "examples.gemma4_eval_worker.PrefetchLoRAWorker",
        }
    quantization_kwargs = (
        {
            "quantization": "fp8_per_tensor",
            "quantization_config": {"linear": {"weight": None, "activation": None}},
        }
        if args.quantization == "fp8-moe"
        else {}
    )
    engine = LLM(
        model=BASE,
        tensor_parallel_size=2,
        dtype="bfloat16",
        **offload_kwargs,
        **quantization_kwargs,
        gpu_memory_utilization=0.91,
        max_model_len=32768,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.max_batched_tokens,
        enforce_eager=True,
        hf_overrides={"architectures": ["Gemma4ForCausalLM"]},
        enable_lora=True,
        max_lora_rank=8,
        enable_mixed_moe_lora_format=True,
        enable_prefix_caching=False,
        disable_custom_all_reduce=True,
        compilation_config={"mode": 0},
        seed=3407,
    )
    states = [("base", None), ("trained", LoRARequest("trained", 1, str(ROOT / "vllm-adapter")))]
    result = {}
    for phase in args.phases.split(","):
        thinking = phase == "thinking"
        prompts = [make_prompt(tok, r, vals, thinking) for r in rs]
        params = SamplingParams(
            temperature=args.temperature if thinking else 0,
            top_p=args.top_p if thinking else 1.0,
            top_k=args.top_k if thinking else -1,
            max_tokens=args.max_new_tokens if thinking else 32,
            stop_token_ids=[1, 106],
            skip_special_tokens=False,
            logprobs=None if thinking else 10,
            seed=3407,
        )
        for name, adapter in states:
            dest = ROOT / f"mmlu-{phase}-{name}-{args.limit}.jsonl"
            if dest.exists():
                done = [json.loads(line) for line in dest.read_text().splitlines()]
                ids = {r["question_id"] for r in done}
            else:
                done = []
                ids = set()
            todo = [(r, p) for r, p in zip(rs, prompts, strict=True) if r["question_id"] not in ids]
            # Enqueue all remaining questions so completed requests are replaced
            # immediately. max_num_seqs still bounds GPU concurrency.
            start = time.time()
            with dest.open("a") as f:

                def save_result(
                    index, output, todo=todo, start=start, done=done, phase=phase, name=name
                ):
                    r, p = todo[index]
                    o = output.outputs[0]
                    pred = extract(o.text, len(r["options"]))
                    first = {
                        str(k): {"lp": v.logprob, "text": v.decoded_token}
                        for k, v in (o.logprobs[0] if o.logprobs else {}).items()
                    }
                    row = {
                        "question_id": r["question_id"],
                        "category": r["category"],
                        "answer": r["answer"],
                        "prediction": pred,
                        "correct": pred == r["answer"],
                        "prompt_tokens": len(output.prompt_token_ids),
                        "output_tokens": len(o.token_ids),
                        "text": o.text,
                        "finish_reason": o.finish_reason,
                        "first_token_logprobs": first,
                        "thought_channel": "<|channel>thought" in o.text,
                        "completed_thought": "<channel|>" in o.text,
                        "prompt_sha256": hashlib.sha256(p.encode()).hexdigest(),
                        "evaluation_elapsed": time.time() - start,
                    }
                    done.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    if len(done) % 16 == 0 or len(done) == len(rs):
                        print(
                            json.dumps(
                                {
                                    "phase": phase,
                                    "model": name,
                                    "progress": len(done),
                                    "summary": summary(done),
                                }
                            ),
                            flush=True,
                        )

                if todo:
                    generate_observed(
                        engine,
                        [p for r, p in todo],
                        params,
                        adapter,
                        ROOT,
                        on_finished=save_result,
                        seeds=[3407 + int(r["question_id"]) for r, p in todo],
                    )
            result[phase + "_" + name] = summary(done)
            (ROOT / f"mmlu-summary-{args.limit}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
