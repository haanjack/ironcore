# IronCore

**A from-scratch LLM training framework for learning & experimentation.**

IronCore is a personal project for practicing AI-systems development — built from the
ground up to understand LLM training internals: distributed training, parallelism,
alignment, and optimization. Inspired by NVIDIA Megatron-LM and HuggingFace Transformers.

## What's inside

- **Training modes** — pretraining, SFT, DPO, and GRPO (Group Relative Policy Optimization)
- **Parallelism** — Tensor (TP), Expert (EP), Data (DP), multi-node, and FSDP
- **Model architectures** — GPT-2/3, LLaMA, Gemma, Qwen, Phi via a single `TransformerModel`
- **Mixture of Experts** — expert routing with load-balance + Z-loss, expert parallelism
- **PEFT / LoRA** — TP-correct, replicated (not sharded) adapters
- **Alignment / RL** — online rollouts, group-relative advantages, KL penalty, multi-backend rewards
- **Optimizer** — Muon (Newton-Schulz) + AdamW hybrid; ZeRO-1 `DistributedOptimizer`
- **Offload** — optimizer-state offload, weight streaming, and activation spilling for single-GPU desktops
- **Checkpointing** — native (universal + distributed TP) and HuggingFace interop

## Where to start

| If you want to… | Go to |
| --- | --- |
| Install and run your first training job | [Getting started](getting_started.md) |
| Use the `ironcore` CLI | [CLI guide](cli_guide.md) · [CLI reference](cli_reference.md) |
| Understand TP/EP/DP/FSDP | [Parallelism](parallelism.md) |
| Fine-tune with DPO/GRPO | [Alignment](alignment.md) · [Reward manager](reward_manager.md) |
| Train beyond your VRAM | [Offload](offload.md) · [Offload design](design/offload.md) |
| Read the design docs | [Design](design/index.md) |
| Contribute | [Contributing](https://github.com/haanjack/ironcore/blob/main/CONTRIBUTING.md) · [CI/CD guide](ci_cd_guide.md) |

> **Hardware note.** IronCore is developed and tested on a dual RTX 3090 (NVLink) rig and
> requires the NGC PyTorch container for full functionality (flash attention ships with the
> base image). See [CONTRIBUTING.md](https://github.com/haanjack/ironcore/blob/main/CONTRIBUTING.md)
> for the container-first setup.
