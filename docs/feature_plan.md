# Feature Plan

## Core Infrastructure
- [x] Configuration system
- [x] Logger integration (WandB)
- [x] Data loader
- [x] Checkpointing (native + HF interop)

## Model Components
- [x] Base model class (Embedding, PE, Transformer, LM Head)
- [x] Activations (GELU, SwiGLU, GeGLU, SiLU, etc.)
- [x] Pre-LN / Post-LN
- [x] LayerNorm / RMSNorm
- [x] MHA / GQA / MQA
- [x] Flash Attention support
- [x] Mixture-of-Experts (DeepSeek-MoE style, shared + routed experts, load-balance loss)
- [ ] Memory efficient attention variants
  - [ ] Sliding window attention
- [ ] Multi-Latent Attention

## Positional Embeddings
- [x] Absolute
- [x] RoPE
- [ ] ALiBi
- [ ] LongRoPE

## Layer Optimizations with DSL
- [ ] Triton
- [ ] TileLang

## Training
- [x] Training loop
- [x] Cross entropy loss
- [x] Automatic Mixed Precision
- [x] Activation checkpointing
- [x] Gradient clipping
- [x] MFU tracking

## Fine-tuning
- [x] Full fine-tuning (SFT, FIM)
- [x] LoRA (rank decomposition, TP-correct, async support)
- [ ] Adapter-based tuning
- [ ] MoRA / DoRA

## Alignment / RL
- [x] DPO (Direct Preference Optimization)
- [x] GRPO (Group Relative Policy Optimization)
  - [x] Online rollout generation (batched, prefix-cached)
  - [x] Group-relative advantage computation
  - [x] KL penalty (approximate + full)
  - [x] PPO-style IS ratio clipping
  - [x] Multi-epoch (offline) training support
  - [x] Reward backends: math, code, API (OpenAI/Anthropic/Google/Zhipu), local endpoint, local model, format, keyword
  - [x] GSM8K training config
  - [x] Strict math reward + strict format reward
  - [ ] Training validation and benchmarking
- [ ] RLAIF (Reinforcement Learning with AI Feedback)
- [ ] RLVR (Reinforcement Learning with Verifiable Rewards)

## Inference / Generation
- [x] HF checkpoint compatibility (save/load)
- [x] KV cache (paged attention, prefix caching)
- [x] Batched generation with sampling (temperature, top-p, top-k)
- [ ] vLLM / SGLang / nano-vllm style full inference engine
- [ ] EOS/stop token handling improvements
- [ ] Speculative decoding

## Distributed Training
- [x] DDP (Data Parallel)
- [x] FSDP integration
- [x] Distributed dataloader
- [x] Tensor Parallelism (synchronous)
- [x] Expert Parallelism (for MoE)
- [x] Asynchronous TP
- [ ] Pipeline Parallelism
- [ ] Context Parallelism

## Data Pipeline
- [x] Dataset preprocessing
- [x] Pretrain task support
- [x] SFT task support
- [x] FIM (Fill-in-the-Middle) task support
- [x] DPO data format support
- [x] GRPO data format support (prompt + verifiable answer / reward metadata)
- [ ] Tokenizer training

## Evaluation
- [x] Perplexity evaluation
- [x] Downstream task evaluation (HellaSwag)

# Design Principles

1. **Modular Components**: Language model components (layers) are prepared as building blocks
2. **Composable Architectures**: Model architectures are compounds of layers for research flexibility
3. **Flexible Parallelism**: DP, TP, PP, and CP options with memory profiling tools
4. **Research-First**: Integrated logging and experiment tracking

# Some Ambitions
- Support for a wide range of model architectures (decoder-only, MoE, etc.)
- State-of-the-art training techniques (DPO, GRPO, etc.)
- Efficient inference with KV caching and speculative decoding
- Seamless integration with Hugging Face ecosystem (checkpoint compatibility, tokenizers, etc.)
- Distributed training support for large models across multiple GPUs and nodes
- DSL-based optimizations for critical components (attention, MLPs) using Triton, TileLang, etc.
- Developing optimization harnesses for GPU optimization and memory efficiency