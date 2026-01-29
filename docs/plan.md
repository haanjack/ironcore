# To Do

## Core Infrastructure
- [x] Configuration system
- [x] Logger integration
- [x] Data loader
- [x] Checkpointing

## Model Components
- [x] Base model class (Embedding, PE, Transformer, LM Head)
- [x] Activations (GELU, SwiGLU, etc.)
- [x] Pre-LN / Post-LN
- [x] LayerNorm / RMSNorm
- [x] MHA / GQA / MQA
- [x] Flash Attention support
- [ ] Memory efficient attention variants
  - [ ] Sliding window attention
- [ ] Parallel layers
- [ ] Multi-Latent Attention
- [ ] Mixture-of-Experts

## Positional Embeddings
- [x] Absolute
- [x] RoPE
- [ ] ALiBi
- [ ] LongRoPE

## Layer Optimizations
- [ ] Triton
- [ ] TileLang
- [ ] Mojo

## Training
- [x] Training loop
- [x] Cross entropy loss
- [x] Automatic Mixed Precision
- [x] Activation checkpointing
- [x] Gradient clipping

## Inference Integration Integration
- [x] HF checkpoint compatibility
- [ ] vLLM / SGLang / nano-vllm style implementation

## Distributed Training
- [x] DDP (Data Parallel)
- [x] FSDP integration
- [x] Distributed Dataloader
- [x] Tensor Parallelism
  - [x] Synchronous TP
  - [ ] Asynchronous TP
- [ ] Pipeline Parallelism
  - [ ] Synchronous PP
  - [ ] ZeroBubble
- [ ] Context Parallelism

## Data Pipeline
- [x] Dataset preprocessing
- [x] Pretrain task support
- [x] SFT task support
- [ ] RLHF task support
- [ ] Tokenizer training

## Fine-tuning
- [x] Full Finetuning
- [ ] LoRA
- [ ] Adapter-based tuning
- [ ] MoRA / DoRA

## RLHF
- [ ] Artifact Store
- [ ] Algorithm Implementation
    - [ ] PPO
    - [ ] DPO

## Evaluation
- [x] Perplexity evaluation
- [x] Downstream task evaluation

# Design Principles

1. **Modular Components**: Language model components (layers) are prepared as building blocks
2. **Composable Architectures**: Model architectures are compounds of layers for research flexibility
3. **Flexible Parallelism**: DP, TP, PP, and CP options with memory profiling tools
4. **Research-First**: Integrated logging and experiment tracking

# Agentic Design
1. Kernel Optimization Experiments
- [ ] Triton Kernel Generation / Optimization
