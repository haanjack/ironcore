# DSL Kernel Development Plan

## Overview

This document describes the plan for implementing custom GPU kernels using DSL frameworks (Triton, TileLang) and building an automated kernel generation pipeline within IronCore.

The project has two goals:
1. **Manual kernel implementation** — Replace PyTorch layer placeholders with fused Triton/TileLang kernels
2. **Automated DSL generation** — Build a harness that enables a coding agent to generate, validate, and benchmark kernels against reference implementations

## Environment

| Component | Version |
|-----------|---------|
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| Triton | 3.4.0 |
| TileLang | TBD (not yet installed) |
| GPU | RTX 4070 Laptop (sm_89, 8GB VRAM) |
| Python | 3.12 |

## Directory Structure

```
ironcore/
    kernels/                        # Kernel implementations (manual + generated)
        __init__.py
        triton/                     # Triton kernel implementations
            __init__.py
            rmsnorm.py              # Fused RMSNorm forward + backward
            layernorm.py            # Fused LayerNorm forward + backward
            softmax.py              # Fused safe-softmax
            rope.py                 # Fused Rotary Position Embedding
            cross_entropy.py        # Fused cross-entropy loss
            activations.py          # Fused SwiGLU, GeGLU, etc.
        tilelang/                   # TileLang kernel implementations
            __init__.py

experiments/
    generation/                     # Automated DSL generation pipeline
        __init__.py
        spec.py                     # KernelSpec dataclass
        harness.py                  # Validation + benchmark harness
        specs/                      # Per-kernel specifications
            __init__.py
            rmsnorm.py              # RMSNorm kernel spec
            layernorm.py            # LayerNorm kernel spec
            softmax.py              # Softmax kernel spec
            rope.py                 # RoPE kernel spec
            activations.py          # Activation kernel specs
            cross_entropy.py        # Cross-entropy kernel spec
        results/                    # Benchmark results and logs
            .gitkeep
```

## Target Kernels

Priority order based on impact and complexity:

### Phase 1 — Foundations
| Kernel | Reference Implementation | Complexity | Expected Speedup |
|--------|--------------------------|------------|------------------|
| Fused RMSNorm | `layers/layernorm/fused_rms_norm.py` | Low | 2-3x |
| Fused LayerNorm | `layers/layernorm/fused_layer_norm.py` | Low | 2-3x |
| Fused Softmax | `layers/attention.py:119-124` | Low | 1.5-2x |

### Phase 2 — Element-wise Fusions
| Kernel | Reference Implementation | Complexity | Expected Speedup |
|--------|--------------------------|------------|------------------|
| Fused SwiGLU | `layers/activations/activations.py:37-38` | Low | 1.5-2x |
| Fused RoPE | `layers/positional_embedding/rotary.py` | Medium | 1.5-2x |

### Phase 3 — Complex Kernels
| Kernel | Reference Implementation | Complexity | Expected Speedup |
|--------|--------------------------|------------|------------------|
| Fused Cross-Entropy | `parallel/tensor_parallel/cross_entropy.py` | High | 1.5-2x |
| Flash Attention (Triton) | `layers/attention.py:68-142` | High | Depends on seq length |

## Generation Pipeline Design

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐
│ Kernel Spec  │────>│ Agent writes │────>│  Harness     │────>│  Result   │
│ (spec.py)    │     │ Triton code  │     │  validates   │     │  pass/fail│
└─────────────┘     └──────────────┘     └──────────────┘     └───────────┘
       │                                        │                     │
       │                                        v                     │
       │                                 ┌──────────────┐             │
       └────── reference fn ────────────>│  Numerical   │             │
                                         │  comparison  │<────────────┘
                                         └──────────────┘
```

### Agent Workflow

1. Agent reads `KernelSpec` — understands input/output shapes, dtypes, reference behavior
2. Agent reads reference PyTorch implementation — understands the algorithm
3. Agent reads existing manual kernels (if any) — learns project patterns
4. Agent writes Triton/TileLang kernel to `ironcore/kernels/`
5. Agent runs `python -m experiments.generation.harness <kernel_name>`
6. Harness reports: correctness (pass/fail), gradient check (pass/fail), performance (speedup)
7. Agent iterates if needed

### Harness CLI

```bash
# Validate a specific kernel
python -m experiments.generation.harness rmsnorm

# Validate all kernels
python -m experiments.generation.harness --all

# Benchmark only (skip correctness)
python -m experiments.generation.harness rmsnorm --benchmark-only

# Verbose output with shapes
python -m experiments.generation.harness rmsnorm --verbose
```

## Integration with Existing Layers

Kernels integrate through the existing layer hierarchy. The layer files remain the public API; kernels are implementation details.

```python
# layers/layernorm/fused_rms_norm.py — BEFORE (current placeholder)
class RmsNorm(BaseModule):
    def __init__(self, config):
        self.layernorm = nn.RMSNorm(config.model.d_model, ...)
    def forward(self, x):
        return self.layernorm(x)

# layers/layernorm/fused_rms_norm.py — AFTER (with Triton kernel)
class RmsNorm(BaseModule):
    def __init__(self, config):
        self.weight = nn.Parameter(torch.ones(config.model.d_model))
        self.eps = config.model.ln_eps
    def forward(self, x):
        return triton_rmsnorm(x, self.weight, self.eps)
```

The switch from PyTorch to Triton is internal to the layer — no changes needed in model code.

## Design Principles

1. **Reference first** — Every kernel has a PyTorch reference. The reference is the spec.
2. **Correctness before speed** — A correct slow kernel beats an incorrect fast one.
3. **Harness-driven** — All kernels are validated through the same pipeline.
4. **Agent-compatible** — Specs and harness output are designed for coding agent consumption.
5. **Incremental adoption** — Each kernel can be swapped in independently. Fallback to PyTorch if kernel is unavailable.
