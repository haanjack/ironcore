# AI-Driven Triton Kernel Generation Pipeline

## Quick Reference

### Available Commands

| Command | Description |
|---------|-------------|
| `ironcore explore <kernel>` | Run exploration phase (7-stage analysis) |
| `ironcore auto <kernel>` | Run full autonomous pipeline (generation + optimization) |
| `python -m experiments.generation.harness_autonomous <kernel> --provider glm` | Direct autonomous generation |
| `python -m experiments.generation.exploration <kernel> --provider glm` | Direct exploration phase |

## Manual Pipeline Execution Guide

### Phase 1: Exploration Stage (Multi-Stage Analysis)

```bash
# Run exploration on softmax
cd /home/hanjack/ironcore-dsl-alpha
python -m experiments.generation.exploration softmax --provider glm
```

**What happens:**
1. **Stage 1**: AI analyzes your algorithm mathematically
2. **Stage 2**: AI identifies operations, reductions, parallelization opportunities
3. **Stage 3**: AI determines optimal tiling strategy (BLOCK_SIZE, register pressure)
4. **Stage 4**: AI creates structured conversion plan (computational passes)
5. **Stage 5**: AI designs detailed kernel code structure
6. **Stage 6**: AI generates complete Triton code
7. **Stage 7**: AI tests code and refines based on diagnostics

**Output:**
- `/experiments/generation/results/exploration/<kernel>_timestamp.json` - Full analysis JSON
- `/ironcore/kernels/triton/<kernel>.py` - Generated kernel code

### Phase 2: Autonomous Generation (Iterative Refinement)

```bash
# Run autonomous pipeline with multiple iterations
python -m experiments.generation.harness_autonomous softmax --provider glm --max-iterations 5
```

**What happens:**
1. Generates initial kernel code
2. Validates correctness (vs PyTorch reference)
3. Validates gradients
4. Runs profiler if correctness passes
5. AI analyzes profiling results
6. AI generates optimized code
7. Repeats until:
   - All performance targets met, OR
   - Max iterations reached

### Phase 3: Performance Comparison

```bash
# Run benchmark comparison
python -c "
import torch
import time

# Reference PyTorch
x = torch.randn(1000, 512, device='cuda')
weight = torch.randn(512, device='cuda')

# Warmup
for _ in range(10):
    y = torch.softmax(x, dim=-1)

# Benchmark
start = time.perf_counter()
for _ in range(100):
    y = torch.softmax(x, dim=-1)
torch.cuda.synchronize()
torch_time = (time.perf_counter() - start) * 1000

print(f'PyTorch softmax: {torch_time:.3f}ms')
"

# Generated Triton
# (load and test your generated kernel)
```

## Current Status (2026-02-11)

### Successfully Generated Kernels

| Kernel | Speedup | Status | Approach |
|--------|--------|--------|----------|
| **RMSNorm** | 2.48x | ✅ Working | Autonomous harness |
| **Softmax** | 0.55x | ⚠️ Correct but slow | Exploration pipeline (7-stage) |
| **LayerNorm** | - | ❌ Failed | Both approaches |
| **GLU** | - | ❌ Failed | Both approaches |
| **CrossEntropy** | - | ❌ Failed | Both approaches |

### Approach Comparison

| Aspect | Exploration (7-stage) | Autonomous (Direct) |
|--------|----------------------|---------------------|
| **Correctness Rate** | Higher (richer context) | Lower |
| **Generation Time** | ~20 seconds | ~2 minutes per iteration |
| **Token Usage** | ~14K tokens | ~10K per iteration |
| **Best For** | Complex kernels | Simple kernels |

**Softmax Case Study (2026-02-11)**:
- **Exploration**: Correct kernel, 14,341 tokens, 7 stages, 0.55x speedup
- **Autonomous**: Failed after 3 iterations, 30,071 tokens, numerical errors

### Pipeline Components

1. **Exploration Phase** (`/experiments/generation/exploration/`)
   - 7-stage AI engagement pipeline
   - Multi-stage reasoning before code generation
   - Rich context propagation

2. **Autonomous Harness** (`/experiments/generation/harness_autonomous.py`)
   - Iterative refinement loop
   - Correctness validation → Profiling → AI optimization

3. **AI Providers** (`/experiments/generation/ai_providers/`)
   - OpenAI-compatible API (GLM-4.7, Kimi, etc.)
   - Code extraction with markdown handling

4. **Specs** (`/experiments/generation/specs/`)
   - rmsnorm, layernorm, softmax, glu, swiglu, cross_entropy

## Development Roadmap

### Phase 1: Core Functionality ✅ COMPLETE
- [x] Basic autonomous generation
- [x] Multi-stage exploration pipeline
- [x] OpenAI-compatible provider support
- [x] Code extraction with markdown handling
- [x] Shape handling for arbitrary inputs
- [x] Triton 3.4.0 compatibility

### Phase 2: Enhancement (In Progress)
- [ ] Multi-stage validation (isolate error components)
- [ ] Profiler integration (GPU metrics for AI)
- [ ] Reference pattern library (proven code snippets)
- [ ] Numerical precision analyzer
- [ ] Hypothesis testing framework
- [ ] Memory layout analyzer

### Phase 3: Advanced Features (Planned)
- [ ] Tile-size auto-tuning
- [ ] Kernel fusion opportunities detection
- [ ] Distributed kernel support
- [ ] Automatic benchmarking suite
- [ ] Performance regression testing

## Creating Custom Kernel Specs

### Template

```python
# File: experiments/generation/specs/your_layer.py

import torch
from experiments.generation.spec import KernelSpec, register_spec

def _reference_your_layer(x, weight, bias):
    """Your PyTorch reference implementation."""
    # Your algorithm here
    return result

def _make_inputs(dtype=torch.float32, device="cuda"):
    """Create test inputs."""
    return (torch.randn(..., device=device), ...)

register_spec(KernelSpec(
    name="your_layer",
    description="Layer description",
    reference_fn=_reference_your_layer,
    input_factory=_make_inputs,
    check_backward=True,
    target_file="ironcore/kernels/triton/your_layer.py",
    kernel_fn_name="triton_your_layer",
))
```

### Add to Registry

Update `/experiments/generation/specs/__init__.py`:
```python
from experiments.generation.specs import rmsnorm, layernorm, softmax, glu, cross_entropy, your_layer
```

## Troubleshooting

### Issue: "Unknown kernel spec"
**Solution**: Import the spec file first or add to `__init__.py`

### Issue: Authentication error (401)
**Solution**: Set `OPENAI_API_KEY` in `.env` for GLM:
```
OPENAI_API_KEY=5f6a5a6800374487b3e67126ed049b16.NQNRPeZ19ISQkquH
```

### Issue: Generated code has numerical errors
**Solution**: Check:
1. Using scalar accumulators: `acc = 0.0; acc += tl.sum(x, axis=0)`
2. Not using `tl.pow()` - use `x * x` instead
3. Float32 buffers for weight gradients

### Issue: Code extraction returns empty
**Solution**: Fixed in latest version - code extraction now handles ````python` correctly

## File Structure

```
ironcore-dsl-alpha/
├── experiments/generation/
│   ├── exploration/
│   │   ├── kernel_explorer.py      # 7-stage explorer
│   │   ├── prompts_exploration.py  # Stage-specific prompts
│   │   └── explorer_cli.py         # CLI interface
│   ├── harness_autonomous.py        # Iterative refinement
│   ├── harness.py                    # Validation & benchmarking
│   ├── ai_providers/               # AI model interfaces
│   ├── specs/                        # Kernel specifications
│   └── results/
│       ├── exploration/             # Exploration results
│       └── autonomous_*             # Autonomous results
├── ironcore/
│   ├── kernels/triton/              # Generated kernels
│   └── __main__.py                 # CLI entry point
└── .env                            # API keys
```

## Common Workflows

### Workflow 1: Generate New Kernel

```bash
# 1. Create spec in experiments/generation/specs/my_layer.py
# 2. Run exploration phase
python -m experiments.generation.exploration my_layer --provider glm

# 3. Check generated kernel
python -c "
from ironcore.kernels.triton.my_layer import triton_my_layer
import torch
x = torch.randn(4, 512, 768, device='cuda')
weight = torch.randn(768, device='cuda')
result = triton_my_layer(x, weight)
print('Success!')
"
```

### Workflow 2: Compare Approaches

```bash
# Direct generation
python -m experiments.generation.harness_autonomous softmax --provider glm --max-iterations 1

# Exploration-enhanced
python -m experiments.generation.exploration softmax --provider glm

# Compare results in:
ls experiments/generation/results/autonomous_softmax_*
ls experiments/generation/results/exploration/softmax_*
```

### Workflow 3: Debug Failed Generation

```bash
# Check debug output
python -m experiments.generation.harness_autonomous softmax --provider glm --max-iterations 1

# Check exploration result (for multi-stage analysis)
cat experiments/generation/results/exploration/softmax_*/json

# Check debug files for raw AI responses
ls experiments/generation/results/debug/
```

## Next Steps

### Immediate (This Week)
1. Fix softmax numerical precision issues (direct generation failed)
2. Add multi-stage validation (Phase 2.1)
3. Test exploration pipeline on layernorm

### Short Term (Next Sprint)
1. Implement profiler integration
2. Create reference pattern library
3. Add numerical precision analyzer

### Long Term
1. Tile-size auto-tuning
2. Kernel fusion opportunities
3. Distributed kernel support

## Contact

For issues or questions, check:
- `/memory/TRITON_KNOWLEDGE.md` - Triton API reference
- `/memory/AI_GENERATION_CHALLENGES.md` - Known limitations
- `/memory/HARNESS_IMPROVEMENTS.md` - Improvement roadmap

### Hybrid Harness (harness_hybrid.py) ✅ NEW

**Combines exploration + closed-loop refinement for optimal results:**

- Phase 1: 7-stage exploration (rich context)
- Phase 2: Closed-loop refinement with exploration-aware prompts
- Best of both worlds

**Usage:**
\`\`\`bash
python -m experiments.generation.harness_hybrid softmax --provider glm --max-iterations 5
python -m experiments.generation.harness_hybrid rmsnorm --provider glm --max-iterations 5
\`\`\`

**Architecture:**
\`\`\`
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Exploration (7-stage)                           │
│ - Algorithm → Graph → Tiling → Plan → Structure → Code          │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Phase 2: Closed-Loop Refinement                         │
│ while not (correct AND fast) AND iterations < max:              │
│   1. Test correctness (with exploration context)        │
│   2. Fix errors (with exploration context)              │
│   3. Profile performance                                  │
│   4. AI analyzes + optimizes (with exploration context) │
└─────────────────────────────────────────────────────────┘
\`\`\`

