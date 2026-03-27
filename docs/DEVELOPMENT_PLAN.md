# IronCore DSL - Development Plan & Status

## Project Overview

**Goal**: Build an AI-driven DSL (Domain-Specific Language) system for automatic Triton kernel generation.

**Vision**: Enable researchers to implement custom neural network layers in high-level Python, and automatically generate optimized GPU kernels.

**Date**: 2026-02-11

---

## Current Status

### Successfully Generated Kernels

| Kernel | Status | Speedup | Approach | Notes |
|--------|--------|---------|----------|-------|
| **RMSNorm** | ✅ Working | 2.48x | Autonomous | Faster than PyTorch |
| **Softmax** | ⚠️ Mixed | 0.55x | Exploration | Correct but slower |
| **LayerNorm** | ❌ Failed | - | Both | Numerical issues |
| **GLU** | ❌ Failed | - | Both | Shape issues |
| **CrossEntropy** | ❌ Failed | - | Both | Complex operation |

### System Components

**Phase 1: Core Infrastructure** ✅ COMPLETE
- [x] Autonomous generation harness (`harness_autonomous.py`)
- [x] 7-stage exploration pipeline (`kernel_explorer.py`)
- [x] OpenAI-compatible AI provider interface
- [x] GLM-4.7 integration with extended reasoning
- [x] Code extraction from markdown responses
- [x] Triton 3.4.0 compatibility layer
- [x] Correctness validation framework
- [x] Gradient checking
- [x] Performance benchmarking

**Phase 2: Enhancement** 🔄 IN PROGRESS
- [ ] Multi-stage validation (error isolation)
- [ ] Profiler integration (GPU metrics for AI)
- [ ] Reference pattern library
- [ ] Numerical precision analyzer
- [ ] Hypothesis testing framework
- [ ] Memory layout analyzer

**Phase 3: Advanced Features** 📋 PLANNED
- [ ] Tile-size auto-tuning
- [ ] Kernel fusion opportunities detection
- [ ] Distributed kernel support
- [ ] Automatic benchmarking suite
- [ ] Performance regression testing

---

## Technical Challenges & Solutions

### Challenge 1: Numerical Precision

**Problem**: AI-generated kernels often have numerical errors due to:
- Using vector accumulators instead of scalar
- Not using float32 for gradient buffers
- Incorrect reduction operations

**Solution**:
```python
# Correct pattern
acc = 0.0  # Scalar accumulator
acc += tl.sum(x, axis=0)

# For gradients
dw = torch.zeros(N, dtype=torch.float32)  # Float32 buffer
```

### Challenge 2: Triton API Limitations

**Problem**: Triton 3.4.0 has missing functions compared to documentation.

**Solution**:
```python
# Wrong - tl.pow() doesn't exist
y = tl.pow(x, 2)

# Correct - use multiplication
y = x * x
```

### Challenge 3: Code Extraction

**Problem**: GLM-4.7 returns code in markdown blocks with language identifier on same line.

**Solution**: Enhanced extraction logic that handles:
- ````python` format
- ```` ` format
- Language identifier on same line as opening marker

### Challenge 4: Shape Handling

**Problem**: Kernels need to handle arbitrary input shapes.

**Solution**:
```python
def forward(ctx, x):
    # Reshape to 2D
    x_reshaped = x.reshape(-1, x.shape[-1]).contiguous()
    M, N = x_reshaped.shape
    # ... process ...
    return output.reshape(original_shape)
```

---

## Comparison: Exploration vs Autonomous

| Aspect | Exploration (7-stage) | Autonomous (Direct) |
|--------|----------------------|---------------------|
| **Correctness Rate** | Higher (richer context) | Lower |
| **Generation Time** | Longer (multi-stage) | Shorter |
| **Token Usage** | Higher (~14K tokens) | Lower (~6K tokens) |
| **Quality** | Better structured | Less structured |
| **Best For** | Complex kernels | Simple kernels |

### Softmax Case Study

**Exploration Pipeline (7 stages)**:
- Stages: Algorithm → Graph → Tiling → Plan → Structure → Code → Refine
- Tokens: 14,341
- Result: Correct, 0.55x speedup
- Time: ~20 seconds

**Autonomous Harness**:
- Iterations: 3
- Tokens: 30,071 (total, failed)
- Result: Numerical errors (max_abs=5.38e-02)
- Time: ~11 minutes

**Lesson**: Multi-stage analysis produces correct kernels more reliably, but takes longer.

---

## Roadmap

### Immediate (This Week)

1. **Fix Softmax Performance** 🎯 HIGH PRIORITY
   - Current: 0.55x slower than PyTorch
   - Target: 2x+ speedup
   - Approach: Reduce memory passes, optimize block size

2. **Improve Error Diagnostics** 🎯 HIGH PRIORITY
   - Multi-stage validation to isolate error components
   - Hypothesis testing framework
   - Better error messages for AI

3. **Create Reference Pattern Library**
   - Proven code patterns for common operations
   - Reduction patterns
   - Element-wise operation patterns

### Short Term (Next Sprint)

1. **Profiler Integration**
   - Capture GPU metrics (memory bandwidth, compute utilization)
   - Feed metrics to AI for optimization decisions
   - Automatic performance bottleneck detection

2. **Numerical Precision Analyzer**
   - Detect precision issues before testing
   - Suggest appropriate accumulator types
   - Validate gradient computation patterns

3. **Hypothesis Testing Framework**
   - Test specific optimization hypotheses
   - A/B testing for kernel variants
   - Automated performance measurement

### Long Term (Next Quarter)

1. **Tile-Size Auto-Tuning**
   - Automatic block size selection
   - Runtime performance measurement
   - Machine learning-based optimization

2. **Kernel Fusion Opportunities**
   - Detect fusion opportunities
   - Multi-kernel generation
   - Fused kernel optimization

3. **Distributed Kernel Support**
   - Multi-GPU kernels
   - NCCL integration
   - Communication optimization

---

## Related Work & Research

### 1. Triton DSL
- **Paper**: "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (2021)
- **Key Insight**: Tiled abstractions for GPU programming
- **Relevance**: Foundation for our kernel generation

### 2. TVM (Tensor Virtual Machine)
- **Paper**: "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning" (2018)
- **Key Insight**: Automatic optimization with search
- **Relevance**: Auto-tuning approaches

### 3. TorchDynamo
- **Paper**: "TorchDynamo: Stable and Fast Compilation for Dynamic Deep Learning" (2022)
- **Key Insight**: Dynamic graph capture
- **Relevance**: Integration patterns

### 4. MLIR (Multi-Level Intermediate Representation)
- **Paper**: "MLIR: A Scalable Infrastructure for Compiler Development" (2020)
- **Key Insight**: Multi-level IR abstractions
- **Relevance**: Compiler infrastructure patterns

### 5. OpenAI Triton Kernels
- **Source**: github.com/openai/triton
- **Key Contribution**: Reference implementations
- **Relevance**: Pattern library source

### 6. Flash Attention
- **Paper**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
- **Key Insight**: Tiling for memory efficiency
- **Relevance**: Advanced tiling strategies

---

## AI Model Selection

### Current: GLM-4.7

**Strengths**:
- Extended reasoning mode
- Strong code generation
- OpenAI-compatible API
- 128K context window

**Weaknesses**:
- Sometimes produces subtle numerical errors
- Requires multiple iterations for complex kernels

### Future Options

1. **Claude 3.5 Sonnet**
   - Stronger reasoning
   - Better code generation
   - Higher cost

2. **GPT-4**
   - Strong code understanding
   - Good at optimization
   - Expensive

3. **Fine-tuned Model**
   - Train on Triton kernels
   - Specialized for DSL tasks
   - Best performance/cost ratio

---

## Performance Targets

### Phase 1: Correctness ✅
- [ ] All kernels pass correctness tests
- [ ] All kernels pass gradient tests
- [ ] Numerical precision within tolerances

### Phase 2: Performance 🔄
- [ ] 2x+ speedup over PyTorch for simple kernels
- [ ] 1.5x+ speedup for complex kernels
- [ ] Memory usage < 2x PyTorch

### Phase 3: Production 📋
- [ ] Auto-tuning for optimal performance
- [ ] Multi-GPU support
- [ ] Automatic fallback to reference

---

## Architecture Decisions

### 1. OpenAI-Compatible API
**Decision**: Use OpenAI-compatible API for all AI providers

**Rationale**:
- Easy provider switching
- Community standard
- Well-supported

### 2. 7-Stage Exploration
**Decision**: Implement multi-stage analysis before code generation

**Rationale**:
- Better correctness
- Richer context for AI
- Debugging insights

### 3. Iterative Refinement
**Decision**: Use test-driven optimization loop

**Rationale**:
- Guarantees correctness
- Data-driven optimization
- Automatic error recovery

---

## Metrics & KPIs

### Success Metrics
- **Correctness Rate**: % of kernels passing tests
- **Average Speedup**: Performance improvement over PyTorch
- **Generation Time**: Time to generate working kernel
- **Token Efficiency**: Correctness per token used

### Current Metrics (as of 2026-02-11)
- Correctness Rate: 40% (2/5 kernels)
- Average Speedup: 1.52x (RMSNorm only)
- Generation Time: 20-600 seconds
- Token Efficiency: ~1.5K tokens per successful kernel

### Target Metrics (Q2 2026)
- Correctness Rate: 80%+
- Average Speedup: 2x+
- Generation Time: <60 seconds
- Token Efficiency: ~1K tokens per successful kernel

---

## Open Questions

1. **Performance Parity**: How to achieve competitive performance with PyTorch's highly optimized kernels?

2. **Complex Operations**: How to handle complex operations like cross-entropy and layer normalization?

3. **Auto-Tuning**: What's the best approach for automatic tile-size selection?

4. **Model Selection**: Should we fine-tune a model for Triton generation?

5. **Fusion**: How to automatically detect and generate fused kernels?

---

## Contributing

See `/docs/PIPELINE_GUIDE.md` for usage instructions.

See `/experiments/generation/SPEC_TEMPLATE_EXAMPLE.py` for adding new kernels.

---

## Contact

- GitHub: github.com/anthropics/ironcore-dsl-alpha
- Issues: github.com/anthropics/ironcore-dsl-alpha/issues
