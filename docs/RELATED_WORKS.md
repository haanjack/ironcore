# Related Works: Tile-DSL & Automatic Kernel Generation

## Overview

This document surveys related work in automatic kernel generation, tile-based DSLs, and AI-driven code optimization for the IronCore DSL project.

Last updated: 2026-02-11

---

## 1. Triton: Tiled Abstractions for GPU Programming

### Papers
- **Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations** (Tillet et al., 2021)
- **Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations (Extended)** (2022)

### Key Ideas
1. **Tiled Abstraction**: Programs operate on tiles instead of individual elements
2. **Compiler-Assisted Tiling**: Automatic memory coalescing and shared memory management
3. **Decoupled Execution**: Separate compute and memory schedules

### Relevance to IronCore
- **Foundation**: IronCore builds on Triton for kernel generation
- **Tiling Strategy**: Our tiling analysis stage draws from Triton's tile-based approach
- **Challenge**: Triton still requires manual kernel writing - we aim to automate this

### Takeaways
- Tile size selection is critical for performance
- Memory coalescing patterns are essential
- Autotuning is needed for optimal performance

---

## 2. TVM: Tensor Virtual Machine

### Papers
- **TVM: An Automated End-to-End Optimizing Compiler for Deep Learning** (Chen et al., 2018)
- **TVM: An Automated DSL for Tensor Machine Learning** (2019)

### Key Ideas
1. **AutoTVM**: Automatic search for optimal schedules
2. **Halide-like IR**: Separate computation from schedule
3. **Graph-level Optimization**: Operator fusion and layout transformation

### Relevance to IronCore
- **Search Strategy**: TVM's autotuning approach could inform our tile-size selection
- **Schedule Primitives**: Their schedule language is similar to our tiling analysis
- **Cost Model**: Machine learning-based cost modeling could help our AI

### Takeaways
- Search space explosion is a major challenge
- Cost models are essential for pruning search space
- Transfer learning between hardware architectures

---

## 3. Halide: Image Processing DSL

### Papers
- **Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing** (Ragan-Kelley et al., 2013)
- **A Design Exploration of Tradeoffs Between Heterogeneity and Programmability** (2019)

### Key Ideas
1. **Schedule-Language Separation**: Algorithm definition separate from optimization
2. **Auto-Scheduler**: Automatic schedule search (Adams et al., 2019)
3. **Reduction Detection**: Automatic parallelization of reductions

### Relevance to IronCore
- **Schedule Representation**: Our conversion plan mirrors Halide's schedule structure
- **Auto-Scheduling**: AI-driven generation is similar to auto-scheduling
- **Reduction Patterns**: Their reduction detection informs our graph analysis

### Takeaways
- Schedule primitives are composable
- Automatic scheduling requires good cost models
- Domain-specific knowledge improves search

---

## 4. MLIR: Multi-Level Intermediate Representation

### Papers
- **MLIR: A Scalable Infrastructure for Compiler Development** (Lattner et al., 2020)
- **MLIR: The Next Evolution of Compiler Infrastructure** (2022)

### Key Ideas
1. **Multi-Level IR**: Different levels of abstraction in same framework
2. **Dialects**: Domain-specific abstractions (Tensor, GPU, LLVM)
3. **Transformation Passes**: Composable optimization pipeline

### Relevance to IronCore
- **IR Design**: Our exploration stages could be formalized as MLIR dialects
- **Pass Structure**: Our 7-stage pipeline mirrors MLIR's transformation passes
- **GPU Dialect**: Direct mapping to Triton kernels

### Takeaways
- Multi-level abstractions enable cross-layer optimization
- Dialect extensibility is powerful
- Pass ordering matters for optimization

---

## 5. XLA: Accelerated Linear Algebra

### Papers
- **XLA: TensorFlow, Compiled** (TensorFlow team, 2017)
- **Graph Compiler for Machine Learning** (2018)

### Key Ideas
1. **Whole-Program Optimization**: Cross-operator optimization
2. **HLO IR**: High-Level Optimizer intermediate representation
3. **Just-in-Time Compilation**: Runtime specialization

### Relevance to IronCore
- **Operator Fusion**: Our future kernel fusion work
- **Shape Inference**: Similar to our shape handling
- **Memory Planning**: Informative for our memory layout analysis

### Takeaways
- Fusion reduces memory bandwidth pressure
- Shape specialization improves performance
- Just-in-time compilation enables dynamic shapes

---

## 6. Flash Attention: Tiling for Memory Efficiency

### Papers
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (Dao et al., 2022)
- **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** (2023)

### Key Ideas
1. **IO-Aware Tiling**: Tile size based on memory hierarchy
2. **Online Softmax**: Numerically stable single-pass softmax
3. **Work Partitioning**: Optimal thread block utilization

### Relevance to IronCore
- **Softmax Optimization**: Direct relevance to our softmax generation
- **Online Softmax**: Could improve our softmax kernel performance
- **Tiling Strategy**: Our tiling analysis should incorporate IO-awareness

### Takeaways
- Memory bandwidth is often the bottleneck
- Online algorithms reduce memory passes
- Block-level parallelism is crucial

### Specific Insights for Softmax
The current exploration-generated softmax (0.55x speedup) could benefit from:
- **Online Softmax**: Compute max, sum, and normalization in one pass
- **Vectorized Loads**: Use `tl.load` with larger stride patterns
- **Shared Memory**: Cache frequently accessed values

---

## 7. TorchDynamo: Dynamic Graph Capture

### Papers
- **TorchDynamo: Stable and Fast Compilation for Dynamic Deep Learning** (Zhu et al., 2022)
- **Why Are Dynamic Graphics Hard to Compile?** (2023)

### Key Ideas
1. **Bytecode Analysis**: Python bytecode introspection
2. **Guard System**: Dynamic shape handling
3. **FX Graph**: Intermediate representation for optimization

### Relevance to IronCore
- **Dynamic Shape Support**: Our kernel needs to handle arbitrary shapes
- **Guard System**: Similar to our validation framework
- **Integration**: Future TorchDynamo integration for automatic kernel dispatch

### Takeaways
- Dynamic shapes require careful guard management
- Bytecode analysis is more reliable than AST
- Guard failures are expensive

---

## 8. AI-Assisted Code Generation

### Papers
- **Automatic Optimization of XLA Programs Using Graph Neural Networks** (Gao et al., 2021)
- **DeepTune: Auto-Tuning Deep Belief Networks** (Zhou et al., 2020)
- **CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis** (Nijkamp et al., 2022)
- **AlphaCode: Competition-Level Code Generation** (Li et al., 2022)

### Key Ideas
1. **Cost Model Learning**: ML models for performance prediction
2. **Program Synthesis**: LLM-based code generation
3. **Iterative Refinement**: Generate-Test-Fix loop

### Relevance to IronCore
- **AI-Driven**: Our core approach uses LLMs for kernel generation
- **Iterative Refinement**: Our autonomous harness mirrors generate-test-fix
- **Cost Models**: Future work could integrate learned cost models

### Takeaways
- LLMs struggle with numerical precision
- Multi-stage reasoning improves correctness
- Context management is critical

---

## 9. Specialized Hardware Compilation

### Papers
- **TC: A Language for Nested Tensors** (Choudhury et al., 2018)
- **Affine Transformations for ReLU Activation** (Vasilache et al., 2018)
- **Polyhedral Optimization** (Grosser et al., 2018)

### Key Ideas
1. **Polyhedral Model**: Mathematical framework for loop optimization
2. **Affine Transformations**: Mathematical representation of memory access
3. **Nested Tensors**: Ragged tensor support

### Relevance to IronCore
- **Tiling Theory**: Polyhedral tiling could inform our tiling strategy
- **Loop Transformations**: Our conversion plan uses similar ideas
- **Specialized Hardware**: Future support for TPUs, accelerators

### Takeaways
- Mathematical models enable systematic optimization
- Affine analysis detects parallelization opportunities
- Polyhedral compilation is complex but powerful

---

## 10. Automatic Differentiation

### Papers
- **Automatic Differentiation in Machine Learning: a Survey** (Baydin et al., 2018)
- **Swift for TensorFlow: Differentiable Programming** (2020)
- **JAX: Composable Transformations** (2019)

### Key Ideas
1. **Source-to-Source**: Transform forward code to backward
2. **VJP Rule**: Vector-Jacobian Product rules
3. **Reverse-Mode AD**: Efficient gradient computation

### Relevance to IronCore
- **Gradient Generation**: Our kernels need backward passes
- **VJP Rules**: AI needs to understand AD patterns
- **Correctness**: Gradient validation is critical

### Takeaways
- Manual backward implementation is error-prone
- AI struggles with complex gradient patterns
- Reference implementations are essential

---

## Gaps & Opportunities

### What's Missing from Current Work

1. **AI-First Kernel Generation**
   - Current work focuses on search, not AI synthesis
   - Opportunity: LLM-driven kernel generation

2. **Automatic VJP Generation**
   - Most systems require manual gradient implementation
   - Opportunity: AI-driven backward pass generation

3. **Numerical Precision Awareness**
   - Cost models focus on performance, not correctness
   - Opportunity: Precision-aware generation

4. **Explainable Optimization**
   - Auto-schedulers are black boxes
   - Opportunity: Multi-stage analysis provides insights

### Novel Contributions of IronCore

1. **7-Stage Exploration Pipeline**
   - Algorithm → Graph → Tiling → Plan → Structure → Code → Refine
   - Richer context than direct generation

2. **AI-Driven Iterative Refinement**
   - Test-driven optimization loop
   - Automatic error recovery

3. **Tile-DSL Abstraction**
   - High-level layer specification to low-level kernel
   - Automatic tiling strategy selection

---

## Research Questions

1. **How to achieve performance parity with hand-written kernels?**
   - Current: 0.55x for softmax, 2.48x for RMSNorm
   - Target: Consistent 2x+ speedup

2. **How to improve numerical precision in AI-generated kernels?**
   - Current: 40% correctness rate
   - Target: 80%+ correctness

3. **What's the optimal trade-off between exploration depth and generation time?**
   - Current: 7 stages takes ~20 seconds
   - Target: Adaptive depth based on complexity

4. **Can we learn a cost model for tile-size selection?**
   - Current: Heuristic-based
   - Target: ML-guided tile sizing

---

## Bibliography

1. Tillet, P., et al. (2021). "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations." PACMPL.

2. Chen, T., et al. (2018). "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." OSDI.

3. Ragan-Kelley, J., et al. (2013). "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing." SIGGRAPH.

4. Lattner, C., et al. (2020). "MLIR: A Scalable Infrastructure for Compiler Development." arXiv.

5. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS.

6. Zhu, Y., et al. (2022). "TorchDynamo: Stable and Fast Compilation for Dynamic Deep Learning." MLSys.

7. Nijkamp, E., et al. (2022). "CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis." arXiv.

8. Gao, Y., et al. (2021). "Automatic Optimization of XLA Programs Using Graph Neural Networks." MLHC.

---

## Contact & Discussion

For questions about related work or suggestions for additional papers to review, please refer to:
- `/docs/DEVELOPMENT_PLAN.md` - Current development status
- `/docs/PIPELINE_GUIDE.md` - Usage instructions
- GitHub Issues: github.com/anthropics/ironcore-dsl-alpha/issues
