# Dense context parallel validation

This records the initial Dense CP validation. Subsequent block-wise MLP and MoE/EP
support and measurements are documented in [block-wise validation](blockwise_mlp_validation.md).

Validation date: 2026-10-08. Configuration and execution contract are in
[Parallelism](../parallelism.md#context-parallelism-cp). The reproducible example is
`configs/experiments/smollm2_135m_cp2.yaml`.

## Environment

- GPU: two RTX 3090s, PyTorch 2.14.0+cu130, NCCL 2.30.7, Transformers 5.17.0.
- CPU: PyTorch 2.13, Gloo, four actual torchrun processes for composed topologies.
- FP32 stored parameters; BF16/FP16 CUDA autocast for ring attention.
- Native ATen FlashAttention, without Transformer Engine or a separate flash-attn package.

## Correctness coverage

`tests/multi_gpu/test_context_parallel.py` compares a full decoder with CP shards:

- Independent full causal attention output and owner-local Q/K/V gradients for MHA,
  GQA and MQA, in BF16 and FP16; perturbing future tokens leaves earlier outputs unchanged.
- Full-parameter and LoRA training, nonzero adapter gradients, three accumulated AdamW
  updates, clipping/norms, standard/reentrant activation checkpointing, chunked CE and
  recomputed linear CE.
- Global positions and shifted labels across an odd sequence length; a CP shard with
  no valid labels; zero loss/gradients for an entirely masked batch; rejection of a
  globally empty training update.
- A parameter used on one CP peer receives the sum everywhere. Globally unused
  parameters keep `grad=None`, preserving optimizer skip/weight-decay semantics.
- Full-sequence evaluation with the default KV-cache configuration. CP disables
  stateful cache construction and rejects explicit generation/cache requests.
- Universal/distributed native checkpoints, CP2 weights loaded at CP1, and CP2
  model/optimizer/trainer-state roundtrips.

CPU TP2+CP2 checks use four processes, including a native TP-sharded model and a full
reference. CPU CP2+DP2 checks use unequal token counts and an entire DP worker whose
labels are ignored, exercising weighted objectives and collective participation.
No GPU count is mocked to claim these four-rank compositions ran on four GPUs.

## Pretrained models

| Model / compute | Check | Outcome |
|---|---|---|
| SmolLM2-135M BF16 | Natural-language LoRA parity; full production trainer | Passed |
| SmolLM2-1.7B FP32 SDPA | Output, adapter gradient, three trainer updates | Passed; update max absolute error <= 5.96e-8 |
| SmolLM2-1.7B FP16 ring | Natural-language LoRA parity, three updates | Passed; gradient relative L2 error about 0.56% |
| SmolLM2-1.7B BF16 ring | Natural-language LoRA parity, three updates | Passed within the BF16 budget; gradient relative L2 error about 6.0% |
| SmolLM2-1.7B BF16 ring | Production LoRA training and evaluation each step | Three training and three evaluation steps completed with finite losses/gradients |

Pretrained parity uses AdamW epsilon 1e-3 and learning rate 1e-5 for CUDA, with the same
settings in the reference and distributed trainers. The production smoke uses the
configured optimizer defaults and a 1e-4 learning rate. Random-token smoke runs check
systems behavior and do not establish language-model convergence.

### Precision interpretation

Sequence partitioning changes GEMM shapes and attention reduction order. BF16 deep
pretrained decoders amplify these rounding differences; bitwise or uniformly tight
per-element agreement is not a valid promise. The BF16 1.7B case did **not** meet a
3% relative gradient threshold: the measured difference was approximately 6.0%.
The pretrained BF16 regression therefore explicitly budgets 8% relative gradient L2,
while requiring close loss, probability KL, gradient norm and overall update agreement.
The measured FP16 gradient difference is considerably smaller.

FP32 references retain tight value/gradient/update checks. CUDA CP initialization also
keeps FP32 GEMM accumulation by disabling reduced-precision reductions, and restores
the original reduced-precision flags when model-parallel groups are destroyed. These choices
follow [PyTorch's numerical-accuracy guidance](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).

## Production execution and limits

The SmolLM2-135M production CLI ran CP2 training, saved step 3, restored model/optimizer/
trainer state, skipped consumed microbatches, and continued to step 6. Native checkpoint
files have one CP writer; rank-local trainer state retains the original topology.

GPU pytest runs cover five distributed tests per rank. The default CPU suite completed
813 passing tests (221 skipped and 5 deselected). The CPU default suite was run
offline to use cached Hugging Face fixtures. Machine-readable local reports and an HTML
study copy live under the ignored `.local/` directory; checkpoints and detailed logs
are local evidence, not model artifacts committed to this repository.

Initial GPU validation is single-node CP2. Four-GPU NCCL composition, multi-node
throughput, long training convergence, dropout, packed/SFT/alignment attention,
Gemma 4, FSDP/offload/distributed optimizers and MoE/EP are outside this initial contract.
Contiguous causal shards have unequal attention work. MLP block chunking and zigzag
attention load balancing are follow-up work.

## 8K system measurement

SmolLM2-135M, sequence length 8192, global batch 2, BF16 compute/FP32 stored weights,
full-parameter Adam training and linear-CE recomputation. The second pair disables
only activation recomputation. Peak memory is the maximum over all ranks; time is the
mean of updates 2 and 3 after CUDA synchronization, taking the slower rank.

| Activation recomputation | Mode | GPUs | Maximum allocated memory per GPU | Update time |
|---|---|---|---|---|
| Enabled | CP1 | 1 | 3170.5 MiB | 1.220 s |
| Enabled | CP2 | 2 | 2876.9 MiB | 0.857 s |
| Disabled | CP1 | 1 | 9077.6 MiB | 1.007 s |
| Disabled | CP2 | 2 | 5782.6 MiB | 0.699 s |

This is one local measurement, not a throughput guarantee. CP replicates model
and optimizer state, so per-GPU memory does not halve. The CP2 peak reduction is
9.3% with activation recomputation and 36.3% without it: recomputation already
removes much of the retained activation memory that CP would otherwise split.
In this full-parameter FP32/Adam setup, weights, gradients and two optimizer
moments alone occupy roughly 2052 MiB per GPU. MLP block chunking can target
remaining intermediate activations but cannot reduce this replicated state.
