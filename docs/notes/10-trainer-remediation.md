# 10. Trainer 보완과 독립 검증

이 단계는 00–09에서 측정한 결과를 보존하면서 후속 보완을 수행한다. 이전 report의 source hash와 loss/성능 값을 변경하지 않는다. 이전 matrix는 같은 optimizer 간 분산 동등성의 근거이며 표준 AdamW 동등성까지 입증한 것은 아니다.

## 검증 현황

| 단계 | 항목 | 현재 근거 | 상세 노트 |
|---|---|---|---|
| A | AdamW/Muon AdamW/offload 수식 | CPU 20개 oracle + CUDA offload 6개 통과 | [11](11-numerics-and-precision.md) |
| B | Nonfinite gradient | 한 rank NaN/Inf 시 두 rank update/scheduler 중단 | [11](11-numerics-and-precision.md), [15](15-distributed-training.md) |
| C | 가변 token/sample/packing 평균 | 독립 global objective와 CPU/DP=2 update 일치 | [11](11-numerics-and-precision.md) |
| D | FP32 weights + BF16 compute | 작은 update 누적 + 네 trainer CUDA/restart matrix 통과 | [11](11-numerics-and-precision.md) |
| E | MoE auxiliary objective | per-layer 독립 식 + DPO/GRPO/GSPO 포함/제외 gradient 통과 | [13](13-alignment-contracts.md) |
| F | Stateful data/checkpoint | worker 0/2·RNG·disk/checksum 오류 및 CUDA worker=2 세 trainer exact resume 통과 | [12](12-stateful-data-and-checkpoints.md) |
| G | GRPO/GSPO·실제 alignment | 수식/warper/EOS·precision 통일·실제 세 pilot 완료; GRPO 품질 개선 미입증 | [13](13-alignment-contracts.md) |
| H | 성능 ablation | 3회 CE/MoE 대조·8K context·compile·EP 포함 HTML 완료 | [14](14-performance-remediation.md) |
| I | EP/FSDP/분산 optimizer | 네 trainer FP32/BF16·대형 EP/FSDP/분산 optimizer 학습/restart 통과 | [15](15-distributed-training.md) |

각 단계의 구현·검증을 완료했다. 실제 GRPO의 reasoning 품질 개선은 입증하지 못했으며 이는 implementation gate의 성공과 별개다. 전체 corpus·장기간 학습·대규모 scaling law를 검증한 결과는 아니다.

최종 CPU 회귀는 **532 passed / 12 skipped / 42 deselected**, profiler/MFU 회귀는 별도로 **80 passed**이다. GPU 검증은 15에 명세한 실제 torchrun/NCCL 실행으로 수행했다. [원본 근거·source snapshot·실패 로그](artifacts/2026-10-07-remediation/README.md)와 [compute/collective HTML](profiling-remediation/profile_report.html)을 보존했다.

실용적인 선택은 FP32 stored weights/BF16 compute의 DP=2를 기본으로 두고, dropout=0인 이 MoE workload에서 batched expert를 선택하는 것이다. Memory가 부족할 때 recomputed CE/FSDP/분산 optimizer/EP를 각각 검토한다. 이들 옵션을 무조건 결합하거나 throughput 이득을 일반화하지 않는다.
