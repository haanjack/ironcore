# 07. 강의 범위와 검증의 경계

## 범위 표

전체 강의의 일부 구성은 직접 검증하고 일부는 후속 실험으로 남긴다. [CS336 2026 강의 일정](https://cs336.stanford.edu/)의 topic을 기준으로 한다. 기존 구현의 존재, 단위 테스트 통과, 실제 GPU 실험, 실제 task 개선을 서로 다른 상태로 기록한다.

| 주제 | 이번 검증 | 아직 필요한 증거 |
|---|---|---|
| tokenization | GPT-2 tokenizer 재사용, 데이터 hash, FIM/collator 회귀 | 새 byte-level BPE 학습·compression·encoding throughput 비교 |
| Transformer | RMSNorm/RoPE/SwiGLU 모델 학습, attention 회귀, causal packing | 각 구성 제거 ablation, numerical gradient oracle 확장 |
| optimizer | full batch/accumulation update, clipping, AMP skip, 재개 | FP32 master weights, Muon/분산 optimizer/offload의 전체 실험 |
| GPU systems | 3090 sweep, memory, synchronized timings, telemetry, trace | 독립 Triton FlashAttention forward/backward 구현과 기준 비교 |
| parallelism | actual DP=2/TP=2 NCCL update/restart | FSDP/ZeRO/EP/다중 node/장애 주입 |
| learning/evaluation | 실제 TinyStories train/valid pilot | corpus 전체·다중 seed·task-level evaluation |
| scaling laws | 두 크기의 parameter/token/시간을 구분해 기록 | isoFLOPs 여러 크기·budget·seed, 회귀와 uncertainty |
| corpus quality | train/valid 분리와 manifest | Common Crawl HTML 처리·language/quality filters·exact/near dedup |
| post-training | SFT/DPO 수치, GRPO/GSPO replay 및 online pipeline | 실제 instruction/preference/reasoning datasets와 보상·성능 개선 |
| MoE | [08의 routing/gradient, DP=2/TP=2, idle experts, 실제 텍스트 pilot](08-moe.md) | EP=2 oracle 실패 및 trainer 실행 차단; fused/grouped expert kernels, 대규모 quality 평가 |
| inference | 기존 코드와 테스트 존재 | 이 실험에서는 inference throughput/quality를 별도로 입증하지 않음 |

## scaling law 실험의 명세

[A3 Scaling](https://github.com/stanford-cs336/assignment3-scaling/blob/main/cs336_assignment3_scaling.pdf)은 isoFLOPs와 compute-optimal 선택을 연결한다. 50M과 130M 두 점만으로 지수나 최적 모델 크기를 추정하지 않는다.

후속 실험은 최소 4개 model size와 여러 token budgets, 각 설정의 반복 seed를 둔다. dense 모델의 대략적인 training compute `C ≈ 6ND`를 출발점으로 사용하되 긴 context에서는 attention term을 별도로 계산한다. 같은 step 수와 같은 FLOP budget은 다르다. 고정 compute에서 model size를 늘리면 training token 수를 조정한다. plot에는 실제 unique/presented tokens, validation NLL과 run별 variance를 표시한다.

현재의 learning pilot은 training tools를 확인하는 실험이며 scaling law fitting dataset이 아니다.

## 데이터 품질 실험의 명세

[A4 Data](https://github.com/stanford-cs336/assignment4-data/blob/main/cs336_assignment4_data.pdf)와 연결하려면 동일 원본 corpus에서 변환/필터/dedup 단계를 분리해 token 유지율과 학습 품질을 측정해야 한다. HTML text extraction, language identification, quality rules/classifier, exact-line dedup, MinHash/LSH near-document dedup 각각의 전후 크기와 수작업 audit 표본을 남긴다.

동일 model/token budget에서 unfiltered, filtered, filtered+deduplicated 데이터를 비교한다. contamination과 train/valid 중복 여부를 검사한다. 이 저장소의 streaming preprocessing이 존재한다는 사실만으로 위 단계를 구현·검증했다고 기록하지 않는다.

## 블로그 작성 시 사용할 주장

검증된 주장에는 실행 ID·설정·허용 오차·artifact를 붙인다. 실패 재현과 수정 후 결과를 함께 설명한다. loss 감소로 gradient/분산 정확성을 대신 설명하지 않는다. throughput 개선은 비교한 workload와 power cap을 명시한다. 미검증 주제와 toy alignment의 의미를 그대로 남긴다.

각 후속 실험은 이 디렉터리에 같은 형식의 노트를 추가한다: 질문 → 명세 → 통제 → 명령 → raw artifact → 수치 → 해석 → 남은 검증. 앞으로 수행할 실험을 완료된 사실로 바꾸지 않는다.

## 10–15 후속 보완

이 표는 최초 00–09 실행 시점의 경계다. 후속 [10](10-trainer-remediation.md)에서 표준 AdamW/offload oracle, FP32 stored weights, stateful cursor·worker prefetch, rank failure, EP/FSDP/분산 optimizer의 네 trainer 검증 및 실제 135M alignment pilot을 추가했다. [14](14-performance-remediation.md)의 batched expert는 native BMM batching이며 자체 Triton grouped-GEMM 구현의 검증은 아니다. 대규모 scaling law·Common Crawl filtering/dedup·장기간 reasoning 성능은 계속 별도 범위다.
