# 15. EP·FSDP·분산 optimizer의 실제 2-GPU 검증

## EP: token dispatch와 ownership

기존 EP는 다른 input을 사용하는 ranks의 forward가 맞지 않았고 all-to-all backward가 빠졌으며 generic DDP initialization이 서로 다른 expert shard를 rank 0 값으로 덮었다. Source token을 expert owner에 variable-split `all_to_all_single`로 보내고 output을 source에 돌려준다. Custom autograd는 반대 방향의 split으로 gradient를 교환한다. 빈 route에서도 같은 collective 순서를 유지한다.

`ExpertParallelModel`은 shared/router/attention parameter만 broadcast·gradient sync하고 owned expert는 해당 shard를 유지한다. Owned gradient는 token source 전체의 합을 가지므로 DP averaging을 한 번 적용한다. 기존 EP oracle의 owned expert gradient 기준도 rank-local full-model gradient에서 source-rank 합으로 바로잡았다. 이는 이전 forward/weight overwrite 실패를 무효화하지 않는다. 원본 실패 report는 08에 그대로 남긴다.

Canonical initialization은 동일 seed의 full expert model에서 global expert id에 해당하는 weights를 가져온다. 초기화 순간에는 full model의 일시적 memory가 필요한다. Counter-based shard-only initialization이나 다중 node scale-out을 구현한 것은 아니다.

현재 지원 계약은 **world=2, EP=2, TP=1**, FSDP·분산 optimizer·weight offload와 결합하지 않는 training이다. EP checkpoint는 global expert names, optimizer parameter identities, topology를 저장하며 같은 topology로 복원한다. Legacy `allreduce`/`alltoall` 설정 둘 모두 올바른 dispatch로 연결되지만 실제 token exchange는 all-to-all이다.

## FSDP와 optimizer state partition

FSDP는 TP=1에 대해 full model + named optimizer state와 rank-local trainer/reference state를 저장한다. `fsdp_mixed_precision: none`을 실제 FP32 경로로 처리하고 explicit dtype 설정을 존중한다. 현재 FSDP1 API를 사용한다. Frozen reference와 full checkpoint는 rank마다 CPU full copy를 보관하므로 큰 model에서는 중복 CPU RAM 비용이 있다.

`DistributedOptimizer`는 whole parameter 단위로 optimizer moments의 owner를 나누고 updated weights를 owner에서 broadcast한다. FSDP와 동시 사용은 중복 partition이므로 거부한다. Native checkpoint가 parameter name을 live Parameter로 바꾼 뒤 partition loader가 다시 name만 찾던 버그를 실제 resume에서 발견했다. 두 key 형태를 받아 owned moments를 복원한다. Whole parameter round-robin은 작은 model에서도 imbalance를 만들 수 있으며 tensor-wise ZeRO partition과 같은 memory 균등성을 보장하지 않는다.

## 수치 gate

FP32 matrix는 loss, 초기 weights, clipping 이전 gradient, 최종 weights, reference 고정, rank equality, scaler/scheduler, exact same-topology resume를 비교한다. Full/DP/EP 비교는 absolute 2e−5 + relative 2e−4 tolerance를 사용하고 resume는 tolerance 0이다. BF16 compute gate는 5e−3 + 5e−2를 사용해 수치 오차를 별도로 기록한다.

| 경로 | CUDA 검증 | Exact resume |
|---|---|---|
| FSDP FP32 | pretrain + SFT/DPO/GRPO full/DP=2 | 8개 |
| Distributed optimizer FP32 | 네 trainer full/DP=2 | 8개 |
| EP FP32 | 네 trainer full/EP=2 | 8개 |
| Recomputed linear CE FP32 | pretrain/SFT full/accum/DP=2/TP=2 | 8개 |
| Batched expert FP32 | 네 trainer full/accum/DP=2/TP=2 | 16개 |
| Batched idle expert FP32 | pretrain, 위 네 cases | 4개 |

Raw EP layer oracle는 두 communication option×동일/다른 input의 4 case×2 rank를 실행했다. Output/input gradient 오차는 약 1e−11 이하, source합 owned expert gradient 오차는 1.86e−9 이하, initialization의 shard overwrite는 0이다. 이 layer gate를 trainer gate로 대체하지 않고 둘 모두 기록한다.

## 한 rank 장애와 불균일 counts

`validate_distributed_contracts.py`는 rank0 sample=3, rank1 sample=5 및 서로 다른 valid target/response 수를 주고 independent full objective의 clipped AdamW update와 비교했다. 최대 weight error는 pretrain 6.98e−10, SFT 1.51e−9이다. 한 rank에만 NaN/Inf gradient, invalid reward, checksum failure, save failure를 주었을 때 두 rank 모두 update/commit 전에 멈췄다. CUDA parameters/CPU optimizer offload의 AMSGrad×gradient scale 6개도 표준 AdamW와 일치했다.

Fault injection은 프로세스가 살아 있는 상태에서의 통제된 오류를 검증한다. GPU reset·worker 강제 kill·network partition으로 process group 자체가 끊기는 장애의 자동 복구를 구현했다고 주장하지 않는다. 해당 경우 NCCL/process timeout 및 외부 launcher의 restart 정책이 필요하다.

대형 실제-text EP/FSDP/분산 optimizer, token GRPO, BF16 및 worker=2 trainer 재개의 후속 결과는 아래의 별도 immutable source snapshot으로 검증했다. Layer/작은-model 수치 검증으로 대형 성능을 추정하지 않는다.

## 최종 BF16와 실제 크기 gate

| 추가 경로 | Tasks/cases | Exact resumes |
|---|---|---:|
| EP BF16 / FP32 weights | 네 trainer, full/EP=2; GRPO는 token objective | 8 |
| FSDP BF16 / FP32 weights | 네 trainer, full/DP=2 | 8 |
| Distributed optimizer BF16 / FP32 weights | 네 trainer, full/DP=2 | 8 |
| Token GRPO BF16 / FP32 weights | MoE full/DP=2/TP=2 | 3 |
| Stateful worker=2 FP32 | pretrain/SFT/GRPO DP=2 | 3 |

BF16 full/EP 비교의 최대 gradient error는 0.002930, 최종 weight error는 0.001551이었다. FSDP/분산 optimizer의 최종 weight error는 0.001727 이하, token GRPO의 DP/TP 비교는 0.001742 이하였다. BF16 tolerance를 FP32 tolerance로 표현하지 않는다. Same-topology 재개는 모든 경우 0이었다.

실제 55.2M EP workload에서 FP32 residual과 BF16 expert result의 `index_copy` dtype 충돌을 발견했다. Expert output을 accumulation dtype으로 명시적으로 cast하고 독립 BF16 sparse forward/gradient oracle를 추가했다. 실패 기록을 보존한 뒤 위 BF16 matrix와 대형 실제-text gate를 통과했다.

동일 context=1024, microbatch=2, global batch=8, 35-step의 실제-text 측정에서 55.2M EP=2는 73,328 tokens/s·2.687GiB/GPU였고 held-out NLL 10.878→6.972였다. 130.4M의 DP baseline/FSDP/분산 optimizer는 각각 60,501/57,607/57,454 tokens/s 및 4.792/3.796/4.551GiB/GPU였다. 이 큰-model system 대조는 한 번의 측정이며 3회 반복인 14의 CE/MoE 대조와 분리한다. 모두 같은 model size의 held-out NLL이 감소했고 실제 학습이 끝났다. FSDP/optimizer shard를 throughput 개선으로 제시하지 않는다.
