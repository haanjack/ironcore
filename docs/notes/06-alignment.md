# 06. SFT, DPO, GRPO의 학습 의미

## 강의와의 연결

[CS336 A5 Alignment](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2026_assignment5_alignment.pdf)은 on-policy policy gradient와 sequence normalization을 다루고, off-policy 단계에서 token-level GRPO와 sequence-level GSPO를 비교한다. [Safety supplement](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)에는 SFT와 DPO가 연결된다. 이번 실험은 이 구성의 수치·시스템 동작을 검증하며 reasoning task 성과는 별도이다.

## SFT

이 구현은 response token을 sample별 평균하고 sample 간 평균한다. prompt labels `-100`, response 길이 차이, train/eval objective 일치, packing의 causal document isolation을 검증한다. fixed row 수의 통제 실험에서 accumulation/DP/TP update를 비교한다. 실제 bin packing이 microbatch마다 row 수를 바꾸면 sample 가중치의 추가 검증이 필요하다.

## DPO

목적함수는 policy/reference의 chosen/rejected sequence log-prob 차이에 대한 logistic preference loss다. reference는 최초 policy/SFT 상태의 고정 복사본이어야 한다.

발견한 오류는 non-FSDP reference의 `requires_grad=False` 누락, TP에서 full-vocabulary logits의 재-gather, 재개 시 reference를 현재 policy로 reset하는 문제다. reference freeze를 적용하고 log-softmax에 logits layout을 명시했다. 실제 shard를 gather하는 helper는 autograd-aware gather를 사용한다. reference를 trainer state로 저장/복원한다.

검증은 초기 loss, frozen reference weights/grad, preference loss 감소, first raw gradient, 최종 policy weights, same-topology 재개를 포함한다. chosen/rejected sample ID를 유지하며 각 DP rank가 서로 다른 pair를 본다.

## GRPO

on-policy는 `-mean(A * sequence_logp / response_length)`를 사용하고 reference KL penalty를 더한다. advantage는 같은 prompt의 completions 내 reward 평균/std로 계산한다. 이 코드의 std는 `torch.std()` 기본 correction=1이다. 일정한 rewards는 zero advantage다.

`grpo_num_epochs > 1`에서 이 구현의 ratio는 `exp((logp_new_sum-logp_old_sum)/response_length)`이다. **sequence geometric-mean ratio인 GSPO 방식**이며 token-level PPO/GRPO clipping과 구분한다. 기존 이름이 GRPO인 것을 이유로 모든 off-policy variant를 구현했다고 주장하지 않는다. sampling temperature/top-p를 바꾸면 recorded behaviour log-prob와 update distribution의 의미도 확인해야 한다.

학습은 rollout당 local prompt batch를 한 번 읽고, completion microbatch들을 sample 수로 가중해 accumulation한 뒤 epoch마다 update한다. `train_batch_size`는 global prompt 수이고 local prompt batch는 DP size로 나눈다. GRPO 재개 data skip은 accumulation count가 아닌 이미 소비한 rollout batch 수를 따른다. evaluation iterator는 eval datasets 설정으로 만들며, `0,0` placeholder 대신 actual held-out generated rewards를 보고한다.

## 통제 실험과 온라인 실험

수치 동등성 실험은 정해진 completion fixture를 사용한다. 실제 LanguageModel/GRPO trainer/reward worker/advantage/log-prob/backward/optimizer를 실행하지만, stochastic generator만 고정된 rollout producer로 바꾼다. 같은 rollout에서 full/accumulation/DP/TP update와 reference 및 재개를 비교한다.

별도 online 실험은 실제 batched generation을 사용하여 full/DP=2/TP=2에서 rollout→reward→advantage→update→evaluation을 실행한다. keyword similarity reward와 tiny model을 사용한다. 이는 reward pipeline의 liveness와 nonzero policy update 증거이며 자연어 추론 성능의 증거는 아니다.

## 실측 결과

FP32와 BF16의 full/accum/DP=2/TP=2 수치·재개 matrix가 모두 통과했다. gradient/최종 weights 오차는 [03의 표](03-distributed-and-resume.md)에 있다. 모든 DPO/GRPO case에서 reference는 초기 policy와 정확히 같았고 reference gradient가 없음을 검사했다. 중단 후 reference를 복원한 loss 궤적도 uninterrupted 실행과 정확히 같았다.

CS336 FP32 full-batch의 통제 DPO loss는 4 steps에서 **0.69315→0.46488→0.33036→0.30812**였으며 최종 fixture 평가 loss는 0.28201, preference accuracy는 1.0이었다. SFT loss는 3.74431→3.62128, fixture 평가 loss는 3.59568이었다. 이는 같은 작은 fixture에 대한 update 검증으로, held-out 자연어 alignment 성과가 아니다.

실제 online generation은 세 matrix 각각 full/DP=2/TP=2에서 2 rollout steps를 실행했다. CS336 FP32 결과:

| 구성 | max policy weight 변화 | held-out mean reward | positive reward rate |
|---|---:|---:|---:|
| full | 0.0017510 | 0.53125 | 1.0 |
| DP=2 | 0.0017506 | 0.50000 | 1.0 |
| TP=2 | 0.0017506 | 0.50000 | 1.0 |

BF16의 세 구성도 실제 online update를 만들었고 최대 weight 변화는 0.001953125였다. generator의 확률적 결과는 topology 간 exact equality로 요구하지 않았다. reward의 양수 비율은 키워드 similarity 함수 특성도 반영하므로 높은 값 자체를 추론 정확도로 사용하지 않는다. [rank별 loss/evaluation 기록](artifacts/2026-10-07/ironcore-validation-cs336-float32/rank_summaries.json)과 [BF16 report](artifacts/2026-10-07/ironcore-validation-cs336-bfloat16/report.json)를 보존했다.

## 추가 gate

실제 reasoning dataset에서는 pretrained model의 zero-shot/SFT baseline과 reward parsing 정확성부터 확보한다. prompt 길이 차이·padding attention, reward worker failure/default reward, empty completion/EOS, sampling warper와 importance ratio, group ID 전역 유일성, off-policy clipping 비율을 별도로 검증한다. toy reward의 상승만으로 GRPO reasoning 학습의 정당성을 확정하지 않는다.
