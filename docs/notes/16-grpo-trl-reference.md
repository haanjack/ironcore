# 16. GRPO를 TRL의 실제 학습 루프와 대조하기

## 무엇을 입증하는가

“수식대로 코드를 작성했다”에서 한 단계 더 나아가, 같은 초기 weights·prompt·completion·reward·optimizer를 넣었을 때 널리 쓰이는 외부 구현과 같은 update를 만드는지 검사한다. 기준은 [TRL 0.29.0 GRPOTrainer](https://huggingface.co/docs/trl/v0.29.0/en/grpo_trainer)와 [해당 release의 실제 코드](https://github.com/huggingface/trl/blob/v0.29.0/trl/trainer/grpo_trainer.py)다. 최신 release 전체의 동등성을 주장하지 않고 설치한 버전과 trainer 파일 SHA256을 고정했다.

독립 수식 oracle에 더해 외부 trainer와의 대조를 확보하면, 선택한 목적함수와 update 경로의 구현을 신뢰할 근거가 강해진다. 수학 reasoning 품질이나 모든 optimizer/precision/rollout 설정의 동등성을 입증하는 실험은 아니다.

## 공통 실험 계약

[`validate_grpo_reference.py`](../../scripts/validate_grpo_reference.py)는 **실제 `IronCore.GRPOTrainer.train()`과 `TRL.GRPOTrainer.train()`을 모두 실행**한다. TRL의 loss·advantage·backward·optimizer 학습 코드를 복사하거나 자체 수식으로 대체하지 않는다. Callback으로 각 optimizer step의 gradient와 전체 trainable weights를 수집하고 HF 이름으로 대응시킨다. Gradient는 clipping 이후 값이다.

생성만 scripted fixture로 바꿔 두 구현에 같은 completion을 공급한다. 각 prompt의 네 completion은 두 후보를 번갈아 사용하며 EOS를 포함한다. 두 구현은 자기 policy에서 old token log-prob를 직접 계산하고 reference는 초기 weights로 고정한다. Scripted completion의 선택 분포는 current policy sampling 분포와 다르므로, 이 실험을 실제 on-policy rollout이나 importance sampling의 통계적 정당성 검증으로 해석하지 않는다.

| 항목 | 동일하게 설정한 값 |
|---|---|
| Objective | token ratio, TRL `loss_type="grpo"`, valid-token mean → completion mean |
| Group | prompt당 4 completions, sample std(correction=1), normalization epsilon 1e−4 |
| Epochs / clipping | rollout당 2 optimizer updates, symmetric epsilon 0.2 |
| KL | beta 0.1, k3, TRL bias correction off; Native log-ratio clamp ±20 바깥의 동등성은 주장하지 않음 |
| Sampling score | temperature 1, top-p 1, top-k 0 |
| Optimizer | AdamW, betas=(0.9,0.95), epsilon 1e−8, weight decay 0.01, gradient clip 1 |
| LR | 작은 모델 1e−3 / 135M 1e−5, constant |
| Stored weights | FP32; dropout 0; 초기 weights와 frozen reference exact 비교 |

Native의 기본 `gspo`, advantage epsilon 1e−8을 그대로 놓고 다른 목적함수의 TRL default와 비교하지 않는다. 이 계약은 명시적으로 token GRPO를 선택하고 epsilon도 맞춘다. 같은 prompt의 reward가 전부 0인 대조는 학습 경로의 증거가 약하므로, 작은 fixture는 `[1,0.5,1,0.5]`, 135M은 `[1,0,1,0]`로 비영 advantage를 만든다. DP=2의 두 rank는 동일한 통제 prompt를 사용한다. Uneven DP partition 검증은 [15](15-distributed-training.md)의 별도 독립 oracle이다.

## FP32 실측 결과

Tensor gate는 `atol=2e−5, rtol=2e−4`, loss scalar는 absolute 2e−5다. 초기 weights와 frozen reference는 tolerance 0이다. 아래 최대값은 모든 optimizer steps와 두 rank의 전체 trainable tensors를 확인한 결과다.

| Workload | Updates | Max loss error | Max clipped-gradient error | Max weight error | Gate |
|---|---:|---:|---:|---:|---|
| 76,096-parameter Llama / CPU | 24 | 1.79e−7 | 2.37e−6 | 2.55e−6 | 통과 |
| 같은 작은 모델 / DP=2 CUDA | 24 | 1.56e−7 | 4.15e−7 | 1.50e−5 | 통과 |
| SmolLM2-135M-Instruct / DP=2 CUDA | 8 | 8.20e−6 | 9.54e−6 | 2.33e−6 | 통과 |

135M은 [13](13-alignment-contracts.md)과 동일한 134,515,008-parameter checkpoint를 Native에 import하고 TRL은 HF model로 읽었다. 따라서 실제 크기의 Native backbone과 외부 HF backbone을 포함하는 비교다.

![FP32 두 trainer의 loss trajectory](artifacts/2026-10-07-grpo-reference/fp32_loss_trajectories.png)

Loss curve의 일치는 동일 update 경로의 보조 근거다. GRPO scalar loss가 매 step 감소해야 하는 것은 아니므로, 이 curve를 teacher-forced NLL이나 생성 성공률의 개선으로 해석하지 않는다.

보상 후보의 조건부 확률 `P(positive)/(P(positive)+P(negative))`도 비교했다. 작은 CPU run은 두 구현 모두 0.52598→0.9999999, 135M FP32는 약 0.0082595→0.9996778이었다. 조건부 확률의 denominator는 지정한 두 completion뿐이며 전체 생성의 성공률이 아니다. 실제 prompt는 하나의 통제 fixture이고 held-out generalization 결과도 아니다. Scripted reward 평균 자체는 고정이므로 이 실험에서 reward curve 상승을 주장하지 않는다.

## BF16에서 확인한 제한

실사용 조건인 FP32 stored weights/BF16 compute에도 같은 135M·8-update 대조를 수행했다. 기존 분산 BF16 gate와 같은 `atol=5e−3, rtol=5e−2`를 유지했지만 **전체 trajectory의 엄격한 gradient/loss 일치 gate는 실패했다**. 허용 오차를 늘려 통과로 바꾸지 않았다.

| BF16 비교 | Max gradient error | Max weight error | Max loss error | Failed tensor/loss checks per rank |
|---|---:|---:|---:|---:|
| Native backbone vs HF/TRL | 0.5063 | 1.34e−4 | 0.1099 | 602 |
| 두 trainer 모두 같은 HF backbone | 0.09606 | 8.05e−5 | 0.04635 | 55 |

Native vs HF는 update 이전부터 positive completion의 sequence log-prob가 −19.7443 / −19.5826으로 달랐다. 같은 HF backbone을 넣는 추가 fixture는 초기 출력이 같고 첫 update의 최대 gradient error가 4.03e−6, weight error가 1.19e−7이었다. 이후에는 차이가 누적돼 strict gate를 벗어났다. 이 adapter는 실험 도구 안에서 backbone 효과를 분리하기 위한 것이며 production trainer의 HF-model 호환성을 새로 선언하는 기능이 아니다.

두 BF16 비교 모두 보상 후보의 확률은 증가했다. Native/HF의 조건부 확률은 각각 0.00695→0.39472 / 0.00850→0.48838, 같은 HF backbone의 IronCore/TRL은 0.00850→0.52324 / 0.00850→0.48838이었다. 이는 같은 학습 방향의 관측이며 여러 seed의 reward 분포가 동등하다는 통계 검증은 아니다.

초기 BF16 출력 차이와 multi-update drift를 관측했으나, packed projection·SDPA kernel·log-softmax·reduction order·AdamW rounding 중 단일 원인으로 분리해 확정하지 않았다. 작은 gradient/weight 차이가 낮은 precision의 forward와 clipping branch에 영향을 줄 수 있다는 설명은 가능하지만 이번 결과만으로 원인을 입증한 것은 아니다. “BF16도 TRL과 수치적으로 동등하다”는 결론은 내리지 않는다. [15](15-distributed-training.md)의 Native끼리 DP/TP/EP/BF16/restart gate와 이 외부-구현 gate는 서로 다른 검증이다.

## 재현과 블로그에서의 결론

![FP32와 BF16의 실제 gradient discrepancy](artifacts/2026-10-07-grpo-reference/precision_discrepancy.png)

Optional experiment dependency는 `trl==0.29.0`이다. Production dependencies에는 추가하지 않는다. [결과·로그·source snapshot·manifest](artifacts/2026-10-07-grpo-reference/README.md)에 성공과 실패를 모두 보존한다.

```bash
pip install trl==0.29.0
CUDA_VISIBLE_DEVICES='' torchrun --standalone --nproc_per_node=1 \
  scripts/validate_grpo_reference.py --output /tmp/grpo-reference-cpu
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/validate_grpo_reference.py --device cuda \
  --model-dir /path/to/SmolLM2-135M-Instruct --lr 1e-5 --rollouts 4 \
  --output /tmp/grpo-reference-135m
```

출력 경로는 매번 새로 지정한다. BF16 실패를 재현하려면 `--precision bfloat16`, 동일 HF backbone 진단에는 추가로 `--backbone hf`를 사용한다. Failed gate는 JSON을 먼저 기록하고 nonzero exit를 반환한다.

현재 뒷받침할 수 있는 문장은 **“지정한 token-GRPO 설정과 고정 rollout에서, 135M Native trainer의 FP32 loss·gradient·optimizer updates가 TRL의 실제 학습 루프와 허용 오차 내에서 일치했다”**이다. GSM8k paired reward 0→0은 그대로 유지한다. 다음 품질 입증 단계는 두 구현이 각자 실제로 rollout하는 쉬운 verifiable task에서 비영 reward를 확보하고, 같은 예산·held-out prompts·여러 seed로 성공률과 KL·길이·clipping 분포를 비교하는 것이다.
