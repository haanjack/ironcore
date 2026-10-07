# 13. Alignment 목적함수·rollout·실제 데이터

## GRPO와 GSPO 구분

`alignment.grpo_objective`는 `grpo`(token ratio) 또는 `gspo`(sequence ratio)이다. 기존 동작의 명시적 기본값은 `gspo`로 보존했다. 실제 reasoning pilot은 `grpo`를 선택한다. [DeepSeekMath](https://arxiv.org/abs/2402.03300)와 [GSPO](https://arxiv.org/abs/2507.18071)의 목적함수 단위를 구분하며, 설정 이름만으로 논문의 전체 training recipe를 재현했다고 주장하지 않는다.

- Token GRPO: 각 response token의 `exp(logπ_new−logπ_old)`와 signed clipped surrogate를 구하고 valid token 평균→completion 평균을 적용한다.
- GSPO: completion의 log-prob 차이를 valid response 길이로 나눈 후 exp와 clipping을 적용한다.
- 두 방식 모두 EOS는 response에 포함하고 이후 padding은 제외한다. 같은 reward의 group은 advantage 0이다.

RolloutBuffer에 token별 old log-prob를 보관하고 select/cat/save/load/to에서 유지한다. Temperature/top-k/top-p를 적용한 실제 behavior 분포와 policy 재점수 분포를 맞춘다. Off-policy token GRPO는 sequence-only legacy buffer를 거부한다. KL은 raw policy/reference log-prob의 clamped k3 추정값이다. Warped/off-policy sampling 및 ratio clamp가 있으므로 unbiased exact KL이라고 해석하지 않는다.

## Padding과 reward 실패

Batched rollout은 nonempty left-padded prompts를 받는다. Prefill의 causal/key mask, decode의 key mask, RoPE logical position, policy/reference full-sequence 재점수의 mask를 동일하게 적용했다. PAD=EOS tokenizer에서도 template 내부의 유효 EOS를 제거하지 않도록 mask를 실제 길이로 만든다. Variable-padding paged rollout은 현재 명시적으로 거부하고 batched 경로를 사용한다.

`test_remediation_contract.py`의 native GQA/RoPE 모델에서 padded batch와 개별 unpadded prompt의 greedy completion·log-prob가 일치했고, temperature=0.7/top-p=0.9/top-k=8의 behavior 재점수도 통과했다. 기존 EOS sanity test와 token GRPO signed clipping 독립 gradient oracle를 함께 실행한다. Reward backend 오류/timeout/NaN/Inf의 trainer 기본 정책은 collective fail-fast이며 fallback은 명시적으로 선택해야 한다. Math reward의 strict mode는 wrong answer에 partial credit을 주지 않고 ground truth 누락을 오류로 처리한다.

## MoE regularizer

`alignment.moe_aux_loss: include`가 기본이며 `disable`로 제외할 수 있다. DPO의 concatenated forward는 해당 policy batch의 aux, separate forward는 chosen/rejected policy aux의 평균을 더한다. Reference aux는 항상 제거하며 reference parameter는 frozen이다. GRPO/GSPO도 policy aux만 추가한다. Include−disable loss와 gradient를 독립 aux 수식과 비교한 네 trainer-path 테스트가 통과했다.

Load-balancing regularizer는 microbatch의 routing 통계에 의존하므로 full batch와 microbatch를 나눈 objective가 일반적으로 같지 않다. Aux>0의 학습에서 DP/accumulation 전체-batch 동등성을 주장하지 않는다. 수치 matrix는 aux=0으로 additive objective를 분리하며, alpha>0는 per-layer 독립 routing/aux gradient oracle와 위 포함/제외 계약으로 검증한다.

## 실제 데이터 pilot

[`benchmark_alignment.py`](../../scripts/benchmark_alignment.py)는 [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), [GSM8k](https://huggingface.co/datasets/openai/gsm8k), [UltraFeedback preference](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized)를 사용한다. 다운로드한 revision/file hash와 고정 subset index를 보존한다. 초기 instruct 모델은 이미 SFT/DPO된 모델이며, 이 실험은 from-scratch alignment나 새 benchmark SOTA를 의미하지 않는다.

Native import의 134,515,008 parameters와 HF FP32 SDPA logits는 전체 유효 vocabulary에서 max error 0.0이다. Missing key는 결정적 RoPE theta buffer 하나이며 trainable missing key는 없다. HF import가 base weights를 빠뜨리면 frozen 여부와 관계없이 중단한다. 새 LoRA adapter는 초기화를 허용하고 wrapper의 base-layer name을 HF 이름에 맞춰 load한다. Tokenizer의 special token attribute를 token 수로 더하던 중복 계산도 제거했다.

SFT/DPO response mask는 template의 문자열 경계와 fast tokenizer offset을 이용한다. BPE가 boundary 주변을 하나의 token으로 합치는 경우, prompt만 tokenization한 길이를 전체 encoding의 prefix 길이라고 가정하지 않는다. Response 문자를 포함하는 token을 학습 대상으로 삼고 prompt-only token을 제외한다. Char prefix 자체가 맞지 않는 template은 오류로 처리한다.

Pilot의 steps·subset·before/after·frozen reference·memory 결과를 아래에 기록했다. GSM8k teacher-forced NLL 감소와 생성 정답률은 서로 다른 지표다. Small sample/짧은 horizon·pretraining contamination 가능성을 분리하고, reward가 모두 0이면 RL reasoning 개선의 증거로 쓰지 않는다.

## 첫 실제-data 결과

FP32 stored weights/BF16 compute, context=512, seed=73, fixed train subset 128 / held-out subset 32이다. Global batch=4, DP=2, local microbatch=1이며 lr=1e−5이다. 초기 weights는 모두 동일 SmolLM2 instruct checkpoint다.

| Pilot | Updates | Held-out before → after | Reference |
|---|---:|---|---|
| SFT / GSM8k | 32 | response NLL 1.175856 → 0.960479 | 해당 없음 |
| DPO / UltraFeedback | 24 | loss 0.693147 → 0.680968, preference win 0 → 0.53125 | parameter exact frozen |
| GRPO / GSM8k 초기 경로 | 8×2 epochs | exact reward 0.0078125 → 0 | parameter exact frozen |

DPO 초기 policy/reference가 같아 reward margin이 tie=0이며 `>0` win은 0이다. 따라서 0→0.53125를 chance 대비 53-point 개선이라고 해석하지 않는다. Small held-out의 통계적 유의성이나 전체 preference 품질을 입증한 것은 아니다. SFT의 response token accuracy 평균은 0.7153→0.7565지만 생성 math accuracy를 대신하지 않는다.

초기 GRPO 경로는 training rollout/reference가 FP32, update가 BF16이었다. IS의 기록된 behavior를 보존하더라도 동일 stored weights의 execution precision이 달라 ratio와 KL에 roundoff 차이가 생길 수 있다. Rollout/reference에도 동일 autocast를 적용하도록 수정했고 별도 후속 run을 기록한다. 최초 negative quality 결과를 숨기거나 precision 수정이 reasoning 품질을 개선했다고 미리 주장하지 않는다.

## 동일 compute precision과 paired 생성 평가

Rollout/reference/policy의 BF16 autocast를 통일한 뒤 네 trainer의 BF16 EP·FSDP·분산 optimizer 및 token GRPO의 full/DP/TP/EP gate와 exact resume를 통과했다. Frozen reference의 parameters는 계속 FP32다. 이후 실제 GRPO 평가는 rank별 `100073 + rank`의 같은 생성 RNG를 before/after에 적용하고 training RNG와 분리했다. 32 held-out prompts×4 completions의 exact reward는 **0→0**, reference는 exact frozen이었다. Implementation이 실행되어 loss/update를 만든 사실로 reasoning 성능 개선을 주장하지 않는다. 135M 모델·128-token horizon·희소 reward·짧은 학습의 한계를 드러낸 negative pilot이다.

Token GRPO는 completion마다 valid-token 평균 후 completion 평균을 적용하고 group standard deviation은 PyTorch sample std(correction=1)이다. 이 normalization 선택과 clamped k3는 기록한 구현의 정의이며 모든 GRPO recipe의 동일한 기본값이라고 주장하지 않는다.

[16. TRL 외부 trainer 대조](16-grpo-trl-reference.md)에서 동일한 completion/reward의 실제 두 학습 루프를 비교했다. 135M FP32의 loss·gradient·update는 허용 오차 내에서 일치했다. BF16 전체 trajectory의 strict 동등성은 통과하지 못했으므로, 위 reasoning pilot과 함께 그 범위를 명시한다.
