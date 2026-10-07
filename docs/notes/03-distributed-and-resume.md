# 03. DP=2, TP=2, checkpoint 재개

## 명세

[A2 Systems의 분산 학습](https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_assignment2_systems.pdf)을 실제 trainer 경로로 검증한다. global batch는 `microbatch × accumulation × DP size`다. TP size는 sample 수를 늘리지 않는다.

`validate_trainers.py`가 단일 GPU full batch를 기준으로 accumulation/DP/TP의 초기 weights·첫 raw gradient·최종 weights·loss 궤적을 비교한다. TP weights는 module의 partition 정보로 재구성한다. DDP의 두 rank weights도 exact equality로 비교한다. 재개는 중간 step에서 멈춘 별도 프로세스를 새로 시작하여 검증한다.

## 실제 발견

TP의 학습 자체는 진행됐지만 universal checkpoint에서 재개한 뒤 Adam update가 `KeyError: exp_avg`로 실패했다. 초기 실패 artifact는 `gpu01` 실험 로그에 남겼다.

원인과 수정:

- load가 full-size optimizer moments를 local parameter 크기로 먼저 reshape하여 버렸음: **TP split 후 shape 검증**으로 변경한다. 잘못된 shape는 조용히 누락하지 않고 실패시킨다.
- 저장 시 optimizer integer ID 순서와 model named parameter 순서를 zip함: decay/no-decay group 등의 순서가 다를 수 있으므로 **parameter name으로 상태를 대응**한다.
- optimizer step을 training step의 integer로 덮어씀: **optimizer가 실제 저장한 step/state 값을 보존**한다. GRPO는 한 rollout step에 여러 update를 하므로 이 구분이 특히 중요하다.
- 지원 state에 `momentum_buffer`를 포함하여 Muon/SGD 계열의 상태를 누락하지 않도록 한다. 이는 그 optimizer들의 전체 GPU 조합 검증을 완료했다는 뜻은 아니다.
- embedding/column bias의 TP shard metadata를 보강한다.
- `latest_step.txt`도 temporary file+replace로 갱신한다.

## rank-local trainer state

native model checkpoint와 함께 `step_N/trainer_rankR.pt`를 저장한다. reference model, AMP scaler, worker별 torch CPU/CUDA RNG, TP RNG tracker를 보존한다. 재개는 native weights/optimizer/scheduler를 복원하고 reference를 만든 뒤 저장된 trainer state를 복원한다. 저장은 atomic rename 후 모든 rank를 동기화하여 native checkpoint marker보다 먼저 완료한다.

final checkpoint에도 trainer state 저장을 적용한다. topology나 task가 다르면 같은 실험의 재개로 취급하지 않는다. 이전 checkpoint에 trainer state가 없으면 경고하고 기존 initialization을 사용하므로 alignment reference와 재개 궤적의 동등성을 보장하지 않는다. 기존 SFT checkpoint에서 새 DPO/GRPO task를 시작하는 것은 별도 경우다.

## 실측 결과

세 가지 CUDA matrix가 통과했다. 각 matrix는 네 task × full/accum/DP=2/TP=2, 4 rollout/training steps를 실행한다. 각 구성마다 2-step checkpoint를 새 프로세스에서 재개하여 uninterrupted 실행과 비교했다. GRPO는 rollout마다 두 optimizer updates를 수행한다.

아래 오차는 단일 full batch 대비 accum/DP/TP 세 경우의 **최댓값**이다. CS336 구성은 RMSNorm/RoPE/SwiGLU 모델이며 통제 실험의 모델은 23,328 parameters다. 대형 모델 학습 실험과 구분한다.

| task | CS336 FP32 raw gradient | CS336 FP32 최종 weights | CS336 BF16 raw gradient | CS336 BF16 최종 weights |
|---|---:|---:|---:|---:|
| pretraining | 4.47e-8 | 5.97e-7 | 9.77e-4 | 1.74e-3 |
| SFT | 2.98e-8 | 1.43e-6 | 1.95e-3 | 1.72e-3 |
| DPO | 1.79e-7 | 8.57e-8 | 7.81e-3 | 2.38e-3 |
| GRPO/GSPO | 1.19e-7 | 1.84e-7 | 3.91e-3 | 1.71e-3 |

FP32는 `atol=2e-5, rtol=2e-4`, BF16은 `atol=5e-3, rtol=5e-2`의 원소별 gate를 통과했다. BF16 DPO의 최대 gradient 오차가 절대 tolerance보다 크더라도 해당 원소의 상대 tolerance를 포함한 기준을 통과한 것이다. BF16을 FP32 수준의 동일성으로 해석하지 않는다.

세 matrix의 총 48개 재개 비교에서 최종 weights·reference·후반 loss 궤적은 exact equality였고 scheduler/scaler 상태도 일치했다. DDP rank 간 최종 weights도 exact equality였다. GPT-2 구성(19,008 parameters)의 FP32 matrix도 통과했으며 최대 weights 오차는 1.30e-6이다. chunk size 7의 pretraining/SFT 추가 matrix도 통과했고 8개 재개 비교의 오차가 0이었다.

근거: [CS336 FP32](artifacts/2026-10-07/ironcore-validation-cs336-float32/report.json), [CS336 BF16](artifacts/2026-10-07/ironcore-validation-cs336-bfloat16/report.json), [GPT-2 FP32](artifacts/2026-10-07/ironcore-validation-final-fp32/report.json), [chunked CE](artifacts/2026-10-07/ironcore-validation-chunked/report.json).

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/validate_trainers.py \
  --device cuda --architecture cs336 --steps 4 --output /tmp/ironcore-correctness-fp32
```

BF16은 `--precision bfloat16`, chunked CE는 `--tasks pretrain,sft --loss-chunk-size 7`로 별도 output에서 실행한다. 이 세 실행과 GPT-2 실행의 report 및 rank별 loss/evaluation 요약을 보존했다.

## 한계

증거는 한 node의 두 3090, uniform local batch, standard optimizer와 DP/TP다. FSDP·optimizer offload·weight streaming·EP·다중 node·topology 변경 재개의 보장은 별도 검증이 필요하다. rank-local reference 파일을 보존하지 않고 model checkpoint만 복사하면 alignment 재개 증거가 사라진다. 오래된 보고서의 loss 감소나 checkpoint timeout을 정상 종료와 동일하게 판단하지 않는다.
