# 11. Optimizer·global objective·precision 보완

## 질문과 독립 기준

DP/TP끼리 일치해도 공통 optimizer 수식이 틀리면 올바른 학습의 증거가 아니다. 표준 [PyTorch AdamW](https://docs.pytorch.org/docs/2.12/generated/torch.optim.AdamW.html)의 `m̂ / (sqrt(v̂) + ε)`와 비교한다. 기존 bias correction을 밖으로 묶은 수식에서는 denominator에도 `ε × sqrt(1−β₂ᵗ)`가 필요했다. AdamW, Muon의 AdamW branch, CPU optimizer offload를 수정했다. Muon 자체의 Newton–Schulz 업데이트는 이 AdamW oracle의 대상이 아니다.

`tests/unit/optimizer/test_adamw_oracle.py`는 gradient 크기 1/1e−8/1e−10, AMSGrad on/off, weight decay, 7-step 및 state reload를 독립 `torch.optim.AdamW`와 비교한다. 18개 조합과 BF16 parameter의 FP32 moment 복원 2개가 통과했다. PyTorch의 기본 optimizer reload가 moment를 parameter dtype으로 cast하는 경우도 별도 보완해 원래 FP32 값과 offload 위치를 보존한다.

[작은 update 실험](artifacts/2026-10-07-remediation/precision_sensitive_updates.json): 첫 gradient 1e−8, lr=1e−3, β=(0.9,0.999), ε=1e−8에서 이전 update는 약 3.065e−5, 표준값은 5e−4이다. `lr=1e−4`의 같은 작은 gradient를 1,000번 적용하면 BF16 stored weights는 1.0에 머무르고 FP32 weights에는 update가 누적된다. 이는 rounding을 분리한 scalar 실험이며 언어모델 품질 수치로 해석하지 않는다.

## 저장 precision과 compute precision

`trainer.parameter_precision: float32`가 기본이다. `model.precision: bfloat16`에서는 FP32 weights/moments를 저장하고 autocast로 BF16 compute를 한다. 이전 저장 방식은 `parameter_precision: model`로 명시한다. FP32 weights가 항상 throughput/VRAM 측면에서 유리하다는 뜻은 아니다. 성능 대조에서 저장 precision을 고정한다. [AMP 지침](https://docs.pytorch.org/docs/2.12/notes/amp_examples.html)의 unscale→finite check→clipping→update 순서를 지킨다.

FP32 weights/BF16 compute의 pretrain/SFT/DPO/GRPO full·accum·DP=2·TP=2 및 16개 same-topology 재개 검증이 통과했다. 이 matrix의 소스는 `ad-bf16-source_hashes.json`으로 보존한다. 후속 checkpoint/alignment 변경은 이후 별도 gate로 검증한다.

## 가변 valid token·sample 수

pretrain은 update 전체의 valid target token 평균, SFT는 응답이 있는 원본 문서별 평균의 평균, DPO는 preference pair 평균이다. Packed SFT의 행은 문서와 같지 않으므로 collator가 `loss_sample_ids`를 기록한다. 빈 response는 denominator에서 제외한다.

Accumulation 전에 CPU input batch들의 count만 모으고 DP 전체 denominator를 all-reduce한다. microbatch backward 가중치는 `local_units × DP_size / global_units`이며 DDP의 gradient 평균을 보상한다. batch를 미리 모을 때 activation graph는 보관하지 않는다. built-in 및 external evaluation도 같은 units로 가중한다. GRPO는 별도의 rollout/epoch loop로 completion 평균을 유지한다.

CPU에서는 1/3/4 sample의 불균일 microbatch, 서로 다른 response 길이, 빈 response, packed 문서의 loss·gradient·optimizer update를 독립 전체 objective와 비교했다. CUDA DP 불균일 rank 실험은 [분산 검증](15-distributed-training.md)에 기록했다. 최대 weight error는 token 평균 6.98e−10, sample 평균 1.51e−9이다.

## 실패 시 update 계약

Forward가 유한하더라도 backward gradient에 NaN/Inf가 있을 수 있다. 모든 gradient를 unscale한 뒤 device scalar로 검사하고 전 rank의 MAX를 교환한다. 한 rank라도 invalid이면 모든 rank의 gradient를 비우고 optimizer/scheduler를 실행하기 전에 예외를 발생시킨다. GRPO override도 같은 경로를 사용한다. 이 검사는 zero gradient나 objective가 0인 정상 상태를 오류로 취급하지 않는다.

`loss.item()`을 microbatch마다 호출하던 기록은 detached device tensor로 누적한 뒤 update당 한 번 변환한다. MoE aux도 detach한 tensor로 보관하고 logging 요청 시에만 host 값으로 바꾼다. Count/finite check·norm·GRPO metrics 등에 필요한 동기화까지 제거했다고 주장하지 않는다.

## Granite MoE: 합산 내부 정밀도와 출력 dtype

Granite 3.1의 현재 HF `grouped_mm` 경로는 각 expert 출력과 routing weight의 곱을 BF16으로 만든 뒤 top-k를 합산하고, residual에 더하기 전에 BF16으로 복원한다. Expert마다 BF16 accumulator에 더하는 eager 경로와 rounding 순서가 다르다. CUDA autocast 안의 `weighted.sum(1)`은 FP32 **출력 tensor**를 만들 수 있다. 이를 그대로 residual에 더하면 이후 hidden states까지 FP32로 승격된다. Granite grouped 경로에서는 `weighted.sum(1, dtype=x.dtype)`로 내부 FP32 reduction을 유지하면서 결과를 BF16으로 직접 저장한다. FP32 activation 크기의 합산 결과를 GPU에 저장한 뒤 `.to(bfloat16)` 하는 단계를 만들지 않는다. Torch의 GEMM/reduction workspace까지 register에만 존재한다고 보장하는 것은 아니다.

Granite expert LoRA는 HF PEFT의 packed parameter 방식과 같이 gate/up이 A를 공유한다. A/B의 저장은 FP32이고 계산용 views는 BF16으로 만든다. `torch.baddbmm(W, B, A, alpha=alpha/r)`를 autocast 없이 BF16 operands에 적용하면 W+delta를 BF16으로 직접 쓸 수 있다. Attention의 q/k/v/o는 기존 activation LoRA를 사용한다. TP2에서는 각 expert의 row projection을 먼저 합친 다음 routing weight를 적용한다.

실제 Granite 1.334B 체크포인트의 첫 MoE에 공통 HF 입력을 넣었을 때 수정된 BF16 출력은 bitwise 일치했다. 전체 FP32 3-update 비교는 TP1/TP2에서 loss 차이 최대 4.44e-6/3.58e-6, 전체 LoRA gradient 상대 L2 최대 2.33e-4/2.24e-4였다. 그러나 BF16 전체 독립 학습에서는 routing 분기가 달라지고 gradient 편차가 남는다. TP1 8 updates의 최대 상대 L2는 약 31.5%다. 같은 adapter를 적용해도 forward 차이가 남아, optimizer trajectory만 원인으로 볼 수 없다. 고정 HF routing이나 HF RMSNorm으로 일부 경계를 바꾼 짧은 대조에서도 모두 제거되지는 않았다. HF 자체 동일-state gradient 반복 편차 약 1.60%도 측정했지만 전체 차이를 설명하지는 않는다. 이 결과를 Gemma 26B의 품질 문제 원인으로 확정하지 않는다.

같은 작업에서 초기화와 offload의 별도 결함도 수정했다. Native LoRA A는 `[input, rank]`이므로 Kaiming fan-in은 A의 transpose에 대해 계산해야 한다. 기존 계산은 rank를 fan-in으로 사용해 H1024/r8에서 초기 A의 범위를 약 11.31배 크게 만들었다. 기존 adapter 파일의 값에는 영향을 주지 않는다. CPU의 trainable adapter와 offload host tile도 처음 등록될 때 storage를 공유하도록 바꿔, scheduler 생성 뒤 load한 adapter가 첫 prefetch의 이전 사본으로 덮이지 않게 했다. 새 초기화가 Gemma SFT의 장기 품질을 회복하는지는 별도 검증이 필요하다.

## Granite BF16 원인 분리: 같은 dtype에서도 GEMM 경계가 중요하다

후속 CUDA TP1 진단은 같은 checkpoint·adapter·첫 microbatch(2×512)를 사용하고 optimizer update 없이 진행했다. 초기 B=0 상태와 실제 HF 8-update adapter를 각각 비교했다. 원자료와 단일 HTML은 Git에서 제외되는 `.local/granite-moe-reference-study/diagnosis-report.html`에 보존한다. 아래 변경은 진단용 monkeypatch이며 production 반영·TP2·독립 다중 update 검증이 완료됐다는 뜻은 아니다.

- **RMSNorm 통계·정규화 계산:** 같은 입력에서 fused norm과 stock HF의 BF16 결과는 소수 값이 다르며, 상대 L2 최대 약2.72e-5였다. HF 모델의 norm만 native fused kernel로 교체한 결과가 native의 expert 미분할 대조와 bitwise 일치했다. 초기 상태에서 stock HF norm과 expert 미분할을 함께 적용하면 전체 logits·loss가 정확히 같았다. 작은 경계 차이가 모델 전체에서는 큰 차이로 전파될 수 있다. 진단의 stock HF norm은 FP32 activation 임시값을 쓰므로 그대로 production 메모리 최적화로 간주하지 않는다.
- **같은 expert의 행 분할:** 4096 row-budget를 채우면서 expert 하나를 두 GEMM segment로 나누면 GEMM M 크기가 바뀐다. Stock HF expert를 같은 방식으로 분할해도 native와 동일한 forward 편차가 재현됐다. 초기 대조에서4096 상한을 유지하고 expert 전체를 한 segment로 묶으면, HF norm과 함께 전체 forward가 정확히 같았다. 이는 개별 expert의 행 수가4096 이하인 microbatch의 결과이며 장문에서 동일 조건을 보장하지 않는다.
- **결합 K/V projection:** 같은 입력과 같은 output cotangent에서 attention forward는 초기 상태의24개 층 모두 정확히 같았지만 dX는 약0.27~0.30% 달랐다. K/V를 각각 계산한 layer0/23 대조에서 dX는 정확히 같거나 매우 작은 반복 오차만 남았다. 각각의 BF16 dX를 더하는 경계와 한 GEMM 안에서 합산하는 경계가 다르다.
- **Flash SDPA backward 비결정성:** HF 자체 같은-state forward는 반복해도 정확히 같았지만 gradient는 달랐다. `ScaledDotProductFlashAttentionBackward0` 실행에만 deterministic 모드를 적용하면 전체 gradient 반복 편차가0이 됐다. `IndexBackward0`만 deterministic으로 만들어서는 없어지지 않았다. 이 환경의 반복 편차를 MoE gather atomic 순서 때문이라고 확정해서는 안 된다.
- **Output head vocabulary padding:** 실제49155와 padded49280의 유효 logits는 같아도 backward GEMM contraction 크기가 달라진다. 같은 head 입력·cotangent 대조의 dX 상대 L2는 약2.41e-4였다. 다른 경계를 맞춘 초기 대조에서 실제 vocabulary 크기로 계산하면 전체 gradient 상대 L2가 약1.65%에서0.0166%로 줄었다. Padding은 token ID/TP 저장 계약상 필요하므로 저장을 없애는 대신 compute view를 별도로 검증해야 한다. 이 홀수 vocabulary 결과를 Gemma에 일반화하지 않는다.

**학습 후 상태에서 남은 결정적 원인은 attention LoRA의 GEMM operand layout이었다.** 위 경계들을 맞춘 뒤에도 실제 HF8 adapter에서는 logit 상대 L2 약2.45%, 전체 gradient 상대 L2 약11.07%가 남았다. Native `x @ A @ B`를 HF `Linear`와 같은 weight stride/layout으로 계산하도록, factor transpose의 contiguous view를 `F.linear`에 전달하자 logits·loss·전체23,396,352개 LoRA parameter의 gradient가 bitwise 일치했다. A/B 모두 nonzero gradient가 있는 상태다. Expert parameter-LoRA folding의 layout만 바꾸는 대조로는 이 편차가 사라지지 않았다.

이 대조의 attention factor 저장은 기존 FP32 leaf 그대로이며 CUDA autocast의 GEMM activation 출력은 BF16이다. 작은 adapter의 operand 배치를 맞추는 데 full activation 크기의 FP32 GEMM 결과가 필요하지 않다. 같은 BF16 dtype과 같은 수식만으로는 reference 수치 경로의 일치를 보장할 수 없다. 실제 operand stride, GEMM segment 크기, 합산·반올림 경계를 함께 기록해야 한다. 초기 B=0 forward 일치만으로 학습 후 LoRA 경로를 인증해서도 안 된다.

이 결과는 고정-state Granite TP1에서 편차 원인을 재현·제거한 증거다. Production kernel에 직접 BF16 출력을 유지하며 반영하는 작업, TP2 및 다중 update trajectory 비교, Gemma26B thinking/benchmark 검증은 별도 후속 범위다.

## Granite production 반영과 독립 학습 검증 (2026-10-10)

위 진단 후 production 경로에 수정하고 실제 Granite 1.334B checkpoint에서 재검증했다. Monkeypatch 없는 `production_validate.py`와 원자료, 소스 SHA256, 단일 HTML은 Git에서 제외되는 `.local/granite-moe-reference-study/production-report.html`에 보존한다. 이전 진단·실패 로그도 유지한다.

Granite RMSNorm은 Triton register 안에서 HF의 FP32 vector-of-4 합산과 rsqrt backward 순서를 재현하고 BF16 결과를 직접 저장한다. Full activation 크기의 FP32 norm 임시 tensor를 생성하지 않는다. Inv-RMS는 token당 FP32 scalar 하나이며 register 내부 FP32 연산은 유지한다. CUDA unit test는 forward·dX bitwise 비교와 TorchDispatch로 activation 크기의 FP32 출력이 없는지를 검사한다. 지원 폭은 128 이상 8192 미만의 4 배수이고, 다른 폭은 기존 kernel을 사용한다. Gemma의 pow 기반 norm backward에는 이 rsqrt 정책을 적용하지 않았다.

Attention LoRA는 FP32 leaf를 유지하면서 작은 factor의 transpose-contiguous view를 `F.linear`에 전달한다. Granite base projection의 배치도 HF에 맞춘다. K/V 각각에 별도의 TP copy autograd node를 둬 BF16 input gradient가 합쳐지는 경계까지 분리했다. Forward만 분리하고 copy node를 공유하면 실제 HF8 adapter에서 전체 gradient 편차 약1.62%가 남았다. Granite CUDA attention은 조건이 맞으면 stock SDPA의 native GQA 경로를 사용한다. Expert weight folding도 HF operand 배치와 BF16 직접 출력을 유지한다.

Granite execution-group planner는 budget 안에 들어가는 expert의 전체 행을 같은 segment에 둔다. Expert 하나가 budget보다 크면 여전히 분할하고 각 group의 총행은 상한을 지킨다. 4096 budget의 작은 검증에서 HF geometry를 맞춘 것이며, 장문에서 모든 expert를 무제한으로 계산한다는 뜻은 아니다. Actual vocabulary만 output-head GEMM에 사용하고 padding logits는 이후 −inf로 복원한다. TP1 CE는 실제 vocabulary에 stock `F.cross_entropy`를 적용하며, TP2 backward는 log-softmax/NLL의 연산 순서대로 `exp(log_probs) × dy` 후 target의 dy를 뺀다.

독립 update에서 추가로 clipping과 AdamW의 FP32 연산 순서를 맞췄다. Packed HF parameter와 native의 expert별 parameter는 FP32 norm 합산 grouping이 달라 1 ULP 차이가 생길 수 있다. `utils.deterministic: true`는 Torch deterministic kernels와 **FP64 clipping-norm accumulation, FP32 최종 scalar/coefficient**를 사용한다. 이번 HF 대조도 같은 clipping 정책을 사용했다. Stock HF 기본 FP32 clipping과 bitwise 같다고 주장하지 않는다. 기본 nondeterministic 설정은 기존 FP32 norm accumulation을 유지한다. FP64를 쓰는 것은 gradient norm의 scalar reduction이며 BF16 full activation을 FP64로 저장하지 않는다.

AdamW, Muon의 AdamW branch, CPU optimizer offload는 Torch와 같은 moment `lerp_`, bias-corrected denominator 및 step-size 순서를 따른다. 수학적으로 같은 식의 재배치를 줄여 BF16 경계에 영향을 주는 미세한 update 차이를 없앴다. CPU offload와 GPU 연산의 bitwise 일치까지 보장하지 않는다.

| 최종 검증 | 독립 updates | 최대 loss 절대 차이 | 전체 LoRA gradient 상대 L2 | 판정 |
| --- | ---: | ---: | ---: | --- |
| BF16 TP1, offload 없음 | 8 | 5.96e−8 | 0 | Gradient·업데이트 adapter·평가 logits bitwise 일치 |
| FP32 TP2, offload 없음 | 3 | 3.28e−7 | 2.36e−5 | 작은 수치 오차로 통과 |
| BF16 TP2, CPU offload·layer spill/recompute·chunked CE | 3 | 3.51e−3 | 0.25835 | 실행·저장/reload 통과, 수치 parity 미통과 |

공통 조건은 microbatch2×512, accumulation2, rank8/alpha16의 q/k/v/o 및 모든 expert gate-up/down LoRA, FP32 adapter, lr2e−5, AdamW β=(0.9,0.95), ε1e−4, clip1이다. BF16 reduced-precision GEMM reduction과 TF32는 껐다. 한 번 초기 상태를 복사한 뒤 양쪽 optimizer가 독립 update하며 중간 resync는 하지 않는다. TP1은 32개 학습 conversation을 소비하고 별도 validation16/test16을 평가했다. Validation CE는 양쪽 모두0.922345→0.851785, test CE는0.613072→0.559525였다. 모든3264 adapter tensor(3072 expert tensor)가 변경됐다. 세 구성 모두 native standalone adapter 및 새 HF base+PEFT reload가 정확히 복원됐고 TP2 replica도 정확히 일치했다. VRAM 수치는 reference와 native를 같이 올린 검증 process의 peak이므로 native 단독 메모리로 읽지 않는다.

**BF16 TP2 forward의 별도 원인 대조:** HF 모델의 projection, attention head, expert intermediate, vocabulary GEMM을 TP2와 같은 폭으로 나누고 BF16 부분 결과를 합쳤다. 초기 상태의 한 microbatch에서 이 HF-only 대조와 native TP2 logits는 bitwise 일치했다. 두 모델 모두 stock 단일-GPU HF 대비 logit 상대 L2 약4.68%였다. Native만의 forward indexing 결함 없이도 topology의 BF16 반올림 경계가 전체 편차를 재현한다. 그러나 이 대조의 gradient 상대 L2는 약1.68%가 남으며 HF expert parameter folding의 backward geometry도 완전히 같지는 않다. 이를 독립 학습에서 측정한25.8% gradient 편차의 전부를 설명하거나 BF16 TP2 인증으로 해석하지 않는다.

선택한 GPU 회귀132개, CPU 회귀197개(9 skipped), smoke recipe config-check와 Ruff/diff-check를 통과했다. Tiny HF oracle에서 `.to(BF16)`로 rotary frequency buffer까지 낮추면 실제 BF16 `from_pretrained`와 다르므로 FP32 inv_freq를 보존해 비교한다. 이는 테스트 기준 수정이며 이번 production RoPE 변경은 아니다. 긴 context·공식 benchmark·Gemma26B thinking/품질 회복은 아직 검증하지 않았다.

## BF16 TP2 후속: 대조군의 topology·CE·optimizer 위치를 분리

후속 조사에서는 **production 구현을 추가로 변경하지 않았다.** 앞 절의 Native TP2를 유지하고 HF-only 대조의 물리적 GEMM 및 autograd 경계를 단계적으로 맞췄다. 원자료·harness·단일 HTML은 Git에서 제외되는 `.local/granite-moe-reference-study/tp2-report.html`에 보존한다. 이전 stock HF 대비 실패 수치도 그대로 보존하며 새 대조로 덮어쓰지 않는다.

1. **Offload/recompute 고정-state 대조:** 초기 adapter와 동일한 두 microbatch에서 recompute on/off의 전체3264개 raw adapter gradient가 bitwise 일치했다. 같은 CE/head chunk 설정끼리 비교하면 CPU weight/adapter streaming·full-layer activation spill을 켜고 끈 gradient도 모두 정확히 같았다. 기본 head와 chunk19 head/CE 사이에는 약1.75% 차이가 있었다. Streaming 손실로 확정하지 않고 GEMM M 크기를 바꾼 별도 구성으로 구분한다.
2. **HF 분할 대조의 backward 경계:** Expert folding만 shard-local로 바꿔서는 앞 절의 약1.68%가 없어지지 않았다. HF concatenate backward가 각 shard의 cotangent를 contiguous로 전달하도록 하고, column projection의 두 input-gradient를 하나의 합산 node에서 더한 뒤 공통 입력으로 전달하자 전체 gradient가 정확히 같아졌다. Native의 TP copy-backward는 projection별 두 rank gradient를 먼저 합친다. HF의 여러 분할 projection이 공통 입력에 직접 이어지면 BF16 gradient가 더해지는 순서가 다르다. Contiguous cotangent만 맞춘 대조는 마지막 norm/head부터 layer23 attention projection까지 정확했지만, 그 입력 norm의 gradient에서 약0.336%가 다시 생겼다. 두 경계를 함께 맞춘 대조는 **초기 B=0과 실제 HF8의 A/B nonzero 상태에서 forward·전체 LoRA gradient가 bitwise 일치**했다.
3. **분산 CE와 독립 trajectory:** 이 HF graph에서도 stock 단일-GPU `F.cross_entropy`를 써서 Native 분산 CE와 독립 학습하면 첫 gradient 차이는 약1.2745%, 8 updates 중 최대는 약21.774%였다. HF-only log-softmax에서 FP32 denominator를 vocabulary 두 partition으로 따로 합산한 뒤 더하도록 바꿨다. Native CE·loss mask·SFT reducer·collective를 호출하지 않는 대조다. 같은 projection 및 partition CE 조건에서는 8 updates 전체 gradient와 업데이트 adapter, train/validation/test logits·greedy·expert 선택이 bitwise 일치했다. 최대 scalar loss 차이는2.98e−8이었다. 이는 분할 CE arithmetic에 대한 일치이며 stock HF fused log-softmax와 bitwise 같은 결과라는 뜻은 아니다.
4. **CPU/GPU optimizer 위치:** Full-offload + chunk19 head/CE에서도 같은-state gradient는 정확했지만 Native CPU optimizer와 HF GPU optimizer를 비교하면 첫 adapter relative L2 약2.60e−11 차이가 생겼다. 이후 BF16 operand 변환과 routing을 거치며 3-step부터 gradient가 갈라지고 최대 약25.09%에 이르렀다. HF adapter leaf와 `torch.optim.AdamW(foreach=False)`도 CPU로 옮겨 계산 위치를 맞추자 **offload 구성의 독립8 updates에서도 gradient·adapter·평가 logits가 모두 bitwise 일치**했다. Native의3264 CPU adapter tensor와 HF의288 packed CPU tensor를 비교했고 standalone adapter reload 및 TP replica도 정확히 복원됐다.

| 8-update 대조 | Native 위치/구성 | HF 대조 조건 | 최대 gradient 상대 L2 | 판정 |
| --- | --- | --- | ---: | --- |
| Projection graph만 맞춤 | GPU TP2, 기본 head | TP2 graph, stock CE, GPU AdamW | 0.217742 | CE 순서 차이가 trajectory로 증폭 |
| CE까지 맞춤 | GPU TP2, 기본 head | TP2 graph, partition CE, GPU AdamW | 0 | Gradient·adapter·logits bitwise 일치 |
| Offload, optimizer 위치 다름 | CPU adapter, spill, chunk19 | TP2 graph, partition CE, chunk19, GPU AdamW | 0.250899 | FP32 update 미세 차이가 trajectory로 증폭 |
| Offload, optimizer 위치도 맞춤 | CPU adapter, spill, chunk19 | TP2 graph, partition CE, chunk19, CPU AdamW | 0 | Gradient·adapter·logits bitwise 일치 |

진단용 partition log-softmax는 독립 stock Torch와 CPU FP64에서 output 최대 절대 차이3.55e−15, 임의 dense cotangent의 gradient 최대 절대 차이5.86e−14였다. CPU FP32에서도 gradient relative L2는 약1.28e−7이었다. 두 CE의 수학적 미분을 확인한 수치 대조이며 BF16 전체 모델의 편차를 이 작은 local error로 상한화하지 않는다.

별도 **순수 Torch CPU/GPU AdamW 대조**는 IronCore/HF 없이 같은 FP32 초기 parameter와 미리 지정한 동일 gradient를 양쪽에 적용했다. 8 updates의 parameter relative L2는 약1.96e−9~4.26e−9였고 BF16 cast 값이 step당6~28개 원소에서 달랐다(총786432개 원소). CPU/GPU 실행만 달라도 quantization 경계가 갈라질 수 있다는 positive control이다. 실제 full-offload의 초기 작은 update 차이를 Native만의 optimizer 수식 결함으로 확정하지 않는다.

조건은 앞 절과 같은 실제 Granite1.334B, BF16 compute/FP32 adapter, deterministic kernels/동일 FP64 clipping norm, 2×512 microbatch와 accumulation2, LoRA r8/alpha16, lr2e−5이다. 소비한 train32 conversation과 별도 validation16/test16의 partition CE는 GPU 대조에서0.919816→0.854746 / 0.612673→0.566494, CPU-offload 대조에서0.919816→0.855498 / 0.612673→0.562410이었다. 두 topology의 학습 결과가 서로 bitwise 같다는 주장은 아니며, 각 topology를 같은 계산 위치의 독립 HF 대조로 확인했다.

**이번에 재현한 Granite BF16 TP2 편차는 계산·합산·저장 경계를 맞춘 독립 대조에서 제거됐다.** Stock 단일-GPU HF와 같은 BF16 trajectory를 요구하는 것은 별도 목표다. 이 결과는 context512의8-step 검증이며 32K/장기 수렴/공식 benchmark나 Gemma26B 품질 회복을 인증하지 않는다. 다음 품질 검증에서는 실제 사용할 TP/offload/CE/optimizer 위치를 고정하고, 짧은 checkpoint부터 held-out CE·생성·thinking을 확인한다.
