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
