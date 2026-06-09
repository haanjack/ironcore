# 옵티마이저

> 이 가이드는 옵티마이저 선택과 설정에 대해 다룹니다. Muon 알고리즘 (Newton-Schulz 직교화), ZeRO-1 파티션 메커니즘, LR 스케줄러 내부 구조는 [옵티마이저 시스템 설계](design/optimizer.md)를 참고하세요.

## 옵티마이저 선택

| | AdamW (`adam`) | Muon (`muon`) |
|---|---|---|
| 적용 대상 | 모든 파라미터 | 2D 가중치 (어텐션 + MLP 프로젝션) — 나머지는 AdamW |
| 메모리 | `2P` 옵티마이저 상태 | 동일 (Muon 버퍼는 온디바이스) |
| 사용 시점 | 범용, 안전한 기본값 | 트랜스포머 훈련; 더 빠른 수렴 경향 |

Muon은 어텐션과 MLP 프로젝션의 2D 가중치 행렬에만 적용됩니다. 임베딩, 바이어스, 노름, 출력 레이어 등 나머지 파라미터는 선택된 옵티마이저에 관계없이 항상 AdamW를 사용합니다.

```yaml
optim:
  optimizer: muon   # adam | muon
```

## DistributedOptimizer (ZeRO-1)

어떤 옵티마이저든 래핑해 옵티마이저 상태를 DP 랭크에 샤딩합니다. 파라미터와 그레이디언트는 완전히 복제 상태를 유지하며, 모멘텀 텐서만 분산됩니다. DP 크기 N에서 `(N-1)/N`의 옵티마이저 상태 메모리를 절약합니다.

```yaml
parallel:
  use_distributed_optimizer: true
  dist_opt_bucket_cap_mb: 25.0   # 파라미터 브로드캐스트 버킷 크기
```

**FSDP와 호환되지 않습니다.** 옵티마이저 상태 샤딩에는 둘 중 하나만 사용하세요.

**참고:** 오프로드 서브시스템과 결합할 때 AdamW 상태만 CPU RAM으로 오프로드할 수 있습니다. Muon의 직교화는 온디바이스 연산이 필요합니다.

## LR 스케줄러

```yaml
optim:
  lr_scheduler: cosine   # cosine | linear
  max_lr: 5e-4
  min_lr: 5e-5
  warmup_steps: 100
  annealing_steps: 0     # 0 = operation.train_steps 사용
```

- **`cosine`** (기본값): 선형 워밍업 후 `max_lr`에서 `min_lr`까지 코사인 감소, 이후 `min_lr` 유지.
- **`linear`**: 선형 워밍업 후 0까지 선형 감소.

## 설정 레퍼런스

| 필드 | 기본값 | 설명 |
|---|---|---|
| `optimizer` | `"adam"` | `"adam"` 또는 `"muon"` |
| `lr_scheduler` | `"cosine"` | `"cosine"` 또는 `"linear"` |
| `max_lr` | `5e-4` | 최대 학습률 |
| `min_lr` | `0.0` | 최소 LR (코사인 바닥) |
| `warmup_steps` | `0` | 선형 워밍업 스텝 수 |
| `annealing_steps` | `0` | 코사인 감소 스텝 수 (0 = `train_steps`) |
| `weight_decay` | `0.01` | 분리된 가중치 감쇠 |
| `adam_beta1` | `0.9` | AdamW β₁ |
| `adam_beta2` | `0.95` | AdamW β₂ |
| `adam_eps` | `1e-8` | AdamW ε |
| `clip_grad` | `1.0` | 그레이디언트 클리핑 임계값 |
| `muon_momentum` | `0.95` | Muon Nesterov 모멘텀 |
| `muon_newton_schulz_steps` | `5` | Newton-Schulz 반복 횟수 |
| `muon_lr_scale` | `1.0` | Muon LR = `max_lr × muon_lr_scale` |
| `adamw_lr_scale` | `1.0` | AdamW LR = `max_lr × adamw_lr_scale` |
| `load_checkpoint_optim_state` | `true` | 체크포인트에서 옵티마이저 상태 복원 |
| `load_checkpoint_lr_scheduler` | `true` | 체크포인트에서 LR 스케줄러 상태 복원 |
| `use_distributed_optimizer` | `false` | ZeRO-1 옵티마이저 상태 파티셔닝 |
| `dist_opt_bucket_cap_mb` | `25.0` | 브로드캐스트 버킷 크기 (MB) |
