# Offload + 분산 학습 아키텍처 설계 (한글판)

> **상태**: ✅ 구현 완료 (Phase A-H 완료)
> **범위**: Optimizer state offload/Weight streaming/Activation spilling offload와 DDP, FSDP, DistributedOptimizer의 상호작용
> **대상 독자**: 멀티-GPU 학습에서 메모리/처리량 트레이드오프를 평가해야 하는 엔지니어
>
> 본 문서는 [offload_fsdp_architecture.md](offload_fsdp_architecture.md) 의 한글 번역 + 코드베이스 검증 결과 + 추가 고려사항을 통합한 버전입니다.

> **빠른 시작**: [§7.2 구성 선택 기준](#72-구성-선택-기준)에서 일반적인 시나리오를 확인하거나, [§7.3 전체 구성 참조](#73-전체-구성-참조)에서 모든 옵션을 확인하세요.

## 목차

1. [요약 (Executive Summary)](#1-요약)
2. [시스템 구성요소](#2-시스템-구성요소)
3. [기능 비교 매트릭스](#3-기능-비교-매트릭스)
4. [구성 카탈로그](#4-구성-카탈로그)
5. [메모리 분석](#5-메모리-분석)
6. [Prefetch 및 Overlap 분석](#6-prefetch-및-overlap-분석)
7. [사용자 의사결정 가이드](#7-사용자-의사결정-가이드)
8. [구현 권고사항](#8-구현-권고사항)
9. [추가 고려사항 (영문판 대비 보강)](#9-추가-고려사항-영문판-대비-보강)
10. [검증 결과 및 갭 요약](#10-검증-결과-및-갭-요약)

---

## 1. 요약

Ironcore의 offload 시스템은 GPU VRAM 사용량을 줄이는 세 가지 독립 모드를 제공합니다.

- **Optimizer state offload (Optimizer state offload)**: AdamW/Muon optimizer state를 CPU에 보관하며 정밀도(fp32/bf16/fp16) 설정 가능
- **Weight streaming (Weight streaming)**: layer 가중치를 CPU→GPU 비동기 prefetch + GPU staging pool 사용
- **Activation spilling (Activation spilling)**: forward 중 중간 activation을 CPU로 spill, backward 시 복원 + 재계산

PyTorch의 FSDP는 파라미터/그래디언트/optimizer state를 rank들로 sharding하여 멀티-GPU에 적용됩니다. FSDP와 offload 시스템은 일부 영역에서 **중복**(둘 다 optimizer state 위치 관리), 일부 영역에서 **상호보완**(FSDP는 activation spill 미지원)됩니다.

본 문서는 모든 조합을 분석하고, 충돌 vs 보완 관계를 식별하며, 호스트 메모리 중복 없이 함께 작동하도록 구현 방향을 제시합니다.

### 핵심 결론

1. **Activation spilling은 모든 병렬 전략과 호환** — activation spilling은 파라미터가 아니라 중간 텐서를 다루므로 DDP/FSDP/단일-GPU 어디서든 사용 가능. 다른 어떤 시스템(FSDP, DeepSpeed, Megatron)도 CPU 기반 activation spilling을 제공하지 않음.

2. **Weight streaming은 DDP/단일-GPU 전용** — weight streaming은 FSDP의 sharding/unsharding과 충돌하므로 이미 차단되어 있음.

3. **Optimizer state offload는 FSDP SHARD_GRAD_OP를 보완** — SHARD_GRAD_OP는 param+grad만 shard하고 optimizer state는 복제 보관함. Optimizer state offload가 이 빈 자리를 CPU offload + 정밀도 설정으로 채움. 단, **FSDP FULL_SHARD와는 호환 금지** (호스트 메모리 중복).

4. **Optimizer state offload + DistributedOptimizer는 곱셈적으로 결합** — DistributedOptimizer가 ZeRO-1처럼 optimizer state를 DP rank들로 분산, Optimizer state offload가 각 rank의 분산분을 bf16으로 CPU offload. rank당 host memory = `1/N × 2 × params × 2B`.

5. **Backward prefetch 갭 존재** — FSDP는 `BACKWARD_PRE`로 all-gather와 backward 연산을 overlap. 우리 offload 시스템은 forward weight prefetch만 있고 backward activation prefetch는 없음. 추가 시 대형 모델에서 5-10% 처리량 향상 기대.

---

## 2. 시스템 구성요소

### 2.1 Offload 모드

#### Optimizer state offload: Optimizer State Offload

**기능**: AdamW의 `exp_avg`, `exp_avg_sq` (옵션으로 `max_exp_avg_sq`)를 GPU 대신 CPU에 보관.

**두 가지 연산 경로**:

| 경로 | 조건 | 방식 | PCIe 트래픽 |
|------|------|-----|------------|
| CPU-compute | params on GPU (Optimizer state offload-only, Weight streaming 비활성) | AdamW 수식이 CPU에서 SIMD/AVX-512(MKL) 실행. grad는 D2H, delta는 H2D. state는 절대 GPU로 안 옮겨짐. | 파라미터당 step당 `2N × dtype_size` |
| GPU-compute | params on CPU (Weight streaming 활성) | state도 이미 CPU. `.to()`는 no-op. CPU에서 native 수식. | 0 (전부 CPU) |

**FSDP optimizer 관리 대비 추가 기능**:

- 정밀도 선택 가능 (`optimizer_state_precision`: fp32/bf16/fp16). bf16은 host memory와 PCIe 트래픽을 절반으로 줄임 (FSDP는 fp32 only).
- 파라미터 단위 offload 임계값 (`optimizer_min_param_elements`: 기본 65536). 작은 파라미터는 전송 오버헤드가 절약량보다 커서 skip.
- LoRA 파라미터 제외: `offloadable=False` 속성으로 GPU 유지 (저지연 fine-tuning).
- CPU-compute AdamW 경로는 SIMD 최적화로 GPU staging 없이 실행 가능.

**설정**:
```yaml
offload:
  enabled: true
  optimizer_offload: true
  optimizer_state_precision: "bf16"     # fp32 / bf16 / fp16
  optimizer_min_param_elements: 65536   # 이보다 작은 param은 offload skip
```

#### Weight streaming: Weight Streaming

**기능**: layer 가중치를 pinned host memory에 저장. 비동기 H2D + pooled staging buffer로 GPU에 stream. forward/backward 후 GPU에서 evict하고 갱신값을 host로 snapshot.

**호환 불가**: FSDP (자체 sharding), activation checkpointing (스케줄러 hook 없는 forward replay).

**Forward prefetch**: N layer 미리 로드 (`weight_prefetch_layers`, 기본 2). 별도 CUDA stream에서 H2D를 현재 layer 연산과 overlap.

**Backward prefetch**: Activation spilling과 결합 시 단일 layer lookahead. layer N의 backward 끝나면 layer N-1 가중치 H2D를 비동기 제출 (autograd가 layer 사이를 traversing 하는 동안).

**설정**:
```yaml
offload:
  enabled: true
  weight_offload: true
  weight_prefetch_layers: 2
  weight_storage_precision: "bf16"
  gpu_staging_pool_mb: 0.0       # 0 = 자동 사이징
  gpu_staging_chunk_mb: 256.0
```

> ⚠️ **자동 활성화 주의**: `weight_offload=true`로 두면 [config/__init__.py:211-219](../ironcore/config/__init__.py#L211-L219) 에서 `activation_spill`이 자동으로 켜집니다 (가중치 evict가 no_autograd_graph를 요구하기 때문). 문서에서 활성화한 적 없는 옵션이 켜져 있다면 이 자동화가 원인입니다.

#### Activation spilling: Activation Spilling

**기능**: activation checkpointing을 대체. forward 시 중간 activation(layer input, post-attention residual)을 비동기 D2H로 pinned host memory에 spill. backward 시 H2D로 복원하고 `torch.enable_grad()` 아래 sub-block을 재계산하여 autograd graph 재구축.

**Granularity**:
- `sub_layer` (기본): attention/MLP 경계마다 spill (layer당 2회).
- `full_layer`: layer 경계마다 spill (layer당 1회, 단 GPU 중간값을 더 보유).

**Free-after-consume**: 각 activation의 pinned host buffer는 backward가 소비한 직후 즉시 풀로 반환. peak host memory는 backward 대기 중인 activation 집합으로 bounded.

**Forward D2H**: 별도 CUDA stream에서 비동기. 다음 sub-layer 연산과 overlap.

**Backward H2D**: 현재 **동기(블로킹 wait)**. 알려진 갭 — [§6 참조](#6-prefetch-및-overlap-분석).

**설정**:
```yaml
offload:
  enabled: true
  activation_spill: true
  activation_spill_granularity: "sub_layer"   # "sub_layer" or "full_layer"
```

### 2.2 병렬 전략

#### DDP (DistributedDataParallel)

모든 rank가 모든 파라미터를 복제. backward 후 all-reduce로 그래디언트 동기화. rank당 param/grad/optimizer state 전체 사본 보유.

**rank당 VRAM**: `params + grads + optimizer_states + activations`

#### FSDP (FullyShardedDataParallel)

파라미터/그래디언트/optimizer state를 rank들로 분산. forward/backward 전 unshard(all-gather), 이후 reshard.

| 전략 | Params | Grads | Optimizer States | 약칭 |
|------|--------|-------|------------------|------|
| `FULL_SHARD` | Sharded (1/N) | Sharded (1/N) | Sharded (1/N) | ZeRO-3 |
| `SHARD_GRAD_OP` | Sharded (1/N) | Sharded (1/N) | **복제 (전체)** | ZeRO-2 |
| `NO_SHARD` | 복제 | 복제 | 복제 | DDP 동등 |
| `HYBRID_SHARD` | 노드 내 FULL_SHARD, 노드 간 복제 | | | ZeRO-3 + DP |

**FSDP CPUOffload** (`cpu_offload=CPUOffload(offload_params=True)`): 비연산 시 파라미터를 CPU로. optimizer step도 CPU에서. 정밀도 설정 불가, activation offload 불가. **제한**: `no_sync()` 밖에서 grad accumulation 동작 안 함.

**FSDP BACKWARD_PRE**: 현재 layer backward 연산 중 다음 layer의 all-gather를 미리 수행. NCCL 통신과 CUDA 연산을 overlap.

**FSDP `use_orig_params`**: 원본 파라미터 객체 보존 (`torch.compile` 필수). FSDP wrapping 후에도 optimizer 참조 유효. **Optimizer state offload + FSDP 호환에 필수**.

#### DistributedOptimizer (ZeRO-1)

기존 optimizer(AdamWOptimizer/MuonOptimizer)를 wrapping하여 round-robin으로 optimizer state를 DP rank들로 분할. 각 rank는 자신 partition(1/N)의 state만 보유/갱신. `optimizer.step()` 후 owner rank가 갱신 파라미터를 broadcast.

**직교(orthogonal)한 관계**: Optimizer state offload(배치), Activation spilling(activation), Weight streaming(weight streaming). DistributedOptimizer는 *어떤* 파라미터를 가질지 결정, Optimizer state offload는 그 state가 *어디에* 있을지(CPU vs GPU) 결정.

**호환 불가**: FSDP (자체 optimizer state sharding이 있음).

---

## 3. 기능 비교 매트릭스

### 3.1 FSDP vs Offload 기능

| 기능 | FSDP FULL_SHARD | FSDP CPUOffload | FSDP SHARD_GRAD_OP | Optimizer state offload | Weight streaming | Activation spilling |
|---|---|---|---|---|---|---|
| Parameter sharding | Yes (ZeRO-3) | No | Yes (ZeRO-2) | No | No | No |
| Gradient sharding | Yes (ZeRO-3) | No | Yes (ZeRO-2) | No | No | No |
| Optimizer state sharding | Yes | No | **No (복제)** | No | No | No |
| Optimizer state CPU offload | No | Yes | No | **Yes** | N/A | N/A |
| 정밀도 설정 | fp32 only | fp32 only | fp32 only | **fp32/bf16/fp16** | N/A | N/A |
| Per-param offload threshold | No | No | No | **Yes** | No | No |
| LoRA 제외 | No | No | No | **Yes** | No | No |
| CPU-compute AdamW (SIMD) | No | No | No | **Yes** | N/A | N/A |
| Weight streaming + GPU staging | No | No | No | No | **Yes** | No |
| Activation D2H/H2D spill | No | No | No | No | No | **Yes** |
| Backward prefetch (param) | **Yes** | **Yes** | **Yes** | No | Partial (1-layer) | No |
| Forward prefetch (param) | **Yes** | **Yes** | **Yes** | No | **Yes** (N-layer) | No |
| 멀티-GPU 필요 | Yes | Yes | Yes | No | No | No |
| Gradient accumulation | Yes | **No (CPUOffload 시 깨짐)** | Yes | Yes | Yes | Yes |

### 3.2 Overlap 분석

| 조합 | 중복 정도 | 위험 | 현재 상태 |
|------|----------|------|----------|
| Weight streaming vs FSDP param sharding | **완전** | 둘 다 파라미터 배치 관리 | **차단됨** (config + runtime) |
| Optimizer state offload vs FSDP FULL_SHARD optim | **완전** | Optimizer state offload가 CPU에 두 번째 사본 생성 | **차단 안 됨 (호스트 OOM 위험)** |
| Optimizer state offload vs FSDP CPUOffload | **부분** | 둘 다 optimizer step을 CPU에서 | **차단 안 됨 (PCIe 낭비)** |
| Optimizer state offload vs FSDP SHARD_GRAD_OP | **없음** | SHARD_GRAD_OP는 optim state 안 건드림 | **호환** |
| Activation spilling vs FSDP | **없음** | 직교 (activation vs param) | **호환** |
| Optimizer state offload vs DistributedOptimizer | **없음** | 직교 (배치 vs 분할) | **호환** |
| Activation spilling vs DistributedOptimizer | **없음** | 직교 | **호환** |

---

## 4. 구성 카탈로그

각 구성마다 메모리 레이아웃과 어디에 무엇이 있는지 설명합니다.

### 4.1 단일 GPU

```
GPU Memory                              Host Memory
+----------------------------------+    +----------------------------------+
| Model parameters (full)          |    | Optimizer states (Optimizer state offload, bf16)      |
| Gradients (full)                 |    | Spilled activations (Activation spilling)         |
| Activations (Activation spilling 미사용 시)       |    | Weight tiles (Weight streaming, bf16)          |
| GPU staging pool (Weight streaming)            |    +----------------------------------+
+----------------------------------+
```

**가용 모드**: Optimizer state offload + Weight streaming + Activation spilling (전체)

**VRAM 분해** (13B model, bf16 params, bf16 optimizer):

| 구성 요소 | Offload 없음 | Optimizer state offload+Weight streaming+Activation spilling |
|-----------|-------------|----------|
| Parameters | 26 GB | ~0.5 GB (3 layer staging pool) |
| Optimizer states | 52 GB (fp32) | 0 GB (CPU에 bf16 = 26 GB host) |
| Activations | 8-20 GB | ~0 GB (host로 spill) |
| Gradients | 26 GB | 26 GB (GPU 잔류 필수) |
| **GPU 합계** | **112-132 GB** | **~27 GB** |
| **Host 합계** | ~0 GB | ~30-50 GB |

**사용 시기**: 모델이 VRAM에 안 들어가는 단일 GPU 학습. 가장 큰 절감 구성.

### 4.2 DDP

```
각 rank의 GPU                            각 rank의 Host
+----------------------------------+    +----------------------------------+
| Model parameters (full replica)  |    | Optimizer states (Optimizer state offload, bf16)      |
| Gradients (full, sync 전)        |    | Spilled activations (Activation spilling)         |
| Activations (Activation spilling 미사용 시)       |    | Weight tiles (Weight streaming, bf16)          |
| GPU staging pool (Weight streaming)            |    +----------------------------------+
+----------------------------------+
         |                                      |
         +------ all-reduce gradients ----------+
```

**단일 GPU와 차이**: rank마다 모델 전체 복제, backward 후 grad all-reduce, 파라미터 sharding 없음.

**rank당 VRAM**: 단일 GPU와 동일.

**사용 시기**: FSDP를 쓰지 않는 멀티-GPU (TP-only, 또는 offload로 per-GPU에 fit).

### 4.3 DDP + DistributedOptimizer (ZeRO-1)

```
각 rank의 GPU                            각 rank의 Host
+----------------------------------+    +----------------------------------+
| Model parameters (full replica)  |    | Optimizer states (Optimizer state offload, bf16)      |
| Gradients (full, sync 전)        |    | local partition (1/N)            |
| Activations (Activation spilling 미사용 시)       |    | Spilled activations (Activation spilling)         |
| GPU staging pool (Weight streaming)            |    | Weight tiles (Weight streaming, bf16)          |
+----------------------------------+    +----------------------------------+
         |                                      |
         +------ all-reduce gradients ----------+
         +------ broadcast updated params ------+
                 (owner rank → others)
```

**DistributedOptimizer + Optimizer state offload 동작**:

```
DistributedOptimizer.step()
  1. non-local param의 grad를 null처리
  2. 내부 AdamWOptimizer.step() 호출
     -> 각 local param에 대해:
        -> offload_enabled? -> _adamw_offloaded_step()
           -> CPU-compute 경로: grad D2H, AdamW CPU, delta H2D
           -> state는 설정 dtype으로 CPU 잔류
  3. owner rank → 다른 rank로 param.data broadcast
```

**rank당 host memory** (13B model, 4 GPUs, bf16):

| 구성 요소 | DDP + Optimizer state offload | DDP + DistOpt + Optimizer state offload |
|-----------|----------|-------------------|
| Optimizer states (CPU) | 26 GB (full) | **6.5 GB** (1/4) |
| Weight tiles (Weight streaming) | 26 GB | 26 GB |
| Spilled activations (Activation spilling) | 4-10 GB | 4-10 GB |
| **Host 합계** | **56-62 GB** | **37-43 GB** |

**사용 시기**: optimizer state가 VRAM 주범인 멀티-GPU 학습. DDP에서 optimizer-heavy workload 최적 구성.

### 4.4 FSDP FULL_SHARD (ZeRO-3)

```
각 rank의 GPU                            각 rank의 Host
+----------------------------------+    +----------------------------------+
| Parameter shard (1/N)            |    | (FSDP 외 사용 없음)               |
| Gradient shard (1/N)             |    |                                  |
| Optimizer state shard (1/N)      |    | (FSDP는 GPU에서 optim 수행)      |
| Unsharded params (transient)     |    |                                  |
| Activations (Activation spilling 미사용 시)       |    +----------------------------------+
+----------------------------------+
         |
         +------ all-gather params (fwd/bwd) --+
         +------ reduce-scatter grads ----------+
```

**가용 offload 모드**: Activation spilling만

**Optimizer state offload 금지 이유**: FSDP FULL_SHARD가 이미 optimizer state를 sharding함. Optimizer state offload가 추가 사본을 CPU에 만들면 host memory 중복. 13B 모델 4-GPU 기준, FSDP는 rank당 13 GB optimizer shard를 GPU 보유, Optimizer state offload가 추가로 52 GB(fp32) 또는 26 GB(bf16)를 host에 둠. 호스트 OOM 위험 실재.

**Weight streaming 금지 이유**: FSDP가 자체 sharding/unsharding 관리 — 충돌.

**Activation spilling 호환 이유**: Activation spilling은 layer 사이의 중간 텐서를 다루지, 파라미터를 다루지 않음. 직교.

**rank당 VRAM** (13B, 4 GPU):

| 구성 요소 | FSDP 단독 | FSDP + Activation spilling |
|-----------|----------|-----------|
| Parameter shard | 6.5 GB | 6.5 GB |
| Optimizer shard | 13 GB (fp32) | 13 GB |
| Gradient shard | 6.5 GB | 6.5 GB |
| Unsharded transient | 26 GB | 26 GB |
| Activations | 8-20 GB | ~0 GB |
| **GPU 합계** | **60-72 GB** | **~52 GB** |

**rank당 Host memory**: spilled activation만 = 4-10 GB.

**사용 시기**: FSDP만으로 충분한 sharding이 되는 멀티-GPU. Activation spilling 추가로 activation 절감. 대부분의 멀티-GPU 권장 구성.

### 4.5 FSDP SHARD_GRAD_OP (ZeRO-2)

```
각 rank의 GPU                            각 rank의 Host
+----------------------------------+    +----------------------------------+
| Parameter shard (1/N)            |    | Optimizer states (Optimizer state offload, bf16)      |
| Gradient shard (1/N)             |    | 전체 모델 분 (sharding 안 됨)     |
| Optimizer states (Optimizer state offload 미사용 시)  |    | Spilled activations (Activation spilling)         |
| Unsharded params (transient)     |    +----------------------------------+
| Activations (Activation spilling 미사용 시)       |
+----------------------------------+
```

**가용 offload 모드**: Optimizer state offload + Activation spilling

**Optimizer state offload 가치**: SHARD_GRAD_OP는 param/grad만 shard, **optimizer state는 안 함**. rank마다 전체 optimizer state 보유 (13B = 52 GB fp32). Optimizer state offload가 이를 CPU bf16(26 GB)으로 offload — 정확히 Optimizer state offload가 메우는 갭.

**`use_orig_params=True` 필수 이유**: Optimizer state offload의 AdamWOptimizer가 원본 파라미터 객체 참조 보유. `use_orig_params=True`이면 sharded FlatParameter의 view로 보존. 미설정 시 FSDP가 FlatParameter로 교체하여 참조가 깨짐.

**rank당 VRAM** (13B, 4 GPU):

| 구성 요소 | SHARD_GRAD_OP | + Optimizer state offload + Activation spilling |
|-----------|---------------|-----------|
| Parameter shard | 6.5 GB | 6.5 GB |
| Optimizer states | 52 GB (fp32 전체!) | 0 GB (CPU bf16) |
| Gradient shard | 6.5 GB | 6.5 GB |
| Unsharded transient | 26 GB | 26 GB |
| Activations | 8-20 GB | ~0 GB |
| **GPU 합계** | **99-111 GB** | **~39 GB** |

**rank당 Host**: optimizer state(26 GB bf16) + activation(4-10 GB) = 30-36 GB.

**사용 시기**: optimizer state가 VRAM 주범인 멀티-GPU. FSDP가 param/grad sharding, Optimizer state offload가 unsharded optimizer state offload. AdamW(state = 2× model)에서 특히 유용.

### 4.6 FSDP FULL_SHARD + CPUOffload

```
각 rank의 GPU                            각 rank의 Host
+----------------------------------+    +----------------------------------+
| (연산 중 transient만)            |    | Parameters (offload)             |
| Unsharded params (transient)     |    | Gradients (offload)              |
| Activations (Activation spilling 미사용 시)       |    | Optimizer states (fp32, 전체)    |
+----------------------------------+    +----------------------------------+
```

**가용 offload 모드**: Activation spilling만

**Optimizer state offload 중복 이유**: FSDP CPUOffload가 이미 fp32 state로 CPU optimizer step 수행. Optimizer state offload 추가 시 동일 데이터 두 곳에서 관리 — 무가치.

**Activation spilling은 가치 있음**: FSDP CPUOffload가 activation 처리 안 함. Activation spilling으로 activation VRAM ~0.

**제약**: FSDP CPUOffload는 `no_sync()` 밖에서 **grad accumulation 미지원**. 미니배치 누적 필요 시 FSDP FULL_SHARD + Activation spilling으로 (CPUOffload 없이).

**사용 시기**: 파라미터 shard도 offload해야 하는 극한 메모리 제약. grad accumulation 필요 시 회피.

### 4.7 체크포인트 호환성 (모든 구성 공통)

위 §4.1-§4.6 의 모든 구성에서 **resume-from-checkpoint이 offload state와 호환됨**. [checkpointing/native.py:380-461](../ironcore/checkpointing/native.py#L380-L461) 가 다음을 자동 처리:

- 저장 시: `optimizer.offload_enabled` 와 param의 `offloadable` 속성, `optimizer_min_param_elements` 임계값을 검사하여 offloaded state는 CPU 위치를 유지한 채 직렬화
- 복원 시: 동일 기준으로 state를 CPU 또는 GPU로 다시 배치. TP-shard도 함께 split (line 414-446)
- HuggingFace interop checkpoint도 동일 기준 적용

**확인된 테스트**: [tests/unit/offload/test_checkpoint_offload.py](../tests/unit/offload/test_checkpoint_offload.py), [tests/integration/offload/test_checkpoint_offload.py](../tests/integration/offload/test_checkpoint_offload.py).

상세 동작은 §9.5 참조.

---

## 5. 메모리 분석

### 5.1 구성 요소별 메모리 공식

P 파라미터, N GPU, D dtype 바이트(bf16=2, fp32=4) 기준:

| 구성 요소 | 공식 | 비고 |
|-----------|------|-----|
| Parameters (full) | `P × D` | bf16: 2P |
| Parameters (sharded) | `P × D / N` | FSDP rank당 |
| Gradients (full) | `P × D` | params와 동일 크기 |
| Gradients (sharded) | `P × D / N` | FSDP rank당 |
| Optimizer states (fp32, AdamW) | `2 × P × 4` | exp_avg + exp_avg_sq |
| Optimizer states (bf16) | `2 × P × 2` | Optimizer state offload with bf16 |
| Optimizer states (AMSGrad fp32) | `3 × P × 4` | + max_exp_avg_sq |
| Optimizer states (AMSGrad bf16) | `3 × P × 2` | Optimizer state offload + AMSGrad with bf16 |
| Optimizer states (sharded fp32) | `2 × P × 4 / N` | FSDP FULL_SHARD rank당 |
| Optimizer states (ZeRO-1 + Optimizer state offload bf16) | `2 × P × 2 / N` | DistOpt + Optimizer state offload |
| GPU staging pool (Weight streaming) | `(prefetch_layers + 1) × layer_bytes` | 연속 layer slide window |
| Activations | `batch × seq_len × hidden × layers × k` | k = 1~5 (checkpointing) |
| Spilled activations on host | GPU activation과 유사 | free-after-consume bounded |

### 5.2 구성별 비교 (13B, 4 GPU, bf16 params)

| 구성 | rank GPU | rank Host | 비고 |
|---|---|---|---|
| No offload, DDP | 112-132 GB | ~0 GB | A100-80GB / H100 필요 |
| Optimizer state offload+Weight streaming+Activation spilling, DDP | ~27 GB | 30-50 GB | 컨슈머 GPU(RTX 4090) 가능 |
| Optimizer state offload+Weight streaming+Activation spilling+DistOpt, DDP | ~27 GB | 37-43 GB | ZeRO-1로 host optim 절감 |
| FSDP FULL_SHARD | 60-72 GB | ~0 GB | A100-80GB 필요 |
| FSDP FULL_SHARD + Activation spilling | ~52 GB | 4-10 GB | A100-80GB 여유 있게 |
| FSDP SHARD_GRAD_OP | 99-111 GB | ~0 GB | 단일 GPU 불가 |
| FSDP SHARD_GRAD_OP + Optimizer state offload + Activation spilling | ~39 GB | 30-36 GB | 컨슈머 GPU 가능 |
| FSDP FULL_SHARD + CPUOffload + Activation spilling | ~30 GB | 36-46 GB | 최대 host offload, grad accum 불가 |

### 5.3 호스트 메모리 중복 회피 규칙

**핵심 제약**: optimizer state를 host memory에 중복 보관하지 말 것.

| 구성 | rank Host optim 메모리 | 중복? |
|---|---|---|
| FSDP FULL_SHARD (GPU optim) | 0 GB | 중복 없음 |
| FSDP FULL_SHARD + CPUOffload | `2 × P × 4 / N` | FSDP 관리, 중복 없음 |
| FSDP FULL_SHARD + Optimizer state offload (잘못) | `2 × P × 4 / N + 2 × P × 2` | **중복!** FSDP shard + Optimizer state offload full |
| FSDP SHARD_GRAD_OP + Optimizer state offload (bf16) | `2 × P × 2` | 중복 없음 (SHARD_GRAD_OP가 optim sharding 안 함) |
| DDP + Optimizer state offload (bf16) | `2 × P × 2` | 중복 없음 |
| DDP + DistOpt + Optimizer state offload (bf16) | `2 × P × 2 / N` | 중복 없음 |

---

## 6. Prefetch 및 Overlap 분석

### 6.1 FSDP의 BACKWARD_PRE

FSDP는 [parallel/parallel.py:153](../ironcore/parallel/parallel.py#L153) 에서 `backward_prefetch=BackwardPrefetch.BACKWARD_PRE` 하드코드. 현재 unit의 grad 연산 **시작 전**에 다음 unit의 all-gather를 미리 수행.

```
Timeline (FSDP backward):
Layer 3 backward: |--all-gather L3--|--compute grad L3--|
Layer 2 backward:                  |--all-gather L2 (prefetch)--|--compute grad L2--|
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                      L3 grad 연산과 overlap
```

**Overlap 대상**: NCCL all-gather (GPU 간 통신) ↔ CUDA grad 연산.

**제한**: 멀티-GPU 통신/연산 overlap만 — offload된 데이터의 CPU/GPU 전송엔 도움 안 됨.

### 6.2 현재 prefetch 능력

| 컴포넌트 | Forward prefetch | Backward prefetch | 메커니즘 |
|---------|-----------------|-------------------|----------|
| Weight streaming weight streaming | **Yes** (N layer 미리, dedicated CUDA stream) | **부분** (Activation spilling 활성 시 1-layer lookahead) | `weight_prefetch_layers`, `MemoryTransferEngine` |
| Activation spilling forward D2H | **Yes** (dedicated stream, fire-and-forget) | N/A | `submit_d2h()` 핸들 반환, backward에서만 wait |
| Activation spilling backward H2D | N/A | **No (동기 블로킹)** | `on_sublayer_backward()`가 H2D 후 즉시 wait |
| Optimizer state offload optimizer offload | N/A | **No (전부 동기, `non_blocking=False`)** | `_adamw_offloaded_step_cpu_compute()` 블로킹 copy |

### 6.3 갭 분석

**Gap 1: Activation backward H2D 완전 블로킹**

[hooks.py:285-334](../ironcore/offload/hooks.py#L285-L334) `on_sublayer_backward()`:
```
1. Forward D2H 완료 대기 (보통 끝나 있음)
2. Activation H2D 제출
3. H2D 즉시 대기                 <-- 블로킹
4. 기본 stream과 sync             <-- 블로킹
5. forward 재계산, backward
```

H2D 전송이 dedicated stream에서 돌지만 즉시 wait해서 연산과 overlap 안 됨.

**영향**: 24-layer 모델, sub_layer granularity 기준
- 48회 activation H2D / backward
- 각 전송: `batch × seq_len × hidden × dtype / PCIe` ≈ 0.13 ms (예: 2×2048×1024×2 / 32 GB/s)
- 총 블로킹: ~6.2 ms / step
- step 200ms 대비 ~3% 오버헤드

48-layer 13B: 96회, 각 ~1.3ms → 총 ~125ms, step 2000ms 대비 ~6%.

**Gap 2: Backward 가중치 prefetch 1-layer 한정**

[scheduler.py:463-464](../ironcore/offload/scheduler.py#L463-L464) `on_backward_layer_end()`가 `layer_idx - 1`만 prefetch (하드코드). `weight_prefetch_layers` config는 backward에서 무시. forward는 N layer prefetch, backward는 1.

**Gap 3: 단일 transfer stream**

`MemoryTransferEngine`이 [transfer_engine.py:77](../ironcore/offload/transfer_engine.py#L77) 에서 `prefetch_streams=1` 하드코드. `OffloadConfig`에서 설정 불가. weight/activation 전송 직렬화. 두 stream이면 동시 overlap 가능.

**Gap 4: `drain_completed()` 사용 안 됨**

[transfer_engine.py:196](../ironcore/offload/transfer_engine.py#L196) 에 `drain_completed()` 정의 (블로킹 없이 완료 폴링). 정의만 있고 호출 없음(테스트 제외). "prefetch 끝났나?"를 hard-wait 없이 확인하는 데 쓸 수 있는데 dead code.

### 6.4 Activation backward prefetch 제안

**아이디어**: Layer N backward 시작 시 layer N-1 activation H2D를 즉시 비동기 제출. dedicated stream에서 N의 backward 연산과 병렬 실행. backward가 layer N-1에 도달할 때 activation이 GPU에 이미 (거의) 있음.

```
현재 (블로킹):
Layer 2 backward: |--H2D act L2 (block)--|--recompute L2--|--backward L2--|
Layer 1 backward:                                          |--H2D act L1 (block)--|--recompute L1--|--backward L1--|

제안 (prefetch):
Layer 2 backward: |--H2D act L2 (block)--|--recompute L2--|--backward L2--|
Layer 1 backward: [H2D act L1 (async, L2 backward 중 시작)]
                                                        (wait, 거의 끝남)--|--recompute L1--|--backward L1--|
```

**구현 스케치**:

`scheduler.py:on_backward_layer_end()` 에서 가중치 prefetch 후 activation도 prefetch:

```python
def on_backward_layer_end(self, layer_idx: int):
    # ... 기존 weight eviction + weight prefetch for layer_idx - 1 ...

    # NEW: layer_idx - 1 activation prefetch
    if self._spill_manager and layer_idx > 0:
        for sub_layer in reversed(range(self._spill_manager.num_sub_layers)):
            key = (self._current_microbatch, layer_idx - 1, sub_layer)
            self._spill_manager.prefetch_activation(key)  # async H2D
```

`hooks.py:on_sublayer_backward()` 에서 prefetch 여부 확인:

```python
def on_sublayer_backward(self, microbatch_idx, layer_idx, sub_layer, gpu_dst):
    key = (microbatch_idx, layer_idx, sub_layer)
    activation = self._spilled[key]

    if activation.is_prefetched:
        self._engine.wait(activation.prefetch_handle)
        self._engine.synchronize_with_default_stream()
    else:
        # fallback: 기존 동기 경로
        self._engine.wait(activation.transfer_handle)
        handle = self._engine.submit_h2d(...)
        self._engine.wait(handle)
        self._engine.synchronize_with_default_stream()
```

**기대 향상**: 대형 모델에서 step당 3-6%. 이전 layer backward 연산 뒤로 H2D latency 숨김.

**리스크**: activation 소비 시점 전에 H2D 완료되어야 정확. 연산이 전송보다 빠르면 어차피 wait — 정확성 위험은 없고 단지 혜택 없음.

### 6.5 Multi-layer backward weight prefetch 제안

**아이디어**: `on_backward_layer_end()`가 `weight_prefetch_layers` 만큼 backward로 prefetch (1 → N).

**기대 향상**: 한계적. 인접 layer 사이 backward traversal은 이미 짧음(autograd overhead). 1-layer lookahead가 대부분 커버. layer당 backward가 매우 빠를 때만(<1ms) 2-3 layer 도움.

**권고**: 우선순위 낮음. activation backward prefetch(§6.4)를 먼저.

### 6.6 DDP에 FSDP-style BACKWARD_PRE를 구현해야 하는가?

**아니오.** FSDP BACKWARD_PRE는 NCCL all-gather와 연산을 overlap. DDP는 all-gather가 없고 backward 끝난 후 all-reduce 일괄 동기화 — prefetch할 게 없음.

DDP + Weight streaming + Activation spilling 등가 최적화는 §6.4-§6.5의 weight/activation backward prefetch. 이건 PCIe DMA(H2D)와 CUDA 연산을 overlap — single-node에서 FSDP의 NCCL/연산 overlap 등가.

---

## 7. 사용자 의사결정 가이드

### 7.1 의사결정 트리

```
GPU 몇 개?
|
+-- 1 GPU
|   |
|   +-- 모델이 VRAM에 들어감?
|       +-- Yes: offload 불필요
|       +-- No: offload.enabled=true
|           |
|           +-- 안 들어가는 게 뭐?
|               +-- optimizer state: Optimizer state offload
|               +-- activation: Activation spilling
|               +-- parameter: Weight streaming
|               +-- 다 안 됨: Optimizer state offload+Weight streaming+Activation spilling
|
+-- 멀티 GPU
    |
    +-- FSDP 사용?
        |
        +-- Yes
        |   |
        |   +-- FULL_SHARD + per-GPU fit?
        |   |   +-- Yes: FSDP 단독 (activation 안 들어가면 + Activation spilling)
        |   |   +-- No (optim OOM): SHARD_GRAD_OP + Optimizer state offload + Activation spilling 고려
        |   |
        |   +-- SHARD_GRAD_OP?
        |   |   +-- optim 안 들어감: + Optimizer state offload + Activation spilling
        |   |   +-- activation만: + Activation spilling
        |   |
        |   +-- CPUOffload (극한)?
        |       +-- Yes: FULL_SHARD + CPUOffload + Activation spilling
        |       +-- 단, CPUOffload는 grad accum 불가
        |
        +-- No (DDP)
            |
            +-- rank당 optim 안 들어감?
                +-- Yes: DDP + DistOpt + Optimizer state offload + Activation spilling
                +-- No: DDP + Optimizer state offload + Activation spilling (또는 Optimizer state offload+Weight streaming+Activation spilling 최대 절감)
```

### 7.2 시나리오별 권장 구성

#### 일반 가이드

| 시나리오 | 권장 | YAML |
|---------|-----|------|
| 1 GPU, 7B, 24GB | Optimizer state offload+Weight streaming+Activation spilling | `offload: {enabled: true, optimizer_offload: true, weight_offload: true, activation_spill: true}` |
| 1 GPU, 7B, 48GB | Optimizer state offload+Activation spilling | `offload: {enabled: true, optimizer_offload: true, activation_spill: true}` |
| 1 GPU, headroom | Optimizer state offload | `offload: {enabled: true, optimizer_offload: true}` |
| 4 GPU FSDP, 13B, 80GB ea | FSDP FULL_SHARD + Activation spilling | `parallel: {use_fsdp: true}, offload: {enabled: true, activation_spill: true}` |
| 4 GPU FSDP, 13B, 48GB ea | SHARD_GRAD_OP + Optimizer state offload + Activation spilling | `parallel: {use_fsdp: true, fsdp_sharding_strategy: shard_grad_op, fsdp_use_orig_params: true}, offload: {enabled: true, optimizer_offload: true, activation_spill: true}` |
| 4 GPU DDP, 13B, 48GB ea | DDP + DistOpt + Optimizer state offload + Activation spilling | `parallel: {use_distributed_optimizer: true}, offload: {enabled: true, optimizer_offload: true, activation_spill: true}` |
| 8 GPU FSDP, 70B | FSDP FULL_SHARD + Activation spilling | (위와 동일 패턴) |

#### 2x RTX 3090 워크스테이션 (NVLink bridge, 128GB+ RAM) — Qwen family 기준

본 codebase의 주 개발 환경(2x RTX 3090, NVLink, 128GB+ host RAM) 기준 Qwen2.5 / Qwen3 family 권장:

| 모델 | 권장 구성 | 핵심 근거 |
|---|---|---|
| Qwen2.5-1.5B / Qwen3-1.7B | DDP, offload 불필요 | 단일 3090에 fit. throughput 위주. Optimizer state offload 선택 |
| Qwen2.5-3B / Qwen3-4B | DDP + Optimizer state offload (Activation spilling 선택) | 24GB 여유 충분. host bf16 ~6GB만 |
| Qwen2.5-7B / Qwen3-8B | DDP + DistOpt + Optimizer state offload + Activation spilling | GPU ~12-15GB. ZeRO-1로 host optim 절감 (~7GB/rank) |
| Qwen2.5-7B + TP=2 | TP=2 + Optimizer state offload + Activation spilling | TP shard로 per-GPU param 절반. NVLink가 TP all-reduce 가속 |
| Qwen2.5-14B / Qwen3-14B | TP=2 + Optimizer state offload + Activation spilling, **또는** FSDP SHARD_GRAD_OP + Optimizer state offload + Activation spilling | param/grad 분할 → GPU ~14-16GB. TP-aware offload 검증 후 권장 (Phase D) |
| Qwen2.5-32B / Qwen3-32B | FSDP FULL_SHARD + CPUOffload + Activation spilling (grad_accum 없이) | params도 CPU. host ~80GB. NVLink가 all-gather 가속. **grad accum 불가 제약** |
| **Qwen3-30B-A3B (MoE)** | FSDP FULL_SHARD + CPUOffload + Activation spilling + EP=2 (실험적) | 활성 3B로 compute 가벼움. 단 전체 30B params host 보관 (~60GB bf16 + 30GB bf16 optim ≈ 90GB host). **MoE × offload 검증 케이스** |
| Qwen2.5-32B + LoRA fine-tuning | DDP + LoRA (`offloadable=False` 어댑터) + Activation spilling | base weight freeze. Optimizer state offload 비활성 (어댑터 GPU 잔류) |
| Qwen3-235B-A22B (MoE) | **디자인 범위 외** (single-node 한계 초과 — host 200GB+ 필요) | 참고용 표시만 |

YAML 예시 (Qwen2.5-7B + DDP + DistOpt + Optimizer state offload + Activation spilling):
```yaml
parallel:
  use_distributed_optimizer: true
offload:
  enabled: true
  optimizer_offload: true
  optimizer_state_precision: "bf16"
  activation_spill: true
  activation_spill_granularity: "sub_layer"
  pinned_memory_pool_gb: -1.0   # auto: psutil 기반 자동 감지 (Phase B-1 후)
```

YAML 예시 (Qwen2.5-14B + FSDP SHARD_GRAD_OP + Optimizer state offload + Activation spilling):
```yaml
parallel:
  use_fsdp: true
  fsdp_sharding_strategy: "shard_grad_op"
  fsdp_use_orig_params: true   # Optimizer state offload+FSDP 필수
offload:
  enabled: true
  optimizer_offload: true
  optimizer_state_precision: "bf16"
  activation_spill: true
```

### 7.3 정밀도 가이드

| 설정 | host memory | 학습 안정성 | 사용 시기 |
|------|------------|------------|----------|
| `optimizer_state_precision: fp32` | 2× (기준) | 최고 | 기본. 항상 안전. |
| `optimizer_state_precision: bf16` | 1× (절반) | 대부분 모델 양호 | host memory 부족 시. loss divergence 모니터링. |
| `optimizer_state_precision: fp16` | 1× (절반) | 위험 (overflow) | AdamW 비권장 (`exp_avg_sq` overflow 가능). |

> **RTX 3090 (Ampere) 특이사항**: bf16은 native FP32 ALU에서 emulation으로 동작하므로 H100/A100보다 약간 느릴 수 있으나 안전성 측면에서 fp16보다 우선 권장. AdamW는 `exp_avg_sq`가 fp16 동적 범위를 초과할 수 있어 fp16 비권장.

### 7.4 Activation spilling granularity 가이드

| Granularity | layer당 host 메모리 | GPU 절감 | 사용 시기 |
|-------------|---------------------|---------|----------|
| `sub_layer` | 2 activations | 최대 | 기본. 대부분 적합. |
| `full_layer` | 1 activation | 적음 (attn/MLP 중간값 잔류) | host memory 매우 부족 + GPU 일부 잔류 허용 |

### 7.5 Telemetry 및 모니터링

Offload 시스템에는 학습 중 H2D/D2H 전송, 대역폭, 정지 이벤트를 모니터링하는 내장 telemetry가 포함되어 있습니다.

#### Telemetry 활성화

학습 전 환경 변수 설정:
```bash
export IRONCORE_OFFLOAD_TELEMETRY=1
ironcore train --config configs/offload.yaml
```

#### 실시간 모니터링 (터미널 시각화)

학습 스크립트에 실시간 메트릭을 위한 visualizer 추가:

```python
from ironcore.utils.offload_visualizer import start_offload_visualizer

# visualizer 시작 (10단계마다 갱신)
viz = start_offload_visualizer(update_interval=10)

try:
    trainer.train()
finally:
    viz.stop()  # 최종 요약 출력
```

**출력 예시**:
```
============================================================
[Offload Telemetry] Step 100
────────────────────────────────────────────────────────────
H2D: 45.23 GB | 28.50 GB/s | 450 transfers
D2H: 12.87 GB | 24.30 GB/s | 225 transfers
Stalls: 3 events | 125.3 ms total
Queue: 2/8 depth
============================================================
```

#### 추적 메트릭

| 메트릭 | 설명 | 정상 범위 |
|--------|------|----------|
| `total_h2d_bytes` | 누적 host→device 데이터 | 추세 모니터링 |
| `total_d2h_bytes` | 누적 device→host 데이터 | 추세 모니터링 |
| `h2d_bandwidth_gb_s` | 유효 H2D 대역폭 | >20 GB/s (PCIe 4.0) |
| `d2h_bandwidth_gb_s` | 유효 D2H 대역폭 | >20 GB/s (PCIe 4.0) |
| `stall_events` | 전송 큐 포화 이벤트 | =0 또는 매우 낮음 |
| `max_queue_depth` | 최대 관찰 큐 깊이 | `prefetch_streams` 이하 |

#### 하드웨어 벤치마크

학습 전 시스템의 PCIe/NVLink 대역폭을 벤치마크:

```bash
# 기본 벤치마크 (10-500 MB 전송)
python scripts/benchmark_offload_pcie.py --sizes 10 50 100 500

# NVLink 켬기/끼기 비교
NCCL_P2P_DISABLE=1 python scripts/benchmark_offload_pcie.py --output no_nvlink.json
NCCL_P2P_DISABLE=0 python scripts/benchmark_offload_pcie.py --output with_nvlink.json

# 사용자 정의 크기 및 출력
python scripts/benchmark_offload_pcie.py --sizes 100 500 1000 --output bandwidth_results.json
```

#### Telemetry 통한 문제 해결

| 증상 | 가능한 원인 | 조치 |
|------|------------|------|
| 낮은 대역폭 (<10 GB/s) | 비-pinned 메모리, 느린 저장소, PCIe 3.0 | `pinned_memory_pool_gb` 설정 확인 |
| 높은 정지 이벤트 | 전송 큐 백업 | `prefetch_streams` 증가 또는 `weight_prefetch_layers` 감소 |
| D2H가 H2D보다 훨씬 느림 | GPU 연산이 전송 차단 | 불필요한 `torch.cuda.synchronize()` 호출 확인 |
| 대역폭이 시간이 지남에 감소 | 열 스로틀링 또는 메모리 단편화 | GPU 온도 모니터링, `nvidia-smi` 확인 |

---

## 8. 구현 상태 (완료된 기능)

> **참고**: 모든 Phase A-H가 완료되었습니다. 이 섹션은 구현된 내용을 문서화합니다.

### 8.1 완료된 기능 목록

| Phase | 기능 | 상태 | 위치 |
|-------|---------|--------|----------|
| A | Optimizer state offload (optimizer offload) | ✅ 완료 | `ironcore/offload/optimizer.py` |
| A | Weight streaming (weight streaming) | ✅ 완료 | `ironcore/offload/scheduler.py` |
| A | Activation spilling (activation spill) | ✅ 완료 | `ironcore/offload/hooks.py` |
| B | Pinned 메모리 풀 자동 감지 | ✅ 완료 | `ironcore/utils/system_info.py` |
| C | 단일 GPU 통합 테스트 | ✅ 완료 | `tests/integration/offload/` |
| D | TP × Offload 지원 | ✅ 완료 | `ironcore/language_model.py` |
| E | 멀티 GPU 설정 가드 | ✅ 완료 | `ironcore/config/__init__.py` |
| E | DDP + Optimizer state offload/Activation spilling 테스트 | ✅ 완료 | `tests/integration/offload/test_ddp_offload.py` |
| E | FSDP SHARD_GRAD_OP + Optimizer state offload | ✅ 완료 | `tests/integration/offload/test_fsdp_shard_grad_op_offload.py` |
| E | FSDP FULL_SHARD + Activation spilling | ✅ 완료 | `tests/integration/offload/test_fsdp_full_shard_activation_spill.py` |
| E | Backward weight prefetch | ✅ 완료 | `ironcore/offload/scheduler.py` |
| F | Telemetry 시스템 | ✅ 완료 | `ironcore/utils/offload_metrics.py` |
| F | Transfer engine 타이밍 | ✅ 완료 | `ironcore/offload/transfer_engine.py` |
| F | 하드웨어 벤치마크 스크립트 | ✅ 완료 | `scripts/benchmark_offload_pcie.py` |
| F | 라이브 visualizer | ✅ 완료 | `ironcore/utils/offload_visualizer.py` |
| G | MoE × Offload 지원 | ✅ 완료 | `tests/integration/offload/test_moe_offload_smoke.py` |
| H | OffloadConfig 재배치 | ✅ 완료 | `ironcore/config/config_offload.py` |

### 8.2 Config 검증 (구현 완료)

**파일**: [ironcore/config/__init__.py](../ironcore/config/__init__.py)

**변경 1**: Optimizer state offload + FSDP FULL_SHARD 차단 (host OOM 위험)

```python
# 기존 weight_offload + FSDP 블록 다음 (line ~210)
if config.offload.optimizer_offload and config.parallel.use_fsdp:
    if config.parallel.fsdp_sharding_strategy == "full":
        raise ValueError(
            "offload.optimizer_offload + FSDP full_shard는 host에 optimizer state를 "
            "중복합니다. fsdp_sharding_strategy: shard_grad_op 로 바꾸거나, "
            "optimizer_offload를 끄고 FSDP가 처리하게 하세요."
        )
```

**변경 2**: Optimizer state offload + FSDP CPUOffload 차단 (중복)

```python
if config.offload.optimizer_offload and config.parallel.use_fsdp:
    if config.parallel.fsdp_offload_params:
        raise ValueError(
            "optimizer_offload는 FSDP CPUOffload와 중복입니다 (둘 다 CPU step). "
            "둘 중 하나만 켜세요."
        )
```

**변경 3**: Optimizer state offload + FSDP without `use_orig_params` 경고

```python
if config.offload.optimizer_offload and config.parallel.use_fsdp:
    if not config.parallel.fsdp_use_orig_params:
        import warnings
        warnings.warn(
            "optimizer_offload + FSDP는 use_orig_params=True 가 필요합니다. "
            "그렇지 않으면 FSDP가 FlatParameter로 교체하여 optimizer 참조가 깨집니다.",
            stacklevel=2,
        )
```

### 8.2 Optimizer 생성 순서 보호

**파일**: [ironcore/trainers/base_trainer.py](../ironcore/trainers/base_trainer.py)

**문제**: optimizer는 line ~314, FSDP wrapping은 line ~358. `use_orig_params=True`이면 동작, 아니면 깨짐.

**수정**: FSDP + Optimizer state offload 활성 시 `use_orig_params=True` 강제.

```python
# _build_model_and_optimizer() 의 get_optimizer() 호출 전:
if self.config.parallel.use_fsdp and not self.config.parallel.fsdp_use_orig_params:
    if self.config.offload.optimizer_offload:
        raise ValueError(
            "FSDP + optimizer_offload는 fsdp_use_orig_params=True 필요. "
            "config의 parallel.fsdp_use_orig_params=true 로 설정하세요."
        )
```

### 8.3 Activation backward prefetch 구현

**우선순위**: 중. Activation spilling 사용 대형 모델 3-6% 처리량 향상.

**수정 파일**:

1. [ironcore/offload/hooks.py](../ironcore/offload/hooks.py) — `ActivationSpillManager`에 `prefetch_activation()` 추가. 비동기 H2D 제출 후 핸들 보관, `SpilledActivation`에 `is_prefetched` flag.

2. [ironcore/offload/scheduler.py](../ironcore/offload/scheduler.py) — `on_backward_layer_end()`에서 이전 layer activation도 prefetch.

3. `on_sublayer_backward()` — `is_prefetched`이면 prefetch 핸들만 wait.

**테스트**: 50 step에서 prefetch on/off의 loss trajectory 비교.

### 8.4 Transfer stream 설정 가능화

**우선순위**: 낮음. weight/activation 동시 overlap 필요할 때만 가치.

**파일**: [ironcore/offload/config.py](../ironcore/offload/config.py)

```python
prefetch_streams: int = 1  # 비동기 전송용 dedicated CUDA stream 수
```

**파일**: [ironcore/offload/transfer_engine.py](../ironcore/offload/transfer_engine.py)

`from_config()` 가 하드코딩 대신 config 값 읽도록.

### 8.5 테스트 요구사항

| 테스트 | 검증 내용 |
|--------|----------|
| `test_optimizer_offload_fsdp_full_shard_blocked` | Optimizer state offload + FULL_SHARD 차단 |
| `test_optimizer_offload_fsdp_cpuoffload_blocked` | Optimizer state offload + CPUOffload 차단 |
| `test_activation_spill_fsdp_integration` | Activation spilling + FSDP FULL_SHARD, 50 step loss parity |
| `test_optimizer_offload_shard_grad_op_fsdp` | Optimizer state offload + SHARD_GRAD_OP + Activation spilling, optimizer state CPU + param shard, loss parity |
| `test_optimizer_offload_distributed_optimizer` | Optimizer state offload + DistOpt, rank당 1/N state on CPU |
| `test_backward_prefetch_correctness` | prefetch on/off에서 동일 그래디언트 |
| `test_backward_prefetch_throughput` | step time 향상 측정 |

### 8.6 우선순위 정리

1. **Config 검증** (§8.1) — 잘못된 설정으로 인한 host OOM 방지. 저위험 고임팩트.
2. **Optimizer 순서 가드** (§8.2) — Optimizer state offload+FSDP 시 `use_orig_params=True` 강제. 저위험.
3. **Activation backward prefetch** (§8.3) — Activation spilling 사용자 3-6% 향상. 중간 노력.
4. **Transfer stream 설정** (§8.4) — 작은 변경. 낮은 우선순위.
5. **통합 테스트** (§8.5) — 모든 구성 end-to-end 검증.

---

## 9. 추가 고려사항 (영문판 대비 보강)

영문 원문이 다루지 않은, 그러나 운영/구현 시 마주치는 항목들:

### 9.1 TP (Tensor Parallel) × Offload 상호작용

현재 `ironcore/offload/` 코드는 TP를 인지하지 않음 (`tp_size`, `expert_parallel` 키워드 grep 결과 없음).

- **Optimizer state offload + TP**: optimizer state는 본래 TP-shard된 weight 형태로 만들어짐. Optimizer state offload는 per-param offload이므로 정확성은 유지되나, `optimizer_min_param_elements` 임계값이 sharded shape로 적용됨. shard로 잘려 임계값 미만이 되어 의도치 않게 GPU에 잔류할 수 있음.
- **Weight streaming + TP**: TP-shard된 가중치를 streaming 시 PCIe 전송량은 작아지지만, all-gather/all-reduce가 weight rematerialization 시점과 충돌하지 않는지 검증 필요.
- **Activation spilling + TP**: activation은 sequence/hidden 차원으로 TP-split되므로 spill 크기도 1/TP. 호환 가능성 높음.

**권고**: TP 결합 통합 테스트 추가.

### 9.2 MoE / Expert Parallelism + Offload

문서에 미언급. EP는 expert를 rank 간 분산. offload의 layer-by-layer 스케줄러가 expert 수준까지 인지하지 못함. MoE 모델 학습 시 동작 여부 명확화 필요.

### 9.3 Pipeline Parallelism

**디자인 범위 외**. 본 codebase는 단일-노드 워크스테이션 최적화를 타겟으로 하며 PP는 현재 미구현이고 향후에도 도입 계획 없음 (단일 노드/2-GPU 환경에서 시험 가치 낮음). 향후 멀티-노드 트랙이 도입될 경우 별도 통합 단계에서 다룬다.

### 9.4 Pinned memory 기본값 100 GB의 위험

`pinned_memory_pool_gb: 100.0` 기본은 컨슈머 시스템(64-128 GB RAM)에서 host OOM 또는 OOM-killer 위험. RAM 자동 감지(예: 가용 RAM의 50% 이내) + 보수적 기본값(예: 32 GB) 권고.

### 9.5 체크포인트 호환성

[checkpointing/native.py:380-461](../ironcore/checkpointing/native.py#L380-L461) 에서 offload state 저장/복원이 자동 처리됨 (영문 원문에 미언급).

**저장 (line 384-394)**:
- `optimizer.offload_enabled`, param의 `offloadable` 속성, `offload_min_param_elements` 임계값 동일 기준으로 검사
- offloaded state(`exp_avg`, `exp_avg_sq`, AMSGrad의 `max_exp_avg_sq`)는 CPU 위치 유지한 채 직렬화 — GPU staging 없음

**복원 (line 415-461)**:
- TP-shard split 처리 (line 414-446) — TP rank 간 state 일관성 유지
- 동일 offload 기준에 따라 CPU 또는 GPU로 재배치 (line 449-460)

**검증된 테스트**:
- [tests/unit/offload/test_checkpoint_offload.py](../tests/unit/offload/test_checkpoint_offload.py)
- [tests/integration/offload/test_checkpoint_offload.py](../tests/integration/offload/test_checkpoint_offload.py)

**HuggingFace interop**: 동일 로직 적용. fp32 state로 export/import 시 자동 dtype 변환.

**주의**: offload config 가 저장/복원 시 동일해야 함 (`optimizer_state_precision`, `optimizer_min_param_elements`). 설정이 다르면 일부 state가 GPU/CPU 잘못된 위치에 복원될 수 있음.

### 9.6 수렴성 회귀

사용자 메모리에 따르면 Weight streaming+Activation spilling+grad_accum>1 device mismatch 버그(커밋 `4a1597f`)와 1000-step loss 발산 이슈가 있었음. 문서에 다음 명시 필요:
- 회귀 테스트 위치
- 검증 step 수 (예: 1000-step 기준)
- 알려진 잔존 caveat

### 9.7 CPU compute 스레드 경합

**언제 critical**: Optimizer state offload-only(CPU compute) 모드 (params on GPU, Weight streaming 비활성). AdamW 수식이 CPU에서 SIMD/AVX-512(MKL)로 실행되며 dataloader worker, gradient all-reduce 백그라운드 스레드, NCCL helper와 OMP/MKL 스레드 풀 경합.

**권고 공식**:
```bash
export OMP_NUM_THREADS=$(python -c "import os; print(max(1, os.cpu_count() - dataloader_workers - 2))")
export MKL_NUM_THREADS=$OMP_NUM_THREADS
```

여기서 `dataloader_workers` 는 `TrainerConfig.dataloader_num_workers` 값.

**예시 (16-core CPU + dataloader_workers=4)**:
```bash
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
```

**검증 방법**: `top -H` 또는 `htop` 으로 학습 중 CPU 스레드 분포 확인. AdamW step 시간이 GPU 연산 시간을 초과하면 스레드 부족 가능성.

**Weight streaming/Activation spilling 활성 시**: AdamW가 GPU-compute 경로(params on CPU 시)이므로 OMP 영향 줄어듦. 단, dataloader는 항상 CPU 사용하므로 기본 권고는 동일.

### 9.8 NUMA-locality

멀티 소켓 시스템에서 pinned memory가 GPU의 NUMA 노드와 일치하지 않으면 cross-socket UPI 트래픽으로 PCIe latency가 증가. **단일 소켓 워크스테이션(예: 2x RTX 3090 일반 환경)에선 무시 가능**.

**확인 방법** (멀티 소켓인지):
```bash
lscpu | grep "NUMA node(s)"   # > 1 이면 멀티 소켓
nvidia-smi topo -m            # GPU↔NUMA 노드 매핑
```

**권고 (멀티 소켓 시스템 한정)**:
```bash
# 각 GPU rank를 가까운 NUMA 노드에 바인드
numactl --cpunodebind=0 --membind=0 \
  torchrun --nproc_per_node=2 -m ironcore train --config configs/...

# 또는 rank별로 바인딩 (script 래퍼 필요)
```

**프로그래밍 방식**: PyTorch 자체에는 NUMA-aware pinned alloc API 없음. `torch.cuda.set_device()` 직후 `os.sched_setaffinity()` 로 CPU affinity를 GPU의 NUMA 노드에 맞추는 것이 차선.

**Phase B-1 (`system_info.py`) 와의 연결**: `psutil.virtual_memory()` 는 NUMA-aware 정보 제공 안 함. 향후 NUMA 감지가 필요하면 `numa` 패키지 또는 `/sys/devices/system/node/` 파싱 추가 검토.

### 9.9 PCIe 대역폭 경합

NVLink 없는 시스템(예: PCIe-only 4×GPU)에서 Optimizer state offload grad D2H/delta H2D가 DDP all-reduce / FSDP all-gather와 PCIe 경합. 정량 분석:
- PCIe Gen4 x16: ~32 GB/s 단방향
- 13B 모델 grad: 26 GB → 0.8s/step의 전송 대역폭 필요

bidirectional + 통신 동시 발생 시 실효 대역폭 절반 가정.

### 9.10 AMSGrad 메모리 보정

§5.1 표는 표준 AdamW(`exp_avg + exp_avg_sq`)만 다룸. AMSGrad는 `max_exp_avg_sq` 를 영구 보관:

| 옵티마이저 | 공식 | 13B 호스트 (Optimizer state offload bf16) | 13B 호스트 (fp32) |
|---|---|---|---|
| AdamW | `2 × P × D` | 26 GB | 52 GB |
| AdamW + AMSGrad | `3 × P × D` | **39 GB** | **78 GB** |

**적용 위치**: [optimizer/](../ironcore/optimizer/) 의 AMSGrad 활성화 시 (config `optimizer.amsgrad: true`). Optimizer state offload offload state는 `max_exp_avg_sq` 도 동일 정밀도로 CPU 저장.

**호스트 메모리 영향**: 1.5×. §5.2 표의 모든 Host per rank 값에 1.5 곱해 재산정 필요. Phase B-1 의 자동 추천 (§9.4) 에서도 AMSGrad 활성화 여부 반영해야 함.

**영문 원본 동기화**: [docs/offload_fsdp_architecture.md](offload_fsdp_architecture.md#51-per-component-memory-formulas) §5.1 표에 AMSGrad 행 추가됨 (이번 변경에서).

### 9.11 CUDA Graph 비호환

**현재 상태**: 코드베이스가 CUDA Graphs를 사용하지 않음 (`torch.cuda.graph`, `make_graphed_callables`, `CUDAGraph` grep 결과 0건). 따라서 실제 충돌은 없음.

**근본 비호환성**: Optimizer state offload/Weight streaming/Activation spilling 의 비동기 prefetch는 매 step마다 다른 텐서 포인터/shape를 만들어내는 동적 동작. CUDA Graphs는 캡처된 커널 시퀀스를 frozen kernel + frozen pointer로 재실행하므로 이와 비호환.

**향후 도입 시 권고**:
- offload 활성 학습에선 `torch.cuda.graph` / `make_graphed_callables` 사용 금지
- Inference path (offload 비활성) 에서만 CUDA Graphs 도입 가능
- 이 비호환성을 config validation에서 명시적으로 차단할 필요는 없음 (코드베이스가 graphs를 도입하기 전까지)

### 9.12 호스트 OOM 복구

pinned pool 고갈 시 동작:
- 현재: 예외 발생 가능
- 권장: 동기 경로 fallback 또는 명확한 에러 메시지 (`pinned_memory_pool_gb` 증액 안내)

### 9.13 운영 telemetry / 메트릭

production 모니터링 권고 지표:
- 전송 큐 depth (제출 vs 완료 차이)
- stall 횟수 (`wait()` 블로킹 시간 누적)
- host memory headroom (`MemAvailable` vs pool 사용량)
- PCIe 사용률 (nvidia-smi `dmon -s p`)

### 9.14 `weight_offload` 자동 활성화 부작용

[config/__init__.py:211-219](../ironcore/config/__init__.py#L211-L219) 에서 `weight_offload=true` 설정 시 `activation_spill`이 **자동 활성화** + `warnings.warn()` 발생.

**근본 원인**: 가중치 eviction은 backward 시 autograd graph가 layer를 단위로 분리되어 있어야 안전 (그렇지 않으면 evicted weight를 backward가 참조하다 segfault). `activation_spill` 의 sub-layer 경계가 이 분리를 제공.

**관찰 가능한 증상**:
- `weight_offload: true` 만 설정했는데 host 메모리에 spilled activation 영역이 생김
- 학습 로그에 `UserWarning: offload.weight_offload requires activation spilling for weight eviction (no_autograd_graph). Enabling offload.activation_spill automatically.` 출력

**대응**:
- 의도된 동작 — 그냥 두면 됨
- 명시적으로 끄고 싶다면 `weight_offload: false` 또는 `activation_spill` 을 함께 명시적 true 로 (silent enable 회피)

**변경 의도 없음**: 이 자동 활성화는 설계상 안전성 가드. 제거하지 말 것. 단 부록 B 표 비고에서 명시 (이 변경에서 적용).

---

## 10. 검증 결과 및 갭 요약

영문 문서의 주장을 코드베이스에 대조 검증한 결과:

| 항목 | 영문 문서 주장 | 실제 코드 | 위치 |
|------|---------------|-----------|------|
| Optimizer state offload + FSDP FULL_SHARD 차단 | "차단 안 됨" | ✅ 일치 — 차단 없음 | [config/__init__.py:206-210](../ironcore/config/__init__.py#L206-L210) |
| Optimizer state offload + FSDP CPUOffload 차단 | "차단 안 됨" | ✅ 일치 — 차단 없음 | 동일 |
| Backward activation H2D | "동기 블로킹 (Gap 1)" | ✅ 일치 — `wait()` + `synchronize_with_default_stream()` | [hooks.py:333-334](../ironcore/offload/hooks.py#L333-L334) |
| Backward weight prefetch | "1-layer 하드코드 (Gap 2)" | ✅ 일치 — `layer_idx - 1`만 | [scheduler.py:463-464](../ironcore/offload/scheduler.py#L463-L464) |
| `prefetch_streams=1` (Gap 3) | "설정 불가" | ✅ 일치 — `from_config`에 하드코드 | [transfer_engine.py:77](../ironcore/offload/transfer_engine.py#L77) |
| `drain_completed()` (Gap 4) | "dead code" | ✅ 일치 — 테스트 외 미호출 | [transfer_engine.py:196](../ironcore/offload/transfer_engine.py#L196) |
| FSDP `BACKWARD_PRE` | "하드코드" | ✅ 일치 | [parallel/parallel.py:153](../ironcore/parallel/parallel.py#L153) |
| Optimizer 생성 순서 | "FSDP wrap 전에 optimizer 생성" | ✅ 일치 — line 314 vs 358 | [trainers/base_trainer.py](../ironcore/trainers/base_trainer.py) |

**테스트 커버리지 갭**: Optimizer state offload+FSDP, Activation spilling+FSDP, Optimizer state offload+DistOpt 통합 테스트 **없음** ([tests/](../tests/) 검색 결과).

**최근 관련 커밋**:
- `932779a` — Optimizer state offload-only 모드 CPU AdamW (40%+ VRAM 절감)
- `4a1597f` — Activation spilling grad_accum>1 device mismatch 수정
- `78adb10` — backward prefetch overlap 시도 (단, activation H2D는 여전히 블로킹)

---

## 부록 A: 현재 enforcement 상태

| 조합 | Config 검증 | Runtime 검사 | 상태 |
|------|------------|-------------|------|
| DistributedOptimizer + FSDP | **차단 (error)** | N/A | 정상 |
| weight_offload + FSDP | **차단 (error)** | **차단 (scheduler)** | 정상 |
| optimizer_offload + FSDP FULL_SHARD | **차단 안 됨** | N/A | **갭 (host OOM)** |
| optimizer_offload + FSDP CPUOffload | **차단 안 됨** | N/A | **갭 (중복)** |
| activation_spill + FSDP | 미차단 (호환) | 미차단 (호환) | 정상 |
| optimizer_offload + DistributedOptimizer | 미차단 (호환) | 미차단 (호환) | 정상 |

## 부록 B: OffloadConfig 필드

| 필드 | 타입 | 기본값 | 설명 |
|-----|------|--------|------|
| `enabled` | bool | False | 모든 offload 마스터 스위치 |
| `optimizer_offload` | bool | False | Optimizer state offload |
| `optimizer_state_precision` | str | "fp32" | fp32/bf16/fp16 |
| `optimizer_min_param_elements` | int | 65536 | 이보다 작은 param skip |
| `weight_offload` | bool | False | Weight streaming. **⚠ true 시 `activation_spill` 자동 활성화** ([config/__init__.py:211-219](../ironcore/config/__init__.py#L211-L219), §9.14) |
| `weight_prefetch_layers` | int | 2 | forward prefetch layer 수 |
| `weight_storage_precision` | str | "bf16" | host weight tile 정밀도 |
| `gpu_staging_pool_mb` | float | 0.0 | 0=자동 |
| `gpu_staging_chunk_mb` | float | 256.0 | staging chunk |
| `activation_spill` | bool | False | Activation spilling |
| `activation_spill_granularity` | str | "sub_layer" | sub_layer / full_layer |
| `pinned_memory_pool_gb` | float | 100.0 | ⚠ 컨슈머 시스템엔 과도. §9.4 참조 (Phase B-1 후 `auto` 권장) |
| `pinned_chunk_gb` | float | 4.0 | pinned chunk |

## 부록 C: ParallelConfig FSDP 필드

| 필드 | 타입 | 기본값 | 설명 |
|-----|------|--------|------|
| `use_fsdp` | bool | False | FSDP wrapping |
| `fsdp_sharding_strategy` | str | "full" | full/hybrid/no_shard/shard_grad_op |
| `fsdp_offload_params` | bool | False | CPUOffload(offload_params=True) |
| `fsdp_use_orig_params` | bool | False | 원본 param 보존 (Optimizer state offload+FSDP 필수) |
| `fsdp_mixed_precision` | str | "native" | native/mixed |
| `use_distributed_optimizer` | bool | False | ZeRO-1 (DDP 전용) |
