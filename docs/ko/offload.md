# 오프로드 (RAM-First 계단식 스케일링)

> 이 가이드는 사용자 대면 내용 (오프로드 설정 및 실행 방법)을 다룹니다. 아키텍처, 메모리 분석, FSDP와의 관계는 [오프로드 시스템 설계](design/offload.md)를 참고하세요.

## 개요

오프로드 서브시스템은 VRAM에 완전히 맞지 않는 모델을 훈련하기 위해 GPU VRAM과 호스트 RAM 사이에서 텐서를 이동합니다. YAML 설정의 `offload:` 섹션으로 구성합니다.

`offload.enabled: true`로 각각 게이팅되는 세 가지 독립적인 기능:

| 기능 | 이동 대상 | 방향 |
|---|---|---|
| 옵티마이저 상태 오프로드 | AdamW/Muon 모멘텀, 분산 상태 | GPU → CPU (스텝 후), CPU → GPU (스텝 전) |
| 가중치 스트리밍 | 레이어 가중치 (어텐션, MLP 프로젝션) | CPU → GPU (계산 전 프리페치), GPU → CPU (스텝 후 스냅샷) |
| 활성화 스필 | 서브레이어 경계에서의 중간 활성화 | GPU → CPU (순전파), CPU → GPU (역전파) |

각각 독립적으로 활성화할 수 있으며, 세 가지를 동시에 실행할 수도 있습니다.

## 빠른 시작

```yaml
offload:
  enabled: true

  # 옵티마이저 상태 오프로드
  optimizer_offload: true

  # 가중치 스트리밍
  weight_offload: true
  weight_prefetch_layers: 2
  weight_storage_precision: fp32

  # 활성화 스필
  activation_spill: false
```

## 옵티마이저 상태 오프로드

옵티마이저 스텝 후 모멘텀과 분산 상태가 CPU RAM으로 이동합니다. 다음 스텝 전에 GPU로 다시 이동합니다. `optimizer_min_param_elements` (기본값 65536) 미만의 파라미터는 GPU에 유지됩니다 (전송 오버헤드가 효과보다 큼).

옵티마이저 상태 오프로드는 `optimizer_state_precision` (기본값: fp32)을 통해 fp32, fp16, bf16 정밀도를 지원합니다. 낮은 정밀도는 RAM 사용량과 PCIe 대역폭을 줄이지만 AdamW의 훈련 안정성에 영향을 줄 수 있습니다.

## 가중치 스트리밍

### 동작 방식

레이어 가중치가 고정 호스트 메모리에 상주합니다. 순전파 중 계산보다 `weight_prefetch_layers` 레이어 앞서 가중치가 비동기적으로 GPU로 프리페치됩니다. 레이어가 실행된 후 스테이징 버퍼가 재사용을 위해 풀에 반환됩니다. 역전파가 완료된 후 업데이트된 가중치가 호스트 메모리로 스냅샷됩니다.

훈련 스텝당 라이프사이클:

1. **`on_training_step_start()`**: 처음 N개 레이어 프리페치
2. **`on_layer_start(i)`**: 레이어 i의 전송 대기, 가중치를 파라미터에 적용 (CPU param.data를 GPU 스테이징 버퍼로 교환), 다음 레이어 프리페치
3. **`on_layer_end(i)`**: 활성화 스필 또는 가중치 스트리밍 재계산 시: GPU에서 가중치 방출 — param.data를 호스트 타일 값으로 교체, GPU 스테이징 버퍼 풀에 반환. 구간 내부에서도 재계산을 통해 레이어별 가중치를 복원하고 방출
4. **`on_backward_layer_start(i)`**: 역전파 재계산에 필요하면 방출된 가중치 다시 로딩
5. **`on_backward_layer_end(i)`**: 역전파 계산 후 가중치 다시 방출
6. **`on_training_step_end()`**: 업데이트된 파라미터(옵티마이저 스텝 후)를 호스트 타일로 스냅샷

### GPU 스테이징 버퍼 풀

가중치 스트리밍은 **풀링 할당** 전략을 사용합니다. 초기화 시 모든 레이어에 영구적인 GPU 스테이징 버퍼를 할당하는 대신, 고정 크기 CUDA 메모리 풀이 레이어 간에 공유됩니다. 버퍼는 H2D 전송 전에 빌려지고, 가중치가 `param.data`로 복사된 즉시 반환됩니다.

따라서 총 레이어 수에 관계없이 어느 시점에서나 `(weight_prefetch_layers + 1)`개 레이어의 GPU 스테이징 메모리만 필요합니다. `weight_prefetch_layers: 2`인 36레이어 모델에서는 3개 레이어 분량의 스테이징 버퍼만 존재합니다.

### 풀 크기 조정

풀 용량은 기본적으로 자동 크기 조정됩니다. 모든 레이어가 가중치를 등록한 후 스케줄러가 계산합니다:

```
budget = max(연속 layer_bytes[i:i+prefetch_layers+1]의 합)
```

이 슬라이딩 윈도우 방식은 `weight_prefetch_layers + 1` 연속 레이어 중 가장 큰 합을 찾아 다양한 레이어 크기(예: 더 큰 전문가 파라미터를 가진 MoE 레이어)를 처리합니다.

**수동 오버라이드** (설정으로):

```yaml
offload:
  gpu_staging_pool_mb: 512.0   # 풀 VRAM 하드 한계 (0 = 자동 크기)
  gpu_staging_chunk_mb: 256.0  # 풀 할당을 위한 CUDA 청크 크기
```

**권장 사항**: 오류가 발생하지 않으면 `gpu_staging_pool_mb: 0` (자동 크기)으로 두세요. 오류가 발생하면 자동 계산값의 `1.1x`로 설정해 여유 공간을 확보하세요.

### 저장 정밀도

호스트의 가중치를 낮은 정밀도로 저장해 RAM 사용량과 PCIe 대역폭을 줄일 수 있습니다:

```yaml
offload:
  weight_storage_precision: bf16  # fp32, fp16, 또는 bf16
```

GPU로 복사할 때 원래 dtype으로 역양자화됩니다. 트레이드오프는 메모리 절약 대 잠재적 정밀도 손실입니다.

### 비호환성

가중치 스트리밍은 다음과 **호환되지 않습니다**:
- **FSDP** (`parallel.use_fsdp: true`): FSDP가 자체 파라미터 샤딩/언샤딩을 관리합니다.

레이어 재계산은 구간 입력을 CPU 또는 GPU에 저장하는 가중치 스트리밍 경로와 함께 사용할 수 있습니다. `weight_offload`는 안전한 재계산을 활성화하지만 `activation_spill`은 자동으로 켜지 않습니다.

### CPU 상주 파라미터 및 옵티마이저

`weight_offload: true`일 때 모델은 CPU에 유지됩니다. 레이어 가중치는 순전파/역전파 중 `param.data` 교체를 통해 일시적으로 GPU 스테이징 버퍼로 교환됩니다. 각 레이어 실행 후 `param.data`는 호스트 타일 값을 지원하는 CPU 텐서로 복원됩니다.

파라미터와 옵티마이저 상태(exp_avg, exp_avg_sq)가 CPU에 상주하므로 AdamW 옵티마이저 스텝이 완전히 CPU에서 실행됩니다. GPU AdamW보다 **20-40배 느리지만** 설계상 내재적입니다. 트레이드오프는 다음의 이유로 허용 가능합니다:

1. 옵티마이저 스텝은 총 스텝 시간(순전파 + 역전파가 지배)의 작은 부분입니다.
2. 대안(오프로드 없음)은 OOM입니다.
3. CPU AdamW 정확도는 GPU AdamW와 수치적으로 동일합니다 (동일한 float32 누적).

## 활성화 스필

활성화 스필을 활성화하면 순전파 중 서브레이어 경계에서 중간 활성화가 CPU로 스필되고, 역전파 중 다시 프리페치됩니다. 소비 후 즉시 해제되어 호스트 메모리를 제한합니다.

`activation_spill: true`는 재계산에 필요한 구간 입력을 CPU에 저장하며 false는 GPU에 유지합니다. `operation.recompute_interval`과 독립적인 선택이고, `activation_recompute`를 비활성화하지 않습니다.

세분도는 `sub_layer`(기본값) 또는 `full_layer`입니다. interval이 1보다 크면 `full_layer`를 사용합니다.

### Recompute interval

```yaml
operation:
  activation_recompute: true
  recompute_interval: 2
offload:
  enabled: true
  weight_offload: true
  activation_spill: true
  activation_spill_granularity: full_layer
```

`recompute_interval`은 재계산 구간에 포함하는 Transformer 레이어 수입니다. 기본값 1은 기존 레이어별 경계를 유지하고, 2는 두 레이어마다 입력 하나를 저장합니다. 0은 전체 레이어 재계산을 비활성화합니다. GPU 상주 모델에서는 `activation_recompute`를 명시적으로 켭니다. 가중치 스트리밍은 양수 interval이 필요하고 안전한 재계산을 활성화하지만, CPU activation 저장은 자동으로 켜지 않습니다.

### 생략할 입력 경계 개수 지정

`operation.recompute_skip_count`는 저장하지 않을 **레이어 입력 경계**의 개수입니다. 기본값 `null`은 interval 방식입니다. 개수 방식은 `recompute_interval: 1`(기본값)과 함께 사용하며, `0 <= recompute_skip_count < num_layers`여야 합니다. 다른 interval을 동시에 지정하면 오류입니다. 최초 레이어 입력은 복원을 위해 항상 저장합니다.

```yaml
operation:
  activation_recompute: true
  recompute_skip_count: 8
offload:
  activation_spill: false  # true이면 구간 입력을 CPU에 저장
```

32레이어에서 8을 지정하면 24개 입력 경계를 저장합니다. 두 레이어 구간 8개와 단일 레이어 구간 16개를 고르게 분산하며, 구간 길이 차이는 최대 1입니다. 더 많이 생략하면 3개 이상 레이어를 복원하는 구간이 필요합니다. 단일 레이어 구간에는 구간 복원용 추가 재계산을 적용하지 않습니다. 가중치 prefetch도 실제 구간 경계에 따라 이전 구간의 시작에서 재개합니다.

이 옵션은 "8개 레이어만 재계산"하는 선택적 recompute가 아닙니다. 모든 레이어 내부는 계속 재계산하며, 저장 경계 생략에 따른 추가 복원 작업을 조절합니다. CPU spill과 독립적으로 GPU 입력 checkpoint에도 적용됩니다. 경계를 생략하는 CPU spill은 `full_layer` granularity가 필요하며, FSDP에서는 아직 지원하지 않습니다. index 수동 지정은 제공하지 않습니다. 공개 예제의 `--recompute-skip-count`와 host RAM 예상치도 동일한 구간 계획을 사용합니다.

가중치와 입력을 모두 GPU에 유지하는 `offload.enabled: false`에서도 interval을 사용할 수 있습니다. GPU 상주 경로는 여러 레이어를 하나의 재계산 구간으로 묶으며 마지막 짧은 구간도 처리합니다. 가중치 스트리밍 경로는 backward에서 구간 입력을 복원한 뒤 GPU 레이어 checkpoint로 구간 내부 입력을 재구성하고, 레이어마다 가중치를 다시 올리고 방출합니다. 원래 forward에서 저장하는 host 경계 수를 줄이면서 여러 레이어의 가중치와 내부 gradient graph가 동시에 GPU에 남는 것을 방지합니다. 구간이 커지면 GPU 입력 보관량과 재계산 비용은 증가합니다.

현재 묶음 단위 FSDP 재계산은 지원하지 않습니다. Gemma offload의 PLE·shared KV 제한은 유지되지만 GPU 상주 Gemma 재계산에서는 둘 다 지원합니다. MLP·MoE 내부의 blocking 및 재계산은 이 전체 레이어 interval과 별도로 적용됩니다.

## 메모리 풀

### PinnedMemoryPool (호스트)

비동기 DMA 전송을 위한 `cudaMallocHost` (페이지 잠금 메모리)를 사용한 사전 할당 호스트 메모리. 병합이 있는 free-list 할당기와 고정 크기 청크. 가중치 타일과 스필된 활성화가 공유합니다.

설정:
```yaml
offload:
  pinned_memory_pool_gb: 100.0  # 총 호스트 메모리 예산
  pinned_chunk_gb: 4.0          # 청크 크기
```

### GPUStagingPool (장치)

가중치 스테이징 버퍼용 사전 할당 CUDA 메모리. PinnedMemoryPool과 동일한 청크 + free-list 패턴이지만 GPU에서. `threading.Lock`으로 스레드 안전.

설정:
```yaml
offload:
  gpu_staging_pool_mb: 0.0      # 0 = 자동 크기
  gpu_staging_chunk_mb: 256.0   # 청크 크기
```

## 구현

주요 파일:

| 파일 | 역할 |
|---|---|
| `ironcore/offload/config.py` | `OffloadConfig` 데이터클래스 |
| `ironcore/offload/memory_pool.py` | `PinnedMemoryPool`, `_PinnedChunk` |
| `ironcore/offload/gpu_staging_pool.py` | `GPUStagingPool`, `_GPUChunk` |
| `ironcore/offload/tile_manager.py` | `WeightTile`, `WeightGroup`, `TileManager` |
| `ironcore/offload/transfer_engine.py` | `MemoryTransferEngine` (비동기 H2D/D2H) |
| `ironcore/offload/scheduler.py` | `ExecutionScheduler` (라이프사이클 오케스트레이션) |
| `ironcore/offload/hooks.py` | `ActivationSpillManager` (activation_spill) |
