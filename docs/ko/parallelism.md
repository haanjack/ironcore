# 병렬화

> 이 가이드는 병렬화 전략의 설정 방법과 조합에 대해 다룹니다. 프로세스 그룹 레이아웃, TP 통신 프리미티브, EP 디스패치 내부 구조는 [병렬화 시스템 설계](design/parallelism.md)를 참고하세요.

## 전략 개요

| 전략 | 분할 대상 | 통신 | 사용 시점 |
|---|---|---|---|
| 데이터 병렬 (DDP) | 배치 | 그레이디언트 all-reduce | 멀티 GPU, 모델이 단일 GPU에 맞는 경우 |
| FSDP | 배치 + 파라미터 | 파라미터 all-gather, 그레이디언트 reduce-scatter | 대형 모델, 전체 상태 샤딩 |
| 텐서 병렬 (TP) | 모델 가중치 (레이어별) | 레이어당 all-gather / all-reduce | 레이어가 단일 GPU에 맞지 않는 경우 |
| 전문가 병렬 (EP) | MoE 전문가 서브셋 | 토큰 디스패치용 all-to-all | Mixture-of-Experts 모델 |
| 분산 옵티마이저 | 옵티마이저 상태만 | 업데이트된 파라미터 브로드캐스트 | 전체 FSDP 오버헤드 없이 ZeRO-1 |

TP와 DP/FSDP는 직교적이며 자유롭게 조합할 수 있습니다. EP는 MoE 모델에서 TP 위에 세 번째 축을 추가합니다.

---

## 프로세스 그룹 레이아웃

랭크는 `[DP × TP]` 2D 그리드로 배치됩니다:

```
World size = TP_size × DP_size

예시: TP=2, DP=2 (world=4)

         TP rank 0   TP rank 1
DP rank 0  [Rank 0]   [Rank 1]   ← TP Group 0: [0, 1]
DP rank 1  [Rank 2]   [Rank 3]   ← TP Group 1: [2, 3]
              │           │
           DP Group 0   DP Group 1
            [0, 2]       [1, 3]
```

- **TP 그룹**: 같은 데이터 샤드를 처리하며, 서로 다른 가중치 샤드를 보유.
- **DP 그룹**: 독립적인 데이터 샤드를 처리하며, 동일한 가중치 샤드를 보유 (같은 TP 위치).

---

## 텐서 병렬 (TP)

TP는 Megatron 스타일의 column/row 병렬 선형 레이어를 사용해 레이어 가중치를 GPU에 분산합니다. 각 레이어는 정확히 한 번의 집합 연산(all-gather 또는 all-reduce)이 필요합니다.

**제약 조건:** `num_attention_heads`, `num_attention_groups`, `vocab_size`는 모두 `tensor_model_parallel_size`로 나누어 떨어져야 합니다.

활성화 방법:

```yaml
trainer:
  tensor_model_parallel_size: 2   # TP 랭크 수
```

---

## 데이터 병렬 (DP) 및 FSDP

### 표준 DDP

`use_fsdp: false`일 때 기본값. 각 DP 랭크가 전체 모델 사본을 보유하며, 역전파 후 그레이디언트를 all-reduce합니다. 모델이 단일 GPU VRAM에 맞을 때 사용합니다.

### FSDP

DP 그룹 전체에 파라미터, 그레이디언트, 옵티마이저 상태를 샤딩합니다.

```yaml
parallel:
  use_fsdp: true
  fsdp_sharding_strategy: full     # full | shard_grad_op | hybrid | no_shard
  fsdp_mixed_precision: mixed      # mixed | fp16 | bf16 | fp32
```

**샤딩 전략:**

| 전략 | 샤딩 대상 | 비고 |
|---|---|---|
| `full` | 파라미터 + 그레이디언트 + 옵티마이저 상태 | 최대 메모리 절약 |
| `shard_grad_op` | 그레이디언트 + 옵티마이저 상태만 | 빠름; CPU 오프로드와 궁합 좋음 |
| `hybrid` | 노드 내 전체 샤딩, 노드 간 복제 | 멀티노드 대형 모델 |
| `no_shard` | 없음 (DDP와 동일) | 디버깅용 |

**체크포인트 형식** (`fsdp_state_dict_type`): `full`은 rank 0으로 수집; `local`은 각 랭크 샤드 저장; `sharded`는 분산 체크포인트 생성.

**참고:** TP > 1일 때 TP 비동기 통신과의 충돌을 피하기 위해 FSDP forward prefetch가 자동으로 비활성화됩니다.

### 분산 옵티마이저 (ZeRO-1)

FSDP 대신 옵티마이저 상태 샤딩에 사용할 수 있는 대안입니다. 파라미터와 그레이디언트는 완전히 복제되며, 모멘텀 텐서만 DP 랭크에 분산됩니다. DP 크기 N에서 `(N-1)/N`의 옵티마이저 상태 메모리를 절약합니다.

```yaml
parallel:
  use_distributed_optimizer: true
  dist_opt_bucket_cap_mb: 25.0    # 브로드캐스트 버킷 크기
```

**FSDP와 호환되지 않습니다.** 둘 중 하나만 사용하세요.

---

## 전문가 병렬 (EP)

MoE 모델에서 EP는 전문가 서브셋을 EP 랭크에 분산합니다. 토큰은 선택된 전문가를 보유한 랭크로 all-to-all을 통해 디스패치되고, 로컬에서 계산된 후 다시 수집됩니다.

공유 전문가(항상 활성화)는 모든 랭크에 복제되며 디스패치에 참여하지 않습니다.

모델 설정에서 구성:

```yaml
model:
  moe:
    use_moe: true
    expert_model_parallel_size: 2
    num_routed_experts: 64
    num_shared_experts: 2
    top_k: 2
```

EP는 TP와 조합할 수 있습니다. World size는 `DP × EP × TP`와 같아야 합니다.

---

## 초기화 순서

트레이너는 이 고정된 초기화 순서를 강제합니다. 순서를 바꾸지 마세요:

```
1. initialize_process()             # dist.init_process_group + cuda.set_device
2. initialize_model_parallel(tp)    # TP/DP 프로세스 그룹 생성
3. initialize_expert_parallel(ep)   # MoE + EP > 1일 때만
4. 모델 빌드 및 dtype 변환
5. (선택) HF 체크포인트 로딩
6. 옵티마이저 빌드
7. torch.compile(model)             # 병렬화 래핑 전에 반드시 수행
8. initialize_parallelism()         # DDP 또는 FSDP 래핑
```

`torch.compile`은 DDP/FSDP 래핑 전에 수행해야 합니다. 래핑 후 컴파일하면 잘못된 결과가 나옵니다.

---

## 설정 레퍼런스

```yaml
trainer:
  tensor_model_parallel_size: 2

parallel:
  # FSDP
  use_fsdp: false
  fsdp_sharding_strategy: full        # full | shard_grad_op | hybrid | no_shard
  fsdp_mixed_precision: mixed         # mixed | fp16 | bf16 | fp32
  fsdp_state_dict_type: full          # full | local | sharded
  fsdp_offload_params: false
  fsdp_use_orig_params: false

  # 분산 옵티마이저
  use_distributed_optimizer: false
  dist_opt_bucket_cap_mb: 25.0

  # 프로세스 그룹
  dist_backend: nccl
  timeout_minute: 10.0
```

---

## 사용 예시

### 단일 GPU

```bash
ironcore train --config configs/example.yaml
```

### 2-GPU 텐서 병렬

```bash
torchrun --nproc_per_node 2 -m ironcore train --config configs/example.yaml \
  --tensor-model-parallel-size 2
```

### 4-GPU: TP=2, DP=2

```bash
torchrun --nproc_per_node 4 -m ironcore train --config configs/example.yaml \
  --tensor-model-parallel-size 2
```

### 4-GPU + FSDP

```yaml
trainer:
  tensor_model_parallel_size: 1
parallel:
  use_fsdp: true
  fsdp_sharding_strategy: full
```

```bash
torchrun --nproc_per_node 4 -m ironcore train --config config.yaml
```

### 멀티노드 (2노드 × 8 GPU)

```bash
# Node 0
torchrun --nproc_per_node 8 --nnodes 2 --node_rank 0 \
  --master_addr <MASTER_IP> --master_port 29500 \
  -m ironcore train --config configs/example.yaml --tensor-model-parallel-size 2

# Node 1
torchrun --nproc_per_node 8 --nnodes 2 --node_rank 1 \
  --master_addr <MASTER_IP> --master_port 29500 \
  -m ironcore train --config configs/example.yaml --tensor-model-parallel-size 2
```

---

## 알려진 제한 사항

- **파이프라인 병렬화 없음.** 모든 트랜스포머 레이어는 같은 TP 그룹에서 실행됩니다.
- **컨텍스트 병렬화 없음.** 시퀀스 길이는 랭크에 분산되지 않습니다.
- **TP 나누어 떨어짐:** `num_attention_heads`, `num_attention_groups`, `vocab_size`는 모두 `tensor_model_parallel_size`로 나누어 떨어져야 합니다.
- **분산 옵티마이저는 FSDP와 호환되지 않습니다.** 둘 중 하나만 사용하세요.
- **EP는 `EP × TP ≤ world_size / DP`를 만족해야 합니다.**
