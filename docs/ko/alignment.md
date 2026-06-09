# 정렬 (DPO 및 GRPO)

> 이 가이드는 정렬 훈련의 설정 및 실행 방법을 다룹니다. 손실 공식, 어드밴티지 계산, KL 추정기, 롤아웃 내부 구조는 [정렬 시스템 설계](design/alignment.md)를 참고하세요.

## DPO vs GRPO

| | DPO | GRPO |
|---|---|---|
| 훈련 중 생성 | 없음 | 있음 |
| 데이터 요구사항 | 사전 수집된 (선택, 거부) 쌍 | 프롬프트만 |
| 보상 소스 | 암묵적 (선호도 쌍) | 명시적 보상 함수 |
| 메모리 오버헤드 | 낮음 — 롤아웃 버퍼 없음 | 높음 — 롤아웃 버퍼 + 참조 모델 |
| 트레이너 | `DPOTrainer` | `GRPOTrainer` |

---

## DPO

`data.task_type: dpo`로 설정하고 `chosen`과 `rejected` 컬럼이 있는 선호도 데이터셋을 제공합니다.

### 주요 설정 결정

`concat_forward_passes: true` (기본값)는 선택 시퀀스와 거부 시퀀스를 단일 순전파로 실행해 TP all-gather를 하나 절약합니다. 디버깅이 아니면 그대로 두세요.

`dpo_label_smoothing`은 선호도 타겟에 ε 스무딩을 추가합니다. 표준 DPO 기본값은 `0.0`이며, 선호도 데이터에 노이즈가 많을 때 0.1–0.2가 도움이 될 수 있습니다.

`dpo_beta`는 손실이 참조 정책으로부터의 편차를 얼마나 강하게 패널티하는지 제어합니다. 일반적인 범위: 0.1–0.5.

### 최소 DPO 설정

```yaml
data:
  task_type: dpo
  datasets:
    - source: my-org/preference-dataset
      chosen_column: chosen
      rejected_column: rejected

alignment:
  dpo_beta: 0.5
  concat_forward_passes: true
```

### DPO 설정 레퍼런스

| 필드 | 기본값 | 설명 |
|---|---|---|
| `alignment.dpo_beta` | `0.5` | β — 선호도 강도 |
| `alignment.dpo_label_smoothing` | `0.0` | 레이블 스무딩 ε |
| `alignment.concat_forward_passes` | `true` | chosen+rejected 단일 순전파 |
| `alignment.metrics_interval` | `0` | N 스텝마다 추가 메트릭 기록 (0 = 매 스텝) |

---

## GRPO

`data.task_type: grpo`로 설정하고 프롬프트 데이터셋을 제공합니다. 트레이너가 완성을 생성하고, 보상 함수로 점수를 매기고, 정책을 훈련합니다.

### 롤아웃 설정

`grpo_group_size` (G)는 프롬프트당 샘플링하는 총 완성 수입니다. 어드밴티지는 그룹 상대적으로 계산됩니다 — 정규화 전에 프롬프트의 G개 완성이 모두 보여야 합니다. DP > 1일 때는 정규화 전 랭크 간에 보상이 all-gather됩니다.

`grpo_rollout_micro_group_size`는 스텝당 병렬로 생성할 완성 수를 제어합니다. 생성 중 피크 메모리를 줄이려면 낮게 설정하세요.

```yaml
alignment:
  grpo_group_size: 8                 # 프롬프트당 G개 완성
  grpo_rollout_micro_group_size: 2   # 한 번에 2개 생성
```

### 멀티 에폭 리플레이 (오프라인 GRPO)

`grpo_num_epochs > 1`로 설정하면 중요도 샘플링된 멀티 에폭 리플레이가 활성화됩니다: 같은 롤아웃 배치가 IS 비율 클리핑과 함께 여러 그레이디언트 업데이트 스텝에 사용됩니다. 약간 오래된 어드밴티지 추정치를 대가로 샘플 효율이 향상됩니다.

```yaml
alignment:
  grpo_num_epochs: 4      # 각 롤아웃을 4번의 그레이디언트 스텝에 재사용
  grpo_clip_eps: 0.2      # PPO 스타일 IS 비율 클리핑 범위
```

### 생성 파라미터

```yaml
alignment:
  generation:
    max_new_tokens: 512
    temperature: 1.0
    top_p: 0.9
    top_k: 0          # 0 = 비활성화
    do_sample: true   # false = greedy
```

### 참조 모델 CPU 오프로드

`alignment.offload_ref_model: true`로 설정하면 참조 모델을 순전파 사이에 CPU에 유지합니다. 참조 로그 확률 계산 시에만 GPU로 이동합니다. 정책 + 참조 모델이 VRAM을 초과하는 단일 GPU GRPO에 유용합니다.

**제한:** FSDP와 호환되지 않습니다. FSDP가 활성화된 경우 경고와 함께 무시됩니다.

### 최소 GRPO 설정

```yaml
data:
  task_type: grpo
  datasets:
    - source: my-org/prompt-dataset
      prompt_column: prompt

alignment:
  grpo_group_size: 8
  grpo_beta: 0.1
  grpo_num_epochs: 1
  generation:
    max_new_tokens: 512
    temperature: 1.0
  reward_manager:
    num_workers: 4
    functions:
      - name: correctness
        type: rule_template
        weight: 1.0
        rule_template: configs/rewards/math_gsm8k.yaml
```

### GRPO 설정 레퍼런스

| 필드 | 기본값 | 설명 |
|---|---|---|
| `alignment.grpo_group_size` | `4` | 프롬프트당 총 완성 수 (G) |
| `alignment.grpo_rollout_micro_group_size` | `1` | 청크당 병렬 생성 완성 수 |
| `alignment.grpo_beta` | `0.1` | KL 패널티 계수 |
| `alignment.grpo_eps` | `1e-8` | 어드밴티지 정규화 ε |
| `alignment.grpo_num_epochs` | `1` | 롤아웃당 그레이디언트 에폭 수 (>1 = 오프라인 리플레이) |
| `alignment.grpo_clip_eps` | `0.2` | IS 비율 클리핑 범위 ε (0 = 클리핑 없음) |
| `alignment.offload_ref_model` | `false` | 참조 모델을 패스 사이에 CPU에 유지 |
| `alignment.generation.max_new_tokens` | `512` | 최대 생성 토큰 수 |
| `alignment.generation.temperature` | `1.0` | 샘플링 온도 |
| `alignment.generation.top_p` | `0.9` | Nucleus 샘플링 임계값 |
| `alignment.generation.top_k` | `0` | Top-k 컷오프 (0 = 비활성화) |
| `alignment.generation.do_sample` | `true` | 확률적 샘플링 (false = greedy) |
| `alignment.reward_manager.*` | — | 보상 함수 목록 및 워커 풀 — [reward_manager.md](reward_manager.md) 참고 |
