# 트레이너

## 트레이너 선택

`ironcore/cli/train.py`는 `data.task_type`에 따라 트레이너를 선택합니다:

| `task_type` | 트레이너 | 사용 사례 |
|---|---|---|
| `pretrain` | `LanguageModelTrainer` | 원시 텍스트 다음 토큰 예측 |
| `sft` | `LanguageModelTrainer` | 응답 토큰만 손실을 계산하는 지도 파인튜닝 |
| `dpo` | `DPOTrainer` | Direct Preference Optimization |
| `grpo` | `GRPOTrainer` | Group Relative Policy Optimization (온라인 롤아웃) |

## BaseTrainer

모든 트레이너는 `BaseTrainer`를 상속합니다. 초기화 순서는 고정되어 있으며 `train()`을 오버라이드하지 마세요:

1. 분산 프로세스 설정
2. TP/DP 프로세스 그룹
3. EP 프로세스 그룹 (MoE 전용)
4. 모델 + 옵티마이저 빌드
5. `torch.compile(model)` — DDP/FSDP 래핑 전에 반드시 수행
6. DDP 또는 FSDP 래핑

`trainer.model_path`에 체크포인트가 있으면 훈련을 자동으로 재개합니다. 체크포인트는 `save_checkpoint_steps`마다, 그리고 훈련 종료 시 저장됩니다.

### 확장 훅

`train()` 대신 이 메서드들을 서브클래스에서 오버라이드하세요:

| 훅 | 호출 시점 |
|---|---|
| `_pre_train_setup()` | 훈련 루프 전 — 체크포인트 로드 후 `_post_checkpoint_load()` 호출 |
| `_post_checkpoint_load(last_step)` | 체크포인트 로드 후 |
| `_on_checkpoint_save(step)` | 각 체크포인트 저장 전 |

## LanguageModelTrainer

`pretrain`과 `sft`에 사용됩니다. 손실은 다음 토큰 교차 엔트로피입니다. 토큰별 정확도가 손실과 함께 기록됩니다.

**SFT 응답 전용 마스킹:** SFT 콜레이터가 `labels`에서 프롬프트 토큰을 `-100`으로 설정합니다. 응답 토큰만 손실에 기여합니다. `pretrain`의 경우 모든 토큰이 마스킹되지 않습니다.

## DPOTrainer

선택/거부 쌍이 있는 선호도 데이터셋이 필요합니다. 훈련 시작 시 로드된 체크포인트로부터 고정된 참조 모델이 생성됩니다. FSDP에서는 FSDP 내부 상태 엉킴을 피하기 위해 `deepcopy` 대신 state dict 사본으로 참조 모델을 빌드합니다.

DPO 설정 옵션은 [alignment.md](alignment.md)를 참고하세요.

## GRPOTrainer

각 훈련 스텝:

1. **롤아웃 단계** — `grpo_rollout_micro_group_size` 단위로 청크 생성
2. **보상 채점** — `RewardManager`가 완성을 채점 (스레드 풀로 병렬화)
3. **업데이트 단계** — 롤아웃 버퍼에 대해 `grpo_num_epochs` 그레이디언트 스텝 실행

설정은 [alignment.md](alignment.md)와 [reward_manager.md](reward_manager.md)를 참고하세요.

## 설정 레퍼런스

| 필드 | 기본값 | 설명 |
|---|---|---|
| `trainer.micro_batch_size` | `2` | 단일 순전파에서 GPU당 배치 크기 |
| `trainer.gradient_accumulation_steps` | `null` | 파라미터 업데이트 전 마이크로 배치 수 (`train_batch_size`에서 파생) |
| `trainer.compile_model` | `false` | `torch.compile` 활성화 |
| `trainer.compile_mode` | `"default"` | `default` \| `reduce-overhead` \| `max-autotune` |
| `trainer.compile_backend` | `"inductor"` | 컴파일러 백엔드 |
| `trainer.compile_dynamic` | `false` | 동적 셰이프 허용 |
| `operation.activation_recompute` | `false` | 활성화 저장 대신 재계산 |
| `operation.train_steps` | — | 총 훈련 스텝 |
| `operation.eval_interval` | `100` | N 스텝마다 평가 |
| `trainer.tensor_model_parallel_size` | `1` | TP 차수 |
