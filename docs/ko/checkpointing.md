# 체크포인팅

> 이 가이드는 체크포인팅 설정 및 사용 방법을 다룹니다. 내부 아키텍처, TP 가중치 gather/split 메커니즘, DistributedOptimizer 상태 처리는 [체크포인팅 시스템 설계](design/checkpointing.md)를 참고하세요.

## 개요

IronCore는 두 가지 형식으로 체크포인트를 저장합니다:

| 형식 | 경로 | 적합한 경우 |
|---|---|---|
| **Universal** | `step_N/pytorch_model.bin` | 이식성, TP 차수 변경 |
| **Distributed** | `step_N/tp{r}/pytorch_model.bin` | 빠른 병렬 저장, 고정 TP |

두 형식 모두 모델 가중치, 옵티마이저 상태, LR 스케줄러 상태, 스텝 카운터를 랭크당 하나의 `torch.save` dict에 저장합니다. `latest_step.txt`는 가장 최근에 기록된 스텝을 추적해 자동 재개에 사용됩니다.

## 기본 설정

`trainer.model_path`를 설정하면 저장 및 로딩이 활성화됩니다:

```yaml
trainer:
  model_path: checkpoints/my-run
  save_checkpoint_steps: 500   # 500 스텝마다 저장
```

재시작 시 가장 최근 체크포인트에서 자동으로 훈련을 재개합니다. 저장을 비활성화하려면 (평가 실행 등):

```yaml
operation:
  no_save: true
```

## 훈련 재개

`train()` 호출 시 IronCore는 `latest_step.txt`를 읽고 첫 번째 스텝 전에 일치하는 체크포인트를 로딩합니다. 훈련은 `step = last_saved_step + 1`부터 계속됩니다. 설정 변경 없이 같은 명령을 다시 실행하면 됩니다.

최신이 아닌 특정 스텝부터 재개하려면:

```python
from ironcore.checkpointing import load_checkpoint
load_checkpoint(config, model, optimizer, lr_scheduler, step=1000)
```

가중치만 재개하고 옵티마이저 상태를 건너뛰려면 (아키텍처 변경 후 또는 사전학습 베이스에서 파인튜닝):

```yaml
optim:
  load_checkpoint_optim_state: false
  load_checkpoint_lr_scheduler: false
```

## Universal 체크포인팅 (TP 차수 변경)

기본 형식 (`save_dist_ckpt: false`)은 TP 차수에 무관합니다. TP=4로 저장하고 TP=1로 재개하거나, TP=2에서 TP=8로 스케일업할 수 있습니다. 별도의 변환 과정이 필요 없습니다.

```yaml
operation:
  save_dist_ckpt: false   # 기본값 — universal 형식
```

저장 시 IronCore는 모든 TP 랭크에서 TP-샤딩된 가중치와 옵티마이저 모멘텀 텐서를 수집해 rank 0에서 단일 파일로 씁니다. 로딩 시 전체 텐서가 새 TP 차수에 맞게 자동으로 분할됩니다.

LoRA 어댑터에도 적용됩니다: `lora_B`는 column-parallel 레이어와 함께 gather/split되고, `lora_A`는 row-parallel 레이어와 함께 처리됩니다.

## Distributed 체크포인팅 (병렬 I/O)

저장 속도가 이식성보다 중요할 때 사용합니다. 각 TP 랭크가 동시에 자신의 샤드를 씁니다:

```yaml
operation:
  save_dist_ckpt: true
```

`step_N/tp{r}/pytorch_model.bin` 형태로 랭크당 파일이 생성됩니다. 규모에 따라 저장이 `TP_size`배 빠르지만, 저장된 TP 차수에 고정됩니다. 다른 TP로 로딩하면 실패합니다.

## HuggingFace 상호 운용

### HuggingFace 체크포인트에서 로딩

```yaml
trainer:
  load_from_hf: meta-llama/Llama-3.1-8B   # HF 모델 ID 또는 로컬 경로
```

`detect_checkpoint_format()`이 safetensors, PyTorch, 단일 파일, 샤딩 형식을 자동으로 처리합니다. 모델 빌드 전에 `detect_bias_from_hf_state_dict()`를 실행해 어느 프로젝션에 바이어스가 있는지 파악하면 `BiasConfig`가 올바르게 설정됩니다.

### HuggingFace 형식으로 내보내기

```python
from ironcore.checkpointing.hf_interop import export_to_huggingface

export_to_huggingface(
    config=config,
    model=model,
    save_directory="exports/llama-finetuned",
    format="safetensors",   # 또는 "pytorch"
    max_shard_size="5GB",
)
```

출력은 표준 HF 레이아웃입니다:

```
exports/llama-finetuned/
├── config.json
├── model.safetensors.index.json
├── model-00001-of-00003.safetensors
└── ...
```

### HF config.json 생성

네이티브 체크포인트 옆에 HF 호환 `config.json`을 쓰려면 (직접 `AutoModel.from_pretrained` 사용에 필요):

```yaml
model:
  hf_model_type: llama
  hf_architecture: LlamaForCausalLM
```

두 필드 모두 설정해야 합니다. 설정되면 매 저장 시 `{model_path}/`에 `config.json`이 기록됩니다.

## LoRA 체크포인트

LoRA 어댑터 가중치는 베이스 모델 가중치와 **함께** 저장됩니다. 별도 어댑터 파일이 없습니다:

```
step_N/pytorch_model.bin
  model.layers.0.linear_q.weight    ← 베이스 가중치
  model.layers.0.linear_q.lora_A    ← LoRA A
  model.layers.0.linear_q.lora_B    ← LoRA B
  ...
```

로딩은 전체 체크포인트와 동일하게 작동합니다. PEFT 설정(`r`, `alpha`, `target_modules`)은 로딩 시 훈련 시와 정확히 일치해야 합니다.

## 체크포인트 검사

```bash
# 요약: 형식, 파라미터 수, 크기, dtype 분포, 스텝
ironcore inspect-checkpoint --path checkpoints/my-run

# 레이어별 셰이프 및 통계
ironcore inspect-checkpoint --path checkpoints/my-run --verbose

# 두 체크포인트 비교 (레이어별 max_abs_diff, mean_abs_diff)
ironcore inspect-checkpoint --path checkpoints/run-a --compare checkpoints/run-b
```

## 설정 레퍼런스

| 필드 | 그룹 | 기본값 | 설명 |
|---|---|---|---|
| `model_path` | `trainer` | `""` | 체크포인트 디렉토리 (비어 있으면 저장/로딩 없음) |
| `load_from_hf` | `trainer` | `null` | 로딩할 HF 모델 ID 또는 로컬 경로 |
| `save_checkpoint_steps` | `trainer` | — | N 스텝마다 저장 |
| `no_save` | `operation` | `false` | 체크포인트 저장 비활성화 |
| `save_dist_ckpt` | `operation` | `false` | `true` = 분산 랭크별, `false` = universal |
| `load_checkpoint_optim_state` | `optim` | `true` | 재개 시 옵티마이저 상태 복원 |
| `load_checkpoint_lr_scheduler` | `optim` | `true` | 재개 시 LR 스케줄러 상태 복원 |
| `hf_model_type` | `model` | `null` | HF 모델 타입 문자열; `config.json` 생성 활성화 |
| `hf_architecture` | `model` | `null` | HF 아키텍처 클래스 이름 |
