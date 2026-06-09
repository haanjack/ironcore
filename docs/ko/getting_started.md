# 시작하기

## 설치

```bash
git clone <repo>
cd ironcore-dev
pip install -e .
```

## 훈련 실행

### 단일 노드, 단일 GPU

```bash
ironcore train --config configs/example.yaml
```

### 분산 훈련 (torchrun)

**데이터 병렬 — GPU 2개, 순수 DP (설정에서 `tensor_model_parallel_size: 1`):**

```bash
torchrun --nproc_per_node 2 -m ironcore train --config configs/example.yaml
```

**텐서 병렬 — GPU 4개:**

```bash
torchrun --nproc_per_node 4 -m ironcore train --config configs/<name>.yaml
```

`RANK`, `LOCAL_RANK`, `WORLD_SIZE`는 `torchrun`이 자동으로 설정하며, 트레이너가 읽어들입니다.

## 설정 구조

설정 파일은 YAML 형식으로 `MainConfig`에 파싱됩니다. 모든 섹션은 선택 사항이며 기본값으로 대체됩니다.

```yaml
# 트레이너 설정
trainer:
  micro_batch_size: 4
  gradient_accumulation_steps: 8
  tensor_model_parallel_size: 1   # TP 차수
  model_path: models/my_run       # 체크포인트 디렉토리
  save_checkpoint_steps: 1000
  use_flash_attn: true

# 훈련 작업 설정
operation:
  train_steps: 10000
  eval_interval: 1000
  activation_recompute: false

# 모델 (내장 설정 이름 또는 인라인 지정)
model: gpt2-small

# 데이터
data:
  task_type: pretrain             # pretrain | sft | dpo | grpo
  seq_length: 1024
  datasets:
    - source: openwebtext
      task_type: pretrain
      ratio: 1.0

# 옵티마이저
optim:
  optimizer: adam                 # adam | muon
  lr_scheduler: cosine
  max_lr: 6e-4
  min_lr: 6e-5
  warmup_steps: 100
  weight_decay: 0.1
  clip_grad: 1.0

# 초기화
init:
  seed: 1337
```

전체 사전학습 예시는 `configs/example.yaml`, LoRA 파인튜닝 예시는 `configs/train_lora_example.yaml`을 참고하세요.

## HuggingFace 사전학습 가중치 불러오기

```yaml
trainer:
  load_from_hf: "Qwen/Qwen2.5-0.5B-Instruct"
```

`huggingface_hub.snapshot_download`으로 가중치를 다운로드하고 `load_from_huggingface()`로 로딩합니다. 아키텍처는 `config.json`에서 자동 감지됩니다. 가중치 매핑 세부 사항은 `docs/checkpointing.md`를 참고하세요.

## 데이터 전처리

```bash
ironcore preprocess --config configs/data/my_data.yaml
```

원시 텍스트 파일을 토크나이즈하고 `StreamingBinaryDataset`용 `.bin`/`.idx` 파일을 생성합니다. FIM 전처리는 데이터 설정에서 `data.fim_rate > 0`으로 활성화합니다.

## 서브시스템 문서

- **체크포인팅** (네이티브 + HF 상호 운용): `docs/checkpointing.md`
- **옵티마이저** (Muon + AdamW, ZeRO-1): `docs/optimizer.md`
- **트레이너** (BaseTrainer 라이프사이클): `docs/trainers.md`
- **정렬** (DPO + GRPO): `docs/alignment.md`
- **데이터로더** (스트리밍 데이터셋, bin-packing): `docs/dataloader.md`
- **추론 & KV 캐시**: `docs/inference.md`
- **평가** (HellaSwag + 퍼플렉서티): `docs/eval.md`
- **보상 시스템** (GRPO 보상): `docs/reward_manager.md`
