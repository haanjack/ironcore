# CLI 가이드

IronCore는 15개의 서브커맨드를 제공합니다. 전체 목록은 `ironcore --help`로 확인하세요.

```
ironcore <command> [options]
```

| 커맨드 | 설명 |
|---------|-------------|
| [`train`](#train-훈련-실행) | 훈련 실행 (pretrain, SFT, FIM, DPO, GRPO) |
| [`preprocess`](#preprocess-데이터셋-전처리-및-검사) | 데이터셋 토크나이즈 및 직렬화; 무결성 검사 |
| [`config-check`](#config-check-설정-검증-및-검사) | 설정 검증, 두 설정 비교, 해석된 YAML 표시 |
| [`tokenize`](#tokenize-토크나이즈-및-통계-표시) | 입력 텍스트나 파일 토크나이즈, 통계 표시 |
| [`inspect-checkpoint`](#inspect-checkpoint-체크포인트-검사) | 체크포인트 내용 검사, 두 체크포인트 비교 |
| [`export`](#export-huggingface-형식으로-변환) | IronCore 체크포인트를 HuggingFace 형식으로 변환 |
| [`generate`](#generate-텍스트-생성) | 대화형 REPL 또는 단일 텍스트 생성 |
| [`track`](#track-로깅-백엔드-설정) | YAML 설정에 로깅 백엔드 설정 패치 |
| [`evaluate`](#evaluate-평가-벤치마크-실행) | 체크포인트에 대해 평가 벤치마크 실행 |
| [`verify-step`](#verify-step-단일-스텝-손실-검증) | 1 훈련 스텝 실행 후 손실 보고 |
| [`verify-parity`](#verify-parity-병렬화-정확성-검증) | TP/DP/FSDP 설정 간 손실 곡선 비교 |
| [`profile`](#profile-훈련-프로파일링) | 모드 프리셋으로 훈련 프로파일링 |
| [`profile-mfu`](#profile-mfu-mfu-프로파일링) | 모델 FLOP 활용도(MFU) 측정 |
| [`analyze-scaling`](#analyze-scaling-스케일링-분석) | 멀티 스케일 훈련 실행 후 스케일링 법칙 피팅 |
| [`gen-report`](#gen-report-실험-보고서-생성) | 마크다운 실험 보고서 생성 |

## 핵심 커맨드

### `train` — 훈련 실행

YAML 설정으로 훈련을 시작합니다. pretrain, SFT, FIM, DPO, GRPO를 지원합니다.

```bash
# 단일 GPU
ironcore train --config configs/example.yaml

# 텐서 병렬 (2 GPU)
torchrun --nproc_per_node 2 -m ironcore train --config configs/example.yaml

# 멀티노드
torchrun --nproc_per_node 8 --nnodes 2 --node_rank 0 \
    --master_addr <IP> --master_port 29500 \
    -m ironcore train --config configs/example.yaml
```

| 플래그 | 필수 | 설명 |
|------|----------|-------------|
| `--config` | 예 | 훈련 설정 YAML 경로 |

### `preprocess` — 데이터셋 전처리 및 검사

훈련용 데이터셋을 토크나이즈하고 직렬화합니다. 선택적 검사 모드로 무결성을 확인하고 통계를 출력합니다.

```bash
# 전처리만
ironcore preprocess --config configs/data/pretrain_example.yaml

# 전처리 후 검사
ironcore preprocess --config configs/data/pretrain_example.yaml --inspect

# 기존 전처리 파일 검사
ironcore preprocess --config configs/data/pretrain_example.yaml --only-inspect

# 샘플 미리보기와 함께 검사
ironcore preprocess --config configs/data/pretrain_example.yaml --only-inspect --preview 5
```

| 플래그 | 필수 | 설명 |
|------|----------|-------------|
| `--config` | 예 | 데이터 설정 YAML 경로 |
| `--inspect` | 아니오 | 전처리 후 검사 실행 |
| `--only-inspect` | 아니오 | 전처리 건너뛰고 검사만 |
| `--preview` | 아니오 | 미리볼 샘플 수 (`--inspect` 포함) |

## 설정 및 데이터 도구

### `config-check` — 설정 검증 및 검사

훈련 설정을 검증하고 각 검사의 통과/실패를 표시합니다. 두 설정 비교 및 완전히 해석된 YAML 출력을 지원합니다.

```bash
# 설정 검증
ironcore config-check --config configs/pretrain_micro.yaml

# 해석된 설정을 YAML로 표시
ironcore config-check --config configs/pretrain_micro.yaml --show

# 두 설정 비교
ironcore config-check --config configs/pretrain_micro.yaml \
    --diff configs/pretrain_tiny.yaml
```

검증 항목: `train_steps > 0`, world size 대 TP size, 배치 크기 일관성, TP 헤드 나누어 떨어짐, 위치 임베딩 타입, 옵티마이저/FSDP 호환성.

| 플래그 | 필수 | 설명 |
|------|----------|-------------|
| `--config` | 예 | 훈련 설정 YAML 경로 |
| `--diff` | 아니오 | 비교할 두 번째 설정 경로 |
| `--show` | 아니오 | 완전히 해석된 설정을 YAML로 출력 |
| `--validate-only` | 아니오 | 검증만 하고 출력 억제 |

### `tokenize` — 토크나이즈 및 통계 표시

입력 텍스트나 파일을 토크나이즈하고 토큰 통계를 보고합니다. 데이터 준비 및 토크나이저 설정 디버깅에 유용합니다.

```bash
# 문자열 토크나이즈
ironcore tokenize --config configs/pretrain_micro.yaml --input "Hello world"

# 파일 토크나이즈
ironcore tokenize --config configs/pretrain_micro.yaml --input data/sample.txt

# 토큰별 분석 표시
ironcore tokenize --config configs/pretrain_micro.yaml \
    --input "Hello world" --show-tokens

# 시퀀스 길이 히스토그램
ironcore tokenize --config configs/pretrain_micro.yaml \
    --input data/sample.txt --histogram
```

출력: vocab 크기, 패딩된 vocab 크기, 총 토큰 수, 고유 토큰 수, 줄당 토큰 수 (평균/최소/최대/중앙값), 압축 비율 (bytes/token).

| 플래그 | 필수 | 설명 |
|------|----------|-------------|
| `--config` | 예 | 훈련 설정 YAML 경로 |
| `--input` | 예 | 텍스트 파일 경로 또는 리터럴 문자열 |
| `--show-tokens` | 아니오 | 토큰별 분석 표시 |
| `--histogram` | 아니오 | 시퀀스 길이 히스토그램 표시 |

## 체크포인트 도구

### `inspect-checkpoint` — 체크포인트 검사

체크포인트 내용 검사: 형식, 파라미터 수, dtype, 훈련 스텝, 아키텍처. 두 체크포인트를 텐서별 가중치 차이로 비교 지원.

```bash
# 기본 검사
ironcore inspect-checkpoint --path models/my_run

# 상세: 레이어별 통계 표시
ironcore inspect-checkpoint --path models/my_run --verbose

# 두 체크포인트 비교
ironcore inspect-checkpoint --path models/run_a \
    --compare models/run_b

# 기계 판독 가능 출력
ironcore inspect-checkpoint --path models/my_run --json
```

| 플래그 | 필수 | 설명 |
|------|----------|-------------|
| `--path` | 예 | 체크포인트 디렉토리 경로 |
| `--compare` | 아니오 | 가중치 차이 비교용 두 번째 체크포인트 |
| `--verbose` | 아니오 | 레이어별 가중치 통계 표시 |
| `--json` | 아니오 | 기계 판독 가능 JSON 출력 |

### `export` — HuggingFace 형식으로 변환

IronCore 체크포인트를 HuggingFace 형식(safetensors 또는 pytorch)으로 변환합니다. `transformers`와 호환되는 `config.json`과 가중치 파일을 생성합니다.

```bash
# safetensors로 내보내기 (기본값)
ironcore export --config configs/example.yaml \
    --checkpoint models/my_run --output-dir exported_model

# pytorch 형식으로 내보내기
ironcore export --config configs/example.yaml \
    --checkpoint models/my_run --output-dir exported_model \
    --format pytorch

# 샤딩 (256 MB당 샤드)
ironcore export --config configs/example.yaml \
    --checkpoint models/my_run --output-dir exported_model \
    --shard-size 256

# 대상 아키텍처 지정
ironcore export --config configs/example.yaml \
    --checkpoint models/my_run --output-dir exported_model \
    --architecture qwen2
```

| 플래그 | 필수 | 설명 |
|------|----------|-------------|
| `--config` | 예 | 훈련 설정 YAML 경로 |
| `--checkpoint` | 아니오 | 체크포인트 경로 (`trainer.model_path` 오버라이드) |
| `--output-dir` | 예 | HuggingFace 체크포인트 출력 디렉토리 |
| `--format` | 아니오 | `safetensors` (기본값) 또는 `pytorch` |
| `--shard-size` | 아니오 | 샤드 크기 (MB) |
| `--architecture` | 아니오 | 대상 아키텍처 (생략 시 자동 감지) |

### `generate` — 텍스트 생성

체크포인트를 로드해 텍스트를 생성합니다. 단일 모드(`--prompt`) 또는 대화형 REPL 지원. 지시 튜닝된 모델용 채팅 템플릿 모드.

```bash
# 단일 생성
ironcore generate --config configs/example.yaml \
    --checkpoint models/my_run \
    --prompt "The meaning of life is"

# 대화형 REPL (--prompt 없음)
ironcore generate --config configs/example.yaml \
    --checkpoint models/my_run

# 채팅 템플릿 모드
ironcore generate --config configs/example.yaml \
    --checkpoint models/my_run --chat \
    --system-prompt "You are a helpful assistant."

# 샘플링 제어
ironcore generate --config configs/example.yaml \
    --checkpoint models/my_run \
    --prompt "Once upon a time" \
    --temperature 0.8 --top-p 0.95 --top-k 50 \
    --max-new-tokens 256
```

REPL 종료: `quit`, `exit`, `q` 입력 또는 Ctrl-C.

| 플래그 | 필수 | 설명 |
|------|----------|-------------|
| `--config` | 예 | 훈련 설정 YAML 경로 |
| `--checkpoint` | 아니오 | 체크포인트 경로 (`trainer.model_path` 오버라이드) |
| `--prompt` | 아니오 | 프롬프트 텍스트 (생략 시 대화형 REPL) |
| `--max-new-tokens` | 아니오 | 최대 생성 토큰 수 (기본값: `128`) |
| `--temperature` | 아니오 | 샘플링 온도 (기본값: `1.0`) |
| `--top-p` | 아니오 | Top-p (nucleus) 샘플링 (기본값: `1.0`) |
| `--top-k` | 아니오 | Top-k 샘플링 (기본값: `0`, 비활성화) |
| `--no-sample` | 아니오 | Greedy 디코딩 사용 |
| `--system-prompt` | 아니오 | 채팅 모드용 시스템 프롬프트 |
| `--chat` | 아니오 | 채팅 템플릿 모드 활성화 |

## 실험 도구

### `track` — 로깅 백엔드 설정

훈련 설정 YAML에 로깅 백엔드 설정을 패치합니다. TensorBoard, MLflow, WandB를 지원합니다. 설정 파일만 수정하며, 실제 백엔드 초기화는 훈련 시작 시 이루어집니다.

```bash
# 대화형 모드 (각 백엔드 프롬프트)
ironcore track --config configs/example.yaml

# 비대화형: 특정 백엔드 활성화
ironcore track --config configs/example.yaml --backends wandb,tensorboard

# 백엔드별 옵션 포함
ironcore track --config configs/example.yaml \
    --backends wandb \
    --wandb-project my-project \
    --wandb-entity my-team
```

### `evaluate` — 평가 벤치마크 실행

훈련된 체크포인트에 대해 평가 태스크를 실행합니다.

```bash
# 기본: HellaSwag
ironcore evaluate --config configs/example.yaml --checkpoint models/my_run

# 커스텀 태스크 및 샘플 수
ironcore evaluate --config configs/example.yaml \
    --task hellaswag --num-samples 500

# 결과를 JSON으로 저장
ironcore evaluate --config configs/example.yaml \
    --checkpoint models/my_run --output eval_results.json
```

### `verify-step` — 단일 스텝 손실 검증

정확히 1 훈련 스텝을 실행하고 손실, 그레이디언트 노름, 타이밍을 보고합니다. 디버깅 및 회귀 테스트에 유용합니다.

```bash
ironcore verify-step --config configs/example.yaml
ironcore verify-step --config configs/example.yaml \
    --reference-loss 10.5432 --tolerance 0.01
```

### `verify-parity` — 병렬화 정확성 검증

같은 시드로 다양한 병렬화 설정 간 손실 곡선을 비교합니다. TP, DP, FSDP가 수치적으로 동등한 결과를 내는지 검증합니다.

```bash
# TP=1과 TP=2 비교 (기본값)
ironcore verify-parity --config configs/example.yaml --num-steps 10

# FSDP on vs off 검증
ironcore verify-parity --config configs/example.yaml --mode fsdp
```

### `profile` — 훈련 프로파일링

네 가지 모드 프리셋을 가진 IronCore 내장 프로파일러 래퍼입니다.

```bash
ironcore profile --config configs/example.yaml --mode quick
ironcore profile --config configs/example.yaml --mode full
ironcore profile --config configs/example.yaml --mode memory
```

| 모드 | 활성화되는 기능 |
|------|-----------------|
| `quick` | 레이어 타이밍 |
| `full` | 레이어 타이밍, torch 프로파일러, GPU 프로파일러, 통신 프로파일러, 메모리 스냅샷, Chrome trace, CSV 내보내기 |
| `comm` | 통신 프로파일러만 |
| `memory` | 메모리 스냅샷 + OOM 모니터 |

### `profile-mfu` — MFU 프로파일링

모델 FLOP 활용도 측정: 달성된 TFLOPS/s를 하드웨어 피크로 나눈 값.

```bash
ironcore profile-mfu --config configs/example.yaml
ironcore profile-mfu --config configs/example.yaml \
    --warmup-steps 5 --measure-steps 10 --hardware-peak 35.6
```

### `analyze-scaling` — 스케일링 분석

여러 모델 또는 배치 크기로 훈련을 실행하고, 최종 손실을 수집해 Chinchilla 스타일 거듭제곱 법칙을 피팅합니다.

```bash
ironcore analyze-scaling --config configs/pretrain_micro.yaml \
    --model-sizes gpt2-micro,gpt2-tiny,gpt2-small-test --num-steps 100 --fit-law --plot
```

### `gen-report` — 실험 보고서 생성

`experiments/<category>/`에 템플릿으로 마크다운 보고서를 생성합니다. 메타데이터(git 해시, 날짜, 설정 정보)가 자동으로 채워집니다.

```bash
ironcore gen-report --name "pretrain_convergence" --category pretrain \
    --config configs/pretrain_micro.yaml
```

## 미니 모델 설정

빠른 반복을 위한 세 가지 소형 모델 설정:

| 설정 | 레이어 | d_model | d_ffn | 헤드 | ~파라미터 | 사용 사례 |
|--------|--------|---------|-------|-------|---------|----------|
| `gpt2-micro` | 2 | 256 | 1024 | 4 | ~2M | 디버깅, 1-스텝 검증 |
| `gpt2-tiny` | 4 | 512 | 2048 | 8 | ~10M | 빠른 검증 실행 |
| `gpt2-small-test` | 8 | 768 | 3072 | 12 | ~40M | 짧은 훈련 실험 |

사용법: 훈련 설정에서 `model: gpt2-micro` (또는 `gpt2-tiny`, `gpt2-small-test`) 설정.

## 실험 설정

`configs/`의 바로 사용 가능한 설정:

| 설정 | 태스크 | 모델 | 데이터셋 | 스텝 |
|--------|------|-------|---------|-------|
| `pretrain_micro.yaml` | Pretrain | gpt2-micro | OpenWebText 10K | 1000 |
| `pretrain_tiny.yaml` | Pretrain | gpt2-tiny | OpenWebText 10K | 2000 |
| `pretrain_small.yaml` | Pretrain | gpt2-small-test | OpenWebText 10K | 3000 |
| `sft_small.yaml` | SFT | gpt2-small-test | UltraChat 5K | 1000 |
| `dpo_small.yaml` | DPO | gpt2-small-test | HH-RLHF 5K | 500 |
| `grpo_small.yaml` | GRPO | gpt2-small-test | GSM8K 1K | 200 |
| `lora_sft_small.yaml` | SFT + LoRA | gpt2-small-test | UltraChat 5K | 1000 |
