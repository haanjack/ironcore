# 데이터로더

## 태스크 타입과 콜레이터 모드

`data.task_type`은 콜레이터 모드와 배치에 포함되는 내용을 결정합니다:

| `task_type` | 배치 키 |
|---|---|
| `pretrain` | `input_ids`, `labels` |
| `sft` | `input_ids`, `labels`, `position_ids`, 선택적으로 `cu_seqlens` (FlashAttention) |
| `dpo` | `chosen_*` 및 `rejected_*` 접두사가 붙은 SFT 키 |
| `grpo` | `input_ids`, `attention_mask`, `metadata` 리스트 |

## 데이터셋 형식

**`StreamingDataset`:** 대용량 텍스트 코퍼스를 위한 블록 기반 셔플링. `ratio` 필드로 여러 데이터셋의 가중치 혼합을 지원합니다.

**`StreamingBinaryDataset`:** `ironcore preprocess`로 생성된 전처리된 `.bin`/`.idx` 파일에 대한 메모리 맵 접근. 대규모 사전학습 코퍼스에 사용합니다.

## SFT bin-packing

SFT 콜레이터는 GPU 활용도를 높이기 위해 여러 짧은 샘플을 `seq_length`까지 각 시퀀스에 패킹합니다. 각 패킹된 샘플은 독립적인 `position_ids`를 가집니다 (`[0, 1, 2, …]`로 리셋). `use_flash_attn: true`일 때는 `cu_seqlens` 경계가 FlashAttention에 전달되어 각 샘플이 자신의 토큰에만 어텐션합니다.

## 컬럼 매핑

데이터셋 컬럼을 역할에 매핑하는 설정:

| 설정 필드 | 기본값 | 사용 위치 |
|---|---|---|
| `text_column` | `"text"` | Pretrain |
| `messages_column` | `"messages"` | SFT |
| `chosen_column` | `"chosen"` | DPO |
| `rejected_column` | `"rejected"` | DPO |
| `prompt_column` | `"prompt"` | GRPO |
| `answer_column` | `"answer"` | GRPO |

## 멀티 데이터셋 혼합

`data.datasets`에 여러 데이터셋을 지정할 수 있습니다. 각 항목에 `ratio` 필드가 있으며, 샘플링은 `dataset_size × ratio`에 비례합니다. 혼합 의미론과 모범 사례는 [multi_dataset_sft.md](multi_dataset_sft.md)를 참고하세요.

## FIM 전처리

Fill-in-the-Middle은 훈련 중이 아닌 **전처리 시점** (`ironcore preprocess`)에 적용됩니다. 훈련 시 콜레이터는 FIM 변환된 시퀀스를 일반 pretrain 시퀀스로 처리합니다.

| 설정 필드 | 기본값 | 설명 |
|---|---|---|
| `data.fim_rate` | `0.0` | FIM으로 변환할 pretrain 문서 비율 (0 = 비활성화) |
| `data.fim_split_type` | `"random"` | `"random"` 또는 `"line_aware"` 분할 전략 |
| `data.fim_prefix_token` | `"<fim_prefix>"` | PSM 형식 접두사 센티넬 |
| `data.fim_suffix_token` | `"<fim_suffix>"` | PSM 형식 접미사 센티넬 |
| `data.fim_middle_token` | `"<fim_middle>"` | PSM 형식 중간 센티넬 |

## 설정 레퍼런스

| 필드 | 기본값 | 설명 |
|---|---|---|
| `data.task_type` | `"pretrain"` | 콜레이터 모드 결정 |
| `data.seq_length` | `1024` | 시퀀스 길이 |
| `data.tokenizer_type` | `"bbpe"` | `"bbpe"`, `"tiktoken"`, 또는 `"sentencepiece"` |
| `data.pad_token_id` | `null` | 패딩 토큰 (null = EOS 사용) |
| `data.vocab_name_or_path` | `"gpt2"` | 토크나이저 vocab 이름 또는 경로 |
| 데이터셋 `ratio` | `1.0` | 이 데이터셋의 샘플링 가중치 |
