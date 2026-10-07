# 02. labels, tokenizer, SFT packing

## 명세

데이터가 제공한 target을 trainer가 다시 shift하지 않는다. `UniversalCollator`와 random dataset은 `input_ids=tokens[:-1]`, `labels=tokens[1:]`를 이미 만든다. response-only labels의 `-100`은 training·evaluation·accuracy에서 같은 의미를 가져야 한다.

[A1의 tokenizer/Transformer](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf)와 [A4의 데이터 처리](https://github.com/stanford-cs336/assignment4-data/blob/main/cs336_assignment4_data.pdf)를 연결하는 입력 계약이다. 새 BPE 학습이나 Common Crawl 필터링을 완료했다는 의미는 아니다.

## 평가의 오류

기존 `LanguageModelTrainer._eval_step`은 이미 shift된 labels를 다시 shift했다. training objective와 다른 target을 평가하고 마지막 위치도 제외했다. 정확도는 원래 labels와 비교하여 loss와 accuracy가 서로 다른 정렬을 사용했다.

수정 후 gathered logits와 already-shifted labels를 같은 위치에서 비교하고, training에 사용한 `loss_fn`으로 평가 loss를 집계한다. labels는 logits device로 이동한다. TP 모델의 inference return은 full-vocabulary logits이므로 accuracy 계산에는 `logits_are_parallel=False`를 명시한다.

## packing의 오류

collator는 packed document별 position IDs와 block-diagonal mask를 만들지만, 기존 `forward_step`은 input IDs/labels만 모델에 전달했다. position reset과 document isolation이 소실됐다. attention의 FlashAttention 경로도 explicit mask를 적용하지 않았다.

수정한 경로:

1. `forward_step`과 evaluation이 position IDs/attention mask를 전달한다.
2. `LanguageModel.forward`가 batch의 mask를 device로 옮기고 causal mask와 결합한다.
3. explicit mask가 있으면 현재 FlashAttention wrapper 대신 mask를 지원하는 SDPA 경로를 사용한다. 이 조건에서는 attention mask를 무시하는 최적화를 적용하지 않는다.

`cu_seqlens` 기반 packed FlashAttention을 새로 구현한 것은 아니다. 현재 정확한 packed SFT 경로는 explicit mask+SDPA이며, 긴 context에서 block mask의 메모리 비용은 추가 최적화 과제다.

## 실험

`test_packed_sft_keeps_documents_isolated_and_causal`은 실제 LanguageModel로 두 문서를 각각 실행한 logits와 한 row에 packing한 logits를 비교한다. 두 번째 문서의 첫 위치에서, 앞 문서와 뒤 token을 바꾸어도 해당 logits가 변하지 않는지 확인한다. FlashAttention을 요청한 설정에서도 explicit mask가 유지돼야 한다.

확장 회귀 실행에서 이 검증을 포함한 SFT masking 4개가 통과했다. tokenizer/FIM/collator 회귀도 함께 실행했다. tokenization과 masking의 근거로 사용하며 자연어 학습 품질의 근거로 확대하지 않는다.

## 실제 corpus manifest

성능/학습 실험은 공개 TinyStories train/valid의 각각 약 8 MiB/1 MiB prefix를 사용한다. 마지막 완전한 `<|endoftext|>` 경계까지 자른 뒤 GPT-2 tokenizer로 uint16 token 파일을 만든다. source URL·byte/token 수·text SHA-256·token SHA-256을 저장한다. source 응답에서 revision을 얻지 못하면 null을 유지하며 commit을 확인했다고 주장하지 않는다.

train/valid의 독립 source 파일을 사용하지만 대규모 near-duplicate 제거나 benchmark contamination 감사는 수행하지 않았다. 실제 dataset 전체에 대한 품질 평가로 확대하지 않는다.
