# 12. Stateful data와 checkpoint 계약

## 명세

정확한 재개는 weights 외에 optimizer, scheduler, scaler, Python/NumPy/Torch/CUDA RNG, TP RNG tracker, frozen reference, 데이터 cursor를 포함한다. [StatefulDataLoader](https://meta-pytorch.org/data/main/torchdata.stateful_dataloader.html)를 사용해 consumed batch 기준 worker prefetch 상태까지 저장한다. `torchdata>=0.11,<0.12` 의존성을 추가했다.

Pretraining은 epoch/block/offset을 저장하고 block별 seed로 shuffle을 재구성해 이전 sample을 읽지 않고 seek한다. SFT/DPO는 weighted dataset 선택 RNG, index shuffle buffer, dataset별 offset, global shard cursor를 저장한다. Token 전체나 activation을 checkpoint에 복제하지 않는다. SFT의 DP rank가 RNG를 서로 다르게 소비하던 경로는 global permutation을 먼저 만들고 분할하도록 수정했다.

GRPO는 shuffled epoch/offset을 저장한다. Worker가 subprocess로 시작되어도 DP topology를 잃지 않도록 dataset 생성 시 rank/world를 기록하고 worker id를 추가한다. Dataset signature, shuffle 설정, worker 수, batch size가 달라지면 같은 궤적의 재개를 거부한다. GRPO JSON은 내용 SHA256, binary data는 파일 경로·크기·mtime를 검사한다. Binary 파일의 동일 크기·동일 mtime 변조까지 검출하는 content hash는 이 runtime cursor의 계약이 아니며 실험 manifest의 SHA256으로 별도 관리한다.

## 검증

`test_stateful_resume.py`: pretrain/SFT 직접 seek, DP permutation 분할, packing 결과(input/labels/sample ids/position/mask)의 worker=0/2 재개, GRPO shuffled epoch·prefetch 재개를 비교했다. `test_checkpoint_contract.py`는 Python/NumPy/Torch RNG와 data state API 복원, 파일 checksum/누락 검사, disk write 실패 시 이전 latest step 유지의 세 조건을 확인했다.

`trainer.num_workers`는 실제 loader에 적용된다. 같은 topology/worker/microbatch의 정확한 재개를 보장하는 범위이며, topology 변경은 model initialization이나 다른 실험이다. Stateful API가 없는 custom iterator의 legacy replay도 유지한다. `/tmp`의 pilot iterator는 cursor와 고정 subset signature를 저장하는 별도 실험용 iterator이며 production binary loader를 대체하지 않는다.

## Atomic commit과 전 rank 실패

Model 및 rank-local trainer sidecar를 temporary file→flush/fsync→replace로 기록한다. Dense native, EP, FSDP 각각 파일 checksum manifest를 만들고 저장이 성공한 뒤 `latest_step.txt`를 바꾼다. 쓰기 또는 integrity 검증이 한 rank에서 실패해도 다른 rank가 FSDP/NCCL load로 진입하지 않도록 오류 여부를 collective로 공유한다. Dense legacy checkpoint에는 manifest가 없을 수 있어 기존 loading 경로를 유지한다.

FSDP reference 저장의 기본 rank0-only full state는 rank 1에 빈 reference를 남겼다. 실제 DPO resume에서 재현하고 두 rank 모두 full reference state를 기록·복원하도록 수정했다. Full reference의 중복 CPU memory 비용은 [15](15-distributed-training.md)에 명시한다. 실패 로그를 삭제하거나 성공 report로 덮지 않는다.

## 최종 GPU 재개와 task 경계

Built-in 평가의 유한 데이터 iterator는 매번 처음부터 다시 생성한다. 한 rank의 데이터가 먼저 끝나면 model forward 전에 전 rank가 함께 종료하여 FSDP/TP collective 순서가 달라지지 않게 한다. 긴 shard의 남은 batch는 이 평가에서 제외하고, 실제로 처리한 objective units로 평균한다. 전 rank의 유효 데이터가 없으면 0이라는 가짜 metric 대신 오류를 낸다. CPU의 반복 평가·valid-count 평균 oracle와 실제 DP=2의 불균일/빈 shard 반복 평가 gate를 통과했다.

실제 torchrun DP=2에서 production stateful iterator를 사용하는 pretrain/SFT/GRPO의 worker=2, 4-step 중단·재개 최종 weights가 tolerance 0으로 일치했다. Worker의 CPU bootstrap 비용을 steady GPU throughput으로 섞지 않는다. 정확한 재개는 동일한 model/objective/hyperparameters와 deterministic reward/data 환경의 계약이다. 구현이 모든 가능한 config 변경을 자동 판별한다는 뜻은 아니다.

Task가 바뀐 checkpoint를 같은 trainer resume로 취급해 이전 global step 때문에 새 학습을 건너뛰는 경로를 거부한다. 새 SFT/DPO/GRPO 단계는 `trainer.load_from_hf`의 초기 weights와 별도 output `model_path`로 시작하고, 같은 task의 native checkpoint를 resume한다. Fresh LoRA adapter는 초기화한 채 HF base weights(동결된 weights 포함)를 모두 가져오도록 검증한다. 서로 다른 task의 optimizer/scheduler/data/reference 상태를 묵시적으로 재사용하지 않는다.
