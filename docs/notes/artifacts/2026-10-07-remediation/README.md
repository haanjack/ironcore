# Trainer remediation 근거

00–09의 artifacts를 보존한 후 실행한 단계 10–15의 결과다. 성공한 gate와 실패한 gate를 같은 파일로 덮지 않는다. 단위 tests의 skip/deselect와 실제 CUDA 검증을 구분한다.

| 디렉터리 | 의미 |
|---|---|
| `ad-bf16` | FP32 stored weights / BF16 compute의 네 trainer DP/TP·16 exact resumes |
| `ep-layer`, `ep-pretrain` | 실제 EP 통신·expert source-gradient 합의 독립 oracle·첫 trainer gate |
| `fsdp-pretrain` | FSDP FP32 pretraining/full-state native 재개 |
| `failed-fsdp-reference` | rank 1의 empty frozen reference를 재현한 실패 |
| `fsdp-fixed-and-failed-distopt` | FSDP alignment 수정 후 성공 + distributed optimizer state 누락 실패 |
| `distributed-correctness` | 분산 optimizer·EP 네 trainer, recomputed CE, batched/idle experts FP32 matrix |
| `failed-alignment-template` | 실험 스크립트가 BatchEncoding을 list로 가정한 실패; rank faults는 통과 |
| `alignment-sft-dpo` | 실제 135M SFT/DPO 결과·선택 row manifest; 초기 GRPO encoding 실패 |
| `followup` | 초기 GRPO·3회 timing 대조·profiler off/on·대형 BF16 EP dtype 실패 |
| `final-precision` | CUDA offload reload/rank faults 통과 + 초기 BF16 EP dtype 실패 |
| `final-bf16-and-failed-worker-cli` | dtype 수정 후 EP BF16 네 trainer 성공 + worker fixture 범위 검사 실패 |
| `final-training` | worker=2 재개·FSDP/distributed optimizer BF16·token GRPO·실제 크기·compile·8K context |
| `final-evaluation` | 동일 생성 seed GRPO before/after·EP SendRecv/AllReduce profiler |
| `sources` | 각 immutable source SHA256 및 portable tar.gz; source 범위가 다른 실행을 섞지 않음 |
| `logs` | 실제 pytest·GPU driver·failure·browser·lint 로그 |

`manifest.json`에는 결과 파일의 SHA256·크기·원래 실행 경로를 기록한다. `huggingface_manifest.json`은 public model/datasets의 다운로드 revision과 파일 hash이다. 실제 weights·corpus·raw Chrome trace는 크기 때문에 저장소에 넣지 않는다. 최종 HTML과 JSON에는 compact CUDA events·trace hash·source provenance를 담았다. Native/HF 초기 logits oracle, scalar precision-sensitive update, 3회 performance CSV/JSON 및 export용 PNG/SVG를 함께 보존한다.

최종 CPU core 회귀는 532 passed / 12 skipped / 42 deselected이다. Markers로 CUDA/MP/HF-hub/e2e를 제외했고 GPU 작업은 별도 torchrun gate를 사용했다. Profiler/MFU unit regression은 80 passed이다. 소스 마지막의 task 경계 검사와 CLI presets도 각각 focused tests/2-GPU smoke로 확인했다.

`summary.json`은 성공한 gate의 상태를 모은 index이며 원본 report의 허용 오차·실패 로그를 대체하지 않는다. 최초 reasoning pilot의 negative 결과와 이후 동일 RNG/compute precision 조건의 0→0 결과를 모두 보존한다. Pipeline completion을 task quality 개선으로 해석하지 않는다.
