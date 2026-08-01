# IronCore 프로덕션 준비 개선 계획

> 작성일: 2026-07-31 · 최종 갱신: 2026-08-01 (실기 검증 결과 반영)
> 범위: `ironcore/` 전체 서브시스템에 대한 코드 감사 기반 버그·개선점·프로덕션 격차 정리
> 방법: 트레이너/옵티마이저/체크포인트, 병렬화/오프로드, 얼라인먼트/데이터로더, 설정/CLI/관측성/패키징의 4개 영역을 각각 정독하고, 최상위 심각도 항목은 실제 코드로 교차검증함.
> 이후 AMD Strix Halo(gfx1151) 단일 GPU 환경에서 **10회의 실기 시험**을 수행해 주요 항목을 실증·반증함 — [§7 실기 검증 결과](#7-실기-검증-결과-2026-08-01) 참조.

---

## 0. 요약 (Executive Summary)

IronCore는 개인 학습용으로 출발했지만 기능 폭(TP/EP/DP/FSDP, MoE, GRPO/DPO, 오프로드, HF 상호운용)이 넓어 사실상 소형 프레임워크 규모다. 코드 구조·문서·CI 골격은 양호하나, **학습 정확성(correctness)을 깨뜨리는 결함이 여러 곳에 잠복**해 있어 현재 상태로는 프로덕션은 물론 신뢰할 수 있는 실험 재현도 어렵다.

가장 중요한 문제는 세 부류다.

1. **조용한 정확성 버그** — 데이터가 DP 랭크 간 중복 학습되고(1/N로 축소), DPO 페어 정렬이 깨지고, GRPO importance ratio가 시퀀스 레벨에서 폭주하며, eval에서 attention dropout이 켜지는 등, 에러 없이 잘못된 학습을 진행하는 문제들. **가장 위험하다 — 실패가 조용하다.**
2. **재개(resume)·체크포인트 신뢰성** — 원자적 저장이 없어 선점(preemption) 시 체크포인트가 손상되고, RNG·데이터로더 위치가 저장되지 않아 재개가 비결정적이며, 보존 정책·검증이 전무하다.
3. **분산 정확성** — DistributedOptimizer의 broadcast가 글로벌 랭크 대신 그룹 랭크를 사용하고, EP>1일 때 DP 그룹 구성이 어긋나며, grad norm 계산에 취약한 휴리스틱이 있다.

아래는 심각도별 항목과 단계별 로드맵이다. 각 항목에는 `파일:라인 — 문제 — 수정 방향`을 명시했다. **모든 P0/P1 수정은 재현 테스트를 먼저 추가**하는 것을 전제로 한다(현재 이 경로들 대부분이 테스트로 보호되지 않음).

---

## 1. P0 — 조용한 정확성 결함 (즉시 수정, 릴리스 차단)

이 항목들은 예외를 던지지 않고 잘못된 결과를 낸다. 실험 결과를 무효화하므로 최우선.

| # | 위치 | 문제 | 수정 방향 |
|---|------|------|-----------|
| P0-1 | `dataloader/dataset.py:437` | `if global_idx % world_size != rank: continue`가 **RNG 소비 이전**에 스킵 → 모든 DP 랭크가 동일한 `rng.choice` 시퀀스를 그려 **같은 샘플을 학습**. 유효 데이터셋이 1/world_size로 축소, 랭크 간 데이터 중복. (검증 완료) | 전역 순열을 먼저 만들고 `[rank::world_size]`로 슬라이스하거나, 먼저 draw 후 non-owned 스킵. |
| P0-2 | `dataloader/collator.py:305-308` (`_collate_dpo`) | chosen/rejected가 **독립적으로** bin-packing·길이정렬되어 row *i*의 chosen과 rejected가 서로 다른 페어. `dpo_loss`가 엇갈린 페어를 뺌 → DPO 학습이 무의미. | `group_id`로 페어 유지, DPO는 packing/정렬 없이 pad, 페어 순서 보존. |
| P0-3 | `alignment/dataset.py`(스트리밍) + `serializer.py:454-478` | DPO chosen/rejected가 독립 스트림 샘플로 방출·셔플·샤딩 → 배치에 매칭 페어가 함께 온다는 보장 없음. | chosen+rejected를 `group_id` 키의 단일 원자 샘플로 방출. |
| P0-4 | `optimizer/lr_scheduler.py:117` | `scheduler = LinearDecayLRScheduler` — **인스턴스가 아니라 클래스**를 대입. `lr_scheduler: linear` 선택 시 `.step()`/`get_last_lr()`에서 즉시 크래시. (검증 완료) | `LinearDecayLRScheduler(optimizer, **lr_scheduler_kwargs)`로 인스턴스화. |
| P0-5 | `layers/attention.py:137` | flash 경로가 `dropout_attn`을 **무조건** 전달 → eval/inference에서도 attention dropout 적용(SDPA 경로는 `self.training`으로 게이트함). 평가 지표 왜곡. (검증 완료) | `dropout_p = dropout_attn if self.training else 0.0`. |
| P0-6 | `checkpointing/native.py:673-674` | `torch.save`가 최종 파일에 **직접** 기록 — 저장 중 크래시/선점 시 잘린 파일이 남아 resume 영구 불가. (검증 완료: tmp+rename·fsync 없음) | `*.tmp`에 저장 → `flush()`+`os.fsync()` → `os.replace()`로 원자 교체. `latest_step`도 동일. |

---

## 2. P1 — 정확성/재개/분산 신뢰성 (프로덕션 전 필수)

### 2.1 체크포인트 & 재개

- **`native.py:660-667` (저장), `load_checkpoint`** — RNG 상태(torch/cuda/numpy/`random`)와 데이터로더/샘플러 위치를 저장·복원하지 않음. 재개가 비결정적이며 **데이터를 처음부터 재처리**. → 체크포인트에 `get_rng_state`류 + consumed offset(epoch, intra-epoch position) 포함, 로드 시 복원.
- **`native.py:600-635`** — universal 체크포인트가 `zip(named_parameters(), state["state"], strict=True)`로 위치 기반 페어링. frozen/LoRA/PEFT 파라미터가 있으면 길이 불일치로 예외 또는 오정렬. → `optimizer.state[param]`로 매칭, `requires_grad=False` 스킵.
- **`native.py:611, 373/416`** — 옵티마이저 상태 직렬화가 `exp_avg`/`exp_avg_sq`/`max_exp_avg_sq`만 처리 → **Muon의 `momentum_buffer`가 resume 시 유실**. → 비-Adam 상태 키 포함.
- **`native.py:677-683`** — `latest_step`을 barrier(690) **이전에** 비원자적으로 기록 → 크래시 시 미완성 체크포인트를 가리킴. → barrier 후 원자적 기록, 모든 샤드 writer 완료 후에만.
- **`native.py:221-226`** — `model_path`가 있는데 체크포인트가 없으면 조용히 -1(신규 시작) 반환 + `no_save=True`로 config 부작용 변경. 의도한 resume이 조용히 재시작되어 덮어쓸 수 있음. → "체크포인트 없음 기대" vs "기대했으나 없음"을 구분해 후자는 명시적 실패.
- **보존/검증 부재** — `keep_last_n` 로테이션 없음(step_N 무한 누적), 저장 후 재로드/체크섬 검증 없음. → 보존 정책 + 저장 후 verify 옵션.

### 2.2 분산(TP/EP/DP) 정확성

- **`optimizer/distributed_optimizer.py:225`** — `dist.broadcast(src=owner_rank)`에 **그룹 상대 랭크**를 전달하나 PyTorch는 **글로벌 랭크**를 기대. TP>1 또는 멀티노드(DP 그룹이 글로벌 0에서 시작하지 않을 때) 파라미터 손상. (검증 완료) → `src=dist.get_global_rank(self.process_group, owner_rank)`.
- **`native.py:148/150`** — `gather_object(dst=0)`(글로벌 0)이나 병합은 `if dp_rank == 0` 가드. DP 그룹의 0이 글로벌 0이 아니면 상태 유실. → `dst`에도 `get_global_rank(dp_group, 0)` 사용.
- **`parallel/parallel_states.py:82`** — DP 그룹을 `world_size // TP`로 구성, EP 무인지. EP>1이면 DP 그룹이 서로 다른 expert 샤드를 걸쳐 grad-norm/DDP SUM이 이질적 grad를 섞음. → `(ep_idx, tp_idx)` 공유 랭크로 DP 그룹 구성.
- **`parallel/grad_norm.py:156`** — ZeRO-3 감지에 `p.grad.numel() < p.numel()` 휴리스틱 → DP 평균화를 잘못 on/off, 전역 norm 오류. → 트레이너에서 명시적 `grads_are_sharded` 플래그 전달.
- **`parallel/grad_norm.py:48`** — grad 없는 랭크가 collective 전에 조기 `return` → 참여 랭크와 데드락, CPU/CUDA 텐서 불일치. → collective는 수행(0 기여), 반환은 `device`에.
- **`parallel/expert_parallel/comm.py:650`** — `num_experts = topk_indices.max()+1`을 배치 데이터에서 유도 → 배치 최대 인덱스 < 설정값이면 로컬 expert 범위 오류(+ GPU sync). → 설정 `num_experts`를 metadata로 전달.
- **`parallel/expert_parallel/comm.py:104`** — `_AllReduceEP.backward`가 `grad_output`을 in-place all_reduce → autograd가 참조 중인 grad 손상 가능. → clone 후 reduce.

### 2.3 얼라인먼트 손실 수학

- **`alignment/loss/grpo.py:52` + `rollout.py:52`** — `group_ids`가 랭크 로컬(`arange(B)`) → all-gather 후 서로 다른 랭크의 동일 로컬 id가 한 그룹으로 병합, 정규화 그룹 오염. → `rank*B + local_id`로 전역 유일화.
- **`alignment/loss/grpo.py:210-211`** — importance ratio를 **시퀀스 레벨** `exp(sum(logp-old_logp))`로 계산 → 수백 토큰 합산으로 분산 폭주. 실측(토큰당 노이즈 std 0.03 × 300토큰)에서 ratio가 0.25~5.0배로 흔들렸고, 토큰 평균 정규화 시엔 0.995~1.005로 안정. → 토큰 레벨 PPO surrogate로 전환.
  - **정정**: 초기 감사는 "기본 `clip_eps=0.0`이라 무클리핑"이라 했으나 **이는 틀렸다**. `grpo.py:173`의 `0.0`은 함수 시그니처 기본값일 뿐이고, 실제 학습 경로는 `config_alignment.py:94`의 `grpo_clip_eps: float = 0.2`가 `grpo_trainer.py:65 → :625`로 전달된다. 즉 **기본 파이프라인은 ε=0.2로 클리핑되며 gradient는 무한 분산이 아니다.** 다만 실측 클립 발동률이 12~100%로, 클리핑이 분산을 상시 억누르는 상태 자체가 시퀀스 레벨 ratio의 부적절함을 방증한다.
- **`alignment/loss/grpo.py:205` + `kl.py:39`** — 정책 손실·KL이 시퀀스 합(길이 정규화 없음) → 긴 완성 쪽으로 gradient/KL 편향. → 응답 길이로 정규화(토큰 평균).
- **`rollout.py:366`** — `old_log_probs`를 raw(비-temperature/필터) logits로 계산하나 토큰은 temperature/top-p 분포에서 샘플 → `temperature != 1`에서 policy/behavior 불일치.

### 2.4 데이터로더/전처리 정합성

- **`collator.py:170-192` (`_collate_sft`)** — 패킹 샘플당 `sample_len-1` 토큰만 기록하나 `current_pos`/`position_ids`/attention/`cu_seqlens`는 `sample_len`만큼 전진 → 시퀀스 중간에 pad 구멍, `cu_seqlens` 과대. → 모든 위치 bookkeeping을 `sample_len-1`로 전진.
- **`collator.py:174-178` vs `serializer.py:637`** — mask_ranges는 원본 위치를 인덱싱하나 labels는 `token_ids[1:]`로 shift → off-by-one, 마지막 prompt 토큰 예측이 손실에 누출되고 첫 완성 토큰이 잘못 마스킹. → mask_range를 -1 shift.
- **`serializer.py:624-641`** — 채팅 템플릿 마스크를 메시지별 개별 `apply_chat_template` 길이 합산으로 구성 → 메시지마다 BOS/헤더 재추가로 `current_pos` drift, 잘못된 토큰 마스킹. → 누적 full-template 출력에 대한 오프셋 diff로 증분 토크나이즈.
- **`dataloader/dataset.py:356-361`** — pretrain 블록 샤딩이 마지막 부분 블록에서 랭크별 불균등 카운트 → 스텝별 마이크로배치 수 desync로 collective 정지 위험. → 나머지 drop 또는 world_size 배수로 pad.
- **`serializer.py:88,96`** — `idx_path=".../data.idx"`인데 `np.save`는 `data.idx.npy` 기록 → "이미 처리됨" 검사가 항상 실패해 매번 재직렬화. → 실제 `.npy` 경로 검사.
- **`preprocessing/inspect.py:148,272`** — `bin_data`를 무조건 `uint16` memmap하나 serializer는 큰 vocab에 `uint32` 기록 → 토큰 카운트·경계 검사 오류. → 파일 크기/vocab로 dtype 추론.

### 2.5 초기화/수치

- **`layers/module.py:127`** — `_init_tp_weight`가 각 TP 가중치 초기화 전 전역 RNG를 동일 `seed`로 리셋 → 같은 shape의 TP 샤드가 **바이트 동일**(대칭성 깨짐), 단일 GPU init과 발산. → 한 번만 seed(또는 파라미터별 파생 seed).
- **`layers/attention.py:168,74`** — `seq_len_q==seq_len_kv` & 비캐싱 시 `is_causal=True`로 `sdpa_mask=None` → **padding mask 폐기**, 패킹 배치가 pad 토큰에 attend. flash 경로도 `attention_mask` 무시. → causal+padding 결합 또는 varlen `cu_seqlens` 사용.
- **`layers/attention.py:44`** — `scale_factor` 계산하나 SDPA/flash에 미전달 → Gemma의 `query_pre_attn_scalar` 등 비기본 스케일이 조용히 무시. → `scale=` 명시 전달.

### 2.6 설정 검증(조기 실패로 전환)

- **`config/config_moe.py:82-101`** — `num_routed_experts % expert_model_parallel_size == 0` 미검증(런타임 심층에서 실패). → `__post_init__`에 divisibility + `EP<=world_size` 검사.
- **`config/config_model.py:187-209`** — GQA sanity 없음: `num_attention_groups`가 `num_attention_heads`를 나누지 않아도 파스 통과. → `heads % groups == 0` assert.
- **`config/__init__.py`** — MoE/EP를 `world_size`에 대해 검증 안 함(`world_size % (TP*EP)` 등). → EP↔world_size 정합 검사 추가.
- **`cli/config_check.py:36`** — `load_full_config`가 충돌 config에서 **raise** → 진단 대상 config에서 config-check가 raw traceback으로 크래시. → 검증 없이 로드 후 방어적으로 검사, `_config_validation`을 단일 소스로 재사용(현재 6개 인라인 검사가 drift).
- **`config/config.py:35-39`** — `BaseConfig.from_dict`가 **클래스**에 `setattr`(인스턴스 아님) → 공유 클래스 상태 변경. → 인스턴스에 설정.
- **`config/config_optim.py`/`config_peft.py`** — optimizer/lr_scheduler 자유 문자열(오타→늦은 KeyError), `min_lr>max_lr`, 음수 warmup, LoRA `r<=0`(scaling 0-division) 미검증. → `__post_init__` enum/bounds.

---

## 3. P2 — 견고성 · 관측성 · 프로덕션 인프라

### 3.1 에러 처리 (조용한 실패 제거)

- **`alignment/rewards/base.py:94`, `builtin.py:375/527`, `model.py:112/133`** — 광범위 `except Exception`이 코드 버그/인증 실패/파싱 오류를 모두 삼키고 `default_reward=0.5` 대체 → 실패가 중립 점수와 구분 불가, reward 신호/advantage 오염. → 좁은 예외 + 컨텍스트 로깅, 지속 실패는 표면화.
- **`__main__.py:24`** — 최상위 try/except 없음, dispatch 반환값 무시 → `return` 상태를 내는 명령이 항상 exit 0, 미포착 예외는 raw traceback. → dispatch 래핑, 예외→비영 exit 매핑.
- **`cli/train.py:40`** — `(ValueError, FileNotFoundError)`만 포착 → 학습 중 RuntimeError/CUDA OOM/NCCL이 정리 없이 탈출. → 광범위 log-and-exit + teardown.
- **`cli/analyze_scaling.py:166/385/431`, `reports.py:255`** — 예외를 `status:"ERROR"`로 기록하나 여전히 exit 0 → 실패한 sweep이 CI에서 성공처럼 보임. → 실패 추적 후 비영 exit.
- **`profiler.py:415/453/488`, `base_trainer.py:280/976-982`, `logger.py:97-117`** — 광범위 except가 MFU init/메트릭 로깅/텐서보드 실패를 조용히 숨김. → 좁히고 최소 1회 WARNING 로깅.

### 3.2 관측성

- **329개 `print()`** (`preprocessing/inspect.py`, `cli/*`, `utils/device.py`, `train.py`, `eval/evaluator.py` 등) — `IronCoreLogger` 우회 → 레벨/타임스탬프/rank 게이팅/싱크 없음. → 로거로 라우팅, `print`는 진짜 CLI stdout 페이로드만.
- **`logger.py:14-48`** — 사람용 포맷만, JSON/구조화 로그 없음 → ELK/Loki 수집 불가. → 옵트인 `JSONFormatter`.
- **`logger.py:204-208`** — `WandbLogger`가 `config.__dict__` 통째 업로드 → reward endpoint/토큰 등 시크릿 유출 가능. → 시크릿 키 스크럽/allowlist.
- **health/heartbeat 부재** — 장시간 분산 run에 hang 감지·readiness 없음. → 주기적 heartbeat + readiness 파일/HTTP.
- **메트릭 텔레메트리 부재** — Prometheus/OTel exporter 없음, throughput/MFU/step-time을 출력 후 regex로 재파싱(`utils/subprocess.py:119-154`). → 구조화 메트릭 직접 방출, GPU/host 메모리 스트림 추가.

### 3.3 패키징 · 재현성

- **`requirements.txt:16` vs `pyproject.toml`** — `wandb`가 requirements에만 있음 → `pip install -e .` 사용자는 `WandbLogger`에서 ImportError. → `[wandb]` extra로 통일.
- **의존성 핀 없음** — `transformers>=4.30.0`, bare `datasets/numpy/pandas`, 락파일 없음 → 비재현 빌드. → uv/pip-tools 락파일 + 릴리스 이미지 핀.
- **`Dockerfile`** — 패키지가 아닌 `requirements.txt` 설치로 extras/entrypoint 불일치, `nvcr.io/.../pytorch:25.12-py3` digest 미핀. → `.[cuda]` 설치 + base digest 핀.

### 3.4 테스트

- **커버리지 측정 전무** — `pytest-cov`/coverage 없음, CI 게이트 없음. → 추가 + 최소 기준.
- **`_config_validation` 직접 단위 테스트 없음** — 오프로드/FSDP/EP 충돌 매트릭스가 간접 노출만. → 테이블 기반 테스트로 각 충돌이 raise하는지 검증.
- **CLI 계층 미테스트** — dispatch/exit code/config-check 출력 테스트 없음. → CLI 스모크 테스트.
- **`tests/property/` 비어 있음** — config roundtrip(`to_yaml`/`from_yaml`) property 테스트 없음.
- **CI 격차** — PR에서는 CPU 로직 테스트만(GPU/멀티GPU는 self-hosted·main only) → 병렬화/오프로드 회귀가 무방비로 main 진입. `tests/multi_gpu/`는 `--ignore`로 제외. 의존성/보안 스캔(pip-audit/Dependabot) 없음.

### 3.5 HF 상호운용/기타

- **`checkpointing/hf_interop.py:463/561/573`** — HF export가 최종 파일명에 직접 기록, 샤드 in-place rename → 크래시 시 index.json과 불일치한 부분 체크포인트. → temp dir 후 디렉터리 원자 rename.
- **`hf_interop.py:239-242,416-429`** — `n_layer=12`, vocab `32000/50257`, `rms_norm_eps=1e-6` 등 하드코딩 폴백이 잘못된 `config.json`을 조용히 생성. → 명시적 실패 또는 실제 model config에서 소싱.
- **`base_trainer.py:871-878`** — "emergency checkpoint" 블록이 로깅만 하고 저장 안 함(주석에 명시), `_handle_training_error`는 어디서도 호출 안 됨 → 안전망처럼 보이나 실제 복구 0. (검증: `train_step`이 try/except로 감싸이지 않음) → 실제 저장하거나 오해 소지 dead code 제거.

---

## 4. 단계별 로드맵

각 단계는 **재현 테스트 → 수정 → CI 편입** 순서로 진행. 병렬화/오프로드 수정은 멀티 GPU 테스트를 PR 게이트에 넣기 전까지 회귀 위험이 크므로 3단계와 함께 묶는다.

### 1단계 — 정확성 응급 (1~2주)
목표: 조용한 오학습 중단. P0 전부 + P1의 손실 수학.
- P0-1 데이터 샤딩 중복, P0-2/3 DPO 페어링, P0-4 LR 스케줄러, P0-5 eval dropout, P0-6 원자적 저장.
- GRPO IS-ratio(2.3), group_id 전역화, SFT 라벨 off-by-one/패킹 구멍.
- 각각 실패를 재현하는 단위 테스트를 먼저 추가(현재 미보호).
- **완료 기준**: `pytest`에 정확성 회귀 테스트 통과, 단일/멀티 GPU에서 loss 재현.

### 2단계 — 재개·체크포인트 신뢰성 (1~2주)
목표: 선점·재시작 안전.
- RNG/데이터로더 위치 저장·복원, Muon momentum 포함, universal 옵티마이저 페어링 수정.
- 원자적 `latest_step`, 로드 실패 명시화, 보존 정책(`keep_last_n`) + 저장 후 verify.
- **완료 기준**: kill-and-resume 통합 테스트에서 loss 곡선 연속성 검증.

### 3단계 — 분산 정확성 + 멀티GPU CI 게이트 (2~3주)
목표: TP/EP/멀티노드 신뢰.
- DistributedOptimizer broadcast 글로벌 랭크, DP↔EP 그룹 구성, grad_norm 플래그화·데드락 제거, EP comm num_experts/in-place 수정.
- attention padding mask/scale, TP init RNG 대칭성.
- `tests/multi_gpu/`를 PR 게이트(또는 nightly required)로 승격, `--ignore` 해제.
- **완료 기준**: 2-GPU parity 테스트(단일 vs TP/DP loss 일치)가 CI에서 강제.

### 4단계 — 설정 검증 & CLI 견고성 (1주)
- `__post_init__`/`_config_validation`에 조기 검사 집중, config-check를 단일 검증기로 통합.
- `__main__`/CLI 예외→exit code 매핑, reward/analyze 실패 표면화.
- CLI 스모크 테스트 + config 충돌 테이블 테스트.

### 5단계 — 관측성 & 패키징 (1~2주)
- print→logger 마이그레이션, JSONFormatter, wandb config 스크럽, heartbeat/health.
- 메트릭 exporter(OTel/Prometheus) 옵션, 의존성 락파일 + Docker digest 핀, `[wandb]` extra 통일.
- `pytest-cov` 도입 + 커버리지 기준선.

### 6단계 — 하드닝 (지속)
- reward code-exec 샌드박스 구현(현재 `NotImplementedError`로 안전하게 미구현), HF export 원자화, 하드코딩 폴백 제거, property 테스트, 보안 스캔(pip-audit/Dependabot).

---

## 5. 우선 착수 Top 10 (한 문장 티켓)

1. `dataloader/dataset.py:437` — DP 랭크 간 데이터 중복 제거(전역 순열 샤딩).
2. `optimizer/lr_scheduler.py:117` — `LinearDecayLRScheduler` 인스턴스화.
3. `layers/attention.py:137` — eval 시 attention dropout 0.
4. `checkpointing/native.py:673` — 원자적 체크포인트 저장(tmp+fsync+replace).
5. `collator.py:305` — DPO chosen/rejected 페어 정렬 보존.
6. `alignment/loss/grpo.py:210` — 토큰 레벨 IS-ratio + 기본 클리핑.
7. `optimizer/distributed_optimizer.py:225` — broadcast src 글로벌 랭크 변환.
8. `native.py` resume — RNG/데이터로더 위치 저장·복원.
9. `collator.py:174` — SFT 라벨 마스크 off-by-one 수정.
10. `alignment/rewards/*` — 광범위 except 제거, reward 실패 표면화.

---

## 7. 실기 검증 결과 (2026-08-01)

AMD Strix Halo(gfx1151, Radeon 8060S) **단일 GPU** WSL2 컨테이너에서 10회 시험을 수행했다.
환경 구축은 [rocm_wsl_setup.md](rocm_wsl_setup.md) 참조. 단일 GPU이므로 TP/EP/DP>1 경로는
검증 불가이며, 해당 항목은 여전히 코드 판독 판정에 머문다.

### 7.1 테스트 스위트 통과 현황

| 시험 | 결과 |
| --- | --- |
| GPU 마크 스위트 (`-m cuda`) | 128 passed / 0 failed |
| 기본 스위트 (unit+regression) | 697 passed / 12 failed / 6 error |
| CPU 전용 CI 선택 | 533 passed / 0 failed |
| offload (unit+integration) | 143 passed / 34 skipped |
| kvcache·attention·eval | 109 passed / 20 skipped |
| MoE | 87 passed / 0 failed |
| GRPO | 24 passed / 2 skipped |
| DPO | 48 passed / 0 failed |
| LoRA | 4 passed / 6 skipped |

기본 스위트의 실패 12건은 모두 **테스트 격리 누수** 한 가지가 원인이다. `integration/optimizer`
테스트가 프로세스 그룹을 정리하지 않아 `dist.is_initialized()`가 참으로 남고, 이후
`rollout.py:275`의 `except (AssertionError, ImportError)`가 `RuntimeError`를 잡지 못해 무관한
rollout 테스트가 연쇄 실패한다. 에러 6건은 컨테이너에서 HF `xet` 백엔드가 404를 내는 문제로,
`HF_HUB_DISABLE_XET=1`로 해소됨(런처에 반영 완료).

### 7.2 실증된 결함 (empirically CONFIRMED)

| 항목 | 실증 증거 |
| --- | --- |
| 체크포인트 비원자적 저장 (P0-6) | 저장 dict 키가 6개뿐이고 `.tmp`/rename 흔적 없음 |
| RNG 상태 미저장 | 체크포인트에 RNG 관련 키 전무 |
| 데이터로더 위치 미저장 | 재개 시 `get_data_iterator()` 무조건 재호출 |
| SFT 라벨 off-by-one | 첫 완성 토큰이 `-100`으로 마스킹됨을 토큰 단위로 확인 |
| SFT 패킹 pad 구멍 | PAD 슬롯이 유효 `position_id`를 갖고 `cu_seqlens` 블록에 포함(9 vs 실제 7) |
| DPO 페어 정렬 붕괴 (P0-2) | 3쌍 중 2쌍 불일치, 나머지 1쌍은 길이 동률로 우연히 일치 |
| GRPO 시퀀스 레벨 IS ratio | std 0.03×300토큰에서 ratio 0.25~5.0배 (토큰 정규화 시 0.995~1.005) |
| GRPO 길이 편향 | log-prob·KL 모두 토큰 차원 `.sum()`, 정규화 없음 |
| reward 예외 은닉 | `compute()`가 예외를 던져도 `[0.5, 0.5, 0.5]` 반환, 로그 없음 |
| offload 풀 미해제 | `shutdown()` 후 `memory_allocated` 352MB 그대로, `del`+GC 후에야 0 |
| `record_stream` 부재 | 저장소 전체 0건 |

### 7.3 신규 발견 (초기 감사에 없던 항목)

**[P0] 생성이 항상 패딩 토큰을 출력한다.** `language_model.py:288`이 `logits[:, -1, :]`를 그대로
샘플링에 넘기는데 이는 패딩된 vocab 전체(gpt2: 50304 vs 실제 50257)다. tied 임베딩의 패딩 행이
0으로 초기화돼 패딩 logit이 정확히 `0.0`인 반면 실제 토큰 logit은 전부 음수(-103~-73)이므로
argmax가 매번 패딩 토큰을 고르고 결과는 빈 문자열이 된다. `[:vocab_size]`로 제한하자 즉시 정상
문장이 생성됐다. **`padding_start_idx`가 이미 존재하며 학습 손실 경로는 이를 올바르게 사용한다 —
생성/평가 경로만 쓰지 않는다.** → 샘플링·argmax 전에 실제 vocab으로 절단.

**[P0] `ironcore evaluate` CLI가 구조적으로 실행 불가능하다.** `evaluate.py:54`가
`train_steps: 0`을 설정하는데 `config/__init__.py:51`이 `train_steps <= 0`을 예외로 거부한다.
두 코드가 정면으로 모순되어 이 서브커맨드는 어떤 입력으로도 동작하지 않는다.

**[P0] `labels=None` 튜플 반환이 여러 호출부를 깨뜨린다.** `language_model.py:170-177`은 캐시가
활성일 때 `(logits, new_key_values)` 튜플을, 아닐 때 텐서를 반환한다. `KVCacheConfig.enabled`
기본값이 True이므로 기본 설정에서 다음이 모두 깨진다.
- `dpo_trainer.py:242/275/277` — `.detach()` → `AttributeError`. **DPO 학습이 end-to-end로 불가능.**
- `language_model_trainer.py:112` — `.argmax()`/슬라이싱 → 트레이너 내장 eval 불가.
→ 반환 타입을 통일하거나 모든 호출부에서 언패킹.

**[P1] LoRA 두 번째 체크포인트 저장이 크래시한다.** `native.py:588-589`가 **frozen 파라미터를
포함한** 전체 `named_parameters()`에 대해 `optimizer.state[param]`을 인덱싱한다.
`optimizer.state`는 `defaultdict`이므로 frozen 파라미터에 대해 **빈 항목을 생성**해 살아 있는
옵티마이저 상태를 오염시키고, 다음 `state_dict()` 호출이 `KeyError`로 죽는다. 전체 학습
모델에서는 재현되지 않는 PEFT 전용 결함. 기존 테스트는 `save_checkpoint`를 **한 번만** 호출해
구조적으로 이 경로를 잡을 수 없다. → frozen 파라미터를 건너뛰고 `param in optimizer.state`로 확인.

**[P1] LoRA 체크포인트가 어댑터만이 아니라 전체 모델을 저장한다.** 292 텐서 중 120개만
`lora_*`(2.36MB)이고 나머지 172개는 frozen base(248.95MB). config 주석은 "LoRA weights will be
saved here"라고 안내하지만 실제로는 매번 전체를 저장한다.

**[P2] mock 데이터가 `data.seq_length`를 무시한다.** `dataloader/__init__.py:50`이
`seq_length=config.model.max_seq_len`을 넘긴다. GPT-2는 `max_seq_len=1024`라 드러나지 않지만
Qwen(32768) 같은 롱컨텍스트 모델에서는 매 배치가 32K 토큰이 되어 사실상 진행이 멈춘다.

**[P2] `inspect-checkpoint` CLI가 실패한다.** 네이티브 체크포인트에 피클된 `pathlib.PosixPath`가
있어 `weights_only=True` 기본값에서 `UnpicklingError`.

**[P2] MoE aux loss가 관측 불가하다.** load-balance aux loss는 스칼라 손실에 합산될 뿐
별도 메트릭으로 노출되지 않는다. 또한 README가 언급하는 **z-loss는 코드에 존재하지 않는다**
(`MoEConfig`에는 `aux_loss_alpha`만 있음).

### 7.4 반증된 항목 (초기 감사의 오류)

**GRPO 기본 무클리핑 주장은 틀렸다.** `grpo.py:173`의 `clip_eps=0.0`은 함수 시그니처 기본값이며,
실제 학습 경로는 `config_alignment.py:94`의 `grpo_clip_eps=0.2`가 `grpo_trainer.py:65 → :625`로
전달된다. 기본 파이프라인은 PPO 방식으로 클리핑되며 gradient는 무한 분산이 아니다.
(§2.3 본문에 정정 반영함.)

**paged attention의 zero-KV 누수 주장은 틀렸다.** 제로 패딩 자체는 실재하나, 유일한 실사용
호출부(`transformer.py:273`)가 per-sequence `cache_position`으로 마스크를 재구성한다. 배치
디코드와 단독 디코드의 logit 차이가 정확히 `0.000e+00`으로 일치했다.

**KV 캐시는 정확하다.** 캐시 사용 생성과 매 스텝 전체 재계산 생성이 20스텝 내내 토큰 완전 일치.

### 7.5 수정 완료 항목

| 파일 | 내용 |
| --- | --- |
| `configs/model/qwen2.5-0.5B.yaml` | 스키마 불일치로 **로드 자체가 불가능**하던 것을 수정(`ln_type`/`ln_eps`, 중첩 `positional_embedding`, 존재하지 않는 `attention_type`·`rope_theta`·`rope_scaling`·`tie_word_embeddings` 제거, `tokenizer_type` 유효값으로 교체, Qwen2.5용 QKV-only bias 명시) |
| `configs/model/gpt2-small-moe.yaml` | `head_dim: 64` 누락으로 QKV 투영(기본값 128 기준)과 어텐션 reshape(`d_model/heads`=64)가 불일치해 크래시하던 것을 수정 |
| `scripts/docker/launch.sh` | `HF_HUB_DISABLE_XET=1` 추가 — HF 다운로드 404 해소 |

전체 모델 config 14개를 전수 검사한 결과 스키마가 깨진 것은 Qwen 하나뿐이었다.

### 7.6 시험이 드러낸 가장 중요한 사실 — 테스트 커버리지의 착시

**총 1,000여 개 테스트가 통과하지만, 위 결함 중 어느 것도 잡지 못한다.** 각 경로에 테스트가
"존재"하지만 결함이 있는 지점을 비껴간다.

| 결함 | 관련 테스트 | 왜 못 잡는가 |
| --- | --- | --- |
| SFT 라벨 off-by-one | 55 passed | `test_sft_masking.py`는 이미 만들어진 `labels`에서 `loss_mask`를 뽑는 다른 단계만 검사, dataloader 테스트는 FIM 전용 |
| DPO 페어 정렬 | 48 passed | `_collate_dpo`를 실제 샘플로 부르는 테스트가 1개뿐이고 **chosen 1 + rejected 1** — 페어가 하나면 정렬이 순서를 못 바꾼다 |
| LoRA 2차 저장 크래시 | 4 passed | `save_checkpoint`를 **한 번만** 호출 |
| offload `shutdown()` 누수 | 143 passed | 해제 여부를 검증하는 테스트 없음 |
| MoE config 크래시 | 87 passed | 테스트가 자체 최소 config를 만들어 **배포된 YAML을 쓰지 않음** |
| Qwen config 로드 불가 | — | `configs/`의 config를 파싱조차 해보는 테스트 없음 |
| GRPO IS-ratio / reward 예외 | 24 passed | 메트릭 키의 **존재**만 확인하고 값은 보지 않음; reward 예외 경로 미검사 |
| GRPO e2e | — | `test_grpo_smoke.py:26`이 `--nproc_per_node=2` 하드코딩, 단일 GPU에서 전부 skip |

→ **1단계 로드맵의 "재현 테스트 먼저"는 선택이 아니라 전제다.** 특히 다음 세 가지는 즉시 도입 가치가 높다.
1. `configs/**/*.yaml` 전체를 파싱만 해보는 테이블 테스트 (Qwen·MoE 결함을 즉시 검출)
2. collator를 **다중 샘플·다중 페어**로 호출하는 테스트 (SFT·DPO 결함 검출)
3. 저장→재개→재저장을 반복하는 체크포인트 테스트 (LoRA 결함·원자성·RNG 검출)

### 7.7 Strix Halo(통합메모리) 특이사항

offload는 GPU 추적 메모리를 11010→1765 MiB로 84% 줄였으나, 같은 실행에서 `host_rss`가
13297MB로 뛴다. 이 APU는 VRAM과 호스트 RAM이 **물리적으로 같은 DRAM 풀**이므로 물리 메모리는
전혀 절약되지 않으며, pinned 풀 오버헤드 탓에 총량은 오히려 baseline(11.4GB)보다 큰 15.3GB가
된다. **discrete GPU에서는 진짜 이득이지만 통합메모리 APU에서는 역효과**다.

성능 참고치: bf16 4096³ matmul 16.3 TFLOP/s, GPT-2 small 6~9k tok/s, Qwen2.5-0.5B 3.2k tok/s
(7.4 TFLOPS/s/GPU). `torch.compile`은 gfx1151에서 정상 동작하며 정상 상태 7~10% 향상(첫 스텝
82.7초 컴파일 후 2.5초). bf16 NaN은 시험 범위에서 관측되지 않았다.

---

## 부록 A. 감사 범위 및 신뢰도

- 4개 영역을 각각 전담 감사해 파일 정독 기반으로 도출. 최상위 P0/P1 6건(LR 스케줄러 클래스 대입, 데이터 샤딩 RNG 스킵, eval dropout, DistributedOptimizer broadcast src, 원자적 저장 부재, DPO 페어링)은 실제 코드로 재확인함.
- 나머지 항목은 감사자 정독 결과이며, **수정 전 재현 테스트로 확정**하는 것을 권장(특히 분산 경로는 멀티 GPU 환경 재현 필요).
- 보안: reward 시스템은 모델 출력에 `eval`/`exec`/`subprocess`를 실행하지 않음(코드 리워드는 샌드박스를 요구하며 `NotImplementedError`로 안전). 유일한 잔여 리스크는 서드파티 LLM judge로의 프롬프트 인젝션(본질적).
