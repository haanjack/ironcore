# 2026-10-07 실험 근거

이 디렉터리는 GPU 실측 결과의 작은 artifact만 보존한다. 전체 해석은 [노트 index](../../README.md)와 각 단계 문서를 따른다.

| 파일 | 의미 |
|---|---|
| `study.json` | 100-step 실제 텍스트 학습 및 model/batch/context sweep의 명령·rank별 설정·원시 기록 |
| `measurements.csv` | 성공한 성능 run의 처리량, rank 중 최대 allocated/reserved, held-out NLL |
| `130m_matched_*.json` | 같은 micro/global batch의 compile 대조 |
| `*chunk4096*.json` 및 failure log | chunked CE 성공 결과와 130m microbatch 32 OOM |
| `ironcore-validation-*/report.json` | FP32/BF16, full/accum/DP/TP, 재개 및 online GRPO gate |
| `ironcore-validation-*/rank_summaries.json` | rank별 loss/evaluation/weight 변화, 메모리 요약; model tensors 제외 |
| `regressions.log`, `profiler_mfu.log` | 각각 233 passed/11 skipped, 69 passed의 pytest 출력 |
| `cli_smoke.json`, `cli_*_smoke.log` | 실제 2-GPU IronCore CLI의 4-step 실행, 변경한 smoke 설정 |
| `corpus_manifest.json` | TinyStories source URL, byte/token 수, raw/token SHA-256; revision null |
| `hardware.txt`, `*_telemetry.csv` | 실제 GPU/NVLink/전력 제한 및 run 중 sampled nvidia-smi 기록 |
| `profile_kernels.json`, `profile_key_averages.csv` | 원본 Chrome trace hash와 kernel 누적 duration, 수정 후 CSV |
| `*_oom.log`, `flash_extension_failure.log`, `tp_resume_initial_failure.log` | 용량/ABI/수정 전 TP Adam 재개 실패 증거 |
| `source_hashes.json` | 실험 종료 후 최종 working tree 파일 SHA-256; run 시작별 commit snapshot은 아님 |
| `course_pdf_hashes.json` | 확인한 CS336 PDF들의 SHA-256; 원문은 index의 공식 링크 참조 |
| `environment_sitecustomize.txt`, `followup_commands.txt` | 이번 환경의 native import workaround 및 추가 검증/최적화 명령 |
| `*.png`, `*.svg` | 블로그에 쓸 수 있는 standalone figure; smoothing 없음 |
| `artifact_hashes.json` | 이 파일과 hash manifest 자체를 제외한 보존 artifact의 무결성 hash |

원본 `study.json`의 최상위 상태는 **failed**다. 마지막 선택적 FlashAttention extension 비교가 ABI 불일치로 실패했기 때문이다. 원본 상태를 completed로 덮어쓰지 않았으며, 앞선 job의 `completed`/`oom` 상태를 각각 판단한다. follow-up의 compile/chunking/profile은 별도 JSON이며 130m microbatch 32 chunking의 OOM도 실패 로그로 보존했다.

각 steady-state sweep은 20 steps 중 첫 10 steps를 제외한 짧은 단일 반복이다. learning pilot은 100 steps, single seed이고 DP=2/single의 평가 token 수는 다르다. 해당 한계와 130m loss spike를 노트에 명시했다. toy alignment의 높은 reward/preference accuracy를 자연어 추론 성과로 해석하지 않는다.

큰 checkpoint tensors, corpus 전문과 token files, Chrome trace 원본은 repository에 넣지 않았다. trace hash와 kernel 요약은 보존하지만 이 요약만으로 timeline/CPU idle을 다시 분석할 수는 없다. `/tmp/ironcore-*` 원본은 이번 세션의 임시 위치이므로 장기 보관 시 별도로 복사해야 한다.
