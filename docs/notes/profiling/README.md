# Compute・collective profiling artifacts

[HTML 보고서](profile_report.html)를 브라우저에서 직접 연다. HTML 안에 세 구성의 두 rank GPU timeline과 요약이 포함돼 서버/네트워크 연결이 필요 없다. 정의·해석·측정 조건은 [09 노트](../09-profiler-html.md)를 따른다.

- `profile_summary.json`: device interval union, category/phase 누적 시간, NCCL kernels/CPU calls/logical payload, 원본 trace SHA-256, 실제 capture update 13·14.
- `study.json`, `commands.py.txt`: 세 20-step 실제 GPU 실행 결과와 명령. exit code 모두 0.
- `regressions.log`: profiler/report/MFU CPU 회귀 80 passed.
- `browser_validation.json`, `browser_validation.py.txt`: desktop/mobile Chromium의 run/rank 선택, 검색·필터·확대·이동·tooltip, JS errors/외부 요청 검사.
- `source_hashes.json`: 보고서 완료 후 코드 snapshot. 각 run 시작의 immutable snapshot은 아니다.
- `artifact_hashes.json`: README와 manifest 자체를 제외한 이 폴더 파일의 SHA-256.

Raw Chrome traces 약 1.8 GB는 JSON에 표시한 `/tmp/ironcore-html-profiles-fixed/` 아래 임시 파일이며 저장소에는 포함하지 않았다. HTML의 compact GPU events로 CUDA timeline을 탐색할 수 있지만 전체 CPU trace, Python stack, memory lifetime을 복원할 수는 없다.

Steady tokens/s는 별도 non-profiled 실험의 값이다. Profile tokens/s와 peak VRAM은 instrumentation overhead를 포함한다. Interval union과 event duration 합, logical tensor bytes와 실제 wire bytes를 구분한다. EP=2는 수치 실패 때문에 trainer에서 거부되며 이 보고서의 MoE는 EP=1이다.
