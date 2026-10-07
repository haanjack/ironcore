import hashlib
import importlib.metadata
import json
import platform
import shutil
import tarfile
from pathlib import Path

repo = Path('/home/hanjack/ironcore/ironcore')
out = repo / 'docs/notes/artifacts/2026-10-07-remediation'
source = Path('/tmp/ironcore-remediation-final-v8-src')
source.mkdir(exist_ok=True)
for name in ['ironcore', 'scripts', 'configs', 'tests']:
    shutil.copytree(repo / name, source / name, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
shutil.copy2(repo / 'pyproject.toml', source / 'pyproject.toml')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(source / 'source_hashes.json').write_text(json.dumps({
    str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*'))
    if p.is_file() and p.name != 'source_hashes.json'}, indent=2) + '\n')
for snapshot in sorted(Path('/tmp').glob('ironcore-remediation*-src')):
    if not (snapshot / 'source_hashes.json').exists():
        continue
    destination = out / 'sources' / snapshot.name
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot / 'source_hashes.json', destination / 'source_hashes.json')
    with tarfile.open(destination / 'source.tar.gz', 'w:gz') as archive:
        archive.add(snapshot, arcname=snapshot.name,
                    filter=lambda info: None if '__pycache__' in info.name or info.name.endswith('.pyc') else info)
shutil.copytree('/tmp/ironcore-remediation-final-eval-contract', out / 'final-eval-contract', dirs_exist_ok=True)
extras = list(Path('/tmp').glob('ironcore-remediation-preset-*_rank*.json'))
extras += [Path('/tmp/ironcore-smollm-native-oracle-final.json'),
           Path('/tmp/ironcore-smollm-native-oracle-final.log'),
           Path('/tmp/validate-smollm-native-final.py'),
           Path('/tmp/smoke-remediation-presets.py'),
           Path('/tmp/check-remediation-html.py'),
           Path('/tmp/summarize-remediation-performance.py'),
           Path('/tmp/archive_remediation.py'), Path('/tmp/ironcore-validation-env/lib/python3.12/site-packages/sitecustomize.py'), Path(__file__)]
extras += list(Path('/tmp').glob('run-remediation*.py'))
for file in extras:
    shutil.copy2(file, out / file.name)
for file in Path('/tmp').glob('ironcore-remediation*.log'):
    shutil.copy2(file, out / 'logs' / file.name)
runtime = {'python': platform.python_version(), 'packages': {
    name: importlib.metadata.version(name) for name in [
        'torch', 'triton', 'transformers', 'torchdata', 'numpy', 'pyarrow',
        'huggingface-hub', 'matplotlib', 'tiktoken', 'pytest', 'playwright']},
    'gpu_scope': 'CUDA_VISIBLE_DEVICES=0,1; 2 x RTX 3090; NVLink NV4; 240 W power limit',
    'cpu_regression_command': "CUDA_VISIBLE_DEVICES='' HF_HOME=/tmp/ironcore-hf TIKTOKEN_CACHE_DIR=/tmp/ironcore-tiktoken MPLCONFIGDIR=/tmp/ironcore-mpl /tmp/ironcore-validation-env/bin/python -m pytest tests/unit/optimizer tests/unit/offload/test_config.py tests/unit/offload/test_system_info.py tests/unit/trainers tests/unit/alignment tests/unit/dataloader tests/unit/moe tests/unit/checkpointing tests/unit/parallel tests/unit/attention tests/unit/reward tests/unit/test_sft_masking.py tests/unit/test_config.py tests/unit/test_fable_regression.py tests/unit/peft/test_merge_lora_weights.py -m 'not cuda and not mp and not hf_hub and not e2e' -o addopts='' -q",
    'final_contract_command': 'CUDA_VISIBLE_DEVICES=0,1 /tmp/ironcore-validation-env/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 scripts/validate_distributed_contracts.py --output /tmp/ironcore-remediation-final-eval-contract',
    'final_source': 'sources/' + source.name + '/source.tar.gz',
    'runtime_bootstrap': 'The validation venv sitecustomize preimports triton and installs tensorboard.compat.notf to avoid host TensorFlow import; runtime is isolated from system Python.'}
(out / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
reports = {}
for label in ['ad-bf16', 'ep-layer', 'ep-pretrain', 'fsdp-pretrain',
              'distributed-correctness', 'final-training', 'final-evaluation']:
    for file in sorted((out / label).rglob('report.json')):
        reports[str(file.relative_to(out))] = json.loads(file.read_text())['status']
assert all(status == 'passed' for status in reports.values())
pilots = {}
for task, file in {
    'sft': out / 'alignment-sft-dpo/sft-real/result.json',
    'dpo': out / 'alignment-sft-dpo/dpo-real/result.json',
    'grpo_paired': out / 'final-evaluation/grpo-paired-evaluation/result.json',
}.items():
    if not file.exists():
        # Earlier pilot labels are preserved in study.json.
        candidates = list((out / 'alignment-sft-dpo').rglob('result.json'))
        file = next(p for p in candidates if json.loads(p.read_text()).get('task') == task)
    result = json.loads(file.read_text())
    assert result['status'] == 'completed'
    pilots[task] = {k: result[k] for k in ['before', 'after', 'steps', 'parameters', 'status']}
browser = json.loads((out / 'browser_check.json').read_text())
summary = {'scope': 'Stages A-I: implementation and bounded CPU/CUDA experiments, not full CS336 curriculum or long-horizon quality proof',
           'source_snapshot': runtime['final_source'], 'passed_reports': reports,
           'cpu_regression': {'passed': 532, 'skipped': 12, 'deselected': 42},
           'profiler_mfu_regression': {'passed': 80},
           'final_distributed_contract': {f'rank{rank}': json.loads((out / f'final-eval-contract/result_rank{rank}.json').read_text()) for rank in [0, 1]},
           'alignment_pilots': pilots,
           'performance_three_repeats': json.loads((out / 'performance_comparison.json').read_text()),
           'html_browser': browser,
           'limitations': ['GRPO paired held-out exact reward 0 to 0: no reasoning quality improvement demonstrated',
                           'FSDP/EP/optimizer combinations and same-topology resume support are bounded by stage 15',
                           'Negative intermediate runs are retained in separate directories',
                           'No absolute hardware optimum, full-corpus run or scaling-law proof']}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
old = json.loads((out / 'manifest.json').read_text())
manifest = {str(p.relative_to(out)): {
    **({'original_path': old[str(p.relative_to(out))]['original_path']} if 'original_path' in old.get(str(p.relative_to(out)), {}) else {}),
    'sha256': sha(p), 'bytes': p.stat().st_size}
    for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
for name, item in manifest.items():
    assert sha(out / name) == item['sha256']
print(json.dumps({'artifact_files_verified': len(manifest), 'passed_reports': len(reports),
                  'browser_checks': len(browser['checks']), 'source': source.name}))
