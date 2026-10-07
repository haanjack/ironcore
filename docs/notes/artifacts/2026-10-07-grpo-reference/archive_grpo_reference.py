import hashlib
import importlib.metadata
import json
import shutil
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('/home/hanjack/ironcore/ironcore/docs/notes/artifacts/2026-10-07-grpo-reference')
labels = ['cpu-final2', 'dp2-v1', '135m-dp2-fp32-v1', '135m-dp2-bf16-v1',
          '135m-dp2-bf16-diag', '135m-hf-dp2-bf16-v1', 'hf-cpu-v3']
summary = {}
for label in labels:
    destination = root / label
    destination.mkdir(parents=True, exist_ok=True)
    source = Path('/tmp/ironcore-grpo-trl-' + label)
    reports = []
    for file in source.glob('rank*.json'):
        shutil.copy2(file, destination / file.name)
        report = json.loads(file.read_text())
        reports.append({'rank': file.stem, 'status': report['status'],
                        'parameters': report['parameters'], 'updates': report['optimizer_updates'],
                        'script_sha256': report['experiment_script_sha256'],
                        'errors_max': {k: max(x[k] for x in report['errors']) for k in [
                            'loss_abs_error', 'gradient_max_abs_error', 'weights_max_abs_error']},
                        'failed_checks': len(report.get('failures', [])),
                        'candidate_scores': report['candidate_scores']})
    shutil.copy2(Path('/tmp/ironcore-grpo-trl-' + label + '.log'), destination / 'run.log')
    summary[label] = reports or [{'status': 'failed_before_summary', 'see': 'run.log'}]
for label, source in [('initial', Path('/tmp/ironcore-grpo-trl-reference-src')),
                      ('final', Path('/tmp/ironcore-grpo-trl-reference-final-src'))]:
    destination = root / 'source' / label
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / 'source_hashes.json', destination / 'source_hashes.json')
    with tarfile.open(destination / 'source.tar.gz', 'w:gz') as archive:
        archive.add(source, arcname=source.name, filter=lambda info: None if '__pycache__' in info.name or info.name.endswith('.pyc') else info)
    shutil.copy2(source / 'scripts/validate_grpo_reference.py', destination / 'validate_grpo_reference.py')
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(root / 'runtime.json').write_text(json.dumps({
    'packages': {name: importlib.metadata.version(name) for name in [
        'torch','transformers','trl','accelerate','datasets','triton','wandb','ruff']},
    'production_base_commit': 'aaec6b925e47ef9257c5b819ae0b3853fbd7badf',
    'reference_docs': 'https://huggingface.co/docs/trl/v0.29.0/en/grpo_trainer',
    'optional_dependency': 'trl==0.29.0',
    'environment': {'HF_HOME': '/tmp/ironcore-hf', 'TIKTOKEN_CACHE_DIR': '/tmp/ironcore-tiktoken',
                    'WANDB_MODE':'disabled', 'HF_HUB_DISABLE_PROGRESS_BARS':'1','OMP_NUM_THREADS':'2'},
    'gpu_scope': '2 x RTX 3090, CUDA_VISIBLE_DEVICES=0,1; TITAN RTX unused',
    'bootstrap': 'Validation venv preimports triton and tensorboard.compat.notf; installed current wandb in venv to resolve broken host optional wandb import. Both trainers set report_to=[]/no external reporting.',
    'limits': 'Shared scripted generation; token GRPO, two updates per rollout; BF16 strict multi-update gates failed.'
}, indent=2) + '\n')
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, label, title in zip(axes, ['cpu-final2','135m-dp2-fp32-v1'], ['76K, CPU / 24 updates','135M, DP=2 / 8 updates']):
    report = json.loads((root / label / 'rank0.json').read_text())
    for trainer, style in [('ironcore','-'),('trl','--')]:
        ys = report['losses'][trainer]
        ax.plot(range(1,len(ys)+1),ys,style,label=trainer,linewidth=1.7)
    ax.set(title=title,xlabel='Optimizer update',ylabel='GRPO fixture loss')
    ax.grid(alpha=.2);ax.legend()
fig.tight_layout()
fig.savefig(root/'fp32_loss_trajectories.png',dpi=160)
fig.savefig(root/'fp32_loss_trajectories.svg')
plt.close(fig)
fig, ax = plt.subplots(figsize=(7,3.6))
for label,title in [('135m-dp2-fp32-v1','FP32 Native vs TRL'),('135m-dp2-bf16-diag','BF16 Native vs TRL'),('135m-hf-dp2-bf16-v1','BF16 same HF backbone')]:
    report=json.loads((root/label/'rank0.json').read_text())
    ax.semilogy([x['optimizer_step'] for x in report['errors']],
                [max(1e-12,x['gradient_max_abs_error']) for x in report['errors']],'-o',label=title,markersize=3)
ax.set(xlabel='Optimizer update',ylabel='Max clipped-gradient absolute difference',title='135M, DP=2: observed reference discrepancies')
ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(root/'precision_discrepancy.png',dpi=160);fig.savefig(root/'precision_discrepancy.svg');plt.close(fig)
shutil.copy2(Path(__file__), root / Path(__file__).name)
manifest={str(p.relative_to(root)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
          for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json'}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print({'archived_and_hashed_files':len(manifest),'cases':len(summary)})
