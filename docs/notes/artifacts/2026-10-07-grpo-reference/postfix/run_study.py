import json
import os
import subprocess
import sys
from pathlib import Path

source = Path('/tmp/ironcore-grpo-trl-shared-logprob-src')
out = Path('/tmp/ironcore-grpo-reference-postfix')
out.mkdir()
launch = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
base = [str(source / 'scripts/validate_grpo_reference.py'), '--device', 'cuda',
        '--model-dir', '/tmp/ironcore-smollm135', '--rollouts', '4', '--lr', '1e-5']
jobs = [
    ('hf-bf16', launch + base + ['--backbone', 'hf', '--precision', 'bfloat16', '--output', str(out/'hf-bf16')]),
    ('native-bf16', launch + base + ['--precision', 'bfloat16', '--output', str(out/'native-bf16')]),
    ('native-fp32', launch + base + ['--output', str(out/'native-fp32')]),
    ('native-moe-bf16-matrix', [sys.executable, str(source/'scripts/validate_trainers.py'),
                              '--device','cuda','--architecture','cs336','--precision','bfloat16',
                              '--parameter-precision','float32','--grpo-objective','grpo','--moe',
                              '--steps','4','--tasks','grpo','--cases','full,dp2,tp2',
                              '--output',str(out/'native-moe-bf16-matrix')]),
]
study = {'status': 'running', 'jobs': []}
for label, cmd in jobs:
    print('Running', label, flush=True)
    with (out / (label + '.log')).open('w') as log:
        run = subprocess.run(cmd, cwd=source, stdout=log, stderr=log, timeout=1200)
    entry = {'label': label, 'command': cmd, 'returncode': run.returncode}
    if 'matrix' not in label:
        reports = list((out/label).glob('rank*.json'))
        if len(reports) != 2:
            raise RuntimeError('Execution failed before recording results: ' + label)
        entry['comparison_status'] = [json.loads(p.read_text())['status'] for p in reports]
    elif run.returncode:
        raise RuntimeError('Regression failed: ' + label)
    study['jobs'].append(entry)
    (out/'study.json').write_text(json.dumps(study,indent=2)+'\n')
study['status']='completed'
(out/'study.json').write_text(json.dumps(study,indent=2)+'\n')
