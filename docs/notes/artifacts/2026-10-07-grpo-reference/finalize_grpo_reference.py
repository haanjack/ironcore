import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/home/hanjack/ironcore/ironcore/docs/notes/artifacts/2026-10-07-grpo-reference')
source=Path('/tmp/ironcore-grpo-reference-postfix')
dest=root/'postfix';dest.mkdir(exist_ok=True)
for p in source.rglob('*'):
    if not p.is_file() or p.suffix not in ['.json','.log']:continue
    if any(part in ['checkpoint','tokenizer','profile'] for part in p.relative_to(source).parts):continue
    out=dest/p.relative_to(source);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
snapshot=Path('/tmp/ironcore-grpo-trl-shared-logprob-src')
src_dest=root/'source/postfix';src_dest.mkdir(parents=True,exist_ok=True)
shutil.copy2(snapshot/'source_hashes.json',src_dest/'source_hashes.json')
with tarfile.open(src_dest/'source.tar.gz','w:gz') as archive:
    archive.add(snapshot,arcname=snapshot.name,filter=lambda info:None if '__pycache__' in info.name or info.name.endswith('.pyc') else info)
shutil.copy2(snapshot/'scripts/validate_grpo_reference.py',src_dest/'validate_grpo_reference.py')
for source,name in [('/tmp/grpo-gradient-probe-before.json','gradient_probe_before.json'),
                    ('/tmp/grpo-gradient-probe-after.json','gradient_probe_after.json'),
                    ('/tmp/gradient_probe.py','gradient_probe.py'),
                    ('/tmp/ironcore-grpo-final-ci-cpu.log','postfix/cpu_regression.log'),
                    ('/tmp/ironcore-grpo-shared-logprob-tests.log','postfix/focused_regression.log'),
                    ('/tmp/run_grpo_reference_postfix.py','postfix/run_study.py'),
                    ('/tmp/ironcore-grpo-reference-postfix.log','postfix/driver.log')]:
    shutil.copy2(source,root/name)
summary=json.loads((root/'summary.json').read_text())
summary['final_decision']={
    'state':'complete_for_documented_contract',
    'required_code_fixes':'completed: reuse one policy/KL log-probability graph when unwarped',
    'remaining_mandatory_fixes':[],
    'native_135m_fp32_reference_equivalence':'passed',
    'same_hf_backbone_135m_bf16_reference_equivalence':'passed_after_fix',
    'native_vs_hf_bf16_backbone_equivalence':'not_provided_by_this_training_contract',
    'operational_rule':'Use Native FP32 for external numerical equivalence. Native BF16 uses its own validated DP/TP/restart contract; do not promise equality with a different HF backbone.',
    'reasoning_quality':'No improvement demonstrated; previous GSM8k paired reward 0 to 0 remains',
}
summary['postfix']={}
for label in ['native-fp32','native-bf16','hf-bf16']:
    records=[]
    for p in sorted((dest/label).glob('rank*.json')):
        r=json.loads(p.read_text());records.append({'rank':p.stem,'status':r['status'],
            'errors_max':{key:max(e[key] for e in r['errors']) for key in ['loss_abs_error','gradient_max_abs_error','weights_max_abs_error']},
            'failed_checks':len(r['failures'])})
    summary['postfix'][label]=records
matrix=json.loads((dest/'native-moe-bf16-matrix/report.json').read_text())
assert matrix['status']=='passed'
assert all(v['status']=='passed' for key in ['native-fp32','hf-bf16'] for v in summary['postfix'][key])
summary['postfix']['native-moe-bf16-matrix']={'status':matrix['status'],
    'exact_resume_errors':[r['resume_max_abs_error'] for r in matrix['checks'] if 'resume_max_abs_error' in r]}
summary['postfix']['cpu_regression']={'passed':692,'skipped':30,'deselected':230}
summary['postfix']['focused_regression']={'passed':96,'skipped':2}
summary['postfix']['gradient_probe']={
    'before':json.loads((root/'gradient_probe_before.json').read_text()),
    'after':json.loads((root/'gradient_probe_after.json').read_text())}
assert summary['postfix']['gradient_probe']['after']['mismatched_gradient_elements']==0
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
runtime=json.loads((root/'runtime.json').read_text())
runtime['postfix_source']='source/postfix/source.tar.gz'
runtime['postfix_regressions']=summary['postfix']['cpu_regression']
runtime['limits']='Shared scripted generation; token GRPO, two updates per rollout. Native FP32 and same-HF-backbone BF16 gates pass after fix. Cross-backbone BF16 equivalence is not provided.'
(root/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(10,3.6))
for ax,p,title in zip(axes,[root/'cpu-final2/rank0.json',dest/'native-fp32/rank0.json'],['76K FP32, CPU / 24 updates (initial)','135M FP32, DP=2 / 8 updates (after fix)']):
    r=json.loads(p.read_text())
    for trainer,style in [('ironcore','-'),('trl','--')]:
        ys=r['losses'][trainer];ax.plot(range(1,len(ys)+1),ys,style,label=trainer,linewidth=1.7)
    ax.set(title=title,xlabel='Optimizer update',ylabel='GRPO fixture loss');ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(root/'fp32_loss_trajectories.png',dpi=160);fig.savefig(root/'fp32_loss_trajectories.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,3.6))
for p,title in [(dest/'native-fp32/rank0.json','Native/HF FP32 after fix'),
                (root/'135m-hf-dp2-bf16-v1/rank0.json','Same HF/BF16 before fix'),
                (dest/'hf-bf16/rank0.json','Same HF/BF16 after fix'),
                (dest/'native-bf16/rank0.json','Different Native/HF BF16 after fix')]:
    r=json.loads(p.read_text());ax.semilogy([e['optimizer_step'] for e in r['errors']],
        [max(1e-12,e['gradient_max_abs_error']) for e in r['errors']],'-o',label=title,markersize=3)
ax.set(xlabel='Optimizer update',ylabel='Max clipped-gradient absolute difference',title='135M, DP=2: before/after and backbone scope')
ax.grid(alpha=.2);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(root/'precision_discrepancy.png',dpi=160);fig.savefig(root/'precision_discrepancy.svg');plt.close(fig)
shutil.copy2(__file__,root/Path(__file__).name)
manifest={str(p.relative_to(root)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
          for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json'}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print({'verified_files':len(manifest),'final_decision':summary['final_decision']['state']})
