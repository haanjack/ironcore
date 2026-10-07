import subprocess,sys,json,shutil,hashlib
from pathlib import Path
root=Path('/home/hanjack/ironcore/ironcore')
source=Path('/tmp/ironcore-remediation-distributed-v2-src');source.mkdir()
for name in ['ironcore','scripts','configs']:
 shutil.copytree(root/name,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(source/'source_hashes.json').write_text(json.dumps({str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()},indent=2)+'\n')
jobs=[('fsdp-alignment',['--fsdp','--cases','full,dp2','--tasks','sft,dpo,grpo']),
      ('distributed-optimizer',['--distributed-optimizer','--cases','full,dp2']),
      ('moe-ep-all',['--moe','--cases','full,ep2']),
      ('recomputed-ce',['--recompute-linear-ce','--loss-chunk-size','7','--tasks','pretrain,sft']),
      ('batched-moe',['--moe','--moe-backend','batched']),
      ('batched-moe-idle',['--moe','--moe-backend','batched','--moe-idle','--tasks','pretrain'])]
report={'status':'running','jobs':[]}
output=Path('/tmp/ironcore-remediation-distributed-v2');output.mkdir()
for label,args in jobs:
 cmd=[sys.executable,str(source/'scripts/validate_trainers.py'),'--device','cuda','--architecture','cs336','--steps','4','--output',str(output/label),*args]
 print('Running',label,flush=True)
 with (output/(label+'.log')).open('w') as log:
  result=subprocess.run(cmd,cwd=source,stdout=log,stderr=log,timeout=1200)
 record={'label':label,'command':cmd,'returncode':result.returncode}
 report['jobs'].append(record);(output/'study.json').write_text(json.dumps(report,indent=2)+'\n')
 if result.returncode:
  report['status']='failed';(output/'study.json').write_text(json.dumps(report,indent=2)+'\n');raise RuntimeError(label)
report['status']='completed';(output/'study.json').write_text(json.dumps(report,indent=2)+'\n')
