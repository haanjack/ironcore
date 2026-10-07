import subprocess,sys,json,shutil,hashlib,time
from pathlib import Path
root=Path('/home/hanjack/ironcore/ironcore')
previous=Path('/tmp/ironcore-remediation-followup-v3/study.json')
while not previous.exists() or json.loads(previous.read_text())['status']=='running':time.sleep(2)
source=Path('/tmp/ironcore-remediation-final-v4-src');source.mkdir()
for name in ['ironcore','scripts','configs']:
 shutil.copytree(root/name,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(source/'source_hashes.json').write_text(json.dumps({str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()},indent=2)+'\n')
output=Path('/tmp/ironcore-remediation-final-precision');output.mkdir()
launch=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2']
jobs=[('final-fault-contracts',launch+[str(source/'scripts/validate_distributed_contracts.py'),'--output',str(output/'final-fault-contracts')]),
('grpo-bf16',[sys.executable,str(source/'scripts/validate_trainers.py'),'--device','cuda','--architecture','cs336','--precision','bfloat16','--steps','4','--tasks','grpo','--grpo-objective','grpo','--moe','--cases','full,dp2,tp2,ep2','--output',str(output/'grpo-bf16')]),
('grpo-consistent-precision',launch+[str(source/'scripts/benchmark_alignment.py'),'--task','grpo','--model-dir','/tmp/ironcore-smollm135','--gsm8k','/tmp/ironcore-gsm8k','--preferences','/tmp/ironcore-ultrafeedback','--output',str(output/'grpo-consistent-precision'),'--steps','8'])]
report={'status':'running','jobs':[]}
for label,cmd in jobs:
 print('Running',label,flush=True)
 with (output/(label+'.log')).open('w') as log:
  result=subprocess.run(cmd,cwd=source,stdout=log,stderr=log,timeout=1800)
 report['jobs'].append({'label':label,'command':cmd,'returncode':result.returncode});(output/'study.json').write_text(json.dumps(report,indent=2)+'\n')
 if result.returncode:
  report['status']='failed';(output/'study.json').write_text(json.dumps(report,indent=2)+'\n');raise RuntimeError(label)
report['status']='completed';(output/'study.json').write_text(json.dumps(report,indent=2)+'\n')
