import subprocess,sys,json,shutil,hashlib,time
from pathlib import Path
root=Path('/home/hanjack/ironcore/ironcore')
previous=Path('/tmp/ironcore-remediation-final-precision/study.json')
while not previous.exists() or json.loads(previous.read_text())['status']=='running':time.sleep(2)
source=Path('/tmp/ironcore-remediation-final-v5-src');source.mkdir()
for name in ['ironcore','scripts','configs']:
 shutil.copytree(root/name,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(source/'source_hashes.json').write_text(json.dumps({str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()},indent=2)+'\n')
output=Path('/tmp/ironcore-remediation-final-validation');output.mkdir()
launch=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2']
jobs=[]
for label,args in [('ep-bf16',['--moe','--cases','full,ep2','--precision','bfloat16','--grpo-objective','grpo']),
 ('stateful-workers2',['--stateful-data','--data-workers','2','--cases','dp2','--tasks','pretrain,sft,grpo']),
 ('fsdp-bf16',['--fsdp','--precision','bfloat16','--cases','full,dp2']),
 ('distopt-bf16',['--distributed-optimizer','--precision','bfloat16','--cases','full,dp2']),
 ('grpo-token-bf16',['--moe','--tasks','grpo','--grpo-objective','grpo','--precision','bfloat16','--cases','full,dp2,tp2'])]:
 jobs.append((label,[sys.executable,str(source/'scripts/validate_trainers.py'),'--device','cuda','--architecture','cs336','--steps','4','--output',str(output/label),*args]))
jobs.append(('grpo-consistent-precision',launch+[str(source/'scripts/benchmark_alignment.py'),'--task','grpo','--model-dir','/tmp/ironcore-smollm135','--gsm8k','/tmp/ironcore-gsm8k','--preferences','/tmp/ironcore-ultrafeedback','--output',str(output/'grpo-consistent-precision'),'--steps','8']))
for label,args in [('moe-ep-large',['--model-size','50m','--moe','--ep','2']),('dense-fsdp',['--model-size','130m','--fsdp']),('dense-distributed-optimizer',['--model-size','130m','--distributed-optimizer']),('dense-baseline',['--model-size','130m']),('dense-compile',['--model-size','130m','--compile']),('dense-capacity',['--model-size','130m','--context','2048','--micro-batch','4','--global-batch','16']),('moe-capacity',['--model-size','50m','--moe','--moe-backend','batched','--context','2048','--micro-batch','4','--global-batch','16']),('dense-long-context',['--model-size','130m','--context','8192','--recompute-linear-ce','--loss-chunk-size','256'])]:
 jobs.append((label,launch+[str(source/'scripts/benchmark_training.py'),'--data-dir','/tmp/ironcore-corpus','--output',str(output/label),'--context','1024','--micro-batch','2','--global-batch','8','--steps','35','--warmup','10','--eval-batches','4',*args]))
report={'status':'running','jobs':[]}
for label,cmd in jobs:
 print('Running',label,flush=True)
 log_path=output/(label+'.log')
 with log_path.open('w') as log:
  result=subprocess.run(cmd,cwd=source,stdout=log,stderr=log,timeout=1800)
 record={'label':label,'command':cmd,'returncode':result.returncode}
 report['jobs'].append(record);(output/'study.json').write_text(json.dumps(report,indent=2)+'\n')
 if result.returncode:
  if 'out of memory' in log_path.read_text().lower() and label in ('dense-capacity','moe-capacity','dense-long-context'):
   record['status']='capacity_limit';continue
  report['status']='failed';(output/'study.json').write_text(json.dumps(report,indent=2)+'\n');raise RuntimeError(label)
report['status']='completed';(output/'study.json').write_text(json.dumps(report,indent=2)+'\n')
