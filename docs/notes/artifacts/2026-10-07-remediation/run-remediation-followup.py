import subprocess,sys,json,shutil,hashlib,time
from pathlib import Path
root=Path('/home/hanjack/ironcore/ironcore')
previous=Path('/tmp/ironcore-remediation-distributed-v3/study.json')
while json.loads(previous.read_text())['status']=='running':time.sleep(2)
if json.loads(previous.read_text())['status']!='completed':raise RuntimeError('Previous study failed')
source=Path('/tmp/ironcore-remediation-final-src');source.mkdir()
for name in ['ironcore','scripts','configs']:
 shutil.copytree(root/name,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(source/'source_hashes.json').write_text(json.dumps({str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()},indent=2)+'\n')
output=Path('/tmp/ironcore-remediation-followup');output.mkdir()
jobs=[]
launch=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2']
jobs.append(('fault-contracts',launch+[str(source/'scripts/validate_distributed_contracts.py'),'--output',str(output/'fault-contracts')]))
for task,steps in [('sft',32),('dpo',24),('grpo',8)]:
 jobs.append((f'alignment-{task}',launch+[str(source/'scripts/benchmark_alignment.py'),'--task',task,'--model-dir','/tmp/ironcore-smollm135','--gsm8k','/tmp/ironcore-gsm8k','--preferences','/tmp/ironcore-ultrafeedback','--output',str(output/f'alignment-{task}'),'--steps',str(steps)]))
for trial in range(3):
 for label,args in [('dense-full',['--model-size','130m']),('dense-recomputed',['--model-size','130m','--recompute-linear-ce','--loss-chunk-size','256']),('moe-loop',['--model-size','50m','--moe','--moe-backend','loop']),('moe-batched',['--model-size','50m','--moe','--moe-backend','batched'])]:
  label=f'{label}-trial{trial}';ctx='2048' if label.startswith('dense') else '1024'
  jobs.append((label,launch+[str(source/'scripts/benchmark_training.py'),'--data-dir','/tmp/ironcore-corpus','--output',str(output/label),'--context',ctx,'--micro-batch','2','--global-batch','8','--steps','35','--warmup','10','--eval-batches','4',*args]))
for label,args in [('dense-recomputed-profile',['--model-size','130m','--recompute-linear-ce','--loss-chunk-size','256']),('moe-loop-profile',['--model-size','50m','--moe','--moe-backend','loop']),('moe-batched-profile',['--model-size','50m','--moe','--moe-backend','batched'])]:
 ctx='2048' if label.startswith('dense') else '1024'
 jobs.append((label,launch+[str(source/'scripts/benchmark_training.py'),'--data-dir','/tmp/ironcore-corpus','--output',str(output/label),'--context',ctx,'--micro-batch','2','--global-batch','8','--steps','16','--warmup','10','--eval-batches','2','--profile','--profile-ranks','0,1',*args]))
for label,args in [('moe-ep-large',['--model-size','50m','--moe','--ep','2']),('dense-fsdp',['--model-size','130m','--fsdp']),('dense-distributed-optimizer',['--model-size','130m','--distributed-optimizer']),('dense-compile',['--model-size','130m','--compile'])]:
 jobs.append((label,launch+[str(source/'scripts/benchmark_training.py'),'--data-dir','/tmp/ironcore-corpus','--output',str(output/label),'--context','1024','--micro-batch','2','--global-batch','8','--steps','35','--warmup','10','--eval-batches','4',*args]))
for label,args in [('final-grpo-token',['--tasks','grpo','--moe','--grpo-objective','grpo','--cases','full,dp2,tp2,ep2']),('final-fsdp-bf16',['--fsdp','--precision','bfloat16','--cases','full,dp2']),('final-distopt-bf16',['--distributed-optimizer','--precision','bfloat16','--cases','full,dp2'])]:
 jobs.append((label,[sys.executable,str(source/'scripts/validate_trainers.py'),'--device','cuda','--architecture','cs336','--steps','4','--output',str(output/label),*args]))
report={'status':'running','jobs':[]}
for label,cmd in jobs:
 print('Running',label,flush=True)
 with (output/(label+'.log')).open('w') as log:
  result=subprocess.run(cmd,cwd=source,stdout=log,stderr=log,timeout=1800)
 report['jobs'].append({'label':label,'command':cmd,'returncode':result.returncode});(output/'study.json').write_text(json.dumps(report,indent=2)+'\n')
 if result.returncode:
  report['status']='failed';(output/'study.json').write_text(json.dumps(report,indent=2)+'\n');raise RuntimeError(label)
report['status']='completed';(output/'study.json').write_text(json.dumps(report,indent=2)+'\n')
