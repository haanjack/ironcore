import json,statistics,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/tmp/ironcore-remediation-followup-v3')
out=Path('/home/hanjack/ironcore/ironcore/docs/notes/artifacts/2026-10-07-remediation')
rows=[]
for name in ['dense-full','dense-recomputed','moe-loop','moe-batched']:
 runs=[json.loads(file.read_text()) for file in sorted(root.glob(name+'-trial*/rank0.json'))]
 if len(runs)!=3:raise ValueError(name)
 timings=[run['mean_step_seconds'] for run in runs]
 tps=runs[0]['global_batch']*runs[0]['context']/statistics.mean(timings)
 peaks=[max(json.loads(file.read_text())['peak_allocated_bytes'] for file in (root/f'{name}-trial{i}').glob('rank*.json')) for i in range(3)]
 rows.append({'workload':name,'parameters':runs[0]['parameters_full'],'context':runs[0]['context'],'global_batch':runs[0]['global_batch'],'micro_batch':runs[0]['micro_batch'],
 'mean_step_seconds':statistics.mean(timings),'tokens_per_second':tps,'trial_min_tps':min(run['global_tokens_per_second'] for run in runs),'trial_max_tps':max(run['global_tokens_per_second'] for run in runs),'peak_allocated_gib':max(peaks)/2**30,'repeats':3,'warmup_excluded':10,'measured_steps_per_repeat':25})
with (out/'performance_comparison.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
(out/'performance_comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
for label,group in [('ce',rows[:2]),('moe',rows[2:])]:
 fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
 names=['Full CE','Recomputed CE'] if label=='ce' else ['Loop experts','Batched experts']
 values=[r['tokens_per_second']/1000 for r in group]
 errors=[[v-r['trial_min_tps']/1000 for v,r in zip(values,group)],[r['trial_max_tps']/1000-v for v,r in zip(values,group)]]
 axes[0].bar(names,values,yerr=errors,capsize=5,color=['#547a99','#258c80'])
 axes[0].set_ylabel('Global throughput (k tokens/s)');axes[0].set_ylim(0,max(values)*1.2)
 for i,v in enumerate(values):axes[0].text(i,v+max(values)*.04,f'{v:.1f}',ha='center')
 memory=[r['peak_allocated_gib'] for r in group]
 axes[1].bar(names,memory,color=['#547a99','#258c80']);axes[1].set_ylabel('Peak allocated per GPU (GiB)');axes[1].set_ylim(0,max(memory)*1.2)
 for i,v in enumerate(memory):axes[1].text(i,v+max(memory)*.04,f'{v:.2f}',ha='center')
 fig.suptitle(f'RTX3090 x2, BF16 compute / FP32 weights, context={group[0]["context"]}\n3 timing repeats; 25 measured updates each (10 warmup excluded)')
 for suffix in ['png','svg']:fig.savefig(out/f'{label}_ablation.{suffix}',dpi=180)
 plt.close(fig)
print(json.dumps(rows,indent=2))
