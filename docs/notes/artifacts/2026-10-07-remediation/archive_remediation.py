import hashlib,json,shutil
from pathlib import Path
root=Path('/home/hanjack/ironcore/ironcore/docs/notes/artifacts/2026-10-07-remediation')
root.mkdir(exist_ok=True)
sources={
'ad-bf16':'/tmp/ironcore-remediation-ad-bf16',
'ep-layer':'/tmp/ironcore-remediation-ep-layer',
'ep-pretrain':'/tmp/ironcore-remediation-ep-pretrain',
'fsdp-pretrain':'/tmp/ironcore-remediation-fsdp-pretrain',
'failed-fsdp-reference':'/tmp/ironcore-remediation-distributed',
'fsdp-fixed-and-failed-distopt':'/tmp/ironcore-remediation-distributed-v2',
'distributed-correctness':'/tmp/ironcore-remediation-distributed-v3',
'failed-alignment-template':'/tmp/ironcore-remediation-followup',
'alignment-sft-dpo':'/tmp/ironcore-remediation-followup-v2',
'followup':'/tmp/ironcore-remediation-followup-v3',
'final-precision':'/tmp/ironcore-remediation-final-precision',
'final-bf16-and-failed-worker-cli':'/tmp/ironcore-remediation-final-validation',
'final-training':'/tmp/ironcore-remediation-final-validation-v2',
'final-evaluation':'/tmp/ironcore-remediation-final-evaluation'}
manifest={}
for label,source in sources.items():
 source=Path(source)
 if not source.exists():continue
 dest=root/label;dest.mkdir(exist_ok=True)
 for file in source.rglob('*'):
  if not file.is_file() or file.suffix not in ['.json','.log','.csv']:continue
  if any(part in ['tokenizer','checkpoint','profile'] for part in file.relative_to(source).parts):continue
  out=dest/file.relative_to(source);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,out)
  manifest[str(out.relative_to(root))]={'original_path':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'bytes':file.stat().st_size}
 for snapshot in Path('/tmp').glob('ironcore-remediation*-src/source_hashes.json'):
  out=root/'sources'/snapshot.parent.name/'source_hashes.json';out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(snapshot,out)
  manifest[str(out.relative_to(root))]={'original_path':str(snapshot),'sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),'bytes':snapshot.stat().st_size}
for file in Path('/tmp').glob('ironcore-remediation*.log'):
 out=root/'logs'/file.name;out.parent.mkdir(exist_ok=True);shutil.copy2(file,out)
 manifest[str(out.relative_to(root))]={'original_path':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'bytes':file.stat().st_size}
for file in [Path('/tmp/ironcore-smollm-native-oracle.json'),Path('/tmp/validate-smollm-native.py'),Path('/tmp/ironcore-smollm-native-oracle-v3.log')]:
 out=root/file.name;shutil.copy2(file,out);manifest[file.name]={'original_path':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'bytes':file.stat().st_size}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('archived files',len(manifest))
