"""Preregister exact-hardware continuation before examining any canary statistics."""
from pathlib import Path
import datetime,hashlib,json,re,shutil,subprocess
b=Path('/ssd/scxi253');p=b/'single-gpu-inference-selector-v42-20260930T230508Z';r=b/'sgi-v42-canary-rpc-recovery-20261001T0430Z-allocation-v3'
r.mkdir();(r/'receipts').mkdir();(r/'logs').mkdir();(r/'runs').mkdir();(r/'analysis').mkdir()
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
for n in ('source','overlays','cache','refs'):(r/n).symlink_to(p/n,target_is_directory=True)
for n in ('resume_measure.py','canary_resume.sbatch'):shutil.copyfile(b/n,r/n)
for f in (p/'receipts').iterdir():
 if f.is_file() and not f.name.startswith('kernel-pipeline') and f.name not in ('controller.lock',):(r/'receipts'/f.name).symlink_to(f)
for d in (p/'analysis').glob('dev-*'):(r/'analysis'/d.name).symlink_to(d,target_is_directory=True)
failures={}
for sh,j in [(6,1644194),(8,1644196),(9,1644197)]:
 log=p/f'logs/canary-{j}.out';s=log.read_text()
 assert 'Unable to confirm allocation' in s and 'Socket timed out on send/recv operation' in s
 assert (p/f'receipts/exit-{j}.txt').read_text().strip()=='1'
 assert 'Traceback (most recent call last)' not in s and 'CUDA error' not in s
 dirs=sorted((p/'runs').glob(f'canary-gpu_4090-s{sh}-*-{j}'))
 env=json.loads((dirs[0]/'environment.json').read_text())
 retained={str(f.relative_to(p)):sha(f) for d in dirs for f in d.iterdir() if f.is_file()}
 missing=[]
 for rep in range(3):
  for mode in (('pristine','paired') if rep in (0,2) else ('paired','pristine')):
   candidates=list((p/'runs').glob(f'canary-gpu_4090-s{sh}-r{rep}-{mode}-*'))
   if candidates:
    assert len(candidates)==1 and (candidates[0]/'complete.json').exists()
   else:missing.append([rep,mode])
 assert missing and all(rep==2 for rep,mode in missing)
 slurm=(p/f'receipts/slurm-{j}.txt').read_text();node=re.search(r' NodeList=(\S+)',slurm)[1]
 failures[str(j)]={'shard':sh,'node':node,'gpu_uuid':env['gpu_uuid'],'cpu_affinity':env['cpu_affinity'],'missing_processes':missing,'retained_files':retained,'failure_log_sha256':sha(log),'metrics_inspected_for_recovery_decision':False}
 shutil.copyfile(log,r/f'logs/original-failure-{j}.out')
receipt={'original_campaign':str(p),'recovery_campaign':str(r),'preregistered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_terminal_receipt_sha256':sha(p/'receipts/kernel-pipeline.json'),'source_commit':(p/'source/SOURCE_COMMIT.txt').read_text().strip(),'source_archive_sha256':(p/'receipts/source-archive.sha256').read_text().strip(),'reason':'Slurm control-plane RPC failed before Python startup; resume only absent repeat2 processes, with identical allocated GPU UUID and CPU affinity. All earlier measurements remain in qualification.','failures':failures,'fresh_cases_consumed':0,'source_changed':False,'analyzer_changed':False,'floors_changed':False,'second_preparation_failure':{'campaign':'/ssd/scxi253/sgi-v42-canary-rpc-recovery-20261001T0430Z-allocation-v2','reason':'Explicit memory flag prohibited; queue assigns 60GB per GPU; no jobs submitted'},'initial_preparation_failure':{'campaign':'/ssd/scxi253/sgi-v42-canary-rpc-recovery-20261001T0430Z','reason':'Requested 96 CPUs violated six CPUs per GPU submission policy; no jobs submitted and no measurements started'},'subsequent_stages_authorized':False,'default_promotion':False,'serving_promotion':False,'resource_request_reason':'Request all 8 GPUs, 48 physical cores (within six cores per GPU policy), and exclusive allocation on each original node to select the original UUID/cores legally; other allocated GPUs remain unused; never overlap another allocation.'}
receipt['scripts_sha256']={n:sha(r/n) for n in ('resume_measure.py','canary_resume.sbatch')}
(r/'recovery-preregistration.json').write_text(json.dumps(receipt,indent=2)+'\n')
# Existing complete processes stay discoverable without changing original artifacts.
for d in (p/'runs').iterdir():(r/'runs'/d.name).symlink_to(d,target_is_directory=True)
for f in (p/'logs').iterdir():
 if f.is_file() and not (r/'logs'/f.name).exists():(r/'logs'/f.name).symlink_to(f)
jobs={}
for old,info in failures.items():
 cmd=['sbatch','--parsable','-p','gpu_4090','--nodelist='+info['node'],'-o',str(r/'logs/resume-%j.out'),str(r/'canary_resume.sbatch'),str(r),old]
 proc=subprocess.run(cmd,capture_output=True,text=True)
 if proc.returncode:
  (r/'receipts/submission-error.json').write_text(json.dumps({'old_job':old,'stdout':proc.stdout,'stderr':proc.stderr,'returncode':proc.returncode},indent=2)+'\n');raise RuntimeError(proc.stderr)
 jobs[old]=proc.stdout.strip().split(';')[0]
 (r/'receipts/resume-jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
print(json.dumps({'recovery':str(r),'jobs':jobs}))
