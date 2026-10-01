"""Finite completeness repair; a known 5090 HOLD permanently blocks later stages."""
from pathlib import Path
import datetime,fcntl,hashlib,json,subprocess,sys,time
r=Path(sys.argv[1]);p=Path(json.loads((r/'recovery-preregistration.json').read_text())['original_campaign'])
lock=(r/'receipts/recovery-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
receipt=r/'receipts/recovery-controller.json'
if receipt.exists():raise RuntimeError('never overwrite an earlier controller run')
pre=json.loads((r/'recovery-preregistration.json').read_text());jobs=json.loads((r/'receipts/resume-jobs.json').read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
state={'state':'WAITING_FOR_ORIGINAL_AND_RESUMED_PROCESSES','terminal':False,'source_commit':pre['source_commit'],'preregistration_sha256':sha(r/'recovery-preregistration.json'),'controller_sha256':sha(Path(__file__)),'resume_jobs':jobs,'known_5090_verdict':False,'fresh_cases_consumed':0,'subsequent_stages_authorized':False,'default_promotion':False,'serving_promotion':False}
def save():
 state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(receipt)
def log(s):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),s,flush=True)
def command(cmd,out):
 with out.open('w') as f:subprocess.run(cmd,cwd=r/'source',stdout=f,stderr=subprocess.STDOUT,check=True)
save();deadline=time.monotonic()+6*3600
try:
 while True:
  original={str(j):(p/f'receipts/exit-{j}.txt').read_text().strip() if (p/f'receipts/exit-{j}.txt').exists() else None for j in range(1644188,1644208)}
  unexpected={j:v for j,v in original.items() if v not in (None,'0') and j not in jobs}
  if unexpected:raise RuntimeError('new original execution failures require separate preregistration: '+str(unexpected))
  resumed={j:(r/f'receipts/resume-exit-{j}.txt').read_text().strip() if (r/f'receipts/resume-exit-{j}.txt').exists() else None for j in jobs.values()}
  state.update(original_process_exits=original,resume_process_exits=resumed);save()
  if any(v not in (None,'0') for v in resumed.values()):raise RuntimeError('recovery process failure; never repeat a started measurement')
  if all(v=='0' or j in jobs for j,v in original.items()) and all(v=='0' for v in resumed.values()):break
  if time.monotonic()>deadline:raise TimeoutError('finite recovery wait exceeded six hours')
  time.sleep(45)
 state['state']='VERIFYING_COMPLETE_UNCHANGED_EVIDENCE';save()
 for old,info in pre['failures'].items():
  for n,h in info['retained_files'].items():
   if sha(p/n)!=h:raise RuntimeError('original retained artifact drift '+n)
 for shard in range(10):
  for rep in range(3):
   for mode in ('pristine','paired'):
    dirs=list((r/'runs').glob(f'canary-gpu_4090-s{shard}-r{rep}-{mode}-*'))
    if len(dirs)!=1 or not (dirs[0]/'complete.json').exists():raise RuntimeError('incomplete/duplicate discovery')
 command(['sha256sum','-c','source.sha256'],r/'logs/recovery-source-final-check.log')
 for old,new in jobs.items():
  command([sys.executable,'-m','research.selector_v4.audit_binary','--workspace',str(r/'cache/gpu_4090/candidate'),'--pristine-workspace',str(r/'cache/gpu_4090/pristine'),'--overlay',str(r/'overlays/candidate'),'--out',str(r/f'receipts/binary-audit-{old}.json')],r/f'logs/resume-binary-audit-{new}.log')
  audit=json.loads((r/f'receipts/binary-audit-{old}.json').read_text());audit['technical_recovery_provenance']={'original_job':old,'resume_job':new,'derived_after_all_measurements':True,'preregistration_sha256':state['preregistration_sha256']};(r/f'receipts/binary-audit-{old}.json').write_text(json.dumps(audit,indent=2)+'\n')
  original=p/f'logs/telemetry-{old}.csv';extra=r/f'logs/resume-telemetry-{new}.csv';dest=r/f'logs/telemetry-{old}.csv'
  if dest.is_symlink():dest.unlink()
  a=original.read_text();b=extra.read_text();assert a.splitlines()[0]==b.splitlines()[0]
  dest.write_text(a.rstrip('\n')+'\n'+'\n'.join(b.splitlines()[1:])+'\n')
  (r/f'receipts/combined-telemetry-{old}.json').write_text(json.dumps({'original':str(original),'original_sha256':sha(original),'resume':str(extra),'resume_sha256':sha(extra),'combined':str(dest),'combined_sha256':sha(dest),'no_rows_removed':True},indent=2)+'\n')
 state['state']='ANALYZING_COMPLETE_4090_WITH_FROZEN_ANALYZER';save()
 command([sys.executable,'-m','research.selector_v4.analyze_v4','--root',str(r),'--out',str(r/'analysis/canary-gpu_4090'),'--stage','canary','--gpu','gpu_4090','--shards','10'],r/'logs/recovery-analysis-4090.log')
 s5090=p/'analysis/canary-gpu_5090';(r/'analysis/canary-gpu_5090').symlink_to(s5090,target_is_directory=True)
 state['verdicts']={}
 for gpu in ('gpu_4090','gpu_5090'):
  s=r/f'analysis/canary-{gpu}/summary.json';x=json.loads(s.read_text());state['verdicts'][gpu]={'pass':x['pass'],'summary_sha256':sha(s),'failed_requirements':[k for k,v in x['requirements'].items() if not v],'metrics':x['metrics']}
 assert not state['verdicts']['gpu_5090']['pass'],'known frozen 5090 HOLD unexpectedly changed'
 state.update(state='CANARY_QUALIFICATION_HOLD_COMPLETENESS_RECOVERED',terminal=True);save();log(state['state'])
except BaseException as exc:
 state.update(state='RECOVERY_EXECUTION_HOLD',terminal=True,error={'type':type(exc).__name__,'message':str(exc)});save();raise
