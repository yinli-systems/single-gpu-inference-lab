"""Collect completed diagnostic artifacts, including failures. Never modify raw runs."""
import argparse,datetime,hashlib,json,subprocess,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
archive=root/'section6-raw-evidence.tar.gz'
if archive.exists():raise FileExistsError('preserve existing archive')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
bounds=[]
for f in sorted((root/'residency-runs').glob('*/launch-*.json')):
 if f.name=='launch-proof.json':continue
 d=json.loads(f.read_text());props=d['deviceProperties'];assert len(props)==1
 es=[e for e in d['traceEvents'] if e.get('cat')=='kernel' and 'BatchPrefillWithRaggedKVCacheKernel' in e.get('name','')];assert len(es)==1
 e=es[0];sm=props[0]['sharedMemPerMultiprocessor'];used=e['args']['shared memory']
 bounds.append(dict(file=str(f.relative_to(root)),sha256=sha(f),gpu=props[0]['name'],shared_bytes_per_SM=sm,
  shared_bytes_per_CTA=used,shared_memory_capacity_upper_bound=sm//used,reported_occupancy=e['args'].get('occupancy'),
  estimator_zero_with_successful_launch=e['args'].get('occupancy',{}).get('activeBlocksPerMultiprocessor')==0,
  interpretation='capacity bound only; no observed CTA-to-SM issue order'))
assert len(bounds)==24
(root/'receipts/resource-bounds.json').write_text(json.dumps(bounds,indent=2)+'\n')
cmd=['sacct','-j','1639150,1639151,1639491,1639492,1639566,1639568','--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES%80,NodeList','-n','-P']
r=subprocess.run(cmd,capture_output=True,text=True,check=True);(root/'receipts/final-accounting.txt').write_text(r.stdout)
jobs=[]
for line in r.stdout.splitlines():
 v=line.split('|')
 if not v or not v[0].isdigit():continue
 jobs.append(dict(job=v[0],state=v[1],exit_code=v[2],allocated_gpu_seconds=int(v[3]),resources=v[4],node=v[5]))
assert len(jobs)==6 and all(j['state'] in ('COMPLETED','TIMEOUT') for j in jobs)
acc=dict(jobs=jobs,allocated_gpu_seconds=sum(j['allocated_gpu_seconds'] for j in jobs),allocated_gpu_hours=sum(j['allocated_gpu_seconds'] for j in jobs)/3600,includes_timeout=True,price_estimate=False)
(root/'receipts/resource-accounting.json').write_text(json.dumps(acc,indent=2)+'\n')
old=Path('/ssd/scxi253/q7b-engines-20260925T1247Z/envs/sglang312/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/attention/prefill.cuh')
new=root/'overlays/fi070/flashinfer/data/include/flashinfer/attention/prefill.cuh'
provenance=[]
for version,f,blob in [('0.6.18',old,'8e7f554cda0e31df879cdc6e85b0c7ef6ff913eb'),('0.7.0',new,'8d17c4674729903ed9f63cca7d6add0e6e4ff33a')]:
 b=f.read_bytes();actual=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();assert actual==blob
 provenance.append(dict(version=version,sha256=sha(f),git_blob=actual,official_tag_blob=blob,official_tag_match=True))
(root/'receipts/official-source-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
for name,src in [('residency-host.patch',root/'overlays/residency070-v2/host-launch-only.patch'),('residency-binding.json',root/'overlays/residency070-v2/SOURCE_BINDING.json'),('native-scheduler.patch',root/'overlays/native-cycle070/native-scheduler.patch'),('native-binding.json',root/'overlays/native-cycle070/NATIVE_BINDING.json')]:
 (root/'receipts'/name).write_bytes(src.read_bytes())
(root/'receipts/collection-failure-history.json').write_text(json.dumps(dict(initial_attempt='TypeError on launch-proof.json: list is not a Chrome trace dictionary',effect='No archive or result modification; corrected collector excludes that non-trace filename',performance_results_changed=False),indent=2)+'\n')
files=[]
for folder in ['runs','residency-runs','native-runs','logs','receipts','available-evidence-v1','residency-analysis-v1','native-cycle-analysis-v1']:
 for f in sorted((root/folder).rglob('*')):
  if f.is_file():assert not f.is_symlink();files.append(f)
files += [root/n for n in ['source.tar','residency-source.tar','residency-v2-source.tar','native-source.tar','analyze_available.py','analyze_residency.py','analyze_native_cycle.py','test_statistics.py']]
manifest=[dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=sha(f)) for f in files]
with tarfile.open(archive,'w:gz',compresslevel=6) as t:
 for f in files:t.add(f,arcname=str(f.relative_to(root)),recursive=False)
meta=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,files=manifest,
 phase_A_C_complete_runs=11,phase_A_C_planned_runs=12,phase_B_complete_runs=6,phase_D_complete_runs=18,phase_E_executed=False,default_promotion=False)
(root/'section6-data-manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
print('ARCHIVE',archive.stat().st_size,meta['archive_sha256'],'FILES',len(files));print('ACCOUNTING',json.dumps(acc));print('RESOURCE_BOUND_ROWS',len(bounds))
