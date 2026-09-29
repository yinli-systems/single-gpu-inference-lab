"""Collect only this campaign's sources and receipts; never model weights or JIT caches."""
from __future__ import annotations
import argparse,datetime,hashlib,io,json,subprocess,tarfile
from pathlib import Path

CAMPAIGNS={
 'http-v1':'single-gpu-inference-parity-remediation-20260930T2200Z',
 'trace-v1':'single-gpu-inference-parity-trace-20260930T2212Z',
 'wire-v2':'single-gpu-inference-parity-wire-20260930T2222Z',
 'trace-v2':'single-gpu-inference-parity-trace2-20260930T2227Z',
 'cleanup':'single-gpu-inference-abort-owner-20260930T2230Z',
 'layers':'single-gpu-inference-parity-layers-20260930T2248Z',
}
JOBS=[1639848,1639849,1639851,1639852,1639860,1639861,1639883]

def collect(parent,out):
 if out.exists():raise FileExistsError('preserve archive')
 result=subprocess.run(['sacct','-X','-j',','.join(map(str,JOBS)),'--format=JobID,JobName,State,ExitCode,ElapsedRaw,AllocTRES','-n','-P'],capture_output=True,text=True,check=True)
 job_rows=[]
 for line in result.stdout.splitlines():
  if not line.strip():continue
  values=line.split('|')
  if values[0].isdigit() and int(values[0]) in JOBS:job_rows.append(values)
 if {int(x[0]) for x in job_rows}!=set(JOBS):raise ValueError('missing accounting')
 if any('gres/gpu=1' not in x[5].split(',') for x in job_rows):raise ValueError('unexpected GPU accounting')
 if any(x[2] not in ('COMPLETED','FAILED','TIMEOUT','CANCELLED') for x in job_rows):raise ValueError('job still active; do not seal')
 selected={};retained_tensors={}
 for alias,dirname in CAMPAIGNS.items():
  root=parent/dirname
  if not root.is_dir():raise ValueError('missing campaign root')
  for child in root.iterdir():
   if child.name in ('source','runs','logs','historical-audit-v1','inspection-2220','abort-source-patch-v1','audit-v1','final-audit','trace-audit','trace-audit-v2','layer-audit') or (child.is_file() and child.name in ('submissions.txt','submission.txt')):
    entries=list(child.rglob('*')) if child.is_dir() else [child]
    for p in entries:
     if '__pycache__' in p.parts:continue
     if p.is_symlink():raise ValueError('unexpected symlink')
     if p.is_file():
      name=alias+'/'+p.relative_to(root).as_posix()
      if alias=='layers' and p.suffix=='.pt':
       data=p.read_bytes();retained_tensors[name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),retained_path=str(p))
      else:selected[name]=p
 manifest={}
 out.mkdir(parents=True)
 with tarfile.open(out/'raw-evidence.tar.gz','w:gz') as tar:
  for name,p in sorted(selected.items()):
   data=p.read_bytes();manifest[name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
   ti=tarfile.TarInfo(name);ti.size=len(data);ti.mode=0o644;tar.addfile(ti,io.BytesIO(data))
  receipt=dict(collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=manifest,
      jobs=job_rows,retained_tensor_files=retained_tensors,all_jobs_terminal=True,total_gpu_seconds=sum(int(x[4]) for x in job_rows),
      archive_scope='all selected campaign evidence except large layer tensor binaries; excluded binaries retain paths and SHA256s',
      assumptions='all listed jobs request one GPU; confirm AllocTRES column',performance_claim=False)
  data=(json.dumps(receipt,indent=2)+'\n').encode();ti=tarfile.TarInfo('MANIFEST.json');ti.size=len(data);tar.addfile(ti,io.BytesIO(data))
 (out/'MANIFEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
 digest=hashlib.sha256((out/'raw-evidence.tar.gz').read_bytes()).hexdigest()
 (out/'archive.sha256').write_text(digest+'  raw-evidence.tar.gz\n')
 print(json.dumps(dict(files=len(manifest),bytes=(out/'raw-evidence.tar.gz').stat().st_size,sha256=digest,jobs=job_rows,total_gpu_seconds=receipt['total_gpu_seconds']),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();collect(a.parent,a.out)
