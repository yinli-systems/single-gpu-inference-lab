"""Archive completed validation attempts without rewriting their evidence."""
import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

a=argparse.ArgumentParser()
a.add_argument('campaign',type=Path)
a.add_argument('out',type=Path)
a.add_argument('--jobs',nargs='+',required=True)
a.add_argument('--reason',required=True)
x=a.parse_args()
x.out.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[]
for directory in ('logs','receipts'):
    if (x.campaign/directory).exists():
        files.extend(p for p in (x.campaign/directory).rglob('*') if p.is_file())
files.extend(p for p in x.campaign.iterdir() if p.is_file())
manifest={str(p.relative_to(x.campaign)):sha(p) for p in sorted(files)}
archive=x.out/'raw-validation-attempt.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for name in manifest:
        t.add(x.campaign/name,arcname=name)
with tarfile.open(archive) as t:
    for name,h in manifest.items():
        assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==h
status=subprocess.check_output(['sacct','-j',','.join(x.jobs),'--format=JobID,State,ExitCode,Elapsed','-n','-X'],text=True)
assert 'RUNNING' not in status and 'PENDING' not in status, status
receipt={'campaign':str(x.campaign),'jobs':x.jobs,'sacct':status,'reason':x.reason,
         'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'files':manifest,
         'fresh_cases_consumed':0,'managed_v2_qualified':False,'default_promotion':False,
         'serving_promotion':False,'historical_token_divergence_resolved':False}
(x.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ('jobs','archive_sha256','archive_bytes')}))
