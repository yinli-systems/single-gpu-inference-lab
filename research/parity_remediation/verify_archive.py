"""Verify every archived member without extracting any paths."""
import argparse,hashlib,json,tarfile
from pathlib import Path,PurePosixPath

def verify(archive):
 with tarfile.open(archive,'r:gz') as t:
  names=[m.name for m in t]
  if len(names)!=len(set(names)):raise ValueError('duplicate archive member')
  m=json.load(t.extractfile('MANIFEST.json'))
  if set(names)!=set(m['files'])|{'MANIFEST.json'}:raise ValueError('manifest coverage')
  for name,info in m['files'].items():
   p=PurePosixPath(name);member=t.getmember(name)
   if p.is_absolute() or '..' in p.parts or not member.isfile():raise ValueError('unsafe archive entry')
   data=t.extractfile(member).read()
   if len(data)!=info['bytes'] or hashlib.sha256(data).hexdigest()!=info['sha256']:raise ValueError('archive corruption')
 return dict(verified_members=len(m['files']),all_jobs_terminal=m['all_jobs_terminal'],total_gpu_seconds=m['total_gpu_seconds'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);a=p.parse_args();print(json.dumps(verify(a.archive),indent=2))
