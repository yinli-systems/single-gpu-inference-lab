"""CPU-only, checksum-gated extraction and analysis of the published evidence.
Never launches GPU jobs, downloads models or executes extracted source code.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def extract(archive,destination,manifest):
    if destination.exists():raise FileExistsError('new extraction directory required')
    if sha(archive)!=manifest['archive_sha256']:raise ValueError('archive SHA256 mismatch')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=set();total=0
        for m in members:
            path=Path(m.name)
            if path.is_absolute() or '..' in path.parts or m.name in names or not (m.isfile() or m.isdir()):
                raise ValueError('unsafe or duplicate archive member')
            names.add(m.name);total+=m.size
            if total>2*1024**3:raise ValueError('unreasonable evidence archive size')
        destination.mkdir(parents=True)
        for m in members:
            target=destination/m.name
            if m.isdir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as source,target.open('xb') as out:
                while True:
                    data=source.read(1<<20)
                    if not data:break
                    out.write(data)
    actual={str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()}
    if actual!=manifest['files']:raise ValueError('extracted file set or content mismatch')

def main():
    D=Path(__file__).resolve().parent
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,default=D/'raw-evidence.tar.gz')
    p.add_argument('--manifest',type=Path,default=D/'data-manifest.json')
    p.add_argument('--extract-to',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('new analysis directory required')
    manifest=json.loads(a.manifest.read_text());extract(a.archive,a.extract_to,manifest)
    a.out.mkdir(parents=True)
    subprocess.run([sys.executable,str(D/'analysis/analyze.py'),'--root',str(a.extract_to),
                    '--stage','test','--out',str(a.out/'formal')],check=True)
    native=a.extract_to/'native-integration'
    if native.exists():
        subprocess.run([sys.executable,str(D/'analysis/analyze_native.py'),'--root',str(native),
                        '--out',str(a.out/'native')],check=True)
    receipt=dict(archive_sha256=manifest['archive_sha256'],files=len(manifest['files']),
                 no_GPU_used=True,outputs={str(p.relative_to(a.out)):sha(p) for p in a.out.rglob('*') if p.is_file()})
    (a.out/'reproduction-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
