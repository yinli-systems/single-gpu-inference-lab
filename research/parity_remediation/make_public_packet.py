"""Keep every JSON/log/source observation; catalogue tensor binaries retained remotely."""
import argparse,hashlib,io,json,tarfile
from pathlib import Path
from collect_evidence import CAMPAIGNS

def run(source,out,parent):
 if out.exists():raise FileExistsError('preserve public packet')
 full_hash=hashlib.sha256(source.read_bytes()).hexdigest();out.mkdir(parents=True)
 with tarfile.open(source,'r:gz') as src:
  original=json.load(src.extractfile('MANIFEST.json'));retained=dict(original.get('retained_tensor_files',{}));selected={}
  with tarfile.open(out/'raw-evidence.tar.gz','w:gz') as dst:
   for name,info in original['files'].items():
    if name.endswith('.pt'):
     alias,rest=name.split('/',1)
     retained[name]=dict(**info,retained_path=str(parent/CAMPAIGNS[alias]/rest));continue
    data=src.extractfile(name).read()
    if len(data)!=info['bytes'] or hashlib.sha256(data).hexdigest()!=info['sha256']:raise ValueError('source corruption')
    selected[name]=info;ti=tarfile.TarInfo(name);ti.size=len(data);ti.mode=0o644;dst.addfile(ti,io.BytesIO(data))
   manifest=dict(original,files=selected,retained_tensor_files=retained,
     original_full_archive=dict(path=str(source),sha256=full_hash,bytes=source.stat().st_size),
     archive_scope='All original JSON/log/source observations. Tensor binary bodies are retained on ParaCloud, with complete paths and hashes; no failed or slow observations filtered.')
   data=(json.dumps(manifest,indent=2)+'\n').encode();ti=tarfile.TarInfo('MANIFEST.json');ti.size=len(data);dst.addfile(ti,io.BytesIO(data))
 (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
 h=hashlib.sha256((out/'raw-evidence.tar.gz').read_bytes()).hexdigest();(out/'archive.sha256').write_text(h+'  raw-evidence.tar.gz\n')
 print(json.dumps(dict(archived_files=len(selected),retained_tensor_files=len(retained),bytes=(out/'raw-evidence.tar.gz').stat().st_size,sha256=h,full_archive_sha256=full_hash),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();run(a.source,a.out,a.parent)
