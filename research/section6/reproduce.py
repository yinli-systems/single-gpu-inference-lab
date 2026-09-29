"""CPU-only, hash-gated replay of actual section-six GPU evidence."""
from __future__ import annotations
import argparse,hashlib,json,math,os,subprocess,sys,tarfile
from pathlib import Path,PurePosixPath

def require(value,msg):
 if not value:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def compare(a,b,path=''):
 if type(a) is not type(b):raise ValueError('type difference at '+path)
 if isinstance(a,dict):
  require(a.keys()==b.keys(),'key difference at '+path)
  return sum((compare(a[k],b[k],path+'/'+k) for k in a),[])
 if isinstance(a,list):
  require(len(a)==len(b),'length difference at '+path)
  return sum((compare(x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))),[])
 if isinstance(a,float):
  require(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),'numeric difference at '+path)
  return [] if a==b else [dict(path=path,original=a,reproduced=b)]
 require(a==b,'value difference at '+path)
 return []
def unpack(archive,manifest,dest):
 m=json.loads(manifest.read_text());require(sha(archive)==m['archive_sha256'],'archive checksum mismatch')
 entries={x['path']:x for x in m['files']};require(len(entries)==len(m['files']),'duplicate manifest entry')
 require(sum(x['bytes'] for x in entries.values())<=1024**3,'unpacked size cap exceeded')
 dest.mkdir();seen=set()
 with tarfile.open(archive,'r:gz') as t:
  for member in t:
   n=member.name;rel=PurePosixPath(n)
   require(not rel.is_absolute() and '..' not in rel.parts and member.isfile() and n in entries and n not in seen,'unsafe/unlisted/duplicate archive member')
   require(member.size==entries[n]['bytes'],'wrong member size')
   target=dest/Path(*rel.parts);target.parent.mkdir(parents=True,exist_ok=True)
   stream=t.extractfile(member);require(stream is not None,'missing member stream')
   h=hashlib.sha256();size=0
   with target.open('xb') as out:
    for data in iter(lambda:stream.read(1024*1024),b''):
     out.write(data);h.update(data);size+=len(data)
   require(size==member.size and h.hexdigest()==entries[n]['sha256'],'member hash mismatch '+n);seen.add(n)
 require(seen==entries.keys(),'archive missing manifest files')
 return m

def run(a):
 require(not a.out.exists(),'refuse overwriting reproduction')
 a.out.mkdir(parents=True);raw=a.out/'raw';m=unpack(a.archive,a.manifest,raw)
 source=Path(__file__).resolve().parent
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH=str(source))
 reports={}
 for script,expected in [('analyze_available.py','available-evidence-v1'),('analyze_residency.py','residency-analysis-v1'),('analyze_native_cycle.py','native-cycle-analysis-v1')]:
  target=a.out/expected
  command=[sys.executable,str(source/script),'--root',str(raw),'--out',str(target)]
  result=subprocess.run(command,env=env,capture_output=True,text=True)
  (a.out/(expected+'.stdout.txt')).write_text(result.stdout)
  (a.out/(expected+'.stderr.txt')).write_text(result.stderr)
  require(result.returncode==0,'analysis failed: '+script+' '+result.stderr[-1000:])
  original=json.loads((raw/expected/'summary.json').read_text());new=json.loads((target/'summary.json').read_text())
  diffs=compare(original,new)
  markdown_same=(raw/expected/'RESULTS.md').read_bytes()==(target/'RESULTS.md').read_bytes()
  require(markdown_same,'result table changed')
  reports[expected]=dict(JSON_byte_identical=sha(raw/expected/'summary.json')==sha(target/'summary.json'),
   markdown_byte_identical=markdown_same,float_differences=diffs,
   original_summary_sha256=sha(raw/expected/'summary.json'),reproduced_summary_sha256=sha(target/'summary.json'))
 receipt=dict(complete=True,CPU_only=True,GPU_rerun=False,independent_third_party=False,
  python=sys.version,archive_sha256=m['archive_sha256'],verified_files=len(m['files']),comparisons=reports,
  failed_original_repeat_preserved=True,full_matrix_claim=False)
 (a.out/'REPRODUCTION.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();base=Path(__file__).parent/'evidence'
 p.add_argument('--archive',type=Path,default=base/'section6-raw-evidence.tar.gz');p.add_argument('--manifest',type=Path,default=base/'section6-data-manifest.json')
 p.add_argument('--out',type=Path,required=True);run(p.parse_args())
