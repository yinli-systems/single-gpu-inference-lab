"""CPU disassembly audit of immutable, already executed private/native modules."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,tarfile
NAMES=('BatchPrefillWithRaggedKVCache','BatchPrefillWithPagedKVCache')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def parse(s):
 out={};arch=None
 for block in re.split(r'(?=\s*Function\s*:\s*)',s):
  found=re.findall(r'arch\s*=\s*(sm_\d+[a-z]?)',block)
  if found:arch=found[-1]
  m=re.search(r'Function\s*:\s*(\S+)',block)
  if not m:continue
  sym=m.group(1);layout=next((n for n in NAMES if n+'Kernel' in sym or n+'ResourceKernel' in sym),None)
  if layout is None:continue
  resource=layout+'ResourceKernel' in sym
  if resource:sym=re.sub(r'(\d+)'+layout+'ResourceKernel',lambda x:str(int(x.group(1))-8)+layout+'Kernel',sym)
  if not arch:raise RuntimeError('missing architecture')
  lines=[]
  for line in block[m.end():].splitlines():
   if not line.strip() or set(line.strip())=={'-'}:continue
   if 'Fatbin' in line or line.strip().startswith(('code for ','arch =','identifier =','host =','compile_size =')):break
   lines.append(' '.join(re.sub(r'/\*[0-9a-fA-F]{4,8}\*/','',line).split()))
  if not any(';' in l for l in lines):raise RuntimeError('empty instructions')
  key=arch+':'+sym;v=hashlib.sha256('\n'.join(lines).encode()).hexdigest()
  if key in out and out[key]!=v:raise RuntimeError('conflicting duplicate '+key)
  out[key]=v
 return out

def run(p,job,out):
 if out.exists():raise FileExistsError('retain immutable audit')
 out.mkdir(parents=True);raw=out/'raw';raw.mkdir();native={};resource={};art=[]
 for i,f in enumerate(sorted((p/'cache'/job).rglob('*.so'))):
  kind='resource' if f.name.startswith('experimental_resource_') else 'native'
  r=subprocess.run(['/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13/bin/cuobjdump','--dump-sass',str(f)],capture_output=True,text=True,timeout=240,env={**__import__('os').environ,'PATH':'/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13/bin:'+__import__('os').environ['PATH']})
  if r.returncode:raise RuntimeError(r.stderr[-2500:])
  records=parse(r.stdout)
  if not records:continue
  target=native if kind=='native' else resource
  for key,v in records.items():
   if key in target and target[key]!=v:raise RuntimeError('cross-module drift')
   target[key]=v
  s=raw/(kind+'-'+str(i)+'.sass');s.write_text(r.stdout)
  art.append({'kind':kind,'binary':str(f),'binary_sha256':sha(f),'sass':str(s.relative_to(out)),'sass_sha256':sha(s),'symbols':len(records)})
 missing=sorted(native.keys()-resource.keys());extra=sorted(resource.keys()-native.keys());mismatch=sorted(k for k in native.keys()&resource.keys() if native[k]!=resource[k])
 passed=bool(native) and all(any(n+'Kernel' in k for k in native) for n in NAMES) and not(missing or extra or mismatch)
 archive=out/'raw-sass.tar.gz'
 with tarfile.open(archive,'w:gz') as t:
  for f in sorted(raw.iterdir()):t.add(f,arcname=f.name)
 d={'pass':passed,'job':job,'campaign':str(p),'source':json.loads((p/'receipts/source.json').read_text()),'paired_kernels':len(native),'missing':missing,'extra':extra,'mismatches':mismatch,'native':native,'resource':resource,'artifacts':art,'raw_archive_sha256':sha(archive),'raw_archive_bytes':archive.stat().st_size,'normalization':'reverse mangled resource-name length for pairing; normalize PC labels and whitespace only; operands, encodings and resource metadata retained','historical_token_divergence_resolved':False}
 (out/'receipt.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({k:d[k] for k in ['pass','job','paired_kernels','raw_archive_bytes','raw_archive_sha256']}))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('campaign',type=Path);a.add_argument('job');a.add_argument('out',type=Path);x=a.parse_args();run(x.campaign,x.job,x.out)
