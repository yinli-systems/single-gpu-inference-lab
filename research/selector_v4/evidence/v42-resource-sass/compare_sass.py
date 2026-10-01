import hashlib,json,re
from pathlib import Path
import argparse
parser=argparse.ArgumentParser();parser.add_argument('raw_root',type=Path);parser.add_argument('--out',type=Path,default=Path('resource-native-sass-comparison.json'));args=parser.parse_args()
root=args.raw_root
NAMES=('BatchPrefillWithRaggedKVCache','BatchPrefillWithPagedKVCache')
def parse(source):
 result={};arch=None
 for block in re.split(r'(?=\s*Function\s*:\s*)',source):
  found=re.findall(r'arch\s*=\s*(sm_\d+)',block)
  if found:arch=found[-1]
  match=re.search(r'Function\s*:\s*(\S+)',block)
  if not match:continue
  symbol=match.group(1)
  layout=next((name for name in NAMES if name+'Kernel' in symbol or name+'ResourceKernel' in symbol),None)
  if layout is None:continue
  is_resource=layout+'ResourceKernel' in symbol
  canonical=re.sub(r'(\d+)'+layout+'ResourceKernel',lambda m:str(int(m.group(1))-8)+layout+'Kernel',symbol) if is_resource else symbol
  lines=[]
  for line in block[match.end():].splitlines():
   if not line.strip() or set(line.strip())=={'-'}:continue
   if 'Fatbin' in line or line.strip().startswith(('code for ','arch =','identifier =','host =','compile_size =')):break
   line=re.sub(r'/\*[0-9a-fA-F]{4,8}\*/','',line)
   lines.append(' '.join(line.split()))
  assert arch is not None
  key=(arch,canonical)
  result.setdefault(key,{})['resource' if is_resource else 'native']=hashlib.sha256('\n'.join(lines).encode()).hexdigest()
 return result
reports=[]
for p in sorted(root.rglob('candidate-*.sass')):
 records=parse(p.read_text());pairs={k:v for k,v in records.items() if set(v)=={'native','resource'}}
 mismatches=[{'arch':k[0],'symbol':k[1],**v} for k,v in pairs.items() if v['native']!=v['resource']]
 reports.append({'file':str(p),'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'paired_kernel_count':len(pairs),'unmatched_kernel_count':len(records)-len(pairs),'identical_count':len(pairs)-len(mismatches),'mismatches':mismatches})
result={'scope':'source b769e7c exact smoke; diagnostic machine-code comparison of native/resource kernels','normalization':'mangled Resource name length reversed for pairing; PC labels and whitespace only in instruction hashes; instruction operands/encoding/resource metadata retained','files':reports,'all_identity':bool(reports) and all(r['paired_kernel_count']>0 and not r['mismatches'] and not r['unmatched_kernel_count'] for r in reports),'historical_token_divergence_resolved':False}
out=args.out;out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'all_identity':result['all_identity'],'files':[{k:v for k,v in r.items() if k!='mismatches'}|{'mismatch_count':len(r['mismatches'])} for r in reports]}))
