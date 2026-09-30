"""Prove that a campaign-private JIT build contains separate native/capped symbols."""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,subprocess

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(a):
    base=Path(a.workspace)/'.cache/flashinfer'
    shared=list(base.rglob('*.so'));generated=list(base.rglob('*.cu'))+list(base.rglob('*.cuh'))
    if not shared or not generated:raise RuntimeError('missing private JIT artifacts')
    source_hits={'ragged_native':0,'ragged_resource':0,'paged_native':0,'paged_resource':0}
    for path in generated:
        text=path.read_text(errors='ignore')
        source_hits['ragged_native']+=text.count('BatchPrefillWithRaggedKVCacheKernel')
        source_hits['ragged_resource']+=text.count('BatchPrefillWithRaggedKVCacheResourceKernel')
        source_hits['paged_native']+=text.count('BatchPrefillWithPagedKVCacheKernel')
        source_hits['paged_resource']+=text.count('BatchPrefillWithPagedKVCacheResourceKernel')
    symbol_hits={k:0 for k in source_hits};binaries=[]
    mapping={'ragged_native':'BatchPrefillWithRaggedKVCacheKernel','ragged_resource':'BatchPrefillWithRaggedKVCacheResourceKernel','paged_native':'BatchPrefillWithPagedKVCacheKernel','paged_resource':'BatchPrefillWithPagedKVCacheResourceKernel'}
    for path in shared:
        proc=subprocess.run(['nm','-C',str(path)],capture_output=True,text=True,timeout=60)
        text=proc.stdout+proc.stderr
        for key,name in mapping.items():symbol_hits[key]+=text.count(name)
        binaries.append({'path':str(path.relative_to(base)),'sha256':sha(path),'bytes':path.stat().st_size,'nm_returncode':proc.returncode})
    if not all(source_hits.values()):raise RuntimeError('generated source lacks isolated symbols '+str(source_hits))
    if not all(symbol_hits.values()):raise RuntimeError('compiled binary lacks isolated symbols '+str(symbol_hits))
    result={'schema':1,'workspace':str(a.workspace),'private_cache':str(base),'source_hits':source_hits,'symbol_hits':symbol_hits,'binaries':binaries,'kernel_symbol_isolation_compiled':True}
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
