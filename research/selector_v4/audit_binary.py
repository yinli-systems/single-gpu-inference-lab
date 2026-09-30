"""Prove that a campaign-private JIT build contains separate native/capped symbols."""
from __future__ import annotations
from .native_sass import compare_native_sass
from pathlib import Path
import argparse,hashlib,json,subprocess

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(a):
    base=Path(a.workspace)/'.cache/flashinfer';overlay=Path(a.overlay)
    shared=list(base.rglob('*.so'))
    if not shared:raise RuntimeError('missing private JIT artifacts')
    header=overlay/'flashinfer/data/include/flashinfer/attention/prefill.cuh'
    if not header.is_file():raise RuntimeError('missing source-bound overlay header')
    source_hits={'ragged_native':0,'ragged_resource':0,'paged_native':0,'paged_resource':0}
    for path in (header,):
        text=path.read_text(errors='ignore')
        source_hits['ragged_native']+=text.count('BatchPrefillWithRaggedKVCacheKernel')
        source_hits['ragged_resource']+=text.count('BatchPrefillWithRaggedKVCacheResourceKernel')
        source_hits['paged_native']+=text.count('BatchPrefillWithPagedKVCacheKernel')
        source_hits['paged_resource']+=text.count('BatchPrefillWithPagedKVCacheResourceKernel')
    symbol_hits={k:0 for k in source_hits};binaries=[]
    mapping={'ragged_native':'BatchPrefillWithRaggedKVCacheKernel','ragged_resource':'BatchPrefillWithRaggedKVCacheResourceKernel','paged_native':'BatchPrefillWithPagedKVCacheKernel','paged_resource':'BatchPrefillWithPagedKVCacheResourceKernel'}
    per_binary=[]
    for path in shared:
        proc=subprocess.run(['nm','-C',str(path)],capture_output=True,text=True,timeout=60)
        text=proc.stdout+proc.stderr;counts={key:text.count(name) for key,name in mapping.items()}
        for key,count in counts.items():symbol_hits[key]+=count
        item={'path':str(path.relative_to(base)),'sha256':sha(path),'bytes':path.stat().st_size,'nm_returncode':proc.returncode,'symbol_hits':counts};binaries.append(item);per_binary.append(counts)
    if not all(source_hits.values()):raise RuntimeError('generated source lacks isolated symbols '+str(source_hits))
    if not all(symbol_hits.values()):raise RuntimeError('compiled binary lacks isolated symbols '+str(symbol_hits))
    ragged_same=any(x['ragged_native'] and x['ragged_resource'] for x in per_binary);paged_same=any(x['paged_native'] and x['paged_resource'] for x in per_binary)
    if not ragged_same or not paged_same:raise RuntimeError('native/resource symbols are not co-resident in compiled tactic module')
    result={'schema':1,'workspace':str(a.workspace),'private_cache':str(base),'overlay':str(overlay),'overlay_header_sha256':sha(header),'source_hits':source_hits,'symbol_hits':symbol_hits,'same_module_pairs':{'ragged':ragged_same,'paged':paged_same},'binaries':binaries,'kernel_symbol_isolation_compiled':True}
    if getattr(a, 'pristine_workspace', None) is not None:
        result['native_sass'] = compare_native_sass(a.pristine_workspace, a.workspace, Path(a.out).with_suffix('.sass'))
        result['native_sass_identity'] = result['native_sass']['native_sass_identity']
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--overlay',type=Path,required=True);p.add_argument('--pristine-workspace',type=Path);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
