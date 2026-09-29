import argparse,hashlib,json,shutil,difflib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists():raise FileExistsError('preserve overlay')
s=a.source/'flashinfer';header=s/'data/include/flashinfer/attention/scheduler.cuh';old=header.read_text()
needle='  // step 4: multiply kv_chunk_size by page_size'
assert old.count(needle)==1
call='  SGICausalOrderWitness(qo_indptr_h, kv_indptr_h, batch_size, gqa_group_size, cta_tile_q, split_kv, enable_cuda_graph, page_size, request_indices, qo_tile_indices, kv_tile_indices);\n\n'
new=old.replace('#include "heap.h"','#include "heap.h"\n#include "sgi_causal_order.cuh"').replace(needle,call+needle)
assert new!=old and new.count('#include "sgi_causal_order.cuh"')==1
shutil.copytree(s,a.out/'flashinfer',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for f in a.source.glob('*.dist-info'):shutil.copytree(f,a.out/f.name)
shutil.copy2(a.source/'SOURCE_BINDING.json',a.out/'SOURCE_BINDING.json')
(a.out/'flashinfer/data/include/flashinfer/attention/scheduler.cuh').write_text(new)
helper=Path(__file__).with_name('sgi_causal_order.cuh');shutil.copy2(helper,a.out/'flashinfer/data/include/flashinfer/attention'/helper.name)
rec=dict(scheduler_original_sha256=hashlib.sha256(old.encode()).hexdigest(),scheduler_modified_sha256=hashlib.sha256(new.encode()).hexdigest(),helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),scope='default-off exact exposed witness only; not public API or thread-safe serving integration')
(a.out/'NATIVE_BINDING.json').write_text(json.dumps(rec,indent=2)+'\n')
(a.out/'native-scheduler.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/scheduler.cuh',tofile='b/scheduler.cuh')))
print(json.dumps(rec))
