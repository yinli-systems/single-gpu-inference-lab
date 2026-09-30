"""Build a source-bound FlashInfer 0.7 overlay with isolated native/capped symbols."""
from __future__ import annotations
from pathlib import Path
import argparse, difflib, hashlib, json, os, shutil
from .kernel_isolation import patch_scheduler, patch_declarations, patch_jinja, patch_run, patch_prefill
from .native_policy import patch_native_entry, patch_native_binding, patch_python_dispatch

EXPECTED = {
 'flashinfer/prefill.py':'bd4aaa0a24462efbb6e6ddb12b3c98da9507e7b6ff6b22da3f40167ca66beb12',
 'flashinfer/data/csrc/batch_prefill_jit_binding.cu':'dad0d81c55b92f71a634cb8240dcb0d150aa6ffd391ef3ef5d87b6207eeca4fc',
 'flashinfer/data/include/flashinfer/attention/prefill.cuh':'e66bce2652c0a3c5b54f88510aff2b4b48ff6a52c4ee23fa97f91fa06a935b87',
 'flashinfer/data/include/flashinfer/attention/scheduler.cuh':'425ba78815ba77ae848c67507d1af6b024d05edd257a7d9b331166afcef9b17c',
 'flashinfer/data/csrc/batch_prefill.cu':'4190105e8b5c93621af44356b9aa50861560a5fbf8ff3279301487860604d932',
 'flashinfer/data/csrc/batch_prefill_paged.cuh':'52fa2dfb066729e7ddac10509d0c2d2a99433622055acb146e2938460aec432e',
 'flashinfer/data/csrc/batch_prefill_paged.cu':'7f9aaa45879034eb1b978ad1a43901fe22a5e84282d61f72e700fd2e77067dbd',
 'flashinfer/data/csrc/batch_prefill_ragged_kernel_inst.jinja':'4e177455f914c3665cf38728883d74c076f368c1bea3736c0033675de342f687',
 'flashinfer/data/csrc/batch_prefill_paged_kernel_inst.jinja':'1e41a0ea9795847272674c1058a9d72f220facf296037cbf1f95ce6b77225a63',
}

def sha_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def sha_text(text: str) -> str: return sha_bytes(text.encode())

def prepare(source: Path, out: Path) -> dict:
    if out.exists(): raise FileExistsError(out)
    originals = {}
    for rel, expected in EXPECTED.items():
        data=(source/rel).read_bytes()
        if sha_bytes(data)!=expected: raise ValueError('unreviewed source '+rel)
        originals[rel]=data.decode()
    shutil.copytree(source/'flashinfer',out/'flashinfer',ignore=shutil.ignore_patterns('__pycache__','*.pyc'),copy_function=os.link)
    for metadata in source.glob('*.dist-info'):
        shutil.copytree(metadata,out/metadata.name,copy_function=os.link)
    patched_prefill,symbol_hashes=patch_prefill(originals['flashinfer/data/include/flashinfer/attention/prefill.cuh'])
    batch=patch_declarations(originals['flashinfer/data/csrc/batch_prefill.cu'])
    batch=patch_run(batch,paged=False);batch=patch_native_entry(batch)
    changes={
      'flashinfer/data/include/flashinfer/attention/scheduler.cuh':patch_scheduler(originals['flashinfer/data/include/flashinfer/attention/scheduler.cuh']),
      'flashinfer/data/include/flashinfer/attention/prefill.cuh':patched_prefill,
      'flashinfer/data/csrc/batch_prefill.cu':batch,
      'flashinfer/data/csrc/batch_prefill_jit_binding.cu':patch_native_binding(originals['flashinfer/data/csrc/batch_prefill_jit_binding.cu']),
      'flashinfer/prefill.py':patch_python_dispatch(originals['flashinfer/prefill.py']),
      'flashinfer/data/csrc/batch_prefill_paged.cuh':patch_run(originals['flashinfer/data/csrc/batch_prefill_paged.cuh'],paged=True),
      'flashinfer/data/csrc/batch_prefill_paged.cu':patch_declarations(originals['flashinfer/data/csrc/batch_prefill_paged.cu']),
      'flashinfer/data/csrc/batch_prefill_ragged_kernel_inst.jinja':patch_jinja(originals['flashinfer/data/csrc/batch_prefill_ragged_kernel_inst.jinja']),
      'flashinfer/data/csrc/batch_prefill_paged_kernel_inst.jinja':patch_jinja(originals['flashinfer/data/csrc/batch_prefill_paged_kernel_inst.jinja']),
    }
    diffs=[];modified={}
    for rel,new in changes.items():
        target=out/rel;target.unlink();target.write_text(new);modified[rel]=sha_text(new)
        diffs.extend(difflib.unified_diff(originals[rel].splitlines(True),new.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
    record={'schema':2,'selector_version':'4.1.0','base_version':'0.7.0','base_hashes':EXPECTED,'modified_hashes':modified,
      'plan_vector_size':16,'policies':{'native':0,'resource_cap':1},'default_policy':'native','native_runtime_policy':True,
      'kernel_symbol_isolation':True,'native_kernel_source_unchanged':True,'symbol_hashes':symbol_hashes,
      'final_tactic':'environment/operation/execution-specific confidence-gated cache; miss/reject => native',
      'device_math_unchanged':True,'descriptor_order_unchanged':True,'production_promoted':False}
    (out/'RESOURCE_BINDING.json').write_text(json.dumps(record,indent=2)+'\n')
    (out/'resource-v4.1.patch').write_text(''.join(diffs));return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    print(json.dumps(prepare(**vars(p.parse_args())),indent=2))
