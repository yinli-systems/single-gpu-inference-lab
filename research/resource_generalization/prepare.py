"""Source-bound, process-immutable native variants; never changes shared installs."""
from pathlib import Path
import argparse,difflib,hashlib,json,shutil
EXPECTED='e66bce2652c0a3c5b54f88510aff2b4b48ff6a52c4ee23fa97f91fa06a935b87'

def sha(x):return hashlib.sha256(x).hexdigest()

def transform(original,mode):
 if mode not in ('off','cap','wide'):raise ValueError('unknown mode')
 if sha(original.encode())!=EXPECTED:raise ValueError('unreviewed official header')
 s=original
 for suffix in ['Ragged','Paged']:
  start=s.index('cudaError_t BatchPrefillWith'+suffix+'KVCacheDispatchedImpl(')
  stop=s.index('cudaError_t BatchPrefillWith'+suffix+'KVCacheDispatched(',start)
  body=s[start:stop]
  marker='  const int max_smem_per_threadblock ='
  if body.count(marker)!=1:raise ValueError('ambiguous resource budget anchor')
  eligible='''
  // Resource-policy experiment: no request identity or exact-length special case.
  // Unknown resource regimes and split-KV preserve the native path.
  const bool sgi_eligible = HEAD_DIM_QK == 128 && HEAD_DIM_VO == 128 &&
      CTA_TILE_Q == 128 && sizeof(DTypeQ) == 2 && sizeof(DTypeKV) == 2 &&
      AttentionVariant::use_softmax && tmp_v == nullptr &&
      max_smem_per_sm == 102400 && max_smem_per_block_optin >= 65536;
'''
  body=body.replace(marker,eligible+marker)
  if mode=='wide':
   old='min(max_smem_per_sm / num_ctas_per_sm, max_smem_per_block_optin);'
   if body.count(old)!=1:raise ValueError('budget equation changed')
   body=body.replace(old,'min(max_smem_per_sm / (sgi_eligible ? 1 : num_ctas_per_sm), max_smem_per_block_optin);')
  needle=('size_t smem_size = sizeof(SmemStorage);' if suffix=='Ragged' else
          'size_t smem_size = sizeof(typename KTraits::SharedStoragePaged);')
  if body.count(needle)!=1:raise ValueError('launch storage anchor changed')
  extra='\n          const size_t sgi_original_smem = smem_size;\n'
  if mode=='cap':extra+='          if (sgi_eligible && smem_size == 49152) smem_size = 65536;\n'
  # Stable allowed maximum across runtime split transitions, no env mutation.
  extra+='''
          const size_t sgi_allowed_smem =
              (HEAD_DIM_QK == 128 && HEAD_DIM_VO == 128 && CTA_TILE_Q == 128 &&
               sizeof(DTypeQ) == 2 && sizeof(DTypeKV) == 2 &&
               max_smem_per_sm == 102400 && max_smem_per_block_optin >= 65536)
              ? std::max<size_t>(smem_size, 65536) : smem_size;
'''
  body=body.replace(needle,needle+extra)
  attr='cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)'
  if body.count(attr)!=1:raise ValueError('attribute anchor changed')
  body=body.replace(attr,'cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sgi_allowed_smem)')
  s=s[:start]+body+s[stop:]
 # All device functions before host dispatch remain text-identical. Wide changes specialization.
 first=original.index('cudaError_t BatchPrefillWithRaggedKVCacheDispatchedImpl(')
 # Host-template declarations immediately precede function; changed body starts afterwards.
 if s[:first]!=original[:first]:raise AssertionError('device source altered')
 return s

def prepare(source,out,mode):
 if out.exists():raise FileExistsError('preserve overlay')
 original=(source/'flashinfer/data/include/flashinfer/attention/prefill.cuh').read_text()
 new=transform(original,mode)
 shutil.copytree(source/'flashinfer',out/'flashinfer',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for f in source.glob('*.dist-info'):shutil.copytree(f,out/f.name)
 h=out/'flashinfer/data/include/flashinfer/attention/prefill.cuh';h.write_text(new)
 rec=dict(mode=mode,base_version='0.7.0',base_sha256=sha(original.encode()),modified_sha256=sha(new.encode()),
          device_source_unchanged=True,device_specialization_changes=(mode=='wide'),
          process_immutable=True,ordered_descriptors_unchanged=True,production_promoted=False)
 (out/'RESOURCE_BINDING.json').write_text(json.dumps(rec,indent=2)+'\n')
 (out/'resource.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile='a/include/flashinfer/attention/prefill.cuh',tofile='b/include/flashinfer/attention/prefill.cuh')))
 return rec
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['off','cap','wide'],required=True)
 a=p.parse_args();print(json.dumps(prepare(a.source,a.out,a.mode)))
