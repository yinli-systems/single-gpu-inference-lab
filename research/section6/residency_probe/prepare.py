"""Build an isolated host-launch-only shared-memory intervention. No live edits."""
import argparse,difflib,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists():raise FileExistsError('preserve existing overlay')
source=a.source/'flashinfer';target=a.out/'flashinfer'
assert source.is_dir()
f=source/'data/include/flashinfer/attention/prefill.cuh';original=f.read_text()
assert hashlib.sha256(original.encode()).hexdigest()=='e66bce2652c0a3c5b54f88510aff2b4b48ff6a52c4ee23fa97f91fa06a935b87', 'unreviewed source'
start=original.index('cudaError_t BatchPrefillWithRaggedKVCacheDispatchedImpl(')
end=original.index('cudaError_t BatchPrefillWithRaggedKVCacheDispatched(',start)
body=original[start:end]
needle='size_t smem_size = sizeof(SmemStorage);'
assert body.count(needle)==1
addition=r"""size_t smem_size = sizeof(SmemStorage);
          // SECTION6 DIAGNOSTIC: host launch intervention only; device kernel unchanged.
          // Explicit FA2/D128/16-bit/unsplit/34-descriptor witness only.
          const bool sgi_eligible = HEAD_DIM_QK == 128 && HEAD_DIM_VO == 128 &&
              CTA_TILE_Q == 128 && sizeof(DTypeQ) == 2 && sizeof(DTypeKV) == 2 &&
              num_qo_heads == 32 && num_kv_heads == 8 && padded_batch_size == 34 &&
              tmp_v == nullptr && smem_size == 49152 && max_smem_per_block_optin >= 65536;
          const char* sgi_env = std::getenv("SGI_FA2_RESIDENCY_PROBE");
          if (sgi_env && !((sgi_env[0] == '0' || sgi_env[0] == '1') && sgi_env[1] == '\0')) {
            return cudaErrorInvalidValue;
          }
          const bool sgi_enabled = sgi_env && sgi_env[0] == '1';
          if (sgi_enabled && !sgi_eligible) return cudaErrorNotSupported;
          // Keep the allowed maximum identical for both captured graphs. Only
          // actual per-launch dynamic bytes differ (48 KiB versus 64 KiB).
          const size_t sgi_launch_limit = sgi_eligible ? 65536 : smem_size;
          if (sgi_enabled) smem_size = 65536;
"""
body=body.replace(needle,addition)
needle2='cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)'
assert body.count(needle2)==1
body=body.replace(needle2,'cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sgi_launch_limit)')
modified=original[:start]+body+original[end:]
modified='#include <cstdlib>\n'+modified
# Device kernel text is unchanged. This is not a new compute kernel.
k0=original.index('template <typename KTraits, typename Params>\n__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchPrefillWithRaggedKVCacheKernel(')
k1=original.index('cudaError_t BatchPrefillWithRaggedKVCacheDispatchedImpl(',k0)
assert original[k0:k1] in modified
# All exact-source checks precede copying or writes.
shutil.copytree(source,target)
for info in a.source.glob("*.dist-info"):
 shutil.copytree(info,a.out/info.name)
f=target/'data/include/flashinfer/attention/prefill.cuh'
f.write_text(modified)
(a.out/'host-launch-only.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='a/data/include/flashinfer/attention/prefill.cuh',tofile='b/data/include/flashinfer/attention/prefill.cuh')))
rec=dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),modified_sha256=hashlib.sha256(modified.encode()).hexdigest(),
 device_region_sha256=hashlib.sha256(original[k0:k1].encode()).hexdigest(),device_region_text_unchanged=True,
 opt_in_env='SGI_FA2_RESIDENCY_PROBE',supported_witness_only=True,production_API=False)
(a.out/'SOURCE_BINDING.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec))
