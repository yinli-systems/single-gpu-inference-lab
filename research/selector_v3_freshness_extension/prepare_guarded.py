"""Create isolated FlashInfer 0.7 overlays for a plan-time guarded resource-cap tactic."""
from __future__ import annotations
from pathlib import Path
import argparse,difflib,hashlib,json,os,shutil
EXPECTED={
 'flashinfer/data/include/flashinfer/attention/prefill.cuh':'e66bce2652c0a3c5b54f88510aff2b4b48ff6a52c4ee23fa97f91fa06a935b87',
 'flashinfer/data/include/flashinfer/attention/scheduler.cuh':'425ba78815ba77ae848c67507d1af6b024d05edd257a7d9b331166afcef9b17c',
 'flashinfer/data/csrc/batch_prefill.cu':'4190105e8b5c93621af44356b9aa50861560a5fbf8ff3279301487860604d932',
 'flashinfer/data/csrc/batch_prefill_paged.cuh':'52fa2dfb066729e7ddac10509d0c2d2a99433622055acb146e2938460aec432e',
 'flashinfer/data/csrc/batch_prefill_paged.cu':'7f9aaa45879034eb1b978ad1a43901fe22a5e84282d61f72e700fd2e77067dbd',
 'flashinfer/data/csrc/batch_prefill_ragged_kernel_inst.jinja':'4e177455f914c3665cf38728883d74c076f368c1bea3736c0033675de342f687',
 'flashinfer/data/csrc/batch_prefill_paged_kernel_inst.jinja':'1e41a0ea9795847272674c1058a9d72f220facf296037cbf1f95ce6b77225a63',
}
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_text(s):return sha_bytes(s.encode())
def one(s,old,new,label):
 if s.count(old)!=1:raise ValueError(f'{label}: expected one anchor, got {s.count(old)}')
 return s.replace(old,new)

def patch_scheduler(s,mode):
 start=s.index('struct PrefillPlanInfo {')
 stop=s.index('template <bool MATERIALIZE',start)
 body=s[start:stop]
 body=one(body,'  bool enable_cuda_graph;\n  bool split_kv;','  bool enable_cuda_graph;\n  bool split_kv;\n  bool resource_cap;','plan field')
 body=one(body,'        enable_cuda_graph(false),\n        split_kv(false) {}','        enable_cuda_graph(false),\n        split_kv(false),\n        resource_cap(false) {}','plan init')
 body=one(body,'            block_valid_mask_offset,\n            enable_cuda_graph,\n            split_kv};','            block_valid_mask_offset,\n            enable_cuda_graph,\n            split_kv,\n            resource_cap};','to vector')
 body=one(body,'    if (vec.size() != 15) {','    if (vec.size() != 16) {','vector size check')
 body=one(body,'      err_msg << "PrefillPlanInfo::FromVector: vec.size() should be 15, but got " << vec.size();','      err_msg << "PrefillPlanInfo::FromVector: vec.size() should be 16, but got " << vec.size();','vector error')
 body=one(body,'    split_kv = vec[14];\n  }','    split_kv = vec[14];\n    resource_cap = vec[15];\n  }','from vector')
 return s[:start]+body+s[stop:]

def patch_prefill(s):
 s=one(s,'float* tmp_s, bool enable_pdl,\n                                                        cudaStream_t stream) {','float* tmp_s, bool enable_pdl,\n                                                        cudaStream_t stream, bool resource_cap) {','ragged impl signature')
 s=one(s,'float* tmp_s, bool enable_pdl,\n                                                    cudaStream_t stream) {','float* tmp_s, bool enable_pdl,\n                                                    cudaStream_t stream, bool resource_cap = false) {','ragged outer signature')
 s=one(s,'      params, tmp_v, tmp_s, enable_pdl, stream))','      params, tmp_v, tmp_s, enable_pdl, stream, resource_cap))','ragged impl call')
 s=one(s,'                                                       bool enable_pdl, cudaStream_t stream) {','                                                       bool enable_pdl, cudaStream_t stream,\n                                                       bool resource_cap) {','paged impl signature')
 s=one(s,'                                                   float* tmp_s, bool enable_pdl,\n                                                   cudaStream_t stream) {','                                                   float* tmp_s, bool enable_pdl,\n                                                   cudaStream_t stream, bool resource_cap = false) {','paged outer signature')
 s=one(s,'                                                                      enable_pdl, stream))','                                                                      enable_pdl, stream, resource_cap))','paged impl call')
 for suffix in ['Ragged','Paged']:
  start=s.index('cudaError_t BatchPrefillWith'+suffix+'KVCacheDispatchedImpl(')
  stop=s.index('cudaError_t BatchPrefillWith'+suffix+'KVCacheDispatched(',start)
  body=s[start:stop]
  marker='  const int max_smem_per_threadblock ='
  eligible=(
 '\n  // Optional, plan-selected resource tactic. Device math and descriptor order are unchanged.\n'
 '  const bool sgi_eligible = HEAD_DIM_QK == 128 && HEAD_DIM_VO == 128 &&\n'
 '      CTA_TILE_Q == 128 && sizeof(DTypeQ) == 2 && sizeof(DTypeKV) == 2 &&\n'
 '      AttentionVariant::use_softmax && tmp_v == nullptr &&\n'
 '      max_smem_per_sm == 102400 && max_smem_per_block_optin >= 65536;\n')
  if body.count(marker)!=1:raise ValueError('resource budget anchor')
  body=body.replace(marker,eligible+marker)
  needle='size_t smem_size = sizeof(SmemStorage);' if suffix=='Ragged' else 'size_t smem_size = sizeof(typename KTraits::SharedStoragePaged);'
  if body.count(needle)!=1:raise ValueError('smem anchor '+suffix)
  extra=(
 '\n          const size_t sgi_original_smem = smem_size;\n'
 '          if (resource_cap && sgi_eligible && smem_size == 49152) smem_size = 65536;\n'
 '          // Stable attribute across graph captures and guarded plan transitions.\n'
 '          const size_t sgi_allowed_smem = sgi_eligible ? std::max<size_t>(smem_size, 65536) : smem_size;\n')
  body=body.replace(needle,needle+extra)
  attr='cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)'
  if body.count(attr)!=1:raise ValueError('attribute anchor '+suffix)
  body=body.replace(attr,'cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sgi_allowed_smem)')
  s=s[:start]+body+s[stop:]
 return s

def patch_declarations(s):
 old='                                                   cudaStream_t stream);'
 count=s.count(old)
 if count not in (1,2):raise ValueError(f'dispatch declarations: {count}')
 return s.replace(old,'                                                   cudaStream_t stream, bool resource_cap);')

def patch_plan_decision(s,mode):
 marker='  TVM_FFI_ICHECK(status == cudaSuccess)\n      << "Failed to plan prefill with error: " << cudaGetErrorString(status);\n\n  return Array(plan_info.ToVector());'
 if mode=='off': decision='  plan_info.resource_cap = false;\n'
 elif mode=='cap': decision='  plan_info.resource_cap = true;\n'
 elif mode=='guarded':
  decision=(
 '  // Selector v3: an execution-plan tactic, not a semantic workload rule.\n'
 '  // It requires anti-monotone q/cached coupling, sufficient strict pairs,\n'
 '  // an unsplit plan, and a hardware-normalized descriptor threshold.\n'
 '  auto* sgi_qo_indptr = static_cast<IdType*>(qo_indptr.data_ptr());\n'
 '  auto* sgi_kv_len = static_cast<IdType*>(kv_len_arr.data_ptr());\n'
 '  uint64_t sgi_max_cached = 0, sgi_discordant = 0, sgi_concordant = 0;\n'
 '  for (int64_t i = 0; i < batch_size; ++i) {\n'
 '    int64_t qi = static_cast<int64_t>(sgi_qo_indptr[i + 1] - sgi_qo_indptr[i]);\n'
 '    int64_t kvi = static_cast<int64_t>(sgi_kv_len[i]);\n'
 '    uint64_t ci = kvi > qi ? static_cast<uint64_t>(kvi - qi) : 0;\n'
 '    sgi_max_cached = std::max(sgi_max_cached, ci);\n'
 '    for (int64_t j = 0; j < i; ++j) {\n'
 '      int64_t qj = static_cast<int64_t>(sgi_qo_indptr[j + 1] - sgi_qo_indptr[j]);\n'
 '      int64_t kvj = static_cast<int64_t>(sgi_kv_len[j]);\n'
 '      int64_t cj = kvj > qj ? kvj - qj : 0;\n'
 '      int64_t dq = qi - qj, dc = static_cast<int64_t>(ci) - cj;\n'
 '      if ((dq > 0 && dc > 0) || (dq < 0 && dc < 0)) ++sgi_concordant;\n'
 '      else if ((dq > 0 && dc < 0) || (dq < 0 && dc > 0)) ++sgi_discordant;\n'
 '    }\n'
 '  }\n'
 '  int sgi_dev_id = 0, sgi_num_sm = 0;\n'
 '  TVM_FFI_ICHECK(cudaGetDevice(&sgi_dev_id) == cudaSuccess) << "Failed to query CUDA device";\n'
 '  TVM_FFI_ICHECK(cudaDeviceGetAttribute(&sgi_num_sm, cudaDevAttrMultiProcessorCount, sgi_dev_id) == cudaSuccess)\n'
 '      << "Failed to query SM count";\n'
 '  const uint64_t sgi_hw_threshold = std::max<uint64_t>(40,\n'
 '      (2 * static_cast<uint64_t>(sgi_num_sm)) / static_cast<uint64_t>(num_kv_heads) + 1);\n'
 '  const bool sgi_pairing_gate = batch_size >= 5 && sgi_max_cached >= 8192 &&\n'
 '      sgi_concordant == 0 && sgi_discordant >= static_cast<uint64_t>(batch_size - 1);\n'
 '  plan_info.resource_cap = sgi_pairing_gate && !plan_info.split_kv &&\n'
 '      static_cast<uint64_t>(plan_info.padded_batch_size) >= sgi_hw_threshold;\n'
  )
 else: raise ValueError(mode)
 replacement=marker.replace('  return Array(plan_info.ToVector());',decision+'\n  return Array(plan_info.ToVector());')
 return one(s,marker,replacement,'plan wrapper decision')

def patch_run(s,paged=False):
 if paged:
  old='PagedParams>(params, tmp_v, tmp_s, enable_pdl, stream);'
  if s.count(old)!=3:raise ValueError(f'paged run calls {s.count(old)}')
  return s.replace(old,'PagedParams>(params, tmp_v, tmp_s, enable_pdl, stream, plan_info.resource_cap);')
 old='RaggedParams>(params, tmp_v, tmp_s, enable_pdl, stream);'
 if s.count(old)!=1:raise ValueError('ragged run call')
 return s.replace(old,'RaggedParams>(params, tmp_v, tmp_s, enable_pdl, stream, plan_info.resource_cap);')

def patch_jinja(s):
 old='bool enable_pdl, cudaStream_t stream);'
 if s.count(old)!=1:raise ValueError('jinja instantiation')
 return s.replace(old,'bool enable_pdl, cudaStream_t stream, bool resource_cap);')

def prepare(source,out,mode):
 if mode not in ('off','cap','guarded'):raise ValueError(mode)
 if out.exists():raise FileExistsError(out)
 originals={}
 for rel,h in EXPECTED.items():
  b=(source/rel).read_bytes()
  if sha_bytes(b)!=h:raise ValueError('unreviewed source '+rel)
  originals[rel]=b.decode()
 shutil.copytree(source/'flashinfer',out/'flashinfer',ignore=shutil.ignore_patterns('__pycache__','*.pyc'),copy_function=os.link)
 for f in source.glob('*.dist-info'):shutil.copytree(f,out/f.name,copy_function=os.link)
 changes={
  'flashinfer/data/include/flashinfer/attention/scheduler.cuh':patch_scheduler(originals['flashinfer/data/include/flashinfer/attention/scheduler.cuh'],mode),
  'flashinfer/data/include/flashinfer/attention/prefill.cuh':patch_prefill(originals['flashinfer/data/include/flashinfer/attention/prefill.cuh']),
  'flashinfer/data/csrc/batch_prefill.cu':patch_plan_decision(patch_run(patch_declarations(originals['flashinfer/data/csrc/batch_prefill.cu'])),mode),
  'flashinfer/data/csrc/batch_prefill_paged.cuh':patch_run(originals['flashinfer/data/csrc/batch_prefill_paged.cuh'],True),
  'flashinfer/data/csrc/batch_prefill_paged.cu':patch_declarations(originals['flashinfer/data/csrc/batch_prefill_paged.cu']),
 }
 for rel in ['flashinfer/data/csrc/batch_prefill_ragged_kernel_inst.jinja','flashinfer/data/csrc/batch_prefill_paged_kernel_inst.jinja']:
  src=(source/rel).read_text(); changes[rel]=patch_jinja(src); originals[rel]=src
 diffs=[]; hashes={}
 for rel,new in changes.items():
  target=out/rel
  target.unlink()  # break hardlink before writing; base wheel stays byte-identical
  target.write_text(new); hashes[rel]=sha_text(new)
  diffs.extend(difflib.unified_diff(originals[rel].splitlines(True),new.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
 rec=dict(mode=mode,base_version='0.7.0',base_hashes=EXPECTED,modified_hashes=hashes,
  plan_vector_size=16,guard_rule='strict_opposite && !split_kv && padded_batch_size>=max(40,floor(2*num_sms/num_kv_heads)+1)',
  selector_version=3,tactic_identity_schema='environment+ordered-geometry+execution-mode+plan-signature',
  descriptor_order_unchanged=True,device_math_unchanged=True,production_promoted=False)
 (out/'RESOURCE_BINDING.json').write_text(json.dumps(rec,indent=2)+'\n')
 (out/'resource-guard.patch').write_text(''.join(diffs))
 return rec
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['off','cap','guarded'],required=True)
 a=p.parse_args();print(json.dumps(prepare(a.source,a.out,a.mode),indent=2))
