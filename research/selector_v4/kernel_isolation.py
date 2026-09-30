"""Source-exact v4 FlashInfer patching with isolated native/capped kernel symbols."""
from __future__ import annotations
import hashlib


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def replace_one(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new)


def template_function_span(source: str, function_name: str) -> tuple[int, int]:
    name_pos = source.index(function_name + "(")
    start = source.rfind("template <", 0, name_pos)
    if start < 0 or (start > 0 and source[start - 1] != "\n"):
        raise ValueError("template start " + function_name)
    brace = source.index("{", name_pos)
    depth = 0
    for pos in range(brace, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return start, pos + 1
    raise ValueError("unterminated function " + function_name)


def duplicate_kernel(source: str, original: str, clone: str) -> tuple[str, str, str]:
    if clone in source:
        raise ValueError("clone already exists")
    start, end = template_function_span(source, original)
    original_block = source[start:end]
    if original_block.count(original) != 1:
        raise ValueError("ambiguous kernel name")
    clone_block = original_block.replace(original, clone, 1)
    patched = source[:end] + "\n\n" + clone_block + source[end:]
    check_start, check_end = template_function_span(patched, original)
    if patched[check_start:check_end] != original_block:
        raise AssertionError("native kernel changed")
    return patched, sha_text(original_block), sha_text(clone_block)


def patch_scheduler(source: str) -> str:
    start = source.index("struct PrefillPlanInfo {")
    stop = source.index("template <bool MATERIALIZE", start)
    body = source[start:stop]
    body = replace_one(body, "  bool enable_cuda_graph;\n  bool split_kv;",
                       "  bool enable_cuda_graph;\n  bool split_kv;\n  bool resource_cap;", "plan field")
    body = replace_one(body, "        enable_cuda_graph(false),\n        split_kv(false) {}",
                       "        enable_cuda_graph(false),\n        split_kv(false),\n        resource_cap(false) {}", "plan init")
    body = replace_one(body, "            block_valid_mask_offset,\n            enable_cuda_graph,\n            split_kv};",
                       "            block_valid_mask_offset,\n            enable_cuda_graph,\n            split_kv,\n            resource_cap};", "to vector")
    body = replace_one(body, "    if (vec.size() != 15) {", "    if (vec.size() != 16) {", "vector size")
    body = replace_one(body, "vec.size() should be 15", "vec.size() should be 16", "vector error")
    body = replace_one(body, "    split_kv = vec[14];\n  }",
                       "    split_kv = vec[14];\n    resource_cap = vec[15];\n  }", "from vector")
    return source[:start] + body + source[stop:]


def patch_declarations(source: str) -> str:
    old = "                                                   cudaStream_t stream);"
    count = source.count(old)
    if count not in (1, 2):
        raise ValueError(f"dispatch declarations: {count}")
    return source.replace(old, "                                                   cudaStream_t stream, bool resource_cap);")


def patch_jinja(source: str) -> str:
    return replace_one(source, "bool enable_pdl, cudaStream_t stream);",
                       "bool enable_pdl, cudaStream_t stream, bool resource_cap);", "jinja")


def patch_run(source: str, *, paged: bool) -> str:
    if paged:
        old = "PagedParams>(params, tmp_v, tmp_s, enable_pdl, stream);"
        if source.count(old) != 3:
            raise ValueError("paged run calls")
        return source.replace(old, "PagedParams>(params, tmp_v, tmp_s, enable_pdl, stream, plan_info.resource_cap);")
    return replace_one(source,
                       "RaggedParams>(params, tmp_v, tmp_s, enable_pdl, stream);",
                       "RaggedParams>(params, tmp_v, tmp_s, enable_pdl, stream, plan_info.resource_cap);",
                       "ragged run call")


def patch_prefill(source: str) -> tuple[str, dict[str, str]]:
    source, ragged_native, ragged_clone = duplicate_kernel(
        source, "BatchPrefillWithRaggedKVCacheKernel", "BatchPrefillWithRaggedKVCacheResourceKernel")
    source, paged_native, paged_clone = duplicate_kernel(
        source, "BatchPrefillWithPagedKVCacheKernel", "BatchPrefillWithPagedKVCacheResourceKernel")
    source = replace_one(source,
        "float* tmp_s, bool enable_pdl,\n                                                        cudaStream_t stream) {",
        "float* tmp_s, bool enable_pdl,\n                                                        cudaStream_t stream, bool resource_cap) {",
        "ragged impl signature")
    source = replace_one(source,
        "float* tmp_s, bool enable_pdl,\n                                                    cudaStream_t stream) {",
        "float* tmp_s, bool enable_pdl,\n                                                    cudaStream_t stream, bool resource_cap = false) {",
        "ragged outer signature")
    source = replace_one(source,
        "      params, tmp_v, tmp_s, enable_pdl, stream))",
        "      params, tmp_v, tmp_s, enable_pdl, stream, resource_cap))",
        "ragged impl call")
    source = replace_one(source,
        "                                                       bool enable_pdl, cudaStream_t stream) {",
        "                                                       bool enable_pdl, cudaStream_t stream,\n                                                       bool resource_cap) {",
        "paged impl signature")
    source = replace_one(source,
        "                                                   float* tmp_s, bool enable_pdl,\n                                                   cudaStream_t stream) {",
        "                                                   float* tmp_s, bool enable_pdl,\n                                                   cudaStream_t stream, bool resource_cap = false) {",
        "paged outer signature")
    source = replace_one(source,
        "                                                                      enable_pdl, stream))",
        "                                                                      enable_pdl, stream, resource_cap))",
        "paged impl call")

    for suffix, original, clone, smem_decl in (
        ("Ragged", "BatchPrefillWithRaggedKVCacheKernel", "BatchPrefillWithRaggedKVCacheResourceKernel",
         "size_t smem_size = sizeof(SmemStorage);"),
        ("Paged", "BatchPrefillWithPagedKVCacheKernel", "BatchPrefillWithPagedKVCacheResourceKernel",
         "size_t smem_size = sizeof(typename KTraits::SharedStoragePaged);"),
    ):
        start = source.index("cudaError_t BatchPrefillWith" + suffix + "KVCacheDispatchedImpl(")
        stop = source.index("cudaError_t BatchPrefillWith" + suffix + "KVCacheDispatched(", start)
        body = source[start:stop]
        marker = "  const int max_smem_per_threadblock ="
        eligibility = (
            "\n  // v4 candidate eligibility; final tactic is selected by deployment-mode calibration.\n"
            "  const bool sgi_eligible = HEAD_DIM_QK == 128 && HEAD_DIM_VO == 128 &&\n"
            "      CTA_TILE_Q == 128 && sizeof(DTypeQ) == 2 && sizeof(DTypeKV) == 2 &&\n"
            "      AttentionVariant::use_softmax && tmp_v == nullptr &&\n"
            "      max_smem_per_sm >= 102400 && max_smem_per_block_optin >= 65536;\n")
        if body.count(marker) != 1:
            raise ValueError("resource budget anchor " + suffix)
        body = body.replace(marker, eligibility + marker)
        if body.count(smem_decl) != 1:
            raise ValueError("smem anchor " + suffix)
        old_kernel = ("auto kernel = " + original + "<KTraits, Params>;" if suffix == "Ragged" else
                      "auto kernel = " + original + "<SAME_KV_STRIDES, KTraits, Params>;")
        native_expr = (original + "<KTraits, Params>" if suffix == "Ragged" else
                       original + "<SAME_KV_STRIDES, KTraits, Params>")
        cap_expr = (clone + "<KTraits, Params>" if suffix == "Ragged" else
                    clone + "<SAME_KV_STRIDES, KTraits, Params>")
        replacement = smem_decl + (
            "\n          const bool sgi_use_cap = resource_cap && sgi_eligible && smem_size == 49152;\n"
            "          if (resource_cap && !sgi_use_cap) return cudaErrorInvalidValue;\n"
            f"          auto native_kernel = {native_expr};\n"
            f"          auto capped_kernel = {cap_expr};\n"
            "          auto kernel = sgi_use_cap ? capped_kernel : native_kernel;\n"
            "          if (sgi_use_cap) smem_size = 65536;\n"
            "          // Attribute state is isolated because the two tactics use distinct symbols.\n")
        body = replace_one(body, smem_decl + "\n          " + old_kernel, replacement,
                           "isolated kernel launch " + suffix)
        source = source[:start] + body + source[stop:]
    return source, {
        "ragged_native_sha256": ragged_native,
        "ragged_clone_sha256": ragged_clone,
        "paged_native_sha256": paged_native,
        "paged_clone_sha256": paged_clone,
    }


def selector_decision_cpp() -> str:
    return '''  // Selector v4 candidate-pool gate. It never overrides a calibrated native decision.
  auto* sgi_qo_indptr = static_cast<IdType*>(qo_indptr.data_ptr());
  auto* sgi_kv_len = static_cast<IdType*>(kv_len_arr.data_ptr());
  uint64_t sgi_max_cached = 0, sgi_discordant = 0, sgi_concordant = 0;
  for (int64_t i = 0; i < batch_size; ++i) {
    int64_t qi = static_cast<int64_t>(sgi_qo_indptr[i + 1] - sgi_qo_indptr[i]);
    int64_t kvi = static_cast<int64_t>(sgi_kv_len[i]);
    uint64_t ci = kvi > qi ? static_cast<uint64_t>(kvi - qi) : 0;
    sgi_max_cached = std::max(sgi_max_cached, ci);
    for (int64_t j = 0; j < i; ++j) {
      int64_t qj = static_cast<int64_t>(sgi_qo_indptr[j + 1] - sgi_qo_indptr[j]);
      int64_t kvj = static_cast<int64_t>(sgi_kv_len[j]);
      int64_t cj = kvj > qj ? kvj - qj : 0;
      int64_t dq = qi - qj, dc = static_cast<int64_t>(ci) - cj;
      if ((dq > 0 && dc > 0) || (dq < 0 && dc < 0)) ++sgi_concordant;
      else if ((dq > 0 && dc < 0) || (dq < 0 && dc > 0)) ++sgi_discordant;
    }
  }
  int sgi_dev_id = 0, sgi_num_sm = 0;
  TVM_FFI_ICHECK(cudaGetDevice(&sgi_dev_id) == cudaSuccess) << "Failed to query CUDA device";
  TVM_FFI_ICHECK(cudaDeviceGetAttribute(&sgi_num_sm, cudaDevAttrMultiProcessorCount, sgi_dev_id) == cudaSuccess)
      << "Failed to query SM count";
  const uint64_t sgi_den = 5 * static_cast<uint64_t>(num_kv_heads);
  const uint64_t sgi_wave_threshold = std::max<uint64_t>(40,
      (12 * static_cast<uint64_t>(sgi_num_sm) + sgi_den - 1) / sgi_den);
  const bool sgi_pairing_gate = batch_size >= 5 && sgi_max_cached >= 8192 &&
      sgi_concordant == 0 && sgi_discordant >= static_cast<uint64_t>(batch_size - 1);
  plan_info.resource_cap = sgi_pairing_gate && !plan_info.split_kv &&
      static_cast<uint64_t>(plan_info.padded_batch_size) >= sgi_wave_threshold;
'''
