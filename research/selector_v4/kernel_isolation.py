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


# V4.2 resource-only additions.  Every official native function/declaration is
# retained byte-for-byte; only parallel resource symbols are appended.
def _clone_template_function(source: str, original: str, clone: str,
                             replacements: tuple[tuple[str, str], ...] = ()) -> tuple[str, str, str]:
    if clone + "(" in source:
        raise ValueError("clone already exists " + clone)
    start, end = template_function_span(source, original)
    native = source[start:end]
    copied = native.replace(original, clone, 1)
    for old, new in replacements:
        if old not in copied:
            raise ValueError("clone transform missing " + old)
        copied = copied.replace(old, new)
    return source[:end] + "\n\n" + copied + source[end:], sha_text(native), sha_text(copied)


def _declaration_span(source: str, function_name: str) -> tuple[int, int]:
    pos = source.index(function_name + "(")
    start = source.rfind("template <", 0, pos)
    if start < 0:
        raise ValueError("template declaration start " + function_name)
    end = source.index(";", pos) + 1
    return start, end


def clone_template_declaration(source: str, original: str, clone: str) -> str:
    if clone + "(" in source:
        raise ValueError("declaration clone exists")
    start, end = _declaration_span(source, original)
    block = source[start:end]
    if block.count(original) != 1:
        raise ValueError("ambiguous declaration " + original)
    return source[:end] + "\n\n" + block.replace(original, clone, 1) + source[end:]


def clone_template_instantiation(source: str, original: str, clone: str) -> str:
    if clone + "<" in source:
        raise ValueError("instantiation clone exists")
    needle = "template cudaError_t " + original + "<"
    pos = source.index(needle)
    start = source.rfind("\n", 0, pos) + 1
    end = source.index(";", pos) + 1
    block = source[start:end]
    if block.count(original) != 1:
        raise ValueError("ambiguous instantiation " + original)
    return source[:end] + "\n" + block.replace(original, clone, 1) + source[end:]


def _resource_eligibility(kernel_line: str) -> str:
    return kernel_line + """
          const bool sgi_eligible = HEAD_DIM_QK == 128 && HEAD_DIM_VO == 128 &&
              CTA_TILE_Q == 128 && sizeof(DTypeQ) == 2 && sizeof(DTypeKV) == 2 &&
              AttentionVariant::use_softmax && tmp_v == nullptr && smem_size == 49152 &&
              max_smem_per_sm >= 102400 && max_smem_per_block_optin >= 65536;
          if (!sgi_eligible) return cudaErrorInvalidValue;
          smem_size = 65536;"""


def patch_resource_prefill(source: str) -> tuple[str, dict[str, str]]:
    source, ragged_native, ragged_clone = duplicate_kernel(
        source, "BatchPrefillWithRaggedKVCacheKernel",
        "BatchPrefillWithRaggedKVCacheResourceKernel")
    source, paged_native, paged_clone = duplicate_kernel(
        source, "BatchPrefillWithPagedKVCacheKernel",
        "BatchPrefillWithPagedKVCacheResourceKernel")

    source, ragged_impl_native, ragged_impl_clone = _clone_template_function(
        source, "BatchPrefillWithRaggedKVCacheDispatchedImpl",
        "BatchPrefillWithRaggedKVCacheResourceDispatchedImpl",
        (("BatchPrefillWithRaggedKVCacheKernel",
          "BatchPrefillWithRaggedKVCacheResourceKernel"),))
    line = "          auto kernel = BatchPrefillWithRaggedKVCacheResourceKernel<KTraits, Params>;"
    if source.count(line) != 1:
        raise ValueError("ragged resource kernel anchor")
    source = source.replace(line, _resource_eligibility(line), 1)
    source, ragged_outer_native, ragged_outer_clone = _clone_template_function(
        source, "BatchPrefillWithRaggedKVCacheDispatched",
        "BatchPrefillWithRaggedKVCacheResourceDispatched",
        (("BatchPrefillWithRaggedKVCacheDispatchedImpl",
          "BatchPrefillWithRaggedKVCacheResourceDispatchedImpl"),))

    source, paged_impl_native, paged_impl_clone = _clone_template_function(
        source, "BatchPrefillWithPagedKVCacheDispatchedImpl",
        "BatchPrefillWithPagedKVCacheResourceDispatchedImpl",
        (("BatchPrefillWithPagedKVCacheKernel",
          "BatchPrefillWithPagedKVCacheResourceKernel"),))
    line = "          auto kernel = BatchPrefillWithPagedKVCacheResourceKernel<SAME_KV_STRIDES, KTraits, Params>;"
    if source.count(line) != 1:
        raise ValueError("paged resource kernel anchor")
    source = source.replace(line, _resource_eligibility(line), 1)
    source, paged_outer_native, paged_outer_clone = _clone_template_function(
        source, "BatchPrefillWithPagedKVCacheDispatched",
        "BatchPrefillWithPagedKVCacheResourceDispatched",
        (("BatchPrefillWithPagedKVCacheDispatchedImpl",
          "BatchPrefillWithPagedKVCacheResourceDispatchedImpl"),))

    start,end = template_function_span(source, "BatchPrefillWithRaggedKVCacheResourceDispatchedImpl")
    ragged_impl_clone = sha_text(source[start:end])
    start,end = template_function_span(source, "BatchPrefillWithPagedKVCacheResourceDispatchedImpl")
    paged_impl_clone = sha_text(source[start:end])
    return source, {
        "ragged_native_sha256": ragged_native,
        "ragged_clone_sha256": ragged_clone,
        "paged_native_sha256": paged_native,
        "paged_clone_sha256": paged_clone,
        "ragged_dispatch_impl_native_sha256": ragged_impl_native,
        "ragged_dispatch_impl_clone_sha256": ragged_impl_clone,
        "ragged_dispatch_native_sha256": ragged_outer_native,
        "ragged_dispatch_clone_sha256": ragged_outer_clone,
        "paged_dispatch_impl_native_sha256": paged_impl_native,
        "paged_dispatch_impl_clone_sha256": paged_impl_clone,
        "paged_dispatch_native_sha256": paged_outer_native,
        "paged_dispatch_clone_sha256": paged_outer_clone,
    }


def patch_resource_declarations(source: str, *, paged_only: bool = False) -> str:
    if not paged_only:
        source = clone_template_declaration(
            source, "BatchPrefillWithRaggedKVCacheDispatched",
            "BatchPrefillWithRaggedKVCacheResourceDispatched")
    return clone_template_declaration(
        source, "BatchPrefillWithPagedKVCacheDispatched",
        "BatchPrefillWithPagedKVCacheResourceDispatched")


def patch_resource_jinja(source: str, *, paged: bool) -> str:
    original = ("BatchPrefillWithPagedKVCacheDispatched" if paged
                else "BatchPrefillWithRaggedKVCacheDispatched")
    clone = ("BatchPrefillWithPagedKVCacheResourceDispatched" if paged
             else "BatchPrefillWithRaggedKVCacheResourceDispatched")
    return clone_template_instantiation(source, original, clone)
