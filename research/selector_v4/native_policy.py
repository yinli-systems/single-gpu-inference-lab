"""Explicit native/cap plan policies for the isolated v4.1 research ABI."""
from __future__ import annotations

POLICY_ARGUMENTS = (
    "float_workspace_buffer", "int_workspace_buffer", "page_locked_int_workspace_buffer",
    "qo_indptr", "kv_indptr", "kv_len_arr", "total_num_rows", "batch_size",
    "num_qo_heads", "num_kv_heads", "page_size", "enable_cuda_graph", "head_dim_qk",
    "head_dim_vo", "causal", "window_left", "fixed_split_size", "disable_split_kv",
    "num_colocated_ctas", "uniform_q_len",
)
PLAN = "BatchPrefillWithKVCachePlan"
RESOURCE = "BatchPrefillWithKVCacheResourcePlan"


def patch_native_entry(source: str) -> str:
    start = source.index("Array<int64_t> " + PLAN + "(")
    end = source.index("Array<int64_t> BatchPrefillWithKVCacheWorkspaceSize(", start)
    function = source[start:end]
    brace = function.index("{")
    signature = function[:brace]
    for name in POLICY_ARGUMENTS:
        if name not in signature:
            raise ValueError("unreviewed plan signature: " + name)
    ret = "  return Array(plan_info.ToVector());"
    if function.count(ret) != 1:
        raise ValueError("unreviewed plan return")
    new_signature = signature.replace(PLAN + "(\n", RESOURCE + "(\n    int64_t resource_policy,", 1)
    body = function[brace:]
    body = body.replace("{\n", "{\n  TVM_FFI_ICHECK(resource_policy == 0 || resource_policy == 1)\n      << \"Invalid experimental resource policy\";\n", 1)
    body = body.replace(ret, "  plan_info.resource_cap = resource_policy == 1;\n" + ret)
    native = new_signature + body
    legacy = signature + "{\n  // Existing entry remains fail-closed and defaults to native.\n  return " + RESOURCE + "(0,\n      " + ", ".join(POLICY_ARGUMENTS) + ");\n}\n\n"
    return source[:start] + native + legacy + source[end:]


def patch_native_binding(source: str) -> str:
    start = source.index("Array<int64_t> " + PLAN + "(")
    end = source.index(");", start) + 2
    original = source[start:end]
    resource = original.replace(PLAN + "(\n", RESOURCE + "(\n    int64_t resource_policy,", 1)
    source = source[:end] + "\n\n" + resource + source[end:]
    anchor = "TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, " + PLAN + ");"
    if source.count(anchor) != 1:
        raise ValueError("unreviewed FFI export")
    return source.replace(anchor, anchor + "\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(plan_resource, " + RESOURCE + ");")


def patch_python_dispatch(source: str) -> str:
    anchor = "        prewarm_paged_kv_stride_variant=prewarm_paged_kv_stride_variant,\n"
    if source.count(anchor) != 1:
        raise ValueError("standard FA2 namespace anchor")
    source = source.replace(anchor, anchor + '        plan_resource=getattr(module, "plan_resource", None) if backend == "fa2" else None,\n        resource_kernel_isolation=True if backend == "fa2" and getattr(module, "plan_resource", None) is not None else False,\n')
    old = "            self._plan_info = self._cached_module.plan(\n                *args,\n            )"
    if source.count(old) != 2:
        raise ValueError("standard ragged/paged plan dispatch anchors")
    new = """            sgi_policy = getattr(self, "_sgi_resource_policy", None)
            if sgi_policy is None:
                self._plan_info = self._cached_module.plan(*args)
            else:
                if type(sgi_policy) is not int or sgi_policy not in (0, 1):
                    raise ValueError("Invalid experimental resource policy")
                if self._backend != "fa2" or self._jit_module is not None:
                    raise RuntimeError("Resource policy requires standard FA2")
                resource_plan = getattr(self._cached_module, "plan_resource", None)
                if resource_plan is None:
                    raise RuntimeError("Resource policy FFI entry unavailable")
                self._plan_info = resource_plan(sgi_policy, *args)"""
    return source.replace(old, new)
