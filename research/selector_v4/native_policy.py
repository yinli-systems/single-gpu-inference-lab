"""V4.2 resource-only host/FFI/Python additions.

The official native plan and run entry points are never rewritten.  New resource
entries accept the official 15-field plan and dispatch only to isolated resource
symbols.
"""
from __future__ import annotations
import ast
import re

RAGGED_RUN = "BatchPrefillWithRaggedKVCacheRun"
RAGGED_RESOURCE_RUN = "BatchPrefillWithRaggedKVCacheResourceRun"
PAGED_RUN = "BatchPrefillWithPagedKVCacheRun"
PAGED_RESOURCE_RUN = "BatchPrefillWithPagedKVCacheResourceRun"
RAGGED_DISPATCH = "BatchPrefillWithRaggedKVCacheDispatched"
RAGGED_RESOURCE_DISPATCH = "BatchPrefillWithRaggedKVCacheResourceDispatched"
PAGED_DISPATCH = "BatchPrefillWithPagedKVCacheDispatched"
PAGED_RESOURCE_DISPATCH = "BatchPrefillWithPagedKVCacheResourceDispatched"


def _function_span(source: str, name: str) -> tuple[int, int]:
    pos = source.index(name + "(")
    line = source.rfind("\n", 0, pos) + 1
    if not source[line:pos].strip() in ("void", "Array<int64_t>"):
        # Walk to a plausible return-type line, but never cross a blank line.
        prev = source.rfind("\n", 0, line - 1) + 1
        if source[prev:pos].lstrip().startswith(("void ", "Array<int64_t> ")):
            line = prev
        else:
            raise ValueError("function start " + name)
    brace = source.index("{", pos)
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == "{": depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0: return line, i + 1
    raise ValueError("unterminated function " + name)


def clone_host_run(source: str, *, paged: bool) -> tuple[str, str, str]:
    original = PAGED_RUN if paged else RAGGED_RUN
    clone = PAGED_RESOURCE_RUN if paged else RAGGED_RESOURCE_RUN
    dispatch = PAGED_DISPATCH if paged else RAGGED_DISPATCH
    resource_dispatch = PAGED_RESOURCE_DISPATCH if paged else RAGGED_RESOURCE_DISPATCH
    if clone in source: raise ValueError("resource host run already exists")
    start, end = _function_span(source, original); native = source[start:end]
    copied = native.replace(original, clone, 1)
    if dispatch not in copied: raise ValueError("host dispatch anchor " + original)
    copied = copied.replace(dispatch, resource_dispatch)
    return source[:end] + "\n\n" + copied + source[end:], native, copied


def _declaration_span(source: str, name: str) -> tuple[int, int]:
    pos = source.index(name + "(")
    start = source.rfind("\n", 0, pos) + 1
    # Declaration return type begins on the same line in this source.
    if not source[start:pos].strip().startswith("void"):
        raise ValueError("declaration start " + name)
    return start, source.index(";", pos) + 1


def patch_native_binding(source: str) -> str:
    """Append resource run declarations/exports; preserve legacy exports."""
    for original, clone in ((RAGGED_RUN, RAGGED_RESOURCE_RUN),
                            (PAGED_RUN, PAGED_RESOURCE_RUN)):
        if clone in source: raise ValueError("resource binding already exists")
        start, end = _declaration_span(source, original); block = source[start:end]
        if block.count(original) != 1: raise ValueError("ambiguous binding declaration")
        source = source[:end] + "\n\n" + block.replace(original, clone, 1) + source[end:]
    ragged_export = "TVM_FFI_DLL_EXPORT_TYPED_FUNC(ragged_run, " + RAGGED_RUN + ");"
    paged_export = "TVM_FFI_DLL_EXPORT_TYPED_FUNC(paged_run, " + PAGED_RUN + ");"
    if source.count(ragged_export) != 1 or source.count(paged_export) != 1:
        raise ValueError("legacy run export anchors")
    source = source.replace(
        ragged_export,
        ragged_export + "\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(ragged_run_resource, " + RAGGED_RESOURCE_RUN + ");")
    return source.replace(
        paged_export,
        paged_export + "\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(paged_run_resource, " + PAGED_RESOURCE_RUN + ");")


def _nested_function_span(source: str, parent: str, name: str) -> tuple[int, int]:
    tree = ast.parse(source); lines = source.splitlines(keepends=True)
    parent_node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == parent)
    node = next(n for n in parent_node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    start_line = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    end_line = node.end_lineno
    starts = [0]
    for line in lines: starts.append(starts[-1] + len(line))
    return starts[start_line], starts[end_line]


def _clone_python_run(source: str, name: str, raw_name: str) -> str:
    start, end = _nested_function_span(source, "get_batch_prefill_module", name)
    block = source[start:end]
    new_name = name + "_resource"
    if "def " + new_name + "(" in source: raise ValueError("resource Python run exists " + name)
    block = block.replace(f'{{uri}}_{name}"', f'{{uri}}_{new_name}"', 1)
    block = block.replace("def " + name + "(", "def " + new_name + "(", 1)
    if raw_name + "(" not in block: raise ValueError("raw Python run anchor " + name)
    block = re.sub(r"\b" + raw_name + r"\b", raw_name.removesuffix("_func") + "_resource_func", block)
    if name == "paged_run":
        block = block.replace('''                assert lazy_independent_module is not None
                routed_paged_run_func = lazy_independent_module.get().paged_run''',
                              '                raise RuntimeError("invalid argument: resource requires matching KV strides")')
    sig_end = block.index("    ) -> None:\n") + len("    ) -> None:\n")
    block = block[:sig_end] + '        if backend != "fa2":\n            raise RuntimeError("resource run requires standard FA2")\n' + block[sig_end:]
    return block


def _clone_python_fake(source: str, name: str) -> str:
    start, end = _nested_function_span(source, "get_batch_prefill_module", name)
    block = source[start:end]; new_name = name + "_resource"
    block = block.replace(f'{{uri}}_{name.removeprefix("_fake_")}"',
                          f'{{uri}}_{name.removeprefix("_fake_")}_resource"', 1)
    return block.replace("def " + name + "(", "def " + new_name + "(", 1)


def patch_python_dispatch(source: str) -> str:
    """Expose resource custom ops while leaving official plan/native ops intact."""
    header = "def get_batch_prefill_module(backend, *args):\n    lazy_independent_module: Optional[_LazyBatchPrefillIndependentModule] = None\n"
    if source.count(header) != 1: raise ValueError("module factory header")
    source = source.replace(
        header,
        header + "    ragged_run_resource_func = None\n    paged_run_resource_func = None\n", 1)
    fa2 = '''    elif backend == "fa2":\n        uri = get_batch_prefill_uri(backend, *args)\n        module = _gen_batch_prefill_primary_module(backend, *args).build_and_load()\n        lazy_independent_module = _LazyBatchPrefillIndependentModule(\n            _gen_batch_prefill_independent_paged_module(backend, *args)\n        )\n        plan_func = module.plan\n        workspace_size_func = getattr(module, "workspace_size", None)\n        ragged_run_func = module.ragged_run\n        paged_run_func = module.paged_run\n'''
    if source.count(fa2) != 1: raise ValueError("FA2 module block")
    source = source.replace(
        fa2,
        fa2 + '''        ragged_run_resource_func = getattr(module, "ragged_run_resource", None)\n        paged_run_resource_func = getattr(module, "paged_run_resource", None)\n''', 1)

    ragged = _clone_python_run(source, "ragged_run", "ragged_run_func")
    fake_ragged = _clone_python_fake(source, "_fake_ragged_run")
    paged = _clone_python_run(source, "paged_run", "paged_run_func")
    fake_paged = _clone_python_fake(source, "_fake_paged_run")
    anchor = ("    # Register the module.\n    #\n"
              "    # Note that plan is not part of model logic. It should not be included in\n"
              "    # Cuda Graph or torch.compile. So, we don't provide a torch library for plan.\n"
              "    def prewarm_paged_kv_stride_variant")
    if source.count(anchor) != 1: raise ValueError("Python resource insertion anchor")
    source = source.replace(anchor, ragged + "\n\n" + fake_ragged + "\n\n" + paged + "\n\n" + fake_paged + "\n\n" + anchor, 1)
    ret = '''        ragged_run=ragged_run,\n        paged_run=paged_run,\n        prewarm_paged_kv_stride_variant=prewarm_paged_kv_stride_variant,\n'''
    if source.count(ret) != 1: raise ValueError("module namespace return")
    source = source.replace(
        ret,
        '''        ragged_run=ragged_run,\n        paged_run=paged_run,\n        ragged_run_resource=ragged_run_resource if backend == "fa2" else None,\n        paged_run_resource=paged_run_resource if backend == "fa2" else None,\n        resource_kernel_isolation=backend == "fa2" and ragged_run_resource_func is not None and paged_run_resource_func is not None,\n        prewarm_paged_kv_stride_variant=prewarm_paged_kv_stride_variant,\n''', 1)

    # Duplicate only the concrete run implementation (overloads remain official).
    for class_name, op in (("BatchPrefillWithRaggedKVCacheWrapper", "ragged"),
                           ("BatchPrefillWithPagedKVCacheWrapper", "paged")):
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        node = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run"][-1]
        lines = source.splitlines(keepends=True)
        start = sum(map(len, lines[:min([node.lineno] + [d.lineno for d in node.decorator_list])-1]))
        end = sum(map(len, lines[:node.end_lineno]))
        block = source[start:end].replace("def run(", "def run_resource(", 1)
        anchor = "self._cached_module." + op + "_run(*run_args)"
        if block.count(anchor) != 1:
            raise ValueError("wrapper resource run anchor " + class_name)
        block = block.replace(anchor, "self._cached_module." + op + "_run_resource(*run_args)")
        # The opt-in entry is restricted before any other backend can run.
        first_stmt = node.body[1] if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) else node.body[0]
        offset = sum(map(len, lines[:first_stmt.lineno-1])) - start + len("_resource")
        guard = ('        if self._backend != "fa2" or self._jit_module is not None:\n'
                 '            raise RuntimeError("resource run requires standard FA2")\n'
                 '        if len(self._plan_info) != 15 or bool(self._plan_info[14]):\n'
                 '            raise RuntimeError("resource run requires official unsplit plan")\n')
        block = block[:offset] + guard + block[offset:]
        source = source[:end] + "\n\n" + block + source[end:]
    ast.parse(source)
    return source
