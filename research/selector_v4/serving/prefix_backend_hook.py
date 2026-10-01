"""Exact-source instance hook for the single SGLang cached-prefix merge call.

Only explicit callers install this research hook. Native wrappers themselves
remain unmodified, so public lease fallback cannot recurse through this router.
No resource training, automatic selection or serving qualification happens here.
"""

import __future__

import ast
import functools
import hashlib
from pathlib import Path

REVIEWED_BACKEND_SHA256 = "5c8baba0d14eeca4c9d4bef8674318be1697b0160f105224282cd4aa27575810"


def transform(source, expected_sha256=REVIEWED_BACKEND_SHA256):
    if hashlib.sha256(source.encode()).hexdigest() != expected_sha256:
        raise ValueError("Unreviewed SGLang attention backend")
    tree = ast.parse(source)
    owner = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlashInferAttnBackend"
    )
    function = next(
        n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "forward_extend"
    )
    matches = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "prefill_wrapper_paged"
        and n.func.attr == "forward_return_lse"
    ]
    if len(matches) != 1:
        raise ValueError("Expected the single reviewed cached-prefix output/LSE merge call")
    call = matches[0]
    call.func = ast.Name(id="_sgi_route_cached_prefix", ctx=ast.Load())
    call.args = [
        ast.Name(id="self", ctx=ast.Load()),
        ast.Name(id="prefill_wrapper_paged", ctx=ast.Load()),
        *call.args,
    ]
    return ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))


def install(backend, router, *, source_path):
    original = backend.forward_extend
    if getattr(original, "_sgi_prefix_backend_hook", False):
        raise RuntimeError("Cached-prefix hook already installed")
    source = Path(source_path).read_text()
    tree = transform(source)
    method = getattr(original, "__func__", original)
    while hasattr(method, "__wrapped__"):
        method = method.__wrapped__
    namespace = dict(method.__globals__)

    def route(instance, owner, q, kv_cache, **options):
        if instance is not backend:
            raise RuntimeError("Instance-owned router used by another backend")
        return router.run(owner, q, kv_cache, **options)

    namespace["_sgi_route_cached_prefix"] = route
    # The exact reviewed source SHA is checked before this single-call AST rewrite.
    future_mask = sum(
        getattr(__future__, name).compiler_flag for name in __future__.all_feature_names
    )
    flags = method.__code__.co_flags & future_mask
    code = compile(tree, str(source_path), "exec", flags=flags, dont_inherit=True)
    exec(code, namespace)  # noqa: S102
    changed = namespace["forward_extend"]

    @functools.wraps(original)
    def forward(*args, **kwargs):
        return changed(backend, *args, **kwargs)

    forward._sgi_prefix_backend_hook = True
    backend.forward_extend = forward
    return original
