"""Reviewed local source AST identity; no SGLang/Torch GPU dependency."""

import __future__

import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.selector_v4.serving.prefix_backend_hook import install, transform

SOURCE = """class FlashInferAttnBackend:
    def forward_extend(self, q, kv):
        prefill_wrapper_paged = self.owner
        marker = self.marker(q)
        output = prefill_wrapper_paged.forward_return_lse(q, kv, causal=False)
        return (marker, output)
"""


def test_one_call_only_is_changed_and_all_other_ast_is_exact():
    changed = transform(SOURCE, hashlib.sha256(SOURCE.encode()).hexdigest())
    routed = next(
        n
        for n in ast.walk(changed)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_sgi_route_cached_prefix"
    )
    assert [a.id for a in routed.args[:2]] == ["self", "prefill_wrapper_paged"]
    routed.func = ast.Attribute(
        value=ast.Name(id="prefill_wrapper_paged", ctx=ast.Load()),
        attr="forward_return_lse",
        ctx=ast.Load(),
    )
    routed.args = routed.args[2:]
    original = ast.parse(SOURCE).body[0].body[0]
    assert ast.dump(changed.body[0]) == ast.dump(original)


def test_source_drift_or_multiple_prefix_calls_is_rejected():
    with pytest.raises(ValueError, match="Unreviewed"):
        transform(SOURCE)
    changed = SOURCE.replace(
        "        return", "        prefill_wrapper_paged.forward_return_lse(q, kv)\n        return"
    )
    with pytest.raises(ValueError, match="single reviewed"):
        transform(changed, hashlib.sha256(changed.encode()).hexdigest())


def test_actual_reviewed_backend_has_exactly_one_target_when_available():
    source = Path(
        os.environ.get(
            "SGI_REVIEWED_BACKEND_SOURCE",
            "/Users/kevin/Projects/sglang-http-ownership-current-20261001/python/sglang/srt/layers/attention/flashinfer_backend.py",
        )
    )
    if not source.exists():
        pytest.skip("Local reviewed SGLang source unavailable; synthetic tests still run")
    changed = transform(source.read_text())
    original = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "FlashInferAttnBackend"
    )
    function = next(
        n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "forward_extend"
    )
    call = next(
        n
        for n in ast.walk(changed)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_sgi_route_cached_prefix"
    )
    call.func = ast.Attribute(
        value=ast.Name(id="prefill_wrapper_paged", ctx=ast.Load()),
        attr="forward_return_lse",
        ctx=ast.Load(),
    )
    call.args = call.args[2:]
    assert ast.dump(changed.body[0]) == ast.dump(function)


def test_install_requires_actual_reviewed_source_without_mutating_instance(tmp_path):
    path = tmp_path / "not-reviewed.py"
    path.write_text(SOURCE)
    original = lambda *a: "native"
    backend = SimpleNamespace(forward_extend=original)
    with pytest.raises(ValueError, match="Unreviewed"):
        install(backend, None, source_path=path)
    assert backend.forward_extend is original


@pytest.mark.parametrize("deferred_annotations", [False, True])
def test_instance_hook_routes_one_call_and_leaves_native_owner_unmodified(
    tmp_path, monkeypatch, deferred_annotations
):
    from research.selector_v4.serving import prefix_backend_hook

    class Backend:
        def forward_extend(self, q, kv):
            prefill_wrapper_paged = self.owner
            marker = self.marker(q)
            output = prefill_wrapper_paged.forward_return_lse(q, kv, causal=False)
            return marker, output

    owner = SimpleNamespace(forward_return_lse=lambda q, kv, **kw: ("native", q, kv, kw))
    backend = Backend()
    backend.owner = owner
    backend.marker = lambda q: "marker"
    original_owner = owner.forward_return_lse
    router = SimpleNamespace(
        run=lambda o, q, kv, **kw: ("routed", o.forward_return_lse(q, kv, **kw))
    )
    path = tmp_path / "synthetic-reviewed.py"
    source = SOURCE
    if deferred_annotations:
        source = SOURCE.replace("self, q, kv", "self, q: MissingType, kv")
        Backend.forward_extend.__code__ = Backend.forward_extend.__code__.replace(
            co_flags=Backend.forward_extend.__code__.co_flags | __future__.annotations.compiler_flag
        )
    path.write_text(source)
    monkeypatch.setattr(
        prefix_backend_hook,
        "transform",
        lambda source: transform(source, hashlib.sha256(source.encode()).hexdigest()),
    )
    original = install(backend, router, source_path=path)
    result = backend.forward_extend("q", "kv")
    assert result == ("marker", ("routed", ("native", "q", "kv", {"causal": False})))
    assert owner.forward_return_lse is original_owner
    assert original("q", "kv") == ("marker", ("native", "q", "kv", {"causal": False}))
    with pytest.raises(RuntimeError, match="already installed"):
        install(backend, router, source_path=path)
    backend.forward_extend = original
    assert backend.forward_extend("q", "kv") == ("marker", ("native", "q", "kv", {"causal": False}))
