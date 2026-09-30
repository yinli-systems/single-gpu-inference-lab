"""Fail-closed certificates for official native plan/run/dispatch source spans."""
from __future__ import annotations
import ast
import hashlib
from .kernel_isolation import template_function_span
from .native_policy import _function_span, _nested_function_span


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def python_spans(source):
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines: offsets.append(offsets[-1] + len(line))
    tree = ast.parse(source)
    result = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in (
            'BatchPrefillWithPagedKVCacheWrapper', 'BatchPrefillWithRaggedKVCacheWrapper'):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name != 'run_resource':
                    start = min([child.lineno]+[d.lineno for d in child.decorator_list])-1
                    key = node.name+'.'+child.name
                    result.setdefault(key, []).append(source[offsets[start]:offsets[child.end_lineno]])
    for name in ('ragged_run', 'paged_run', '_fake_ragged_run', '_fake_paged_run', 'prewarm_paged_kv_stride_variant'):
        start,end = _nested_function_span(source, 'get_batch_prefill_module', name)
        result['module.'+name] = [source[start:end]]
    return result


def audit_native_sources(originals, candidates):
    certificates = {}
    def check(key, before, after):
        if before != after:
            raise ValueError('native source changed: '+key)
        certificates[key] = sha(before)
    for rel,before in originals.items():
        if rel.endswith('.py'): continue
        after = candidates[rel]
        # All C++/Jinja transforms must be additive at whole-line boundaries.
        remaining = iter(after.splitlines(keepends=True))
        if not all(any(line == observed for observed in remaining) for line in before.splitlines(keepends=True)):
            raise ValueError('non-additive native source transform: '+rel)
    rel='flashinfer/data/include/flashinfer/attention/scheduler.cuh'
    check('scheduler_entire_file', originals[rel], candidates[rel])
    for rel,names,span in (
        ('flashinfer/data/csrc/batch_prefill.cu', ['BatchPrefillWithKVCachePlan','BatchPrefillWithRaggedKVCacheRun'], _function_span),
        ('flashinfer/data/csrc/batch_prefill_paged.cuh', ['BatchPrefillWithPagedKVCacheRun'], _function_span),
        ('flashinfer/data/include/flashinfer/attention/prefill.cuh', [
            'BatchPrefillWithRaggedKVCacheKernel','BatchPrefillWithPagedKVCacheKernel',
            'BatchPrefillWithRaggedKVCacheDispatchedImpl','BatchPrefillWithPagedKVCacheDispatchedImpl',
            'BatchPrefillWithRaggedKVCacheDispatched','BatchPrefillWithPagedKVCacheDispatched'], template_function_span),
    ):
        for name in names:
            a,b=span(originals[rel],name); c,d=span(candidates[rel],name)
            check(name, originals[rel][a:b], candidates[rel][c:d])
    rel='flashinfer/prefill.py';before=python_spans(originals[rel]);after=python_spans(candidates[rel])
    for name,spans in before.items():
        check(name, '\n'.join(spans), '\n'.join(after.get(name, [])))
    return {'native_source_identity':True,'plan_vector_size':15,'sha256':certificates,
            'python_factory_additions_only_outside_native_ops':True}
