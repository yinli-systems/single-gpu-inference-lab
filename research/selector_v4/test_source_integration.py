"""Run against the pinned pristine package; no CUDA or FlashInfer import needed."""
import ast
import os
import tempfile
import unittest
from pathlib import Path
from .prepare_overlay import prepare, EXPECTED
from .source_audit import audit_native_sources

@unittest.skipUnless(os.environ.get('SGI_PRISTINE_SOURCE'), 'explicit pinned pristine package required')
class SourceIntegrationTests(unittest.TestCase):
    def test_official_source_identity_and_resource_ops_are_distinct(self):
        source=Path(os.environ['SGI_PRISTINE_SOURCE'])
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'candidate';receipt=prepare(source,out)
            self.assertTrue(receipt['native_source_audit']['native_source_identity'])
            self.assertEqual(receipt['plan_vector_size'],15)
            py=(out/'flashinfer/prefill.py').read_text();tree=ast.parse(py)
            factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_batch_prefill_module')
            for name in ('ragged','paged'):
                for prefix in ('','_fake_'):
                    node=next(n for n in factory.body if isinstance(n,ast.FunctionDef) and n.name==prefix+name+'_run_resource')
                    self.assertIn('{uri}_'+name+'_run_resource',ast.get_source_segment(py,node.decorator_list[0]))
                op=next(n for n in factory.body if isinstance(n,ast.FunctionDef) and n.name==name+'_run_resource')
                calls=[n.func.id for n in ast.walk(op) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
                self.assertNotIn('routed_paged_run_resource_func',calls)
            for node in tree.body:
                if isinstance(node,ast.ClassDef) and node.name.startswith('BatchPrefillWith'):
                    ops=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='run_resource']
                    for op in ops:
                        self.assertIsInstance(op.body[1],ast.If)
                        self.assertIn('self._backend',ast.get_source_segment(py,op.body[1]))
            before={rel:(source/rel).read_text() for rel in EXPECTED}
            after={rel:(out/rel).read_text() for rel in EXPECTED}
            after['flashinfer/data/csrc/batch_prefill.cu']=after['flashinfer/data/csrc/batch_prefill.cu'].replace('BatchPrefillWithKVCachePlan(', 'BrokenNativePlan(',1)
            with self.assertRaises((ValueError,AssertionError)):audit_native_sources(before,after)
