import unittest
from research.selector_v4.native_policy import clone_host_run, patch_native_binding, patch_python_dispatch

class NativePolicyTests(unittest.TestCase):
    def test_native_host_run_preserved(self):
        src='void BatchPrefillWithRaggedKVCacheRun(int x) {\n  BatchPrefillWithRaggedKVCacheDispatched(x);\n}\n'
        out,native,resource=clone_host_run(src,paged=False)
        self.assertTrue(out.startswith(native));self.assertIn('BatchPrefillWithRaggedKVCacheResourceDispatched(x)',resource)
        self.assertNotIn('resource_cap',out)
    def test_binding_keeps_legacy_and_adds_resource_run_only(self):
        src='void BatchPrefillWithRaggedKVCacheRun(int x);\nvoid BatchPrefillWithPagedKVCacheRun(int x);\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(ragged_run, BatchPrefillWithRaggedKVCacheRun);\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(paged_run, BatchPrefillWithPagedKVCacheRun);'
        out=patch_native_binding(src)
        self.assertIn('ragged_run_resource, BatchPrefillWithRaggedKVCacheResourceRun',out)
        self.assertIn('paged_run_resource, BatchPrefillWithPagedKVCacheResourceRun',out)
        self.assertNotIn('plan_resource',out)
    def test_unknown_python_anchor_rejected(self):
        with self.assertRaises(ValueError):patch_python_dispatch('wrong source')
