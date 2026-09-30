import unittest
from research.selector_v4.native_policy import patch_native_entry, patch_native_binding, patch_python_dispatch, POLICY_ARGUMENTS

class NativePolicyTests(unittest.TestCase):
    def test_legacy_entry_is_native_and_only_two_policies(self):
        sig='Array<int64_t> BatchPrefillWithKVCachePlan(\n    '+', '.join('int64_t '+x for x in POLICY_ARGUMENTS)+') '
        src=sig+'{\n  initialize();\n  return Array(plan_info.ToVector());\n}\n\nArray<int64_t> BatchPrefillWithKVCacheWorkspaceSize(\n'
        out=patch_native_entry(src)
        self.assertIn('resource_policy == 0 || resource_policy == 1',out)
        self.assertIn('plan_info.resource_cap = resource_policy == 1;',out)
        self.assertIn('return BatchPrefillWithKVCacheResourcePlan(0,',out)
    def test_binding_keeps_legacy_and_adds_explicit_entry(self):
        src='Array<int64_t> BatchPrefillWithKVCachePlan(\n    int x);\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, BatchPrefillWithKVCachePlan);'
        out=patch_native_binding(src);self.assertIn('plan_resource, BatchPrefillWithKVCacheResourcePlan',out)
    def test_unknown_python_anchor_rejected(self):
        with self.assertRaises(ValueError):patch_python_dispatch('wrong source')
if __name__=='__main__':unittest.main()
