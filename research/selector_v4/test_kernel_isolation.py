import unittest
from kernel_isolation import duplicate_kernel,template_function_span,selector_decision_cpp

SOURCE='''template <typename T>\n__global__ void Kernel(const T x) {\n  if (x) { int y = 1; }\n}\n\ntemplate <typename T>\nvoid next(T x) {}\n'''

class KernelIsolationTests(unittest.TestCase):
    def test_original_kernel_is_byte_identical(self):
        start,end=template_function_span(SOURCE,'Kernel');original=SOURCE[start:end]
        patched,native_hash,clone_hash=duplicate_kernel(SOURCE,'Kernel','ResourceKernel')
        pstart,pend=template_function_span(patched,'Kernel')
        self.assertEqual(original,patched[pstart:pend]);self.assertNotEqual(native_hash,clone_hash)
        self.assertEqual(patched.count('ResourceKernel('),1)
    def test_duplicate_rejected(self):
        patched,_,_=duplicate_kernel(SOURCE,'Kernel','ResourceKernel')
        with self.assertRaises(ValueError):duplicate_kernel(patched,'Kernel','ResourceKernel')
    def test_cpp_gate_uses_2_4_waves(self):
        code=selector_decision_cpp();self.assertIn('12 * static_cast<uint64_t>(sgi_num_sm)',code)
        self.assertIn('5 * static_cast<uint64_t>(num_kv_heads)',code)
        self.assertIn('plan_info.resource_cap',code)

if __name__=='__main__': unittest.main()
