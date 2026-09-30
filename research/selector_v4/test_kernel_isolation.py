import unittest
from research.selector_v4.kernel_isolation import duplicate_kernel,template_function_span

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


if __name__=='__main__': unittest.main()
