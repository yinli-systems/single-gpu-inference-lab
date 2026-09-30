import unittest
from research.selector_v4.native_sass import parse_sass

SASS='''arch = sm_89
Function : _ZN_BatchPrefillWithRaggedKVCacheKernel_x
.headerflags @"EF_CUDA_SM89"
/*0000*/ MOV R1, c[0x0][0x28]; /* 0x12340000 */
/*0010*/ EXIT; /* 0x00000000 */
'''
class NativeSassTests(unittest.TestCase):
    def test_pc_and_spacing_normalized_but_operands_and_encoding_retained(self):
        expected=parse_sass(SASS)
        self.assertEqual(expected,parse_sass(SASS.replace('/*0000*/','/*0020*/').replace('MOV R1','MOV   R1')))
        self.assertNotEqual(expected,parse_sass(SASS.replace('R1,','R2,')))
        self.assertNotEqual(expected,parse_sass(SASS.replace('0x12340000','0x12340001')))
    def test_resource_symbols_excluded(self):
        self.assertEqual({},parse_sass(SASS.replace('RaggedKVCacheKernel','RaggedKVCacheResourceKernel')))
    def test_empty_disassembly_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'empty native'):parse_sass(SASS.split('/*0000*/')[0])
