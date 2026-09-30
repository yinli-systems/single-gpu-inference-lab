import importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('forced_analysis',Path(__file__).with_name('analyze.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Analysis(unittest.TestCase):
 def test_numeric_layer_order(self):
  keys=['model.layers.10|input|0','model.layers.2|input|0','model.layers.2.self_attn.attn|output|0','model.layers.2.self_attn.qkv_proj|output|0']
  self.assertEqual(m.ordered_shadow_keys(keys),[keys[1],keys[3],keys[2],keys[0]])
 def test_kinds(self):self.assertLess(m.kind_of('model.layers.0.self_attn.qkv_proj|output|0'),m.kind_of('model.layers.0.self_attn.attn|output|0'))
if __name__=='__main__':unittest.main()
