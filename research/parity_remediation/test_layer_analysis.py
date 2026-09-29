import copy,unittest
from analyze_layers import pair,position

def fixture():
 return dict(history_sha256='history',batch_signature='batch',graph_used=True,graph_rows=8,row=0,
  shadow_sha256={'model.layers.0.input_layernorm|input|0.0':'input','model.layers.0.input_layernorm|output|0':'norm','model.layers.1.self_attn.attn|output|0':'out'},cache_sha256={'0.k':'k','0.v':'v'},logits_sha256='logits')
class LayerAnalysisTests(unittest.TestCase):
 def test_equal(self):self.assertTrue(pair(fixture(),fixture())['all_observed_inputs_equal'])
 def test_wrong_graph(self):
  b=fixture();b['graph_rows']=4
  with self.assertRaises(ValueError):pair(fixture(),b)
 def test_first_difference(self):
  b=fixture();b['shadow_sha256']['model.layers.0.input_layernorm|output|0']='bad';d=pair(fixture(),b);self.assertEqual(d['first_observed_shadow_difference'],'model.layers.0.input_layernorm|output|0')
 def test_KV_mismatch_not_hidden(self):
  b=fixture();b['cache_sha256']['0.k']='other';d=pair(fixture(),b);self.assertFalse(d['all_observed_inputs_equal']);self.assertEqual(d['changed_KV_count'],1)
 def test_missing_operator(self):
  b=fixture();b['shadow_sha256'].pop('model.layers.0.input_layernorm|output|0')
  with self.assertRaises(ValueError):pair(fixture(),b)
 def test_numeric_layer_order(self):self.assertLess(position('model.layers.2.self_attn.attn|output|0'),position('model.layers.11.input_layernorm|input|0'))
 def test_no_causal_claim(self):self.assertFalse(pair(fixture(),fixture())['old_uninstrumented_failure_attributed'])
if __name__=='__main__':unittest.main()
