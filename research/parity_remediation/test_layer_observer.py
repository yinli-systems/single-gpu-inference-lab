import importlib.util,types,unittest

@unittest.skipUnless(importlib.util.find_spec('torch'),'torch unavailable on this host')
class ShadowTests(unittest.TestCase):
 def model(self,layers=36):
  import torch
  class Block(torch.nn.Module):
   def __init__(self):
    super().__init__();self.input_layernorm=torch.nn.Identity();self.post_attention_layernorm=torch.nn.Identity()
    self.self_attn=torch.nn.Module();self.self_attn.qkv_proj=torch.nn.Identity();self.self_attn.attn=torch.nn.Identity();self.self_attn.o_proj=torch.nn.Identity()
    self.mlp=torch.nn.Module();self.mlp.gate_up_proj=torch.nn.Identity();self.mlp.down_proj=torch.nn.Identity()
   def forward(self,x):
    x=self.input_layernorm(x);x=self.self_attn.qkv_proj(x);x=self.self_attn.attn(x);x=self.self_attn.o_proj(x);x=self.post_attention_layernorm(x);x=self.mlp.gate_up_proj(x);return self.mlp.down_proj(x)
  class Model(torch.nn.Module):
   def __init__(self):
    super().__init__();self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([Block() for _ in range(layers)])
   def forward(self,x,positions,forward_batch):
    for layer in self.model.layers:x=layer(x)
    return x
  return Model()
 def test_unchanged_values_and_shadow_copy(self):
  import torch
  from layer_observer import install_model
  r=types.SimpleNamespace(model=self.model());install_model(r)
  x=torch.arange(8.).reshape(2,4);before=x.clone();fb=types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda:True))
  out=r.model.forward(x,None,fb);self.assertIs(out,x);self.assertTrue(torch.equal(out,before))
  bank=r._sgi_shadow_banks['eager',2];self.assertEqual(len(bank),504)
  for t in bank.values():self.assertNotEqual(t.data_ptr(),x.data_ptr());self.assertTrue(torch.equal(t,before))
 def test_prefill_not_in_decode_bank(self):
  import torch
  from layer_observer import install_model
  r=types.SimpleNamespace(model=self.model());install_model(r)
  r.model.forward(torch.ones(2,4),None,types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda:False)))
  self.assertEqual(r._sgi_shadow_banks,{})
 def test_unknown_structure_rejected(self):
  from layer_observer import install_model
  with self.assertRaises(RuntimeError):install_model(types.SimpleNamespace(model=self.model(2)))
 def test_idempotent(self):
  from layer_observer import install_model
  r=types.SimpleNamespace(model=self.model());install_model(r);f=r.model.forward;install_model(r);self.assertIs(f,r.model.forward)
if __name__=='__main__':unittest.main()
