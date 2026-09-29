import importlib.util,json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest import mock

@unittest.skipUnless(importlib.util.find_spec('torch'),'torch unavailable on this host')
class TraceTests(unittest.TestCase):
 def test_observer_preserves_logits_result_and_history(self):
  import torch
  import server_entry
  class Runner:
   calls=0
   def sample(self,logits_output,forward_batch):
    self.calls+=1
    return torch.tensor([7])
  fake=types.ModuleType('sglang.srt.model_executor.model_runner');fake.ModelRunner=Runner
  with tempfile.TemporaryDirectory() as td, mock.patch.dict(sys.modules,{'sglang.srt.model_executor.model_runner':fake}), mock.patch.dict(os.environ,{'SGI_TRACE_PATH':td}):
   server_entry.install();fn=Runner.sample;server_entry.install();self.assertIs(Runner.sample,fn)
   x=torch.arange(20,dtype=torch.float32).reshape(1,20);original=x.clone();r=Runner()
   mode=types.SimpleNamespace(is_decode=lambda:False)
   fb=types.SimpleNamespace(batch_size=1,input_ids=torch.tensor([5]),seq_lens=torch.tensor([1]),req_pool_indices=torch.tensor([0]),forward_mode=mode,extend_seq_lens_cpu=[1])
   lo=types.SimpleNamespace(next_token_logits=x)
   self.assertEqual(r.sample(lo,fb).tolist(),[7]);self.assertTrue(torch.equal(x,original))
   fb.input_ids=torch.tensor([7]);fb.seq_lens=torch.tensor([2]);fb.forward_mode=types.SimpleNamespace(is_decode=lambda:True)
   self.assertEqual(r.sample(lo,fb).tolist(),[7]);self.assertEqual(r.calls,2)
   rows=[json.loads(s) for f in Path(td).glob('*.jsonl') for s in f.read_text().splitlines()]
   self.assertEqual(len(rows),2);self.assertTrue(all(x['all_histories_complete'] for x in rows))
   self.assertNotEqual(rows[0]['history_sha256'],rows[1]['history_sha256']);self.assertEqual(rows[0]['logit_sha256'],rows[1]['logit_sha256'])
if __name__=='__main__':unittest.main()
