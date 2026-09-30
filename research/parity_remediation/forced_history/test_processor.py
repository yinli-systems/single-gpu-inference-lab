import unittest
try:
 import torch
 from force_processor import ForcedHistoryProcessor
except ModuleNotFoundError:
 torch=None;ForcedHistoryProcessor=None
class Req:
 def __init__(self,ids):self.output_ids=list(ids)
@unittest.skipIf(torch is None,'torch unavailable')
class ProcessorTests(unittest.TestCase):
 def test_forces_declared_prefix_only(self):
  x=torch.tensor([[1.,2.,3.],[3.,2.,1.]])
  p=[{'forced_tokens':[1,2],'__req__':Req([])},{'forced_tokens':[1],'__req__':Req([1])}]
  y=ForcedHistoryProcessor()(x.clone(),p)
  self.assertEqual(int(y[0].argmax()),1)
  self.assertTrue(torch.equal(y[1],x[1]))
 def test_advances_with_output_history(self):
  x=torch.zeros(1,5);p=[{'forced_tokens':[3,4],'__req__':Req([3])}]
  y=ForcedHistoryProcessor()(x,p);self.assertEqual(int(y.argmax()),4)
 def test_rejects_missing_state(self):
  with self.assertRaises(RuntimeError):ForcedHistoryProcessor()(torch.zeros(1,3),[{}])
if __name__=='__main__':unittest.main()
