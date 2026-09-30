import json,unittest
from pathlib import Path
class Cases(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.x=json.loads(Path(__file__).with_name('cases.json').read_text())
 def test_contract(self):
  self.assertEqual(len(self.x['targets']),2);self.assertEqual(self.x['concurrency'],8)
  for t in self.x['targets']:
   self.assertEqual(len(t['rows']),16)
   self.assertTrue(all(len(r['forced_tokens'])==128 and not r['natural'] and r['max_new_tokens']==128 for r in t['rows'][:8]))
   self.assertTrue(all(len(r['forced_tokens'])==t['output_index_zero_based'] and r['natural'] and r['max_new_tokens']==t['output_index_zero_based']+1 for r in t['rows'][8:]))
 def test_historical_witnesses(self):
  q={x['id']:x for x in self.x['targets']};self.assertEqual(q['decode-11']['historical_tokens'],[315,304]);self.assertEqual(q['decode-13']['historical_tokens'],[30280,785])
if __name__=='__main__':unittest.main()
