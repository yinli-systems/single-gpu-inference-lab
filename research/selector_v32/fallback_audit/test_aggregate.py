import unittest,math
from analyze_diagnostic import aggregate_groups
class AggregateTests(unittest.TestCase):
 def rows(self):
  rows=[]
  for p in range(3):
   for b in range(8):
    seq=('off','guarded','guarded','off') if b%2==0 else ('guarded','off','off','guarded')
    for pos,arm in enumerate(seq):
     rows.append(dict(case='exposed',dtype='bfloat16',layout='ragged',split='auto',scenario='independent',timer='pooled_events',execution='graph1_replay',rep=p,block=b,position=pos,arm=arm,wall_us=100.,device_us=99.,wall_minus_device_us=1.,host_span_separation=False,positive_control=False))
  return rows
 def test_single_slow_sample_is_not_removed(self):
  rows=self.rows();rows[1]['wall_us']=150.
  r=aggregate_groups(rows)[0]
  self.assertAlmostEqual(r['paired_wall_ratio'],1.5**(-1/48),places=12);self.assertEqual(r['rows'],96)
 def test_missing_sample_rejected(self):
  rows=self.rows();rows.pop()
  with self.assertRaises(ValueError):aggregate_groups(rows)
 def test_duplicate_sample_rejected(self):
  rows=self.rows();rows.append(dict(rows[0]))
  with self.assertRaises(ValueError):aggregate_groups(rows)
 def test_wrong_treatment_label_rejected(self):
  rows=self.rows();rows[0]['arm']='guarded'
  with self.assertRaises(ValueError):aggregate_groups(rows)
if __name__=='__main__':unittest.main()
