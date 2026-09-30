import math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from analyze import process_grid,duplicate_grid

def row(k,b,execution,group,pos,arm,value):
 return ((k,b,execution,group,pos),{'arm':arm,'wall_us':value})
class AnalyzerHelperTests(unittest.TestCase):
 def test_canary_two_block_position_and_pairing(self):
  k=('c','float16','ragged','auto');idx={}
  for b in range(2):
   values=[100,102,102,100]
   for p,v in enumerate(values):idx.update([row(k,b,'eager_full_call','position',p,'pristine',v)])
   seq=('off','guarded','guarded','off') if b==0 else ('guarded','off','off','guarded')
   for p,arm in enumerate(seq):idx.update([row(k,b,'eager_full_call','guarded',p,arm,100 if arm=='off' else 80)])
  run={'index':idx}
  self.assertEqual(process_grid(run,k,'eager_full_call','pristine','position',2).shape,(2,))
  self.assertAlmostEqual(process_grid(run,k,'eager_full_call','off','guarded',2)[0],100)
  self.assertAlmostEqual(process_grid(run,k,'eager_full_call','guarded','guarded',2)[0],80)
  self.assertEqual(duplicate_grid(run,k,'eager_full_call','pristine','position',2).shape,(2,))
  self.assertEqual(duplicate_grid(run,k,'eager_full_call','off','guarded',2).shape,(2,))
 def test_geometric_pair_aggregation(self):
  k=('c','float16','paged','unsplit');idx={}
  for b in range(8):
   for p,v in enumerate((90,110,110,90)):idx.update([row(k,b,'graph16_replay','position',p,'pristine',v)])
  g=process_grid({'index':idx},k,'graph16_replay','pristine','position',8)
  self.assertTrue(all(abs(x-math.sqrt(90*110))<1e-9 for x in g))
if __name__=='__main__':unittest.main()
