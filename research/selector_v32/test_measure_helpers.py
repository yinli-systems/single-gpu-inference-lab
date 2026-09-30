import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from measure import replace_plan_flag
class ImmutableArray(tuple):
 def __new__(cls,values):return super().__new__(cls,values)
class PlanHelperTests(unittest.TestCase):
 def test_rebuilds_immutable_runtime_type(self):
  original=ImmutableArray(range(16));rebuilt,values=replace_plan_flag(original,True)
  self.assertIs(type(rebuilt),ImmutableArray);self.assertEqual(values[:15],list(range(15)));self.assertEqual(values[15],1);self.assertEqual(original[15],15)
  rebuilt0,values0=replace_plan_flag(rebuilt,False);self.assertEqual(values0[15],0);self.assertEqual(rebuilt[15],1)
 def test_rejects_wrong_vector_size(self):
  with self.assertRaises(RuntimeError):replace_plan_flag(ImmutableArray(range(15)),True)
if __name__=='__main__':unittest.main()
