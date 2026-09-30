import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from measure import Runtime
from analyze_calibration import clock_stable

class RuntimePlanTests(unittest.TestCase):
    def runtime(self,policy,flag):
        rt=Runtime.__new__(Runtime);rt.policy=policy
        rt.wrapper=SimpleNamespace(_plan_info=[0]*15+[flag]);rt._plan=lambda:None
        return rt
    def test_forced_native_and_cap_are_exact(self):
        self.assertEqual(self.runtime(0,0).plan()[-1],0)
        self.assertEqual(self.runtime(1,1).plan()[-1],1)
        with self.assertRaises(RuntimeError):self.runtime(0,1).plan()
        with self.assertRaises(RuntimeError):self.runtime(1,0).plan()
    def test_candidate_pool_accepts_native_or_cap_decision(self):
        self.assertEqual(self.runtime(2,0).plan()[-1],0)
        self.assertEqual(self.runtime(2,1).plan()[-1],1)

class TelemetryTests(unittest.TestCase):
    def write(self,rows):
        tmp=tempfile.NamedTemporaryFile(mode='w',delete=False,suffix='.csv')
        tmp.write('timestamp, utilization.gpu [%], clocks.current.sm [MHz]\n')
        for util,clock in rows:tmp.write(f't,{util} %, {clock} MHz\n')
        tmp.close();return Path(tmp.name)
    def test_idle_samples_do_not_poison_active_clock_check(self):
        p=self.write([(0,210)]*20+[(100,2700)]*20)
        stable,receipt=clock_stable(p);self.assertTrue(stable);self.assertEqual(receipt['active_samples'],20)
        p.unlink()
    def test_unstable_active_clocks_fail_closed(self):
        p=self.write([(100,2000)]*10+[(100,2800)]*10)
        stable,_=clock_stable(p);self.assertFalse(stable);p.unlink()
    def test_insufficient_active_samples_fail_closed(self):
        p=self.write([(100,2700)]*5+[(0,200)]*20)
        stable,receipt=clock_stable(p);self.assertFalse(stable);self.assertEqual(receipt['reason'],'insufficient-active-telemetry');p.unlink()

if __name__=='__main__':unittest.main()
