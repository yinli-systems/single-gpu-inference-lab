import unittest
from pathlib import Path

class LauncherV41Tests(unittest.TestCase):
    def test_candidate_jit_is_persistent_and_auditable(self):
        text=Path(__file__).with_name('run_v4.sbatch').read_text()
        self.assertIn('XDG_CACHE_HOME="$P/cache/$SLURM_JOB_PARTITION/$OVERLAY/xdg-$MODE-r$REP"',text)
        self.assertNotIn('XDG_CACHE_HOME="$LOCAL/cache-$MODE-r$REP"',text)
        self.assertIn('--workspace "$P/cache/$SLURM_JOB_PARTITION/candidate"',text)
        self.assertLess(text.index('measure_v4'),text.index('audit_binary'))
if __name__=='__main__':unittest.main()
