import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from manifest import tactic_identity
class IdentityTests(unittest.TestCase):
 def test_environment_operation_and_execution_bound(self):
  env=dict(gpu_uuid='u',gpu_name='g',num_sms=1,driver='d',cuda='c',torch='t',flashinfer='f',python='p',cudnn='cd',nvcc='n',backend_sha256='h',official_overlay_sha256='o',source_archive_sha256='s',selector_version='3.2',measurement_policy='paired',profiling_mode='unprofiled')
  op=dict(execution_mode='eager_full_call',backend='fa2',causal=True,layout='paged',dtype='bfloat16',requested_split='auto',actual_split='unsplit',num_qo_heads=32,num_kv_heads=8,head_dim_qk=128,head_dim_vo=128,page_size=16,q=[1,2],cached=[4,3],ordered_pairs_sha256='x',plan_signature=[1,2],window_repeats=16,candidate_binary_sha256='b')
  a=tactic_identity(environment=env,operation=op)
  self.assertNotEqual(a,tactic_identity(environment={**env,'driver':'e'},operation=op))
  self.assertNotEqual(a,tactic_identity(environment=env,operation={**op,'execution_mode':'graph1_replay'}))
  self.assertNotEqual(a,tactic_identity(environment=env,operation={**op,'q':[2,1]}))
if __name__=='__main__':unittest.main()
