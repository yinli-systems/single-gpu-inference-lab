import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from autotune import canonical_hash
from evidence import load_run,EXECUTIONS

CASE={'id':'c0','family':'canary','regime':'opposite','batch_size':5,'q':[1,2,3,4,5],'cached':[9000,8000,7000,6000,5000],'generator_variant':0}
MANIFEST={'case_hash':'casehash','release_hash':'releasehash','dtypes':['float16'],'layouts':['ragged'],'splits':['unsplit']}

def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class EvidenceFixture(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.path=self.root/'runs/r';self.path.mkdir(parents=True);(self.root/'source').mkdir();(self.root/'receipts').mkdir();(self.root/'overlays/v4').mkdir(parents=True)
  (self.root/'source/measure.py').write_text('source');(self.root/'receipts/source-archive.sha256').write_text('archive\n');(self.root/'overlays/v4/RESOURCE_BINDING.json').write_text('{}')
  env={'mode':'calibration','stage':'canary','rep':0,'shard':0,'shards':1,'measurement_revision':'4.0','profiled':False,'gpu_name':'NVIDIA GeForce RTX 4090','gpu_uuid':'g','driver':'d','num_sms':128,'cuda':'13','torch':'t','flashinfer':'0.7.0','python':'3.12','cudnn':'x','nvcc':'x','measurement_policy':'m','source_archive_sha256':'archive','overlay_sha256':file_sha(self.root/'overlays/v4/RESOURCE_BINDING.json'),'case_hash':'casehash','release_hash':'releasehash','cases':[CASE],'source':{'measure.py':file_sha(self.root/'source/measure.py')}}
  plan=[0]*16;cap=plan.copy();cap[-1]=1;pool=plan.copy();identity={}
  for ex in EXECUTIONS:
   payload={'environment':{'gpu':'g'},'operation':{'execution_mode':ex}};identity[ex]={'payload':payload,'sha256':canonical_hash(payload)}
  qual={'case':'c0','dtype':'float16','layout':'ragged','split':'unsplit','exact':True,'cap_supported':True,'native_plan':plan,'cap_plan':cap,'pool_plan':pool,'candidate_pool':{'eligible':False},'identities':identity,'chosen':{},'out_sha256':'a','lse_sha256':'b','eager_calls':16}
  rows=[]
  for block in range(8):
   seqA=(0,3) if block%2==0 else (1,2)
   for ex in EXECUTIONS:
    for group in ('candidate','null'):
     for pos in range(4):
      role='A' if pos in seqA else 'B';arm='native' if role=='A' or group=='null' else 'cap'
      rows.append({'case':'c0','dtype':'float16','layout':'ragged','split':'unsplit','mode':'calibration','rep':0,'block':block,'execution_mode':ex,'comparison_group':group,'position':pos,'role':role,'arm':arm,'actual_arm':arm,'wall_us':1000.0,'device_us':999.0,'kernel_calls':16,'identity_sha256':identity[ex]['sha256']})
  memory=[{'case':'c0','dtype':'float16','layout':'ragged','split':'unsplit','allocated_after_gc':0,'reserved_after_gc':0,'total_device_bytes':100}]
  self.objects={'environment.json':env,'rows.json':rows,'qualification.json':[qual],'memory.json':memory,'progress.json':{'rows':len(rows)}};self.flush()
 def flush(self):
  for name,obj in self.objects.items():(self.path/name).write_text(json.dumps(obj))
  c={'complete':True,'rows':len(self.objects['rows.json']),'qualifications':len(self.objects['qualification.json']),'files':{n:file_sha(self.path/n) for n in self.objects}}
  (self.path/'complete.json').write_text(json.dumps(c))
 def validate(self,**kwargs):
  with patch('evidence.expected_cases',return_value=[CASE]),patch('evidence.load',return_value=MANIFEST):
   return load_run(self.root,self.path,mode='calibration',stage='canary',gpu='gpu_4090',rep=0,shard=0,shards=1,**kwargs)
 def test_complete_contract_passes(self):self.assertEqual(len(self.validate()[1]),192)
 def test_tampered_artifact_rejected(self):
  (self.path/'rows.json').write_text('[]')
  with self.assertRaisesRegex(ValueError,'artifact digest'):self.validate()
 def test_missing_coordinate_rejected(self):
  self.objects['rows.json'].pop();self.flush()
  with self.assertRaisesRegex(ValueError,'row completeness'):self.validate()
 def test_wrong_actual_arm_rejected_for_unsupported_cap(self):
  q=self.objects['qualification.json'][0];q['cap_supported']=False;q['native_plan'][14]=1;q['cap_plan']=q['native_plan'].copy();q['pool_plan']=q['native_plan'].copy()
  self.flush()
  with self.assertRaisesRegex(ValueError,'unsupported cap executed'):self.validate()
 def test_role_tamper_rejected(self):
  self.objects['rows.json'][0]['role']='B';self.flush()
  with self.assertRaisesRegex(ValueError,'ABBA role'):self.validate()

if __name__=='__main__':unittest.main()
