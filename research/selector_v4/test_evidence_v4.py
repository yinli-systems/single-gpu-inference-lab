from __future__ import annotations
import hashlib, json, tempfile, unittest
from pathlib import Path
from research.selector_v4.evidence_v4 import EXECUTIONS, expected_sequence, validate_run
from research.selector_v4.eligibility import evaluate_eligibility, effective_eligibility
from research.selector_v4.identity import TacticIdentity
from research.selector_v4.schema import QUALIFICATION_REVISION, TACTIC_CAP, TACTIC_NATIVE

class EvidenceContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)
        self.case={'id':'fixture-v4','family':'dev','q':[3,35,99,163,259],'cached':[32768,16384,8192,2048,64]}
        self.manifest={'case_hash':'fixture-hash','cases':[self.case],'dtypes':['float16'],'layouts':['paged'],'requested_splits':['unsplit'],'blocks':2}
        self.env={'mode':'paired','stage':'dev','rep':0,'shard':0,'shards':1,'measurement_contract_revision':QUALIFICATION_REVISION,
            'profiled':False,'kernel_symbol_isolation':True,'case_hash':'fixture-hash','cases':[self.case]}
        environment={'gpu_name':'NVIDIA Test','gpu_uuid':'GPU-fixture','num_sms':170,'driver':'580.82.07','cuda':'13.0','torch':'2.13','flashinfer':'0.7.0','nvcc':'13.0',
            'backend_source_sha256':'a'*64,'official_overlay_sha256':'b'*64,'resource_binding_sha256':'c'*64,
            'max_smem_per_sm':102400,'max_smem_per_block_optin':101376}
        core=[40]+[0]*14
        self.qual={'case':'fixture-v4','dtype':'float16','layout':'paged','split':'unsplit','measurement_contract_revision':QUALIFICATION_REVISION,
            'candidate_plan_core_equal':True,'cap_supported':True,'cap_probe_error':None,'native_after_cap_exact':True,
            'arms':{},'identities':{},'eligibility':{}}
        for arm,flag,tactic in [('off',0,TACTIC_NATIVE),('cap',1,TACTIC_CAP)]:
            self.qual['arms'][arm]={'pristine_exact':True,'execution_checks':dict.fromkeys(EXECUTIONS,True),'out_sha256':'1'*64,'lse_sha256':'2'*64,'plan_info':list(core),'actual_tactic':tactic}
            self.qual['identities'][arm]={}
            for execution in EXECUTIONS:
                operation={'execution_mode':execution,'backend':'fa2','causal':True,'layout':'paged','dtype':'float16','actual_split':'unsplit','num_qo_heads':32,'num_kv_heads':8,
                    'head_dim_qk':128,'head_dim_vo':128,'page_size':16,'q':self.case['q'],'cached':self.case['cached'],'plan_signature':core,'window_config':{'calls':16}}
                policy={'execution_mode':execution,'timer':'deployment_wall','qualification_revision':QUALIFICATION_REVISION}
                identity=TacticIdentity(environment,operation,policy)
                self.qual['identities'][arm][execution]={'key':identity.key,'payload':identity.to_dict()}
                self.qual['eligibility'][execution]=effective_eligibility(identity,cap_supported=True).to_dict()
        self.rows=[]
        for block in range(2):
            for execution in EXECUTIONS:
                for position,(arm,role) in enumerate(expected_sequence('paired',block)):
                    self.rows.append({'case':'fixture-v4','dtype':'float16','layout':'paged','split':'unsplit','block':block,'execution_mode':execution,'position':position,
                        'arm':arm,'actual_tactic':self.qual['arms'][arm]['actual_tactic'],'role':role,'comparison_group':'cap','wall_us':1000.0,'kernel_calls':128,
                        'window_elapsed_us':128000.0,'tactic_identity':self.qual['identities'][arm][execution]['key']})
        self.flush()

    def flush(self):
        artifacts={'environment.json':self.env,'measurements.json':self.rows,'qualification.json':[self.qual],'memory.json':[{'case':'fixture-v4'}],'progress.json':{'rows':len(self.rows)}}
        for name,value in artifacts.items():(self.path/name).write_text(json.dumps(value))
        complete={'complete':True,'rows':len(self.rows),'expected':len(self.rows),'qualifications':1,'files':{name:hashlib.sha256((self.path/name).read_bytes()).hexdigest() for name in artifacts}}
        (self.path/'complete.json').write_text(json.dumps(complete))

    def validate(self):return validate_run(self.path,mode='paired',stage='dev',rep=0,shard=0,shards=1,manifest=self.manifest)
    def test_complete_fixture_passes(self):
        result=self.validate();self.assertEqual(len(result['rows']),24);self.assertEqual(len(result['qualifications']),1)
    def test_tactic_plan_flag_corruption_rejected(self):
        self.qual['arms']['cap']['plan_info'][0]=41;self.flush()
        with self.assertRaisesRegex(ValueError,'cap/fallback tactic contract'):self.validate()
    def test_native_fallback_for_unsupported_cap_passes(self):
        self.qual['cap_supported']=False;self.qual['cap_probe_error']=None;self.qual['arms']['cap']['plan_info'][-1]=0;self.qual['arms']['cap']['actual_tactic']=TACTIC_NATIVE
        for execution in EXECUTIONS:
            raw=self.qual['identities']['cap'][execution];identity=TacticIdentity(raw['payload']['environment'],raw['payload']['operation'],raw['payload']['measurement_policy'])
            self.qual['eligibility'][execution]=effective_eligibility(identity,cap_supported=False,runtime_reason='static_or_plan_ineligible').to_dict()
        for row in self.rows:
            if row['arm']=='cap':row['actual_tactic']=TACTIC_NATIVE
        self.flush();self.validate()
    def test_abba_label_corruption_rejected(self):
        self.rows[0]['role']='B';self.flush()
        with self.assertRaisesRegex(ValueError,'ABBA/BAAB'):self.validate()
    def test_artifact_tamper_rejected(self):
        (self.path/'measurements.json').write_text('[]')
        with self.assertRaisesRegex(ValueError,'artifact hash mismatch'):self.validate()
