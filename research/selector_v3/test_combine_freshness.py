import copy,json,math,tempfile,unittest
from pathlib import Path
from combine_freshness import combine

MODES=('pristine','off','cap','guarded')

def basis(value):
    return {'main_log':[math.log(value)]*24,'repeat_log':[math.log(value)]*24}

def make_cells(cases,selected_gain=1.08):
    cells=[]
    for case,selected in cases:
        for dtype in ('float16','bfloat16'):
            for layout in ('ragged','paged'):
                for split in ('auto','unsplit'):
                    for calls in (0,1,16):
                        for metric in ('run_device_us','cycle_us'):
                            cap_value=1/selected_gain
                            guarded_value=cap_value if selected else 1.0
                            cell={'case':case,'dtype':dtype,'layout':layout,'split':split,'calls':calls,'metric':metric,'shard':0,'selected':selected,
                                  'comparisons':{'off':{'ratio':1.0,'controls_resolve':True},'cap':{'ratio':selected_gain,'controls_resolve':True},'guarded':{'ratio':selected_gain if selected else 1.0,'controls_resolve':True}},
                                  'bootstrap_basis':{'pristine':basis(1.0),'off':basis(1.0),'cap':basis(cap_value),'guarded':basis(guarded_value)}}
                            cells.append(cell)
    return cells

def numerics(n):
    return {mode:{'qualifications':n,'exact_full_outputs':n,'max_abs_vs_pristine':0.0,'FP32_max_abs':0.01,'FP32_vectors':100} for mode in MODES}

def source_hashes(measure='a'*64,prepare='b'*64):
    return {'measure.py':measure,'prepare_guarded.py':prepare,'manifest.py':'c'*64}

def environment(case_hash,source):
    return {'common':{'case_hash':case_hash,'source_archive_sha256':source,'gpu':'NVIDIA GeForce RTX 5090','driver':'580.82.07','num_sm':170,'cuda':'13.0','torch':'2.13.0+cu130','flashinfer':'0.7.0','official_overlay_sha256':'o','profiling_mode':'unprofiled-release-timing'},
            'mode_builds':{mode:{'mode':mode,'header_sha256':'p' if mode=='pristine' else 'm','measurement_source_digest':source} for mode in MODES},'gpu_uuids':{'0':['GPU-a']}}

class CombineFreshnessTests(unittest.TestCase):
    def test_combines_independent_strict_fresh_cases_without_promotion(self):
        primary_cases=[(f'p{i}',i<8) for i in range(28)]+[('dup0',True),('dup1',True)]
        extension_cases=[('e0',True),('e1',False)]
        primary={'stage':'release','gpu':'gpu_5090','manifest_hash':'pm','measurement_source_commit':'1'*40,'analysis_source_commit':'2'*40,
                 'deployment_environment':environment('pm','ps'),'source_hashes':source_hashes(),'cells':make_cells(primary_cases),'numerics':numerics(100),'release_gate':{'pass':True}}
        extension={'stage':'release','gpu':'gpu_5090','manifest_hash':'em','measurement_source_commit':'3'*40,'analysis_source_commit':'4'*40,
                   'deployment_environment':environment('em','es'),'source_hashes':source_hashes(),'cells':make_cells(extension_cases),'numerics':numerics(10),'extension_candidate_gate':{'pass':True}}
        amendment={'original_30_case_primary_gate_unchanged':True,'secondary_strict_freshness_sensitivity_excludes':['dup0','dup1']}
        prereg={'primary_manifest_case_hash':'pm','extension_case_hash':'em','primary_measurement_source_commit':'1'*40,
                'selector_rule_changed':False,'primary_manifest_changed':False,'merge_into_primary_claim':False,
                'excluded_primary_duplicates':['dup0','dup1'],'cases':[{'id':'e0'},{'id':'e1'}]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[]
            for name,obj in [('p.json',primary),('e.json',extension),('a.json',amendment),('r.json',prereg)]:
                path=root/name;path.write_text(json.dumps(obj));paths.append(path)
            result=combine(*paths,'5'*40)
        self.assertTrue(result['combined_candidate_gate_pass'])
        self.assertTrue(result['performance_evidence_support'])
        self.assertFalse(result['single_gpu_release_gate_pass'])
        self.assertFalse(result['serving_validation_eligible'])
        self.assertEqual(len(result['strictly_fresh_primary_cases']),28)

    def test_rejects_measurement_program_drift(self):
        primary_cases=[(f'p{i}',i<8) for i in range(28)]+[('dup0',True),('dup1',True)]
        extension_cases=[('e0',True),('e1',False)]
        primary={'stage':'release','gpu':'gpu_5090','manifest_hash':'pm','measurement_source_commit':'1'*40,'analysis_source_commit':'2'*40,'deployment_environment':environment('pm','ps'),'source_hashes':source_hashes(),'cells':make_cells(primary_cases),'numerics':numerics(100),'release_gate':{'pass':True}}
        extension={'stage':'release','gpu':'gpu_5090','manifest_hash':'em','measurement_source_commit':'3'*40,'analysis_source_commit':'4'*40,'deployment_environment':environment('em','es'),'source_hashes':source_hashes(measure='d'*64),'cells':make_cells(extension_cases),'numerics':numerics(10),'extension_candidate_gate':{'pass':True}}
        amendment={'original_30_case_primary_gate_unchanged':True,'secondary_strict_freshness_sensitivity_excludes':['dup0','dup1']}
        prereg={'primary_manifest_case_hash':'pm','extension_case_hash':'em','primary_measurement_source_commit':'1'*40,'selector_rule_changed':False,'primary_manifest_changed':False,'merge_into_primary_claim':False,'cases':[{'id':'e0'},{'id':'e1'}]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[]
            for name,obj in [('p.json',primary),('e.json',extension),('a.json',amendment),('r.json',prereg)]:path=root/name;path.write_text(json.dumps(obj));paths.append(path)
            with self.assertRaisesRegex(ValueError,'measurement program'):combine(*paths,'5'*40)

    def test_rejects_environment_drift(self):
        primary={'stage':'release','gpu':'gpu_5090','manifest_hash':'pm','measurement_source_commit':'1'*40,'analysis_source_commit':'2'*40,'deployment_environment':environment('pm','ps'),'source_hashes':source_hashes(),'cells':[],'numerics':numerics(0),'release_gate':{'pass':False}}
        extension=copy.deepcopy(primary);extension.update(manifest_hash='em',measurement_source_commit='3'*40,analysis_source_commit='4'*40);extension['deployment_environment']=environment('em','es');extension['deployment_environment']['common']['driver']='581.0';extension['extension_candidate_gate']={'pass':False}
        amendment={'original_30_case_primary_gate_unchanged':True,'secondary_strict_freshness_sensitivity_excludes':['dup0','dup1']}
        prereg={'primary_manifest_case_hash':'pm','extension_case_hash':'em','primary_measurement_source_commit':'1'*40,'selector_rule_changed':False,'primary_manifest_changed':False,'merge_into_primary_claim':False,'excluded_primary_duplicates':['dup0','dup1'],'cases':[{'id':'e0'},{'id':'e1'}]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[]
            for name,obj in [('p.json',primary),('e.json',extension),('a.json',amendment),('r.json',prereg)]:path=root/name;path.write_text(json.dumps(obj));paths.append(path)
            with self.assertRaisesRegex(ValueError,'environment'):combine(*paths,'5'*40)

if __name__=='__main__':unittest.main()
