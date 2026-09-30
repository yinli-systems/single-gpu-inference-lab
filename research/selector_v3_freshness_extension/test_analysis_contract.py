import copy,json,tempfile,unittest
from pathlib import Path

from analyze import deployment_environment_identity,freshness_sensitivity_amendment,mode_build_identity


def environment(uuid='GPU-a',driver='580.82.07',header='a'*64,mode='guarded'):
    return {
        'partition':'gpu_5090','case_hash':'b'*64,'gpu':'NVIDIA GeForce RTX 5090','num_sm':170,
        'cuda':'13.0','torch':'2.13.0+cu130','flashinfer':'0.7.0','official_overlay_sha256':'c'*64,
        'source_archive_sha256':'d'*64,'profiling_mode':'unprofiled-release-timing','profiled':False,
        'nsight_compute_excluded':True,'clocks_locked':False,'whole_node_exclusive':False,
        'full_model':False,'output_seed':'fixed','mode':mode,'header_sha256':header,
        'source':{'measure.py':'e'*64,'manifest.py':'f'*64},
        'hardware':{'out':f'name, uuid, driver_version, power.limit [W]\nNVIDIA GeForce RTX 5090, {uuid}, {driver}, 575.00 W\n'},
    }


class AnalysisContractTests(unittest.TestCase):
    def test_deployment_identity_allows_uuid_but_not_driver_drift(self):
        a=deployment_environment_identity(environment(uuid='GPU-a'))
        b=deployment_environment_identity(environment(uuid='GPU-b'))
        self.assertEqual(a,b)
        c=deployment_environment_identity(environment(uuid='GPU-c',driver='581.00.00'))
        self.assertNotEqual(a,c)

    def test_mode_build_identity_is_source_order_stable_and_header_bound(self):
        a=environment();b=copy.deepcopy(a);b['source']={'manifest.py':'f'*64,'measure.py':'e'*64}
        self.assertEqual(mode_build_identity(a),mode_build_identity(b))
        b['header_sha256']='1'*64
        self.assertNotEqual(mode_build_identity(a),mode_build_identity(b))

    def test_freshness_amendment_is_measurement_commit_bound(self):
        commit='1'*40
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'receipts').mkdir()
            receipt={
                'schema':2,'measurement_source_commit':commit,'selector_rule_changed':False,
                'measurement_manifest_changed':False,'original_30_case_primary_gate_unchanged':True,
                'registered_before_any_release_performance_ratio_analysis':True,'canary_performance_used':False,
                'secondary_strict_freshness_sensitivity_excludes':['duplicate-a','duplicate-b'],
                'strictly_fresh_relative_to_canary_case_count':2,
            }
            (root/'receipts'/'freshness-sensitivity-amendment-v2.json').write_text(json.dumps(receipt))
            out=freshness_sensitivity_amendment(root,commit,{'duplicate-a','duplicate-b','fresh-a','fresh-b'})
            self.assertEqual(out['excluded_cases'],['duplicate-a','duplicate-b'])
            with self.assertRaises(ValueError):
                freshness_sensitivity_amendment(root,'2'*40,{'duplicate-a','duplicate-b','fresh-a','fresh-b'})


if __name__=='__main__':unittest.main()
