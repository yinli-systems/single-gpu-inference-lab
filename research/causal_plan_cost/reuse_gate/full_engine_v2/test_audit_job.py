"""Whole-job parser tests on explicitly synthetic temporary files; no GPU data."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from audit_observations import audit_job, ARMS
from test_audit_observations import fixture


class JobAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        payload,steps,cells=fixture();self.cells=cells
        marker=dict(all_arms_complete=True,greedy_outputs_identical=True)
        (self.root/'complete.json').write_text(json.dumps(marker))
        for arm in ARMS:
            folder=self.root/arm;folder.mkdir()
            summary=dict(complete=True,arm=arm,vllm='0.29.0',flashinfer='0.6.18',
                dtype='float16',eager=True,requested_KV_blocks=2048,page_size=16,scratch_MiB=512,
                source_sha256={'/fake/measure_engine.py':'synthetic-source-hash'},
                model_config_sha256='synthetic-config-hash',actual_config_KV_blocks=2048)
            (folder/'summary.json').write_text(json.dumps(summary))
            for phase in ('warmup','measured'):
                for name in ('burst','staggered'):
                    record=copy.deepcopy(payload);record.update(phase=phase,workload=name)
                    (folder/f'{phase}-{name}.json').write_text(json.dumps(record))
                    (folder/f'{phase}-{name}-steps.jsonl').write_text('\n'.join(map(json.dumps,steps))+'\n')
        self.module=patch.dict('sys.modules',{'engine_screen':SimpleNamespace(workload=lambda name:self.cells)})
        self.module.start();self.addCleanup(self.module.stop)
    def change(self,path,mutate):
        p=self.root/path;x=json.loads(p.read_text());mutate(x);p.write_text(json.dumps(x))
    def test_complete_triplet_not_a_performance_claim(self):
        r=audit_job(self.root);self.assertTrue(r['evidence_audit_passed'])
        self.assertTrue(r['greedy_outputs_identical']);self.assertFalse(r['performance_promoted'])
        self.assertEqual(len(r['work']),12)
    def test_missing_arm_is_not_omitted(self):
        (self.root/'fixed_budget/summary.json').unlink()
        with self.assertRaises(FileNotFoundError):audit_job(self.root)
    def test_missing_warmup_is_not_omitted(self):
        (self.root/'auto/warmup-burst.json').unlink()
        with self.assertRaises(FileNotFoundError):audit_job(self.root)
    def test_missing_completion_is_not_success(self):
        (self.root/'complete.json').unlink()
        with self.assertRaises(ValueError):audit_job(self.root)
    def test_wrong_engine_version(self):
        self.change('native_cycle/summary.json',lambda x:x.update(vllm='other'))
        with self.assertRaises(ValueError):audit_job(self.root)
    def test_changed_source_between_arms(self):
        self.change('native_cycle/summary.json',lambda x:x['source_sha256'].update({'/fake/measure_engine.py':'different'}))
        with self.assertRaises(ValueError):audit_job(self.root)
    def test_changed_model_between_arms(self):
        self.change('native_cycle/summary.json',lambda x:x.update(model_config_sha256='different'))
        with self.assertRaises(ValueError):audit_job(self.root)
    def test_unknown_actual_capacity_fails_closed(self):
        self.change('native_cycle/summary.json',lambda x:x.update(actual_config_KV_blocks=None))
        r=audit_job(self.root);self.assertFalse(r['capacity_verified']);self.assertFalse(r['performance_promoted'])
    def test_token_difference_preserved(self):
        self.change('native_cycle/measured-burst.json',lambda x:x['requests_detail'][0]['token_ids'].__setitem__(0,8))
        self.change('complete.json',lambda x:x.update(greedy_outputs_identical=False))
        r=audit_job(self.root);self.assertFalse(r['greedy_outputs_identical'])
    def test_falsely_claimed_token_parity_rejected(self):
        self.change('native_cycle/measured-burst.json',lambda x:x['requests_detail'][0]['token_ids'].__setitem__(0,8))
        with self.assertRaises(ValueError):audit_job(self.root)


if __name__=='__main__':unittest.main()
