"""CPU-only full pipeline tests on explicitly synthetic serving artifacts."""
import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest

from research.execution_contracts import audit, pipeline
from research.execution_contracts.test_contracts import make_job, save, complete


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)/'campaign'
        self.repo = Path(self.tmp.name)/'repo'
        self.out = Path(self.tmp.name)/'output'
        (self.root/'serving-runs').mkdir(parents=True)
        producer = self.repo/'research/resource_generalization/serving'
        producer.mkdir(parents=True)
        for name in ('http_bench.py', 'lifecycle.py'):
            (producer/name).write_text('synthetic producer fixture: '+name)
        analysis = self.repo/'research/resource_generalization/analysis'
        analysis.mkdir(parents=True)
        original = Path(__file__).resolve().parents[2]/'research/resource_generalization/analysis/analyze_http.py'
        shutil.copyfile(original, analysis/'analyze_http.py')
        for rep, job in enumerate((100, 101, 102)):
            path = make_job(self.root/'serving-runs'/str(job), 'formal')
            (path/'hardware.csv').write_text('name, uuid, driver_version, power.limit\nfixtureGPU, fixture-'+str(rep)+', fixture-driver, 450.00 W\n')
            for mode in ('pristine', 'off', 'cap'):
                p = path/mode/'environment.json'
                env = audit.load(p)
                env['rep'] = rep
                env['source_sha256'] = audit.digest_bytes((producer/'http_bench.py').read_bytes())
                save(p, env)
                complete(p.parent, HTTP=True, full_model=True, mode=mode, stage='formal', rep=rep)
                p = path/('lifecycle-'+mode)/'complete.json'
                life = audit.load(p)
                life['source_sha256'] = audit.digest_bytes((producer/'lifecycle.py').read_bytes())
                save(p, life)

    def tearDown(self):
        self.tmp.cleanup()

    def run_pipeline(self, jobs=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return pipeline.run(self.root, jobs or [100, 101, 102], 'formal', self.repo, self.out)

    def test_complete_pipeline(self):
        self.assertEqual(self.run_pipeline(), 0)
        gate = audit.load(self.out/'gate.json')
        self.assertTrue(gate['integrity_pass'])
        self.assertFalse(gate['serving_promotion'])
        stats = audit.load(self.out/'statistics/summary.json')
        self.assertEqual(stats['total_successful_requests'], 1296)
        self.assertTrue(all(x.get('favorable_ratio') == 1 for x in stats['comparisons']))

    def test_parity_quarantines_statistics(self):
        path = self.root/'serving-runs/100/cap'
        p = path/'decode-b0.json';data = audit.load(p)
        data['requests'][0]['tokens'][0] += 1
        save(p, data)
        complete(path, HTTP=True, full_model=True, mode='cap', stage='formal', rep=0)
        self.assertEqual(self.run_pipeline(), 2)
        report = audit.load(self.out/'gate.json')
        self.assertEqual(report['status'], 'QUARANTINED')
        self.assertEqual(len(report['audits'][0]['mismatches']), 1)
        self.assertFalse((self.out/'statistics').exists())

    def test_duplicate_jobs_rejected(self):
        self.assertEqual(self.run_pipeline([100, 100, 102]), 2)
        self.assertFalse((self.out/'statistics').exists())

    def test_missing_repeat_rejected(self):
        self.assertEqual(self.run_pipeline([100, 101]), 2)

    def test_source_mutation_rejected(self):
        (self.repo/'research/resource_generalization/serving/http_bench.py').write_text('changed')
        self.assertEqual(self.run_pipeline(), 2)
        self.assertIn('source', audit.load(self.out/'gate.json')['message'])

    def test_statistical_source_change_rejected(self):
        p = self.repo/'research/resource_generalization/analysis/analyze_http.py'
        with p.open('a') as f:
            f.write('\n# unreviewed change\n')
        self.assertEqual(self.run_pipeline(), 2)
        self.assertIn('re-review', audit.load(self.out/'gate.json')['message'])

    def test_cross_driver_rejected(self):
        p = self.root/'serving-runs/101/hardware.csv'
        p.write_text(p.read_text().replace('fixture-driver', 'other-driver'))
        self.assertEqual(self.run_pipeline(), 2)
        self.assertIn('mixed', audit.load(self.out/'gate.json')['message'])

    def test_multiple_visible_gpus_rejected(self):
        p = self.root/'serving-runs/101/hardware.csv'
        with p.open('a') as f:
            f.write('fixtureGPU, anotherGPU, fixture-driver, 450.00 W\n')
        self.assertEqual(self.run_pipeline(), 2)
        self.assertIn('single visible', audit.load(self.out/'gate.json')['message'])

    def test_out_never_overwritten(self):
        self.out.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_pipeline()

    def test_nvidia_smi_actual_units_header(self):
        for job in (100, 101, 102):
            p = self.root/'serving-runs'/str(job)/'hardware.csv'
            p.write_text(p.read_text().replace('power.limit\n', 'power.limit [W]\n'))
        self.assertEqual(self.run_pipeline(), 0)


class DiagnosticTests(unittest.TestCase):
    def test_difference_and_length(self):
        from research.execution_contracts.diagnose import difference
        self.assertEqual(difference([1, 2], [1, 3, 4]),
                         dict(first_difference=1, differing_positions=2, lengths=[2, 3]))
        self.assertIsNone(difference([1], [1])['first_difference'])

    def test_localize_without_promoting(self):
        from research.execution_contracts.diagnose import diagnose
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'serving-runs').mkdir()
            p = make_job(root/'serving-runs/101', 'formal')
            f = p/'cap/decode-b1.json'
            data = audit.load(f)
            data['requests'][0]['tokens'][74] += 1
            save(f, data)
            complete(f.parent, mode='cap', stage='formal', rep=0, full_model=True, HTTP=True)
            result = diagnose(root, [101])
            self.assertEqual(result['jobs'][0]['cross_arm'][0]['first_difference'], 74)
            self.assertEqual(result['jobs'][0]['within_mode'][0]['first_difference'], 74)
            self.assertFalse(result['causality_established'])
            self.assertEqual(result['new_gpu_runs'], 0)
            with self.assertRaises(ValueError):
                diagnose(root, [101, 101])
