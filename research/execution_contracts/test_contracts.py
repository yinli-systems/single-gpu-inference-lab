"""Synthetic protocol/fault-injection tests; never GPU performance evidence."""
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from research.execution_contracts import audit
from research.execution_contracts.core import Geometry, Interval, net_value, reversal_witness


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, allow_nan=False))


def complete(path, **kwargs):
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob('*.json')
             if p.name != 'complete.json'}
    save(path/'complete.json', dict(complete=True, files=files, **kwargs))


def make_batch(spec):
    rows = []
    for i, cell in enumerate(spec['cells']):
        times = [.1+j*.01 for j in range(cell['output_tokens'])]
        rows.append(dict(id=cell['id'], input_tokens=len(cell['input_ids']),
                         tokens=list(range(cell['output_tokens'])), token_times=times,
                         events=[dict(received=t, count=j+1, new=1) for j, t in enumerate(times)],
                         ttft=times[0], tpot=(times[-1]-times[0])/(len(times)-1),
                         latency=times[-1]+.01, request_start=0., completion=times[-1]+.01,
                         finish_reason={'type': 'length'}))
    elapsed = 2.
    output = sum(len(r['tokens']) for r in rows)
    return dict(workload=spec['name'], concurrency=spec['concurrency'], elapsed=elapsed,
                requests=rows, errors=[], output_tokens=output,
                output_tokens_per_second=output/elapsed, requests_per_second=len(rows)/elapsed,
                strict_slo_goodput=len(rows)/elapsed,
                TTFT={str(p): audit.quantile([r['ttft'] for r in rows], p) for p in (.5, .95, .99)},
                TPOT={str(p): audit.quantile([r['tpot'] for r in rows], p) for p in (.5, .95, .99)},
                workload_sha256=audit.digest_bytes(json.dumps(spec, sort_keys=True).encode()))


def make_job(root, stage='smoke'):
    root.mkdir()
    (root/'launcher-complete.txt').write_text('synthetic fixture, not a real job\n')
    (root/'exit.txt').write_text('0\n')
    model = {name: dict(bytes=123, sha256='a'*64) for name in
             ['config.json', 'model.safetensors.index.json'] + [f'model-{i:05d}-of-00003.safetensors' for i in range(1, 4)]}
    save(root/'model-binding.json', model)
    for mode in ('pristine', 'off', 'cap'):
        life = root/('lifecycle-'+mode)
        records = [dict(key=f'{d}-shared{s}-stream{i}-transition{t}', graph_eager_exact=True,
                        pristine_exact=True, full_max_abs=0.,
                        FP32=dict(vectors=32, max_abs=.001, rmse=.0001, lse_max_abs=.001))
                   for d in ('float16', 'bfloat16') for s in (False, True)
                   for i in range(2) for t in range(4)]
        save(life/'qualification.json', records)
        complete(life, checks=32, mode=mode, header_sha256='a'*64)
        env = dict(mode=mode, stage=stage, rep=0, HTTP=True, full_model=True, all_model_layers=36,
                   flashinfer='0.7.0', sglang='0.5.20', config_sha256='a'*64,
                   model_index_sha256='a'*64, source_sha256='b'*64,
                   prefill_sha256='a'*64, input_API='native token IDs, tokenizer initialization skipped')
        save(root/mode/'environment.json', env)
        for name, spec in audit.workloads(stage).items():
            for b in range(1 if stage == 'smoke' else 3):
                save(root/mode/f'{name}-b{b}.json', make_batch(spec))
        complete(root/mode, HTTP=True, full_model=True, mode=mode, stage=stage, rep=0)
    return root


class CoreTests(unittest.TestCase):
    def test_exact_covariance_10000(self):
        rng = random.Random(930331)
        for _ in range(10000):
            n = rng.randrange(1, 17)
            q = tuple(rng.randrange(1, 4097) for _ in range(n))
            k = tuple(rng.randrange(32769) for _ in range(n))
            g = Geometry(q, k)
            self.assertEqual(g.work-g.independent_work,
                             sum(a*b for a, b in zip(q, k))-Fraction(sum(q)*sum(k), n))

    def test_equal_marginals_different_work(self):
        a, b = Geometry((1, 3), (2, 8)), Geometry((1, 3), (8, 2))
        self.assertEqual(a.features('marginals'), b.features('marginals'))
        self.assertNotEqual(a.work, b.work)
        self.assertNotEqual(a.features('joint_work'), b.features('joint_work'))

    def test_permutation_is_not_order_identity(self):
        a, b = Geometry((1, 3), (2, 8)), Geometry((3, 1), (8, 2))
        self.assertEqual(a.features('unordered_pairs'), b.features('unordered_pairs'))
        self.assertNotEqual(a.features('ordered_pairs'), b.features('ordered_pairs'))

    def test_minimax(self):
        a, b = Geometry((1, 3), (2, 8)), Geometry((1, 3), (8, 2))
        result = reversal_witness(a, b, 'marginals', 'fixed', 'fixed',
                                  (Interval(9, 10), Interval(13, 15)),
                                  (Interval(15, 17), Interval(10, 11)))
        self.assertAlmostEqual(result['two_action_minimax_regret_lower_bound'], 12/7)
        self.assertFalse(result['statistical_guarantee'])
        self.assertFalse(result['measured_gpu_witness'])
        for i in range(10001):
            p = i/10000
            self.assertGreaterEqual(max((1-p)*3, p*4)+1e-12, 12/7)

    def test_context_mismatch_rejected(self):
        g = Geometry((1,), (2,))
        with self.assertRaises(ValueError):
            reversal_witness(g, g, 'joint_work', 'graph', 'eager',
                             (Interval(1, 2), Interval(4, 5)), (Interval(4, 5), Interval(1, 2)))

    def test_different_features_rejected(self):
        with self.assertRaises(ValueError):
            reversal_witness(Geometry((1,), (2,)), Geometry((2,), (1,)), 'ordered_pairs', 'x', 'x',
                             (Interval(1, 2), Interval(4, 5)), (Interval(4, 5), Interval(1, 2)))

    def test_aggregate_features(self):
        self.assertEqual(Geometry((1, 3), (2, 8)).features('aggregate'), (2, 4, 10))

    def test_overlap_is_not_a_witness(self):
        g = Geometry((1,), (2,))
        with self.assertRaisesRegex(ValueError, 'not separated'):
            reversal_witness(g, g, 'aggregate', 'x', 'x',
                             (Interval(1, 3), Interval(2, 4)), (Interval(4, 5), Interval(1, 2)))

    def test_cost_overflow(self):
        with self.assertRaisesRegex(ValueError, 'overflow'):
            net_value(guaranteed_reuses=10, native_lower_us=1e308, candidate_upper_us=1,
                      probe_us=0, setup_us=0, dispatch_per_call_us=0,
                      context_matches=True, qualified=True)

    def test_unknown_representation(self):
        with self.assertRaises(ValueError):
            Geometry((1,), (2,)).features('unknown')

    def test_numeric_faults(self):
        for bad in (float('nan'), float('inf'), -1, True, '3', None):
            with self.subTest(bad=repr(bad)), self.assertRaises((ValueError, TypeError)):
                Interval(bad, 10)
        for args in (((), ()), ((True,), (1,)), ((1,), (-1,)), ([1], [2]), ((1, 2), (1,))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Geometry(*args)

    def test_budget_gates(self):
        kwargs = dict(native_lower_us=100, candidate_upper_us=80, probe_us=10, setup_us=20,
                      dispatch_per_call_us=2, sunk_us=5, context_matches=True, qualified=True)
        yes = net_value(guaranteed_reuses=10, **kwargs)
        self.assertEqual(yes['net_lower_bound_us'], 145.)
        self.assertEqual(yes['decision'], 'candidate_eligible')
        self.assertFalse(yes['runtime_promoted'])
        for changed in ({'guaranteed_reuses': None}, {'guaranteed_reuses': 1},
                        {'guaranteed_reuses': 10, 'qualified': False},
                        {'guaranteed_reuses': 10, 'context_matches': False}):
            result = net_value(**(kwargs | changed))
            self.assertEqual(result['decision'], 'native')
            self.assertEqual(result['fallback_cost_us'], 35.)
        for value in (0, -1, True, 3., '10'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                net_value(guaranteed_reuses=value, **kwargs)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_job(Path(self.tmp.name)/'job')
        self.spec = audit.workloads('smoke')['smoke']

    def tearDown(self):
        self.tmp.cleanup()

    def test_smoke_integrity_no_promotion(self):
        result = audit.audit_http_job(self.root, 'smoke')
        self.assertEqual(result['successful_requests'], 12)
        self.assertTrue(result['token_parity'])
        self.assertFalse(result['serving_promotion'])
        self.assertFalse(result['candidate_launch_attributed'])

    def test_formal_fixture(self):
        root = make_job(Path(self.tmp.name)/'formal', 'formal')
        result = audit.audit_http_job(root, 'formal')
        self.assertEqual(result['successful_requests'], 432)
        self.assertFalse(result['serving_promotion'])

    def test_token_mismatch_is_blocked_not_dropped(self):
        p = self.root/'cap'/'smoke-b0.json'
        data = audit.load(p)
        data['requests'][0]['tokens'][0] += 1
        save(p, data)
        complete(p.parent, HTTP=True, full_model=True, mode='cap', stage='smoke', rep=0)
        result = audit.audit_http_job(self.root, 'smoke')
        self.assertFalse(result['token_parity'])
        self.assertEqual(len(result['mismatches']), 1)
        self.assertEqual(result['successful_requests'], 12)

    def test_missing_manifest_entry(self):
        p = self.root/'cap'/'complete.json'
        c = audit.load(p)
        c['files'].pop('smoke-b0.json')
        save(p, c)
        with self.assertRaisesRegex(ValueError, 'hash-bound'):
            audit.audit_http_job(self.root, 'smoke')

    def test_symlink_manifest_target(self):
        p = self.root/'cap'/'environment.json'
        q = Path(self.tmp.name)/'elsewhere.json'
        p.rename(q)
        p.symlink_to(q)
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            audit.audit_http_job(self.root, 'smoke')

    def test_path_traversal(self):
        p = self.root/'cap'/'complete.json'
        c = audit.load(p)
        c['files']['../x'] = 'a'*64
        save(p, c)
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            audit.audit_http_job(self.root, 'smoke')

    def test_duplicate_json_keys(self):
        p = Path(self.tmp.name)/'duplicate.json'
        p.write_text('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            audit.load(p)

    def test_exponent_overflow_json(self):
        p = Path(self.tmp.name)/'exp.json'
        p.write_text('{"a":1e999}')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            audit.load(p)

    def test_cli_in_process(self):
        out = Path(self.tmp.name)/'in-process.json'
        with patch.object(sys, 'argv', ['audit', '--out', str(out), 'http', '--job', str(self.root), '--stage', 'smoke']):
            self.assertEqual(audit.main(), 0)
        self.assertTrue(audit.load(out)['integrity_pass'])

    def test_cli_records_integrity_failure(self):
        out = Path(self.tmp.name)/'failure-audit.json'
        (self.root/'exit.txt').write_text('1')
        with patch.object(sys, 'argv', ['audit', '--out', str(out), 'http', '--job', str(self.root), '--stage', 'smoke']):
            self.assertEqual(audit.main(), 2)
        self.assertFalse(audit.load(out)['integrity_pass'])

    def test_nonfinite_json(self):
        p = Path(self.tmp.name)/'nan.json'
        p.write_text('{"a":NaN}')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            audit.load(p)

    def test_partial_job(self):
        (self.root/'exit.txt').write_text('1')
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            audit.audit_http_job(self.root, 'smoke')

    def test_lifecycle_false_pass(self):
        p = self.root/'lifecycle-cap'/'qualification.json'
        data = audit.load(p)
        data[0]['graph_eager_exact'] = False
        save(p, data)
        complete(p.parent, checks=32, mode='cap', header_sha256='a'*64)
        with self.assertRaisesRegex(ValueError, 'not exact'):
            audit.audit_http_job(self.root, 'smoke')

    def test_environment_mismatch(self):
        p = self.root/'cap'/'environment.json'
        data = audit.load(p)
        data['sglang'] = 'different'
        save(p, data)
        complete(p.parent, HTTP=True, full_model=True, mode='cap', stage='smoke', rep=0)
        with self.assertRaisesRegex(ValueError, 'environment mismatch'):
            audit.audit_http_job(self.root, 'smoke')

    def test_model_shard_missing(self):
        p = self.root/'model-binding.json'
        data = audit.load(p)
        data.pop('model-00001-of-00003.safetensors')
        save(p, data)
        with self.assertRaisesRegex(ValueError, 'missing model shard'):
            audit.audit_http_job(self.root, 'smoke')

    def test_cli_does_not_overwrite(self):
        out = Path(self.tmp.name)/'audit.json'
        cmd = [sys.executable, '-m', 'research.execution_contracts.audit', '--out', str(out),
               'http', '--job', str(self.root), '--stage', 'smoke']
        first = subprocess.run(cmd, capture_output=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        before = out.read_bytes()
        second = subprocess.run(cmd, capture_output=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(before, out.read_bytes())


def batch_corruptions():
    return {
        'ttft': lambda d: d['requests'][0].__setitem__('ttft', 0.),
        'tpot': lambda d: d['requests'][0].__setitem__('tpot', 0.),
        'latency': lambda d: d['requests'][0].__setitem__('latency', .01),
        'tail': lambda d: d['TTFT'].__setitem__('0.99', .001),
        'goodput': lambda d: d.__setitem__('strict_slo_goodput', 999.),
        'request_rate': lambda d: d.__setitem__('requests_per_second', 99.),
        'output_count': lambda d: d.__setitem__('output_tokens', 1),
        'throughput': lambda d: d.__setitem__('output_tokens_per_second', 999.),
        'bool_timing': lambda d: d['requests'][0].__setitem__('ttft', True),
        'nan_timing': lambda d: d['requests'][0].__setitem__('ttft', float('nan')),
        'inf_elapsed': lambda d: d.__setitem__('elapsed', float('inf')),
        'zero_elapsed': lambda d: d.__setitem__('elapsed', 0.),
        'request_id': lambda d: d['requests'][0].__setitem__('id', 'not-requested'),
        'duplicate_request': lambda d: d['requests'].__setitem__(1, d['requests'][0]),
        'bool_token': lambda d: d['requests'][0]['tokens'].__setitem__(0, True),
        'negative_token': lambda d: d['requests'][0]['tokens'].__setitem__(0, -1),
        'input_count': lambda d: d['requests'][0].__setitem__('input_tokens', 1),
        'missing_finish': lambda d: d['requests'][0].__setitem__('finish_reason', None),
        'event_count': lambda d: d['requests'][0]['events'][0].__setitem__('count', 2),
        'event_time': lambda d: d['requests'][0]['events'][0].__setitem__('received', .11),
        'time_reversal': lambda d: d['requests'][0]['token_times'].__setitem__(1, 0.),
        'end_outside': lambda d: d['requests'][0].__setitem__('completion', 3.),
        'errors': lambda d: d['errors'].append('preserved request failure'),
        'concurrency': lambda d: d.__setitem__('concurrency', 100),
        'workload_hash': lambda d: d.__setitem__('workload_sha256', 'a'*64),
        'missing_quantile': lambda d: d['TTFT'].pop('0.95'),
        'token_count': lambda d: d['requests'][0]['tokens'].pop(),
    }


def add_fault_test(name, mutate):
    def test(self):
        data = make_batch(self.spec)
        mutate(data)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            audit.validate_batch(data, self.spec)
    setattr(AdapterTests, 'test_fault_'+name, test)


for _name, _mutate in batch_corruptions().items():
    add_fault_test(_name, _mutate)


class ScanTests(unittest.TestCase):
    def data(self):
        cases = [dict(id='left', family='development', q=[1, 3], cached=[2, 8]),
                 dict(id='right', family='development', q=[1, 3], cached=[8, 2])]
        h = audit.digest_bytes(json.dumps(cases, sort_keys=True, separators=(',', ':')).encode())
        cells = [dict(case=c['id'], shard=0, dtype='float16', layout='ragged', split='unsplit',
                      calls=1, metric='cycle_us', comparisons={'cap': dict(ratio=r,
                          CI95=[r-.01, r+.01], controls_resolve=True)})
                 for c, r in zip(cases, (.8, 1.2))]
        return dict(cases=cases, case_hash=h), dict(complete=True, stage='development', gpu='fixture',
                 source_hashes={'fixture': 'synthetic'}, hardware={'0': ['synthetic']}, manifest_hash=h, cells=cells)

    def test_exact_reversal_candidate_only(self):
        manifest, summary = self.data()
        result = audit.scan_summary(manifest, summary, 'marginals')
        self.assertEqual(result['candidate_count'], 1)
        self.assertFalse(result['measured_gpu_witness'])
        self.assertFalse(result['serving_promotion'])

    def test_stronger_representation_no_false_collision(self):
        manifest, summary = self.data()
        result = audit.scan_summary(manifest, summary, 'joint_work')
        self.assertEqual(result['candidate_count'], 0)
        self.assertTrue(result['no_witness_is_not_sufficiency_proof'])

    def test_controls_unresolved(self):
        manifest, summary = self.data()
        summary['cells'][0]['comparisons']['cap']['controls_resolve'] = False
        self.assertEqual(audit.scan_summary(manifest, summary, 'marginals')['candidate_count'], 0)

    def test_cross_dtype_not_a_collision(self):
        manifest, summary = self.data()
        summary['cells'][0]['dtype'] = 'bfloat16'
        self.assertEqual(audit.scan_summary(manifest, summary, 'marginals')['candidate_count'], 0)

    def test_duplicate_cells_rejected(self):
        manifest, summary = self.data()
        summary['cells'].append(summary['cells'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            audit.scan_summary(manifest, summary, 'marginals')

    def test_overlapping_intervals_not_a_reversal(self):
        manifest, summary = self.data()
        for cell in summary['cells']:
            cell['comparisons']['cap'].update(ratio=1., CI95=[.9, 1.1])
        self.assertEqual(audit.scan_summary(manifest, summary, 'marginals')['candidate_count'], 0)

    def test_cli_scanner(self):
        manifest, summary = self.data()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            save(p/'manifest.json', manifest)
            save(p/'summary.json', summary)
            with patch.object(sys, 'argv', ['audit', '--out', str(p/'out.json'), 'scan',
                                          '--manifest', str(p/'manifest.json'), '--summary', str(p/'summary.json'),
                                          '--representation', 'marginals']):
                self.assertEqual(audit.main(), 0)
            self.assertEqual(audit.load(p/'out.json')['candidate_count'], 1)

    def test_stage_leakage_rejected(self):
        manifest, summary = self.data()
        summary['stage'] = 'confirmatory'
        with self.assertRaisesRegex(ValueError, 'stage leakage'):
            audit.scan_summary(manifest, summary, 'marginals')


if __name__ == '__main__':
    unittest.main()
