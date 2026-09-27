"""Synthetic parser/contract unit fixtures, never benchmark evidence."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from audit_canaries import audit_run, ratio_interval

class AuditTests(unittest.TestCase):
    def fixture(self, rows, **changes):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        raw = root / 'measurements.jsonl'
        raw.write_text(''.join(json.dumps(x) + '\n' for x in rows))
        summary = {'records': len(rows), 'planned_records': len(rows),
                   'completed_records': sum(x['status'] == 'complete' for x in rows),
                   'measurements_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
                   'policies': ['auto', 'none']}
        summary.update(changes)
        (root / 'summary.json').write_text(json.dumps(summary))
        return raw
    def row(self):
        return {'shape': {'name': 'unit_fixture'}, 'hq': 16, 'policy': 'auto',
                'status': 'complete', 'planner_match': True,
                'full_output_comparison': {'passed': True, 'max_abs': 0},
                'fp32_reference': {'vectors': 1, 'max_abs': 0}}
    def test_valid_counts(self):
        result = audit_run(self.fixture([self.row()]))
        self.assertEqual(result['complete_records'], 1)
        self.assertEqual(result['non_auto_full_output_comparisons'], 0)
    def test_hash_mismatch(self):
        with self.assertRaises(ValueError):
            audit_run(self.fixture([self.row()], measurements_sha256='bad'))
    def test_count_mismatch(self):
        with self.assertRaises(ValueError):
            audit_run(self.fixture([self.row()], completed_records=99))
    def test_duplicate(self):
        with self.assertRaises(ValueError):
            audit_run(self.fixture([self.row(), self.row()]))
    def test_reference_failure(self):
        row = self.row(); row['full_output_comparison']['passed'] = False
        with self.assertRaises(ValueError):
            audit_run(self.fixture([row]))
    def test_errors_are_retained(self):
        row = self.row(); row.update(status='error', error='unit-test error')
        result = audit_run(self.fixture([row]))
        self.assertEqual(result['failed_records'], 1)
        self.assertEqual(result['errors'][0]['error'], 'unit-test error')
    def test_bootstrap_constant(self):
        self.assertEqual(ratio_interval([5., 5.], [4., 4.]), [1.25, 1.25])
    def test_invalid_timing(self):
        with self.assertRaises(ValueError):
            ratio_interval([0.], [1.])

if __name__ == '__main__':
    unittest.main()
