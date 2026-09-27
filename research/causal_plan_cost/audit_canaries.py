"""Read-only audit of completed FA2 canaries. No GPU calls, fitting, or job submission.

Usage: python audit_canaries.py --root /path/to/plan-cost-project --out NEW.json
The optional output is create-only. Bootstrap intervals are descriptive within-run
block resampling, not independent-process/hardware confidence intervals.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any


def ratio_interval(xs: list[float], ys: list[float]) -> list[float]:
    if not xs or not ys or min(xs + ys) <= 0:
        raise ValueError('Positive timing blocks are required')
    rng = random.Random(20260928)
    draws = sorted(statistics.median(rng.choices(xs, k=len(xs))) /
                   statistics.median(rng.choices(ys, k=len(ys)))
                   for _ in range(5000))
    return [draws[124], draws[4874]]


def audit_run(raw: Path) -> dict[str, Any]:
    blob = raw.read_bytes()
    summary = json.loads((raw.parent / 'summary.json').read_text())
    digest = hashlib.sha256(blob).hexdigest()
    if digest != summary['measurements_sha256']:
        raise ValueError(f'Raw data hash mismatch: {raw}')
    rows = [json.loads(line) for line in blob.splitlines() if line.strip()]
    keys = [(r['shape']['name'], r['hq'], r['policy']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate shape/head/policy result')
    passed = [r for r in rows if r['status'] == 'complete']
    if len(rows) != summary['records'] or len(passed) != summary['completed_records']:
        raise ValueError('Summary/record count mismatch')
    for row in passed:
        if not row.get('planner_match') or not row['full_output_comparison']['passed']:
            raise ValueError('Incomplete numerical/planner evidence')
    result = {
        'run': raw.parent.name, 'raw_sha256': digest,
        'planned_records': summary['planned_records'], 'returned_records': len(rows),
        'complete_records': len(passed), 'failed_records': len(rows) - len(passed),
        'non_auto_full_output_comparisons': sum(r['policy'] != 'auto' for r in passed),
        'max_abs_vs_auto': max((r['full_output_comparison']['max_abs'] for r in passed), default=None),
        'selected_FP32_vectors': sum(r.get('fp32_reference', {}).get('vectors', 0) for r in passed),
        'max_selected_FP32_abs_error': max((r.get('fp32_reference', {}).get('max_abs', 0) for r in passed), default=None),
        'errors': [{'shape': r['shape']['name'], 'hq': r['hq'], 'policy': r['policy'],
                    'error': r.get('error')} for r in rows if r['status'] != 'complete'],
        'equal_work_contrasts': [], 'complete_training_fixture': [],
        'claim_boundary': 'Recovered incomplete canary; not held-out, new GPU execution, or serving speedup.'
    }
    lookup = {(r['shape']['name'], r['hq'], r['policy']): r for r in passed}
    for hq in (16, 32):
        for policy in ('auto', 'none'):
            ak = ('discovery-eq255-A', hq, policy)
            bk = ('discovery-eq255-B', hq, policy)
            if ak not in lookup or bk not in lookup:
                continue
            a, b = lookup[ak], lookup[bk]
            result['equal_work_contrasts'].append({
                'hq': hq, 'policy': policy, 'A_ms': a['median_ms'], 'B_ms': b['median_ms'],
                'ratio_A_B': a['median_ms'] / b['median_ms'],
                'within_run_block_resampling_interval': ratio_interval(a['cuda_ms'], b['cuda_ms']),
                'A_cycle_ms': a['median_cycle_ms'], 'B_cycle_ms': b['median_cycle_ms'],
                'A_plan': a['plan_summary'], 'B_plan': b['plan_summary'],
                'independent_process_replicates_per_hardware': 1
            })
        cells = [lookup.get(('Q512-K1536-n8-A', hq, m)) for m in summary['policies']]
        if all(r is not None for r in cells):
            result['complete_training_fixture'].append({'hq': hq, 'policies': [
                {'policy': r['policy'], 'run_ms': r['median_ms'],
                 'plan_run_sync_ms': r['median_cycle_ms']} for r in cells]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    files = sorted((args.root / 'runs').glob('canary-*/measurements.jsonl'))
    if not files:
        raise SystemExit('No completed canary artifacts found')
    report = {'audit_kind': 'reanalysis_only', 'runs': [audit_run(p) for p in files]}
    text = json.dumps(report, indent=2) + '\n'
    if args.out:
        with args.out.open('x') as out:
            out.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
