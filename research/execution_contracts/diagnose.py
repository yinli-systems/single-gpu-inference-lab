"""Read-only parity localization after integrity verification; never performance promotion."""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
from pathlib import Path
from .audit import audit_http_job, load
from .core import require


def difference(left: list[int], right: list[int]) -> dict:
    changed = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
    changed += list(range(min(len(left), len(right)), max(len(left), len(right))))
    return dict(first_difference=changed[0] if changed else None,
                differing_positions=len(changed), lengths=[len(left), len(right)])


def diagnose(root: Path, jobs: list[int]) -> dict:
    require(jobs and all(type(j) is int and j > 0 for j in jobs)
            and len(set(jobs)) == len(jobs), 'unique positive jobs required')
    result = dict(schema=1, artifact_integrity_pass=True, new_gpu_runs=0,
                  performance_promotion=False, causality_established=False, jobs=[])
    for job in jobs:
        path = root/'serving-runs'/str(job)
        audit = audit_http_job(path, 'formal')
        blocks = {mode: {f.name: load(f) for f in sorted((path/mode).glob('*-b*.json'))}
                  for mode in ('pristine', 'off', 'cap')}
        cross, within, first_block, profiles = [], [], [], {}
        for name, base in blocks['pristine'].items():
            b = {r['id']: r for r in base['requests']}
            for mode in ('off', 'cap'):
                for row in blocks[mode][name]['requests']:
                    diff = difference(b[row['id']]['tokens'], row['tokens'])
                    if diff['first_difference'] is not None:
                        cross.append(dict(mode=mode, file=name, request=row['id'], **diff))
        for mode, arm in blocks.items():
            for work in ('prefill', 'decode', 'mixed'):
                base = arm[f'{work}-b0.json']
                by_id = {r['id']: r for r in base['requests']}
                for index in (1, 2):
                    target = arm[f'{work}-b{index}.json']
                    require(target['workload_sha256'] == base['workload_sha256'],
                            'different repeated workload: not a within-mode comparison')
                    for row in target['requests']:
                        diff = difference(by_id[row['id']]['tokens'], row['tokens'])
                        if diff['first_difference'] is not None:
                            within.append(dict(mode=mode, work=work, from_block=0,
                                               to_block=index, request=row['id'], **diff))
                later = statistics.median(arm[f'{work}-b{i}.json']['elapsed'] for i in (1, 2))
                first_block.append(dict(mode=mode, work=work,
                    first_over_later_elapsed=base['elapsed']/later,
                    interpretation='descriptive order effect; not proof of JIT or cache causality'))
            p = path/mode/'profile-evidence.json'
            profiles[mode] = (dict(recorded_64KiB_launches=load(p).get('launch_hit_count_64KiB'),
                                  attribution_established=False) if p.exists() else None)
        result['jobs'].append(dict(job=job, token_parity=audit['token_parity'],
            successful_requests=audit['successful_requests'], cross_arm=cross,
            within_mode=within, first_block=first_block, profiles=profiles,
            input_receipts_sha256={mode: hashlib.sha256((path/mode/'complete.json').read_bytes()).hexdigest()
                                   for mode in blocks}))
    result['boundary'] = ('A within-mode mismatch demonstrates output instability in that mode, '
        'not the cause of a between-mode difference. Missing batch signatures/logits prevent '
        'first-operator attribution. All original measured blocks remain included.')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--jobs', type=int, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = diagnose(args.root, args.jobs)
        code = 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        result = dict(artifact_integrity_pass=False, performance_promotion=False,
                      error=type(exc).__name__, message=str(exc))
        code = 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
