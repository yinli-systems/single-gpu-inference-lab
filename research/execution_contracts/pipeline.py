"""Strict validation BEFORE the existing frozen HTTP statistical analyzer.

Read-only for campaign inputs. Output is exclusive-create. Token disagreement,
missing/hash-mismatched evidence, and protocol disagreement quarantine the
comparison instead of producing a speedup table. No runtime promotion occurs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from .audit import audit_http_job, digest_bytes, load
from .core import require

LEGACY_BLOB = '369441f58807d279aac54b967474fdd097f70c46'


def validate_group(root: Path, jobs: list[int], stage: str, repo: Path) -> list[dict]:
    require(jobs and len(set(jobs)) == len(jobs)
            and all(type(j) is int and j > 0 for j in jobs), 'duplicate/invalid jobs')
    require(stage in ('smoke', 'formal'), 'invalid stage')
    if stage == 'formal':
        require(len(jobs) == 3, 'three formal allocations required')
    reports, reps, identities = [], set(), set()
    producer = repo/'research/resource_generalization/serving'
    source_hash = digest_bytes((producer/'http_bench.py').read_bytes())
    lifecycle_hash = digest_bytes((producer/'lifecycle.py').read_bytes())
    for job in jobs:
        path = root/'serving-runs'/str(job)
        report = audit_http_job(path, stage)
        require(report['token_parity'], f'token parity failed in job {job}')
        reports.append(dict(job=job, **report))
        env = load(path/'pristine/environment.json')
        require(type(env['rep']) is int and (env['rep'] not in reps or stage == 'smoke'),
                'duplicate formal repeat')
        reps.add(env['rep'])
        rows = list(csv.DictReader(io.StringIO((path/'hardware.csv').read_text())))
        require(len(rows) == 1, 'single visible GPU required')
        hardware = {k.strip(): v.strip() for k, v in rows[0].items()}
        # nvidia-smi emits the unit suffix unless nounits was requested.
        if 'power.limit [W]' in hardware:
            require('power.limit' not in hardware, 'ambiguous power header')
            hardware['power.limit'] = hardware.pop('power.limit [W]')
        require(all(k in hardware for k in ('name', 'uuid', 'driver_version', 'power.limit')),
                'hardware identity incomplete')
        per_mode = []
        for mode in ('pristine', 'off', 'cap'):
            e = load(path/mode/'environment.json')
            l = load(path/('lifecycle-'+mode)/'complete.json')
            require(e['source_sha256'] == source_hash and l['source_sha256'] == lifecycle_hash,
                    'producer source not bound to reviewed repository')
            per_mode.append((mode, e['prefill_sha256']))
        identities.add(json.dumps((hardware['name'], hardware['driver_version'],
                                  hardware['power.limit'], per_mode, load(path/'model-binding.json')),
                                 sort_keys=True))
    require(len(identities) == 1, 'mixed model/device family/driver/power/source context')
    require(stage != 'formal' or reps == {0, 1, 2}, 'incomplete formal repeat set')
    return reports


def run(root: Path, jobs: list[int], stage: str, repo: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 1, 'stage': stage, 'jobs': jobs, 'serving_promotion': False}
    try:
        # Preserve per-job audit evidence even when a later group gate fails.
        audits = [dict(job=job, **audit_http_job(root/'serving-runs'/str(job), stage)) for job in jobs]
        report['audits'] = audits
        validate_group(root, jobs, stage, repo)
        source = repo/'research/resource_generalization/analysis/analyze_http.py'
        raw = source.read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        require(blob == LEGACY_BLOB, 'statistical analyzer changed: re-review required')
        spec = importlib.util.spec_from_file_location('frozen_http_statistics', source)
        require(spec is not None and spec.loader is not None, 'cannot load reviewed analyzer')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.run(SimpleNamespace(root=root, jobs=jobs, stage=stage, out=out/'statistics'))
        report.update(status='INTEGRITY_PASSED_UNPROMOTED', integrity_pass=True, audits=audits,
                      statistics_blob=blob, boundary='archived HTTP samples, not a new GPU run')
        code = 0
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        report.update(status='QUARANTINED', integrity_pass=False,
                      error=type(exc).__name__, message=str(exc))
        code = 2
    with (out/'gate.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return code


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--jobs', type=int, nargs='+', required=True)
    p.add_argument('--stage', choices=['smoke', 'formal'], required=True)
    p.add_argument('--repo', type=Path, default=Path('.'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    raise SystemExit(run(a.root, a.jobs, a.stage, a.repo, a.out))
