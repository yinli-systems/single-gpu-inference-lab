"""Reproduce synthetic corruption-rejection comparisons, not actual data corruption."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from . import audit
from .pipeline import LEGACY_BLOB
from .test_contracts import make_job, complete, batch_corruptions, save


def run(repo: Path) -> dict:
    path = repo/'research/resource_generalization/analysis/analyze_http.py'
    raw = path.read_bytes()
    if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() != LEGACY_BLOB:
        raise ValueError('unreviewed legacy reader')
    spec = importlib.util.spec_from_file_location('legacy_fault_reader', path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    cases = []
    mutations = list(batch_corruptions().items()) + [('empty_hash_manifest', None)]
    for name, mutate in mutations:
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp)/'job')
            file = job/'pristine/smoke-b0.json'
            if mutate is not None:
                obj = audit.load(file)
                mutate(obj)
                # Deliberate NaN/Infinity injections are part of this test only.
                file.write_text(json.dumps(obj, allow_nan=True))
                complete(file.parent, HTTP=True, full_model=True, mode='pristine', stage='smoke', rep=0)
            else:
                obj = audit.load(file.parent/'complete.json')
                obj['files'] = {}
                save(file.parent/'complete.json', obj)
            row = dict(fault=name)
            for label, call in [('legacy', lambda: legacy.read_mode(file.parent, 'smoke')),
                                ('new', lambda: audit.audit_http_job(job, 'smoke'))]:
                try:
                    call()
                    row[label] = 'accepted'
                except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                    row[label] = 'rejected'
                    row[label+'_reason'] = type(exc).__name__+': '+str(exc)
            cases.append(row)
    return dict(kind='synthetic_fault_injection_not_actual_experiment_corruption',
                legacy_git_blob=LEGACY_BLOB, faults=len(cases),
                legacy_accepted=sum(c['legacy']=='accepted' for c in cases),
                new_accepted=sum(c['new']=='accepted' for c in cases), cases=cases, new_gpu_runs=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path('.'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = run(args.repo)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print({k:v for k,v in result.items() if k!='cases'})
