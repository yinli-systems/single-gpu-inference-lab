"""Read-only adapter for resource_generalization HTTP and cost-result artifacts.

Does not import campaign code, alter frozen results, submit jobs, or dispatch a
candidate. Integrity checks authenticate neither the author nor physical timing.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
from typing import Any

from .core import Geometry, Interval, number, require, reversal_witness


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pairs(items: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in items:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"missing/unsafe file: {path}")
    require(path.stat().st_size <= 128*1024*1024, "JSON exceeds audit size limit")
    def invalid(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")
    def finite_float(value: str) -> float:
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON number")
        return result
    return json.loads(path.read_text(), object_pairs_hook=_pairs, parse_constant=invalid, parse_float=finite_float)


def bundle(path: Path, required: set[str]) -> dict:
    require(path.is_dir() and not path.is_symlink(), f"missing/unsafe bundle: {path}")
    require(not (path/'failure.json').exists(), f"preserved failure: {path}")
    receipt = load(path/'complete.json')
    require(receipt.get('complete') is True, "bundle incomplete")
    files = receipt.get('files')
    require(type(files) is dict and required <= files.keys(), "required files not hash-bound")
    for name, sha in files.items():
        require(type(name) is str and name not in ('', '.', '..', 'complete.json')
                and Path(name).name == name and '/' not in name and '\\' not in name,
                "unsafe manifest filename")
        require(type(sha) is str and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha),
                "invalid sha256")
        target = path/name
        require(target.is_file() and not target.is_symlink(), f"missing/unsafe bound file: {name}")
        require(digest_bytes(target.read_bytes()) == sha, f"hash mismatch: {name}")
    return receipt


def close(actual: Any, expected: float, name: str) -> None:
    actual = number(actual, name)
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9),
            f"derived metric mismatch: {name}")


def quantile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = (len(ordered)-1)*p
    lo = int(index)
    return ordered[lo] + (ordered[min(lo+1, len(ordered)-1)]-ordered[lo])*(index-lo)


def workloads(stage: str) -> dict[str, dict]:
    """Exact fixture recipe from collector at 42a000d, not a new workload."""
    require(stage in ('smoke', 'formal'), "invalid stage")
    result = {}
    for kind, lengths, output, concurrency in (
        ('prefill', [256, 512, 1024, 2048], 32, 4),
        ('decode', [128], 128, 8), ('mixed', [64, 2048, 256, 4096], 64, 8)):
        cells = []
        for i in range(16):
            rng = random.Random(936644+i)
            cells.append(dict(id=f'{kind}-{i:02d}',
                              input_ids=[rng.randrange(200, 16000) for _ in range(lengths[i % len(lengths)])],
                              output_tokens=output))
        result[kind] = dict(name=kind, concurrency=concurrency, cells=cells)
    if stage == 'smoke':
        return {'smoke': dict(name='smoke', concurrency=4,
                             cells=[dict(c, id='smoke-'+c['id']) for c in result['mixed']['cells'][:4]])}
    return result


def validate_batch(data: dict, spec: dict) -> dict:
    require(data.get('errors') == [], "request failures retained: not a comparable complete batch")
    require(data.get('workload') == spec['name'] and data.get('concurrency') == spec['concurrency'],
            "workload/concurrency mismatch")
    expected_hash = digest_bytes(json.dumps(spec, sort_keys=True).encode())
    require(data.get('workload_sha256') == expected_hash, "workload binding mismatch")
    elapsed = number(data['elapsed'], 'elapsed', positive=True)
    rows = data['requests']
    expected = {c['id']: c for c in spec['cells']}
    require(type(rows) is list and len(rows) == len(expected), "request coverage")
    seen = set()
    for row in rows:
        rid = row['id']
        require(rid in expected and rid not in seen, "unexpected/duplicate request ID")
        seen.add(rid)
        cell = expected[rid]
        require(type(row['input_tokens']) is int and row['input_tokens'] == len(cell['input_ids']),
                "input length mismatch")
        tokens, times = row['tokens'], row['token_times']
        require(type(tokens) is list and len(tokens) == cell['output_tokens'], "output token count mismatch")
        require(all(type(t) is int and t >= 0 for t in tokens), "invalid token IDs")
        require(type(times) is list and len(times) == len(tokens), "missing token arrival times")
        for t in times:
            number(t, 'arrival')
        require(all(a <= b for a, b in zip(times, times[1:])), "nonmonotonic arrivals")
        start = number(row['request_start'], 'start')
        end = number(row['completion'], 'completion')
        latency = number(row['latency'], 'latency', positive=True)
        require(end <= elapsed+1e-8 and end >= start and times[-1] <= latency+1e-8,
                "request outside measured horizon")
        close(latency, end-start, 'latency')
        close(row['ttft'], times[0], 'ttft')
        close(row['tpot'], (times[-1]-times[0])/(len(times)-1) if len(times) > 1 else 0., 'tpot')
        require(row.get('finish_reason') is not None, "missing finish reason")
        events = row['events']
        require(type(events) is list and bool(events), "missing SSE events")
        reconstructed, count, previous = [], 0, 0.
        for event in events:
            stamp = number(event['received'], 'event time')
            added, total = event['new'], event['count']
            require(type(added) is int and type(total) is int and added >= 0,
                    "invalid SSE event counts")
            require(total == count+added and total <= len(tokens) and previous <= stamp <= latency+1e-8,
                    "SSE accounting mismatch")
            reconstructed.extend([stamp]*added)
            count, previous = total, stamp
        require(count == len(tokens) and reconstructed == times, "SSE/token time mismatch")
    output = sum(len(r['tokens']) for r in rows)
    require(type(data['output_tokens']) is int and data['output_tokens'] == output, "throughput numerator")
    close(data['output_tokens_per_second'], output/elapsed, 'throughput')
    close(data['requests_per_second'], len(rows)/elapsed, 'request throughput')
    good = sum(r['ttft'] <= 2. and r['tpot'] <= .05 for r in rows)
    close(data['strict_slo_goodput'], good/elapsed, 'joint SLO goodput')
    for field, key in [('TTFT', 'ttft'), ('TPOT', 'tpot')]:
        require(set(data[field]) == {'0.5', '0.95', '0.99'}, "quantile coverage")
        for p in (.5, .95, .99):
            close(data[field][str(p)], quantile([r[key] for r in rows], p), field+str(p))
    return {'requests': len(rows), 'output_tokens': output, 'good_requests': good,
            'elapsed_seconds': elapsed, 'goodput': good/elapsed,
            'tokens_by_id': {r['id']: r['tokens'] for r in rows}}


def audit_http_job(path: Path, stage: str) -> dict:
    """Validate a single existing serving-runs/<job>, preserving all failures."""
    require(path.is_dir() and not path.is_symlink(), "unsafe job root")
    require((path/'launcher-complete.txt').is_file(), "launcher incomplete")
    require((path/'exit.txt').read_text().strip() == '0', "nonzero job exit")
    specs = workloads(stage)
    names = {f'{w}-b{b}.json': spec for w, spec in specs.items()
             for b in range(1 if stage == 'smoke' else 3)}
    model = load(path/'model-binding.json')
    weights = sorted(k for k in model if k.startswith('model-') and k.endswith('.safetensors'))
    require(bool(weights) and set(model) == {'config.json', 'model.safetensors.index.json', *weights},
            "model binding file coverage")
    require(weights == [f'model-{i:05d}-of-{len(weights):05d}.safetensors' for i in range(1, len(weights)+1)],
            "missing model shard")
    for entry in model.values():
        require(type(entry['bytes']) is int and entry['bytes'] > 0, "invalid model size")
        require(type(entry['sha256']) is str and len(entry['sha256']) == 64
                and all(c in '0123456789abcdef' for c in entry['sha256']), "invalid model hash")
    arms, environments = {}, {}
    for mode in ('pristine', 'off', 'cap'):
        life = bundle(path/('lifecycle-'+mode), {'qualification.json'})
        require(type(life['checks']) is int and life['checks'] == 32 and life.get('mode') == mode,
                "lifecycle coverage/identity")
        qualification = load(path/('lifecycle-'+mode)/'qualification.json')
        expected_keys = {f'{d}-shared{s}-stream{i}-transition{t}' for d in ('float16', 'bfloat16')
                         for s in (False, True) for i in range(2) for t in range(4)}
        require(len(qualification) == 32 and {q['key'] for q in qualification} == expected_keys,
                "lifecycle record coverage")
        for q in qualification:
            require(q.get('graph_eager_exact') is True and q.get('pristine_exact') is True
                    and number(q['full_max_abs'], 'full_max_abs') == 0, "lifecycle not exact")
            require(type(q['FP32']['vectors']) is int and q['FP32']['vectors'] > 0,
                    "missing FP32 references")
            for k in ('max_abs', 'rmse', 'lse_max_abs'):
                number(q['FP32'][k], 'FP32 '+k)
        modepath = path/mode
        receipt = bundle(modepath, set(names) | {'environment.json'})
        require(receipt.get('stage') == stage and receipt.get('mode') == mode
                and receipt.get('HTTP') is True and receipt.get('full_model') is True,
                "completion identity mismatch")
        env = load(modepath/'environment.json')
        require(env.get('mode') == mode and env.get('stage') == stage
                and env.get('rep') == receipt.get('rep'), "environment identity mismatch")
        require(env.get('HTTP') is True and env.get('full_model') is True
                and env.get('all_model_layers') == 36, "full-model identity mismatch")
        require({p.name for p in modepath.glob('*-b*.json')} == set(names), "workload matrix mismatch")
        arms[mode] = {name: validate_batch(load(modepath/name), spec) for name, spec in names.items()}
        require(env['config_sha256'] == model['config.json']['sha256']
                and env['model_index_sha256'] == model['model.safetensors.index.json']['sha256'],
                "model/environment hash mismatch")
        require(life['header_sha256'] == env['prefill_sha256'], "lifecycle/header mismatch")
        environments[mode] = env
    identity = ('rep', 'flashinfer', 'sglang', 'config_sha256', 'model_index_sha256', 'source_sha256', 'input_API')
    for key in identity:
        require(all(key in env for env in environments.values()), f"missing environment field: {key}")
        require(len({json.dumps(env[key], sort_keys=True) for env in environments.values()}) == 1,
                f"cross-arm environment mismatch: {key}")
    mismatches = []
    for name, baseline in arms['pristine'].items():
        for mode in ('off', 'cap'):
            for rid, tokens in baseline['tokens_by_id'].items():
                if arms[mode][name]['tokens_by_id'][rid] != tokens:
                    mismatches.append({'mode': mode, 'file': name, 'id': rid})
    for arm in arms.values():
        for result in arm.values():
            del result['tokens_by_id']
    return {'schema': 1, 'integrity_pass': True, 'stage': stage, 'token_parity': not mismatches,
            'mismatches': mismatches, 'batches': arms,
            'successful_requests': sum(x['requests'] for arm in arms.values() for x in arm.values()),
            'source_provenance_authenticated': False, 'candidate_launch_attributed': False,
            'serving_promotion': False,
            'limits': ['not a full-model rerun', '64KiB profile counts alone do not prove candidate attribution',
                       'requires model-file/source binding and independent formal statistics',
                       'SSE token arrival timing, not on-device per-token compute time']}


def scan_summary(manifest: dict, summary: dict, representation: str) -> dict:
    """Find exact-feature ranking reversals in a DERIVED summary; never promotes it.

    Cost unit is dimensionless, normalized to pristine in each state. Bounds
    come from nominal ratio intervals; multiplicity and source provenance are
    not validated here. Candidates are hypotheses for fresh GPU confirmation.
    """
    require(summary.get('complete') is True and summary.get('stage') in ('development', 'confirmatory'),
            "incomplete or unsupported summary")
    cases = manifest['cases']
    require(digest_bytes(json.dumps(cases, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
            == manifest['case_hash'] == summary['manifest_hash'], "manifest hash mismatch")
    by_id = {c['id']: c for c in cases}
    require(len(by_id) == len(cases), "duplicate cases")
    groups = {}
    seen = set()
    for cell in summary['cells']:
        case = by_id[cell['case']]
        require(case['family'] == summary['stage'], "stage leakage")
        geo = Geometry(tuple(case['q']), tuple(case['cached']))
        context = json.dumps((summary['gpu'], summary['source_hashes'],
                              summary['hardware'][str(cell['shard'])],
                              cell['dtype'], cell['layout'], cell['split'], cell['calls'], cell['metric']),
                             sort_keys=True)
        identity = (cell['case'], context)
        require(identity not in seen, "duplicate summary cell")
        seen.add(identity)
        comp = cell['comparisons']['cap']
        lo, hi = comp['CI95']
        Interval(lo, hi)
        number(comp['ratio'], 'ratio', positive=True)
        if comp.get('controls_resolve') is not True:
            continue
        key = (context, geo.features(representation))
        groups.setdefault(key, []).append((cell, geo, (Interval(1., 1.), Interval(1./hi, 1./lo))))
    witnesses = []
    for (context, _), entries in groups.items():
        for left, right in itertools.combinations(entries, 2):
            for a, b in ((left, right), (right, left)):
                try:
                    witness = reversal_witness(a[1], b[1], representation, context, context, a[2], b[2])
                except ValueError:
                    continue
                witness.update(left_case=a[0]['case'], right_case=b[0]['case'],
                               units='pristine-normalized cost', evidence='unverified derived-summary candidate')
                witnesses.append(witness)
                break
    return {'representation': representation, 'cells_seen': len(seen), 'candidates': witnesses,
            'candidate_count': len(witnesses), 'measured_gpu_witness': False,
            'no_witness_is_not_sufficiency_proof': True, 'serving_promotion': False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    http = sub.add_parser('http')
    http.add_argument('--job', type=Path, required=True)
    http.add_argument('--stage', choices=['smoke', 'formal'], required=True)
    scan = sub.add_parser('scan')
    scan.add_argument('--manifest', type=Path, required=True)
    scan.add_argument('--summary', type=Path, required=True)
    scan.add_argument('--representation', default='joint_work', choices=['aggregate', 'marginals', 'joint_work', 'unordered_pairs', 'ordered_pairs'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = (audit_http_job(args.job, args.stage) if args.command == 'http' else
                  scan_summary(load(args.manifest), load(args.summary), args.representation))
        code = 0 if result.get('token_parity', True) else 2
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        result = {'integrity_pass': False, 'serving_promotion': False,
                  'error': type(exc).__name__, 'message': str(exc)}
        code = 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
