"""Independent request/step-evidence audit; never modifies measured workloads.

No torch/vLLM imports, no fabricated timing, no fitting, no GPU submission.
A validation failure is retained, not omitted from a better-looking subset.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ARMS = ('auto', 'fixed_budget', 'native_cycle')
METRIC_FIELDS = ('ttft_seconds', 'offered_ttft_seconds', 'tpot_seconds',
                 'request_latency_seconds', 'offered_latency_seconds')


def finite(value: Any, label: str, minimum: float = 0.) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{label}: numeric value required')
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f'{label}: invalid value')
    return float(value)


def close(actual: Any, expected: float, label: str) -> None:
    actual = finite(actual, label)
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError(f'{label}: raw evidence does not reproduce summary')


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError('quantiles require observations')
    xs = sorted(finite(x, 'quantile input') for x in values)
    result = {}
    for name, q in (('p50', .5), ('p95', .95), ('p99', .99)):
        pos = q * (len(xs) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        result[name] = xs[lo] + (pos - lo) * (xs[hi] - xs[lo])
    return result


def audit_workload(payload: dict, steps: list[dict], cells: list[dict]) -> dict:
    if payload.get('complete') is not True:
        raise ValueError('workload is incomplete')
    wall = finite(payload['elapsed_seconds'], 'elapsed', 1e-15)
    expected = {c['id']: c for c in cells}
    records = {r['id']: r for r in payload['requests_detail']}
    if not cells or len(expected) != len(cells) or len(records) != len(payload['requests_detail']):
        raise ValueError('empty or duplicate request IDs')
    if set(records) != set(expected):
        raise ValueError('planned request coverage mismatch')
    whash = hashlib.sha256(json.dumps(cells, sort_keys=True).encode()).hexdigest()
    if payload['workload_sha256'] != whash:
        raise ValueError('workload definition changed')
    reconstructed = defaultdict(list)
    finished = set()
    positive_events = defaultdict(list)
    previous_end = 0.
    for step in steps:
        start = finite(step['started'], 'step start')
        end = finite(step['completed'], 'step end')
        if start + 1e-9 < previous_end or end < start or end > wall + 1e-9:
            raise ValueError('step clock order/interval violated')
        previous_end = end
        seen = set()
        for event in step['outputs']:
            rid = event['id']
            if rid not in expected or rid in seen or rid in finished:
                raise ValueError('unknown, duplicate or post-finish output')
            seen.add(rid)
            count = event['added']
            if type(count) is not int or count < 0 or count > expected[rid]['max_tokens']:
                raise ValueError('invalid number of newly observed tokens')
            if type(event['finished']) is not bool:
                raise ValueError('finished flag is not Boolean')
            if end < finite(records[rid]['admission'], 'admission'):
                raise ValueError('output precedes admission')
            reconstructed[rid].extend([end] * count)
            if count:
                positive_events[rid].append(end)
            if event['finished']:
                finished.add(rid)
    if finished != set(expected):
        raise ValueError('step trace did not finish all requests')
    metrics = {key: [] for key in METRIC_FIELDS}
    strict = lenient = total = 0
    bundles = 0
    for rid, record in records.items():
        cell = expected[rid]
        tokens = record['token_ids']
        times = record['token_times']
        n = cell['max_tokens']
        if record['finished'] is not True or len(tokens) != n or len(times) != n:
            raise ValueError('incomplete token payload')
        if any(type(x) is not int or x < 0 for x in tokens):
            raise ValueError('invalid generated token ID')
        if record['expected_output_tokens'] != n or record['prompt_tokens'] != len(cell['prompt']):
            raise ValueError('request lengths disagree with plan')
        if len(reconstructed[rid]) != n:
            raise ValueError('step trace token count differs')
        for a, b in zip(times, reconstructed[rid]):
            close(a, b, 'token observation timestamp')
        admission = finite(record['admission'], 'admission')
        arrival = finite(cell['arrival'], 'offered arrival')
        if admission + 1e-9 < arrival:
            raise ValueError('request admitted before scheduled arrival')
        close(record['scheduled_arrival'], arrival, 'scheduled arrival')
        values = dict(ttft=times[0]-admission, offered_ttft=times[0]-arrival,
                      latency=times[-1]-admission, offered_latency=times[-1]-arrival,
                      tpot=(times[-1]-times[0])/(n-1) if n > 1 else 0.)
        for key, val in values.items():
            close(record[key], val, key)
        smet = values['offered_ttft'] <= 2. and values['tpot'] <= .05
        lmet = values['offered_ttft'] <= 5. and values['tpot'] <= .1
        if record['strict_slo'] is not smet or record['lenient_slo'] is not lmet:
            raise ValueError('SLO classification disagrees with observation times')
        strict += int(smet)
        lenient += int(lmet)
        total += n
        bundles += n - len(positive_events[rid])
        for key, val in zip(METRIC_FIELDS, (values['ttft'], values['offered_ttft'],
                            values['tpot'], values['latency'], values['offered_latency'])):
            metrics[key].append(val)
    computed = dict(requests=len(cells), output_tokens=total, elapsed_seconds=wall,
                    tokens_per_second=total/wall, requests_per_second=len(cells)/wall,
                    strict_slo_count=strict, lenient_slo_count=lenient,
                    strict_slo_goodput=strict/wall, lenient_slo_goodput=lenient/wall)
    for key, value in computed.items():
        close(payload[key], value, key)
    for key, values in metrics.items():
        computed[key] = percentiles(values)
        for q, value in computed[key].items():
            close(payload[key][q], value, key+'/'+q)
    if payload['router']['counts'].get('qualified_plans', 0) <= 0:
        raise ValueError('no qualified prefill planning actually executed')
    return dict(passed=True, requests=len(cells), tokens=total, steps=len(steps),
                metrics=computed, bundled_extra_tokens=bundles,
                note='TPOT is per-request mean; no fabricated per-token ITL samples.')


def audit_job(path: Path) -> dict:
    from engine_screen import workload
    result = dict(path=str(path), work={}, raw_sha256={}, parity={}, errors=[])
    complete = path/'complete.json'
    if not complete.exists():
        raise ValueError('no complete three-arm job marker')
    marker = json.loads(complete.read_text())
    if marker['all_arms_complete'] is not True:
        raise ValueError('not every model process completed')
    signatures = set()
    configs = set()
    capacities = []
    measured = {}
    for arm in ARMS:
        summary_path = path/arm/'summary.json'
        summary = json.loads(summary_path.read_text())
        if summary.get('complete') is not True or summary['arm'] != arm:
            raise ValueError('invalid arm summary')
        if (summary['vllm'], summary['flashinfer'], summary['dtype'], summary['eager'],
            summary['requested_KV_blocks'], summary['page_size'], summary['scratch_MiB']) != (
                '0.29.0', '0.6.18', 'float16', True, 2048, 16, 512):
            raise ValueError('engine, precision or resource configuration changed')
        signatures.add(json.dumps({Path(k).name: v for k,v in summary['source_sha256'].items()}, sort_keys=True))
        configs.add(summary['model_config_sha256'])
        capacities.append(summary['actual_config_KV_blocks'])
        for phase in ('warmup','measured'):
            for name in ('burst','staggered'):
                raw = path/arm/f'{phase}-{name}.json'
                trace = path/arm/f'{phase}-{name}-steps.jsonl'
                payload = json.loads(raw.read_text())
                if payload['phase'] != phase or payload['workload'] != name:
                    raise ValueError('mislabelled warmup/measurement')
                steps = [json.loads(line) for line in trace.read_text().splitlines()]
                key = arm+'/'+phase+'/'+name
                result['work'][key] = audit_workload(payload, steps, workload(name))
                for f in (raw, trace, summary_path):
                    result['raw_sha256'][str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
                if phase == 'measured':
                    measured[arm,name] = {r['id']:r['token_ids'] for r in payload['requests_detail']}
    if len(signatures) != 1 or len(configs) != 1:
        raise ValueError('model or source differs between methods')
    for name in ('burst','staggered'):
        result['parity'][name] = {arm:measured[arm,name]==measured['auto',name] for arm in ARMS}
    same = all(v for w in result['parity'].values() for v in w.values())
    if same != marker['greedy_outputs_identical']:
        raise ValueError('token evidence disagrees with job parity marker')
    result.update(evidence_audit_passed=True, greedy_outputs_identical=same,
                  actual_KV_blocks=capacities, capacity_verified=all(x==2048 for x in capacities),
                  performance_promoted=False, independent_GPU_repeat_claim=False)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('preserve existing audit')
    report = audit_job(a.job)
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('work','raw_sha256')}))
