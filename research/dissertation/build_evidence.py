"""Freeze and verify the dissertation's versioned, scoped evidence.

No benchmark or promotion state is changed. This records source bytes and JSON
pointers, then derives presentation units without changing stored estimators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(document, path):
    current = document
    for part in path.strip('/').split('/'):
        current = current[part.replace('~1', '/').replace('~0', '~')]
    return current


def freeze():
    sources = {
        'authority': 'research/selector_v4/STATUS.json',
        'static4090': 'research/selector_v32/release_v322/evidence/final/4090-summary.json',
        'static5090': 'research/selector_v32/release_v322/evidence/final/5090-summary.json',
        'v4114090': 'research/selector_v4/evidence/v411-exact-dev/analysis/gpu_4090-summary.json',
        'v4115090': 'research/selector_v4/evidence/v411-exact-dev/analysis/gpu_5090-summary.json',
        'dev4090': 'research/selector_v4/evidence/v42-exact-dev/gpu_4090-summary.json',
        'dev5090': 'research/selector_v4/evidence/v42-exact-dev/gpu_5090-summary.json',
        'public5090': 'evidence/v42-public-path-5090-supplement/public-path-5090-column-arithmetic-supplement/analysis/summary.json',
        'intervention': 'research/selector_v4/evidence/v42-attribute-launch-dual-controls/receipt.json',
        'history': 'research/parity_remediation/evidence/forced-history-v3/summary.json',
    }
    inputs = HERE / 'inputs'
    inputs.mkdir(exist_ok=True)
    index = {}
    docs = {}
    for key, original in sources.items():
        src = ROOT / original
        dst = inputs / (key + '.json')
        dst.write_bytes(src.read_bytes())
        index[key] = {'original_path': original, 'snapshot_path': str(dst.relative_to(HERE)),
                      'sha256': digest(dst), 'bytes': dst.stat().st_size}
        docs[key] = json.loads(dst.read_text())
    facts = []

    def add(identifier, source, path, scope, unit='ratio', transform=None):
        raw = pointer(docs[source], path)
        value = 100 * raw if transform == 'fraction_to_percent' else raw
        facts.append({'id': identifier, 'source': source, 'json_pointer': path,
                      'scope': scope, 'unit': unit, 'raw_value': raw,
                      'transform': transform, 'value': value})

    for key in ('default_promotion', 'serving_promotion', 'historical_token_divergence_resolved'):
        add(key, 'authority', '/' + key, 'Authoritative old v4.2 qualification state', 'boolean')
    add('old_canary_state', 'authority', '/v42/canary_state', 'Old consumed ten-case canary', 'status')
    for gpu in ('4090', '5090'):
        src = 'static' + gpu
        scope = 'v3.2.2 fresh release; 30 geometries; 720 cells/card; selected subset'
        for key in ('count', 'ratio', 'worst_point_ratio', 'simultaneous_worst_CI95', 'controls_failed'):
            add('static_' + gpu + '_' + key, src, '/release_gate/paired_selected/' + key,
                scope, 'fold/cell count' if key in ('count', 'controls_failed') else 'ratio')
        add('static_' + gpu + '_pass', src, '/single_gpu_release_gate_pass', scope, 'boolean')
        for family in ('v411', 'dev'):
            src = family + gpu
            scope = ('v4.1.1' if family == 'v411' else 'v4.2 b769e7c') + ' exposed exact development; two geometries'
            for key in ('selected_records', 'fold_records', 'selected_geomean', 'selected_joint_min_lcb95',
                        'native_overlay_joint_min_lcb95'):
                add(family + '_' + gpu + '_' + key, src, '/metrics/' + key, scope,
                    'fold records' if key.endswith('records') else 'ratio')
            add(family + '_' + gpu + '_pass', src, '/pass', scope, 'boolean')
        scope = 'Old v4.2 b769e7c ten-case consumed canary; qualification HOLD'
        for key in ('selected_records', 'fold_records', 'selected_geomean', 'selected_joint_min_lcb95',
                    'native_overlay_joint_min_lcb95', 'actual_policy_vs_pristine_worst'):
            add('canary_' + gpu + '_' + key, 'authority', '/v42/canary_' + gpu + '/metrics/' + key,
                scope, 'fold records' if key.endswith('records') else 'ratio')
        add('canary_' + gpu + '_failures', 'authority', '/v42/canary_' + gpu + '/failed_requirements', scope, 'gate names')
        for layout in ('paged1_cached_prefix', 'ragged_projection_suffix'):
            for arm in ('native', 'attribute49152_launch49152', 'attribute65536_launch49152', 'attribute65536_launch65536'):
                base = '/results/gpu_' + gpu + '/layouts/' + layout + '/cupti/' + arm
                scope = '9f7b80a diagnostic; fixed exposed BF16; 16 CUPTI launches/arm/layout/card'
                add('intervention_' + gpu + '_' + layout + '_' + arm, 'intervention', base + '/median_kernel_us', scope, 'microseconds')
    for key in docs['public5090']['metrics']:
        if key == 'regret':
            continue
        add('public5090_' + key, 'public5090', '/metrics/' + key,
            'c6879fb original complete RTX 5090 only; separate frozen arithmetic/CSV-column supplement; exposed two cases',
            'fold records' if key in ('records', 'selected') else 'ratio')
    for key in ('p50', 'p90', 'p99', 'worst'):
        add('public5090_regret_' + key + '_percent', 'public5090', '/metrics/regret/' + key,
            'Available certified held-out oracle; oracle never trains; RTX 5090 exposed supplement only',
            'percent', 'fraction_to_percent')
    for key in ('original_failure_reproduced', 'first_state_attributed', 'diagnostic_only', 'performance_claim'):
        add('history_' + key, 'history', '/' + key, 'Forced-history v3 job1640986; two target conditions; dependent pairs', 'boolean')
    ledger = {'schema': 1, 'evidence_cutoff': '2026-10-02', 'qualification': 'HOLD',
              'sources': index, 'facts': facts,
              'warnings': ['Legacy v4.2 dev regret uses oracle/chosen-1 and is excluded from regret comparisons.',
                           'Selected subsets and versioned populations are not a paired improvement trend.',
                           'Frozen report inputs include a dated authority snapshot; live state may change.',
                           'Source-byte verification authenticates inputs, not physical or statistical truth.']}
    (HERE / 'claim_ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    verify()


def verify():
    path = HERE / 'claim_ledger.json'
    ledger = json.loads(path.read_text())
    docs = {}
    for key, item in ledger['sources'].items():
        src = HERE / item['snapshot_path']
        assert digest(src) == item['sha256'], ('source_hash', key)
        assert src.stat().st_size == item['bytes'], ('source_bytes', key)
        docs[key] = json.loads(src.read_text())
    values = {}
    for fact in ledger['facts']:
        raw = pointer(docs[fact['source']], fact['json_pointer'])
        assert raw == fact['raw_value'], ('raw_pointer', fact['id'])
        value = 100 * raw if fact['transform'] == 'fraction_to_percent' else raw
        assert value == fact['value'], ('unit_conversion', fact['id'])
        assert fact['id'] not in values, ('duplicate_id', fact['id'])
        values[fact['id']] = value
    assert ledger['qualification'] == 'HOLD'
    for key in ('default_promotion', 'serving_promotion', 'historical_token_divergence_resolved'):
        assert values[key] is False, key
    assert values['old_canary_state'] == 'QUALIFICATION_HOLD'
    assert values['static_4090_pass'] is False and values['static_5090_pass'] is False
    assert values['dev_4090_pass'] is True and values['dev_5090_pass'] is True
    assert values['history_original_failure_reproduced'] is False
    assert values['history_first_state_attributed'] is False
    assert values['history_diagnostic_only'] is True
    assert values['history_performance_claim'] is False
    expected = 0.042468246096653167
    assert abs(values['public5090_regret_p99_percent'] - expected) < 1e-15
    report = {'pass': True, 'ledger_sha256': digest(path), 'source_count': len(docs),
              'fact_count': len(values), 'qualification': 'HOLD',
              'default_promotion': False, 'serving_promotion': False,
              'historical_token_divergence_resolved': False,
              'verification_scope': 'Frozen source bytes, JSON pointers, exact derived units and scoped state invariants; no new GPU evidence.'}
    (HERE / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return ledger, values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true', help='Explicitly replace the dated report snapshot.')
    arguments = parser.parse_args()
    freeze() if arguments.freeze else verify()
