"""Experimental, CPU-only order proposals and empirical admission gates.

No GPU-memory mutation or kernel execution occurs here. Logical pair counts and
locality proxies are not GPU latency models; admission is not a runtime guarantee.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
import statistics
from typing import Sequence

MAX_DESCRIPTORS = 131072
PROPOSALS = ('identity', 'causal_heavy', 'locality_packet8')


def _int(value, label, minimum=1, maximum=2**31-1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('invalid ' + label)
    return value


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Geometry:
    query: tuple[int, ...]
    total_kv: tuple[int, ...]
    group: int
    tile: int
    chunk: int
    split: bool

    def __post_init__(self):
        if type(self.query) is not tuple or type(self.total_kv) is not tuple:
            raise ValueError('immutable geometry required')
        if not self.query or len(self.query) != len(self.total_kv):
            raise ValueError('geometry length mismatch')
        _int(self.group, 'group', maximum=256)
        _int(self.tile, 'tile', maximum=1024)
        if type(self.split) is not bool:
            raise ValueError('split must be bool')
        _int(self.chunk, 'chunk', minimum=1 if self.split else -1)
        count = 0
        for q, k in zip(self.query, self.total_kv):
            _int(q, 'query'); _int(k, 'total_kv')
            if k < q:
                raise ValueError('KV length shorter than causal query')
            count += (q*self.group+self.tile-1)//self.tile * (
                (k+self.chunk-1)//self.chunk if self.split else 1)
            if count > MAX_DESCRIPTORS:
                raise ValueError('descriptor allocation limit')

    def descriptors(self) -> tuple[tuple[int, int, int], ...]:
        return tuple((r, t, s) for r, (q, k) in enumerate(zip(self.query, self.total_kv))
                     for t in range((q*self.group+self.tile-1)//self.tile)
                     for s in range((k+self.chunk-1)//self.chunk if self.split else 1))


def _positive_sum(n: int, a: int) -> int:
    """Sum max(a+j, 0) for j=0..n-1 using integer arithmetic."""
    first = max(0, 1-a)
    count = max(0, n-first)
    return count*(2*a+first+n-1)//2


def causal_pairs(g: Geometry, desc: tuple[int, int, int]) -> int:
    """Exact LOGICAL causal-pair count for contiguous GQA-packed query rows.

Not padded tensor-core FLOPs, memory traffic, issue order or elapsed time.
"""
    if len(desc) != 3 or any(type(v) is not int or v < 0 for v in desc):
        raise ValueError('invalid descriptor')
    r, tile_id, split_id = desc
    if r >= len(g.query):
        raise ValueError('request index out of range')
    q, k = g.query[r], g.total_kv[r]
    left, right = tile_id*g.tile, min((tile_id+1)*g.tile, q*g.group)
    start = split_id*g.chunk if g.split else 0
    end = min((split_id+1)*g.chunk, k) if g.split else k
    if left >= right or start >= end or (not g.split and split_id):
        raise ValueError('tile index out of range')
    a, cap = k-q+1-start, end-start
    def prefix(n):
        full, rem = divmod(n, g.group)
        return g.group*(_positive_sum(full, a)-_positive_sum(full, a-cap)) + \
            rem*min(cap, max(0, a+full))
    return prefix(right)-prefix(left)


def _validate(desc, g):
    expected = g.descriptors()
    if len(desc) != len(expected) or any(len(d) != 3 or any(type(v) is not int for v in d) for d in desc):
        raise ValueError('invalid descriptor multiset')
    if Counter(map(tuple, desc)) != Counter(expected):
        raise ValueError('not a complete descriptor bijection')


def transitions(desc, order):
    keys = [(desc[i][0], desc[i][2]) for i in order]
    return sum(a != b for a, b in zip(keys, keys[1:]))


def propose(g: Geometry, desc: Sequence[tuple[int, int, int]], policy: str) -> tuple[int, ...]:
    """Return indices into the ORIGINAL descriptor sequence. No live-plan cache.

locality_packet8 preserves native order inside 8-descriptor packets, fixes the
first packet, limits subsequent movement to 32-descriptor windows, and rejects
any proposal increasing request/KV-split boundary crossings. These are structural
constraints, not a claim that real cache misses or elapsed time cannot increase.
"""
    _validate(desc, g)
    identity = tuple(range(len(desc)))
    if policy not in PROPOSALS:
        raise ValueError('unknown proposal')
    if policy == 'identity':
        return identity
    work = [causal_pairs(g, tuple(d)) for d in desc]
    if policy == 'causal_heavy':
        return tuple(sorted(identity, key=lambda i: (-work[i], i)))
    packet, window = 8, 32
    order = list(identity[:packet])
    for start in range(packet, len(identity), window):
        chunks = [list(identity[i:min(i+packet, start+window)])
                  for i in range(start, min(start+window, len(identity)), packet)]
        chunks.sort(key=lambda xs: (-Fraction(sum(work[i] for i in xs), len(xs)), xs[0]))
        order.extend(i for xs in chunks for i in xs)
    if sorted(order) != list(identity):
        raise RuntimeError('internal permutation violation')
    if transitions(desc, order) > transitions(desc, identity):
        return identity
    return tuple(order)


@dataclass(frozen=True)
class Context:
    """Exact execution identity; source/geometry changes invalidate admission."""
    gpu: str
    driver: str
    torch: str
    cuda: str
    backend_sha256: str
    policy_sha256: str
    geometry_sha256: str
    dtype: str
    mode: str
    cache_regime: str
    timing_scope: str
    calls_per_plan: int

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == 'calls_per_plan':
                _int(value, key)
            elif type(value) is not str or not value:
                raise ValueError('missing context ' + key)
        if self.mode not in ('eager', 'graph'):
            raise ValueError('unsupported execution mode')
        if self.timing_scope not in ('run_only', 'plan_plus_run'):
            raise ValueError('unsupported timing boundary')
        for value in (self.backend_sha256, self.policy_sha256, self.geometry_sha256):
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('SHA256 identity required')

    @property
    def key(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class Probe:
    trial_id: str
    context_key: str
    policy: str
    baseline_us: float
    candidate_us: float
    exact_output: bool
    exact_lse: bool

    def __post_init__(self):
        if not self.trial_id or not self.context_key or not self.policy:
            raise ValueError('probe identity missing')
        for value in (self.baseline_us, self.candidate_us):
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError('invalid timing')
        if self.exact_output is not True or self.exact_lse is not True:
            raise ValueError('unqualified numerical result')


def select(context: Context, probes: Sequence[Probe]) -> str:
    """Choose once on selection data; never try a runner-up on confirmation."""
    by_policy = {}; ids = set()
    for p in probes:
        if p.context_key != context.key or p.trial_id in ids:
            raise ValueError('mixed context or duplicate selection sample')
        if p.policy == 'identity':
            raise ValueError('identity should be the paired reference')
        ids.add(p.trial_id)
        by_policy.setdefault(p.policy, []).append(p.baseline_us/p.candidate_us)
    eligible = {name: values for name, values in by_policy.items() if len(values) >= 6}
    if not eligible:
        return 'identity'
    name = max(sorted(eligible), key=lambda p: statistics.median(eligible[p]))
    return name if statistics.median(eligible[name]) > 1.01 else 'identity'


def admit(context: Context, selected: str, selection_ids: set[str],
          confirmation: Sequence[Probe], aa: Sequence[Probe], *,
          probe_cost_us: float, extra_setup_us: float, dispatch_us: float,
          expected_reuses: int | None) -> dict:
    """Conservative empirical screen, NOT a latency or probability guarantee.

Exactly six independent confirmation pairs, no optional stopping, all >1% gains;
A/A must be inside a fixed observed envelope. Reject if the observed gain cannot
repay probing/setup/dispatch within the supplied reuse budget. Unknown reuse
means identity. Future timing may still regress. Calls-per-plan is keyed separately.
"""
    _int(len(selection_ids), 'selection sample count')
    for value in (probe_cost_us, extra_setup_us, dispatch_us):
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid charged cost')
    result = dict(policy='identity', reason='not_admitted', production_qualified=False)
    if selected == 'identity':
        return dict(result, reason='no_selection_gain')
    if expected_reuses is None:
        return dict(result, reason='unknown_reuse')
    _int(expected_reuses, 'expected_reuses')
    if len(confirmation) != 6 or len(aa) != 6:
        return dict(result, reason='six_confirmation_and_AA_pairs_required')
    all_ids = set(selection_ids)
    for group in (confirmation, aa):
        for p in group:
            if p.trial_id in all_ids:
                raise ValueError('selection/confirmation/A-A sample leakage')
            all_ids.add(p.trial_id)
            if p.context_key != context.key:
                return dict(result, reason='context_mismatch')
    if any(p.policy != selected for p in confirmation) or any(p.policy != 'identity_repeat' for p in aa):
        raise ValueError('wrong confirmation policy')
    if any(abs(p.baseline_us-p.candidate_us) > max(3., .02*p.baseline_us) for p in aa):
        return dict(result, reason='AA_observed_envelope_failed')
    if any(p.baseline_us/p.candidate_us <= 1.01 for p in confirmation):
        return dict(result, reason='confirmation_gain_failed')
    gain = min(p.baseline_us-p.candidate_us for p in confirmation)-dispatch_us
    if gain <= 0:
        return dict(result, reason='dispatch_exceeds_gain')
    breakeven = max(1, math.ceil((probe_cost_us+extra_setup_us)/gain))
    if breakeven > expected_reuses:
        return dict(result, reason='insufficient_reuse', break_even_reuses=breakeven)
    return dict(result, policy=selected, reason='empirically_admitted',
                min_observed_net_gain_us=gain, break_even_reuses=breakeven,
                sign_test_p_under_independent_median_null=1/64,
                caveat='six signs are not a worst-case or future-runtime guarantee')
