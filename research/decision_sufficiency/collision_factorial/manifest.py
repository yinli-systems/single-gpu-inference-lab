"""Frozen exact-collision families for the decision-sufficiency factorial assay.

No GPU timing is used to construct or select these states.  Each A/B pair has
identical request count, query multiset, cached-KV multiset, aggregate sums,
exact analytical attention work, pinned-Vidur-style lookup key, and augmented
marginal moments.  Only the request-level q<->cached pairing differs.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

from l20_stack.decision_sufficiency import (
    Geometry,
    aggregate_sums,
    analytical_work,
    marginal_moments,
    paired_multiset,
    vidur_prefill_lookup_key,
)


SCHEMA = 1

# Equal-dot four-request cores found by exhaustive permutation over the declared
# q/k sets.  This table is frozen before any collision-factorial GPU timing.
_CORES = {
    "canary": {
        "q": (48, 96, 192, 384),
        "cached_a": (6144, 1536, 14336, 512),
        "cached_b": (14336, 512, 1536, 6144),
    },
    "formal-01": {
        "q": (48, 96, 192, 384),
        "cached_a": (8192, 0, 16384, 2048),
        "cached_b": (16384, 0, 2048, 8192),
    },
    "formal-02": {
        "q": (32, 80, 176, 352),
        "cached_a": (0, 8192, 16384, 2048),
        "cached_b": (2048, 16384, 0, 8192),
    },
    "formal-03": {
        "q": (64, 112, 240, 496),
        "cached_a": (6144, 1536, 14336, 512),
        "cached_b": (14336, 1536, 512, 6144),
    },
    "formal-04": {
        "q": (48, 128, 224, 448),
        "cached_a": (3072, 12288, 256, 768),
        "cached_b": (12288, 768, 256, 3072),
    },
    "formal-05": {
        "q": (48, 128, 224, 448),
        "cached_a": (4096, 16384, 128, 1024),
        "cached_b": (16384, 1024, 128, 4096),
    },
}

# Anchors are appended identically to A and B.  They increase request count
# without changing the exact feature-collision proof.
_ANCHORS = {
    "canary": ((64, 256), (128, 2048)),  # n=6, qualification-only
    "formal-01": ((80, 512), (160, 4096)),  # n=6
    "formal-02": ((80, 512), (160, 4096)),  # n=6
    "formal-03": (
        (24, 256),
        (40, 768),
        (72, 1536),
        (144, 3072),
        (288, 6144),
        (576, 12288),
    ),  # n=10
    "formal-04": (
        (24, 256),
        (40, 768),
        (72, 1536),
        (144, 3072),
        (288, 6144),
        (576, 12288),
    ),  # n=10
    "formal-05": (
        (24, 0),
        (40, 256),
        (56, 512),
        (72, 768),
        (88, 1536),
        (104, 3072),
        (144, 4096),
        (208, 6144),
        (304, 8192),
        (416, 12288),
    ),  # n=14
}


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _state(pair_id: str, arm: str) -> dict:
    core = _CORES[pair_id]
    if arm not in ("a", "b"):
        raise ValueError("arm must be a or b")
    anchors = _ANCHORS[pair_id]
    q = tuple(core["q"]) + tuple(x[0] for x in anchors)
    cached = tuple(core["cached_" + arm]) + tuple(x[1] for x in anchors)
    return {
        "id": pair_id + "-" + arm,
        "pair_id": pair_id,
        "arm": arm,
        "stage": "canary" if pair_id == "canary" else "formal",
        "q": list(q),
        "cached": list(cached),
    }


def _feature_receipt(case: dict) -> dict:
    g = Geometry(tuple(case["q"]), tuple(case["cached"]))
    return {
        "n": g.n,
        "aggregate_sums": list(aggregate_sums(g)),
        "analytical_work": g.attention_work,
        "vidur64": list(vidur_prefill_lookup_key(g, 64)),
        "marginal_moments": list(marginal_moments(g)),
        "paired_geometry_sha256": digest(list(paired_multiset(g))),
    }


def _validate_pair(a: dict, b: dict) -> dict:
    if a["pair_id"] != b["pair_id"] or a["arm"] != "a" or b["arm"] != "b":
        raise ValueError("misaligned pair")
    ga = Geometry(tuple(a["q"]), tuple(a["cached"]))
    gb = Geometry(tuple(b["q"]), tuple(b["cached"]))

    equalities = {
        "n": ga.n == gb.n,
        "query_multiset": sorted(ga.query) == sorted(gb.query),
        "cached_multiset": sorted(ga.cached) == sorted(gb.cached),
        "aggregate_sums": aggregate_sums(ga) == aggregate_sums(gb),
        "analytical_work": analytical_work(ga) == analytical_work(gb),
        "vidur64": vidur_prefill_lookup_key(ga, 64)
        == vidur_prefill_lookup_key(gb, 64),
        "marginal_moments": marginal_moments(ga) == marginal_moments(gb),
    }
    if not all(equalities.values()):
        raise AssertionError("declared exact collision is invalid: " + str(equalities))
    if paired_multiset(ga) == paired_multiset(gb):
        raise AssertionError("pairing was not changed")

    return {
        "pair_id": a["pair_id"],
        "equalities": equalities,
        "pairing_differs": True,
        "feature_a": _feature_receipt(a),
        "feature_b": _feature_receipt(b),
    }


def build() -> dict:
    cases = []
    pairs = []
    for pair_id in _CORES:
        a = _state(pair_id, "a")
        b = _state(pair_id, "b")
        pairs.append(_validate_pair(a, b))
        cases.extend((a, b))

    ids = [x["id"] for x in cases]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate state id")
    geometry_hashes = [digest((x["q"], x["cached"])) for x in cases]
    if len(geometry_hashes) != len(set(geometry_hashes)):
        raise AssertionError("duplicate geometry")

    return {
        "schema": SCHEMA,
        "kind": "exact feature-collision factorial assay",
        "construction": (
            "CPU-only equal-dot cores plus identical anchors; frozen before new GPU timing"
        ),
        "primary_pair_ids": [
            "formal-01",
            "formal-02",
            "formal-03",
            "formal-04",
            "formal-05",
        ],
        "canary_pair_ids": ["canary"],
        "resource_modes": ["pristine", "cap"],
        "order_actions": ["identity", "heavy_first"],
        "control_action": "identity_repeat",
        "dtypes": ["float16", "bfloat16"],
        "split_modes": ["auto", "unsplit"],
        "execution_modes": ["eager_one", "graph16"],
        "blocks": 12,
        "decision_margin": 0.01,
        "cases": cases,
        "pair_proofs": pairs,
        "case_hash": digest(cases),
    }


def cases_for(stage: str) -> list[dict]:
    if stage not in ("canary", "formal"):
        raise ValueError("stage must be canary or formal")
    return [x for x in build()["cases"] if x["stage"] == stage]


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, allow_nan=False))
