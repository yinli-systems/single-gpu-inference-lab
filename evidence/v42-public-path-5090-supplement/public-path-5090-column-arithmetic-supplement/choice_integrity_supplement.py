"""Original choice bytes and strict gates across float64 arithmetic paths.

Only four derived floating fields permit four IEEE-754 adjacent values.
Input digests, thresholds, checks, winner, seed and artifacts remain exact.
This comparison tolerance never changes any qualification threshold.
"""

import math
import struct

from research.selector_v4.public_qualification.contract import digest

DERIVED = frozenset(("geomean", "gain_ci95", "control_ci90", "block_minimum"))


def _derived_equal(actual, expected):
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_derived_equal(a, e) for a, e in zip(actual, expected))
        )
    if not (
        type(actual) is float
        and type(expected) is float
        and math.isfinite(actual)
        and math.isfinite(expected)
        and actual > 0
        and expected > 0
    ):
        return False
    bits = lambda x: struct.unpack(">Q", struct.pack(">d", x))[0]
    return abs(bits(actual) - bits(expected)) <= 4


def verify_choice_record(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("Complete original choice fields required")
    original = {k: v for k, v in actual.items() if k != "checksum"}
    if actual["checksum"] != digest(original):
        raise ValueError("Original serialized choice checksum changed")
    changed = []
    for key, value in expected.items():
        if key == "checksum":
            continue
        if key in DERIVED:
            if not _derived_equal(actual[key], value):
                raise ValueError("Original derived choice differs beyond four ULP: " + key)
            if actual[key] != value:
                changed.append(key)
        elif actual[key] != value:
            raise ValueError("Strict original choice provenance/decision differs: " + key)
    return {"original_checksum_verified": True, "roundoff_fields": changed, "maximum_ulp": 4}
