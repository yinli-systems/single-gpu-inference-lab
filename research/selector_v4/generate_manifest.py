"""Deterministically generate v4 canary and untouched release geometries."""
from __future__ import annotations
from pathlib import Path
import hashlib, json, random

SEED_TEXT = "selector-v4-isolated-kernel-autotune-holdout-2026-09-30"
SEED = int(hashlib.sha256(SEED_TEXT.encode()).hexdigest()[:16], 16)
REGIMES = ("opposite", "tied-opposite", "near-opposite", "same-paired", "mixed", "bimodal")
BATCHES = (5, 6, 7, 8, 9, 10)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def increasing_q(batch: int, rng: random.Random, variant: int) -> list[int]:
    start = rng.choice((2, 3, 5, 17, 33, 65))
    steps = [rng.choice((28, 36, 52, 68, 84, 100)) for _ in range(batch - 1)]
    if variant % 3 == 0 and len(steps) >= 3:
        steps[1] = rng.choice((20, 28, 36)); steps[-1] += rng.choice((32, 64, 96))
    q = [start]
    for step in steps: q.append(q[-1] + step)
    return q


def cache_levels(batch: int, rng: random.Random, variant: int) -> list[int]:
    anchors = [64, 192, 640, 1792, 4352, 8960, 14592, 22016, 28672, 36864]
    chosen = sorted(rng.sample(anchors, batch))
    out = []
    for i, value in enumerate(chosen):
        jitter = 32 * rng.randrange(0, 8)
        out.append(value + jitter + 16 * ((variant + i) % 3))
    return out


def make_case(case_id: str, family: str, regime: str, batch: int,
              rng: random.Random, variant: int) -> dict:
    q = increasing_q(batch, rng, variant)
    ascending = cache_levels(batch, rng, variant)
    descending = list(reversed(ascending))
    if regime == "opposite":
        cached = descending
    elif regime == "tied-opposite":
        cached = descending
        mid = batch // 2
        cached[mid] = cached[mid - 1]
        q[mid] = q[mid - 1]
    elif regime == "near-opposite":
        cached = descending
        i = 1 + (variant % max(1, batch - 2))
        cached[i], cached[i + 1] = cached[i + 1], cached[i]
    elif regime == "same-paired":
        cached = ascending
    elif regime == "mixed":
        cached = ascending[:]
        rng.shuffle(cached)
        if cached in (ascending, descending): cached = cached[1:] + cached[:1]
    elif regime == "bimodal":
        cached = descending
        odds = cached[1::2]; evens = cached[::2]
        cached = odds + evens
    else:
        raise ValueError(regime)
    return {"id": case_id, "family": family, "regime": regime,
            "batch_size": batch, "q": q, "cached": cached,
            "generator_variant": variant}


def build() -> dict:
    rng = random.Random(SEED)
    cases = []
    for i in range(8):
        regime = REGIMES[i % len(REGIMES)]
        batch = BATCHES[(i * 2 + 1) % len(BATCHES)]
        cases.append(make_case(f"canary-v4-{i:02d}", "canary", regime,
                               batch, rng, 1000 + i))
    for i in range(48):
        regime = REGIMES[i % len(REGIMES)]
        batch = BATCHES[(i * 5 + i // len(REGIMES)) % len(BATCHES)]
        cases.append(make_case(f"holdout-v4-{i:02d}", "release", regime,
                               batch, rng, 2000 + i))
    signatures = [(tuple(c["q"]), tuple(c["cached"])) for c in cases]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("duplicate geometry")
    release = [c for c in cases if c["family"] == "release"]
    manifest = {
      "schema": 1, "selector_version": "4.0", "generator_seed_text": SEED_TEXT,
      "generator_seed": SEED, "case_hash": digest(cases),
      "release_hash": digest(release), "primary": "autotuned",
      "dtypes": ["float16", "bfloat16"], "layouts": ["ragged", "paged"],
      "splits": ["auto", "unsplit"], "execution_modes":
          ["eager_full_call", "graph1_replay", "graph16_replay"],
      "tactics": ["native", "cap", "autotuned"],
      "calibration_process_repeats": 3, "calibration_blocks": 8,
      "evaluation_process_repeats": 3, "evaluation_blocks": 8,
      "full_model": False, "default_promotion": False, "cases": cases,
    }
    return manifest


if __name__ == "__main__":
    out = Path(__file__).with_name("manifest.json")
    out.write_text(json.dumps(build(), indent=2) + "\n")
    print(out)
