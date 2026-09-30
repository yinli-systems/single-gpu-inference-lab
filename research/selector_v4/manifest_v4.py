from __future__ import annotations
import hashlib, json, math, random
from pathlib import Path
from typing import Any, Iterable

SCHEMA = 4
GENERATOR_SEED = 73194721

def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _tiles(total: int, batch: int, rng: random.Random) -> list[int]:
    if batch < 4 or total < batch:
        raise ValueError("invalid descriptor allocation")
    weights = [(i + 1) ** 1.35 for i in range(batch)]
    extras = total - batch
    raw = [extras * w / sum(weights) for w in weights]
    values = [1 + int(x) for x in raw]
    remainder = total - sum(values)
    order = sorted(range(batch), key=lambda i: (raw[i] - int(raw[i]), rng.random()), reverse=True)
    for i in order[:remainder]:
        values[i] += 1
    if sum(values) != total:
        raise AssertionError("descriptor allocation drift")
    return values

def _queries(total: int, batch: int, rng: random.Random) -> list[int]:
    tiles = _tiles(total, batch, rng)
    values = [32 * (tile - 1) + rng.randint(2, 30) for tile in tiles]
    return sorted(values)

def _cached(batch: int, maximum: int, rng: random.Random) -> list[int]:
    if maximum < 1024:
        raise ValueError("invalid cached maximum")
    minimum = 64
    if batch == 1:
        return [maximum]
    ratio = (minimum / maximum) ** (1.0 / (batch - 1))
    values = [max(minimum, int(round(maximum * ratio**i / 64)) * 64) for i in range(batch)]
    values[0], values[-1] = maximum, minimum
    for i in range(1, len(values) - 1):
        lower = minimum + (len(values) - 1 - i) * 64
        upper = values[i - 1] - 64
        jittered = values[i] + rng.randint(-7, 7) * 64
        values[i] = max(lower, min(upper, jittered))
    values[-1] = minimum
    if any(values[i] <= values[i + 1] for i in range(len(values) - 1)):
        raise AssertionError("cached prefix ordering lost")
    return values

def _regime(values: list[int], regime: str, rng: random.Random) -> list[int]:
    out = list(values)
    if regime == "opposite":
        return out
    if regime == "near-opposite":
        i = max(1, len(out) // 3)
        out[i], out[i + 1] = out[i + 1], out[i]
        return out
    if regime == "tied-opposite":
        i = len(out) // 2
        out[i] = out[max(0, i - 1)]
        return out
    if regime == "same-paired":
        return list(reversed(out))
    if regime == "mixed":
        rng.shuffle(out)
        return out
    raise ValueError(regime)

def _historical(repo: Path, exclude: Path | None = None) -> tuple[set[tuple[tuple[int, ...], tuple[int, ...]]], set[tuple[int, ...]], set[tuple[int, ...]], dict[str, str]]:
    keys: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    q_vectors: set[tuple[int, ...]] = set()
    cached_vectors: set[tuple[int, ...]] = set()
    hashes: dict[str, str] = {}
    excluded = None if exclude is None else exclude.resolve()
    self_manifest = (repo / "research/selector_v4/manifest.json").resolve()
    for path in sorted(repo.glob("research/**/manifest.json")):
        if path.resolve() == self_manifest or (excluded is not None and path.resolve() == excluded):
            continue
        try:
            raw = path.read_bytes(); data = json.loads(raw)
        except Exception:
            continue
        hashes[str(path.relative_to(repo))] = hashlib.sha256(raw).hexdigest()
        for case in data.get("cases", []):
            if "q" in case and "cached" in case:
                q = tuple(map(int, case["q"])); cached = tuple(map(int, case["cached"]))
                keys.add((q, cached)); q_vectors.add(q); cached_vectors.add(cached)
    return keys, q_vectors, cached_vectors, hashes

def _make_case(case_id: str, family: str, index: int, *, batch: int, descriptors: int,
               maximum: int, regime: str, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    q = _queries(descriptors, batch, rng)
    cached = _regime(_cached(batch, maximum, rng), regime, rng)
    actual = sum((x + 31) // 32 for x in q)
    if actual != descriptors:
        raise AssertionError("descriptor count mismatch")
    return {
        "id": case_id, "family": family, "index": index, "regime": regime,
        "batch_size": batch, "descriptor_count": descriptors,
        "q": q, "cached": cached, "max_cached": max(cached),
        "shape_eligible": batch >= 5 and max(cached) >= 8192,
        "generator_seed": seed,
    }

def _specs(family: str, count: int, offset: int) -> Iterable[dict[str, Any]]:
    batches = [5, 6, 7, 8, 10, 12]
    descriptors = [28, 32, 36, 40, 44, 48, 52, 56, 60, 66, 72, 80]
    maxima = [12288, 16384, 24576, 32768, 49152, 65536]
    regimes = ["opposite", "near-opposite", "tied-opposite", "mixed", "same-paired"]
    for i in range(count):
        batch = batches[(i * 5 + offset) % len(batches)]
        desc = descriptors[(i * 7 + offset * 3) % len(descriptors)]
        desc = max(desc, batch + 4)
        maximum = maxima[(i * 3 + offset) % len(maxima)]
        regime = regimes[(i * 2 + offset) % len(regimes)]
        if family == "canary" and i in (0, 6):
            maximum = 6144
        if family == "canary" and i == 1:
            batch = 4; desc = max(desc, 24)
        if family == "release" and i % 13 == 0:
            maximum = 6144
        if family == "release" and i in (17, 41):
            batch = 4; desc = max(desc, 28)
        if family == "stress":
            batch = [9, 11, 12][i % 3]
            maximum = [49152, 65536, 98304][i % 3]
            desc = [64, 76, 88, 96][i % 4]
        yield dict(batch=batch, descriptors=desc, maximum=maximum, regime=regime)

def generate(repo: Path, output: Path) -> dict[str, Any]:
    history, history_q, history_cached, history_hashes = _historical(repo, output)
    cases: list[dict[str, Any]] = []
    seen = set(history); seen_q = set(history_q); seen_cached = set(history_cached)
    dev_cases = [
        {"id":"dev-v4-v32-below","family":"dev","index":0,"regime":"historical-below","batch_size":7,"descriptor_count":39,
         "q":[3,39,107,143,211,247,379],"cached":[22528,16512,11840,8256,2624,672,80],"max_cached":22528,
         "shape_eligible":True,"generator_seed":None,"exposed_development":True,"provenance":"selector-v3.2 canary-v32-below"},
        {"id":"dev-v4-v32-both","family":"dev","index":1,"regime":"historical-selected","batch_size":7,"descriptor_count":44,
         "q":[3,39,107,175,243,311,411],"cached":[22528,16512,11840,8256,2624,672,80],"max_cached":22528,
         "shape_eligible":True,"generator_seed":None,"exposed_development":True,"provenance":"selector-v3.2 canary-v32-both"},
    ]
    cases.extend(dev_cases)
    families = (("canary", 10, 11), ("release", 48, 29), ("stress", 12, 47))
    for family, count, offset in families:
        for i, spec in enumerate(_specs(family, count, offset)):
            for attempt in range(1000):
                seed = GENERATOR_SEED + offset * 100000 + i * 1009 + attempt
                case = _make_case(f"{family}-v4-{i:02d}", family, i, seed=seed, **spec)
                q_key = tuple(case["q"]); cached_key = tuple(case["cached"]); key = (q_key, cached_key)
                if key not in seen and q_key not in seen_q and cached_key not in seen_cached:
                    break
            else:
                raise RuntimeError("failed to generate fresh case")
            seen.add(key); seen_q.add(q_key); seen_cached.add(cached_key); cases.append(case)
    payload: dict[str, Any] = {
        "schema": SCHEMA, "generator_seed": GENERATOR_SEED,
        "qualification_revision": "4.0.0",
        "dtypes": ["float16", "bfloat16"], "layouts": ["ragged", "paged"],
        "requested_splits": ["auto", "unsplit"],
        "executions": ["eager_full_call", "graph1_replay", "graph16_replay"],
        "blocks": 8, "repeats": 3,
        "families": {"dev": 2, "canary": 10, "release": 48, "stress": 12},
        "historical_manifest_hashes": history_hashes, "cases": cases,
    }
    payload["case_hash"] = digest(cases)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload

def load(path: Path | None = None) -> dict[str, Any]:
    path = path or Path(__file__).with_name("manifest.json")
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA or data.get("case_hash") != digest(data.get("cases")):
        raise ValueError("manifest contract mismatch")
    return data
