"""Validate immutable native metadata observations, not HTTP performance."""

from pathlib import Path
import hashlib, json, collections

b = Path("/ssd/scxi253")
p = b / "sgi-http-geometry-native-unpack-8f9ab99-eb4-20261001"
old = b / "sgi-http-geometry-261123b-eb4-20261001"
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
assert (p / "receipts/exit-1645581.txt").read_text().strip() == "0"
r = p / "runs/full-http-1645581"
c = json.loads((r / "complete.json").read_text())
assert c["complete"] and c["pass_"]
assert sum(x["samples"] for x in c["requests"]) == 7
rows = []
for name, h in c["geometry_files"].items():
    f = r / name
    assert sha(f) == h
    rows += [json.loads(x) for x in f.read_text().splitlines()]
assert len(rows) == 54
for x in rows:
    assert x["native_kv_unpacking_applied"] == (x["layout"] == "paged")
    assert x["qo_lengths"] == [b - a for a, b in zip(x["q_indptr"], x["q_indptr"][1:])]
    if x["layout"] == "paged":
        assert x["page_size"] == 1 and x["inputs"][1]["shape"][1] == 1
        assert x["kv_lengths"] == [
            (b - a - 1) * x["page_size"] + n
            for a, b, n in zip(x["kv_indptr"], x["kv_indptr"][1:], x["last_page_lengths"])
        ]
    else:
        assert x["kv_lengths"] == [b - a for a, b in zip(x["kv_indptr"], x["kv_indptr"][1:])]
    assert x["metadata_version_counters_detect_foreign_cuda_writes"] is False
    assert not x["resource_tactic_selected"] and not x["timing_valid"]
parity = []
for f in sorted(r.glob("*.json")):
    if f.name in ["complete.json", "environment.json"]:
        continue
    a = json.loads(f.read_text())
    z = json.loads((old / "runs/full-http-1645470" / f.name).read_text())
    assert a["payload"] == z["payload"]
    ar = json.loads(a["response"])
    zr = json.loads(z["response"])
    ar = ar if isinstance(ar, list) else [ar]
    zr = zr if isinstance(zr, list) else [zr]
    for i, (x, y) in enumerate(zip(ar, zr, strict=True)):
        assert len(x["output_ids"]) == len(y["output_ids"]) == 16
        parity.append(
            dict(request=f.name, sample=i, exact=x["output_ids"] == y["output_ids"], tokens=16)
        )
assert len(parity) == 7
out = p / "derived-native-geometry"
out.mkdir()
result = dict(
    observer_harness_commit=json.loads((p / "binding.json").read_text())["observer_harness_commit"],
    job="1645581",
    observed_records=len(rows),
    raw_geometry_files=c["geometry_files"],
    native_page_sizes=sorted({x["page_size"] for x in rows}),
    layout_causal_split_counts={
        str(k): v
        for k, v in collections.Counter(
            (x["layout"], x["causal"], x["actual_split"]) for x in rows
        ).items()
    },
    noncausal_prefix_query_longer_than_kv=sum(
        any(q > k for q, k in zip(x["qo_lengths"], x["kv_lengths"]))
        for x in rows
        if not x["causal"]
    ),
    ragged_kv_strides=sorted(
        {tuple(x["inputs"][1]["stride"]) for x in rows if x["layout"] == "ragged"}
    ),
    observer_repair_native_token_controls=parity,
    all112_native_control_tokens_exact=all(x["exact"] for x in parity),
    resource_tactic_selected=False,
    timing_valid=False,
    fresh_cases_consumed=0,
    full_http_resource_qualified=False,
    historical_token_divergence_resolved=False,
    limitations=[
        "Observations include startup/capture warmup as well as HTTP;54 records are not54 live HTTP samples.",
        "Observer sync/readbacks/write costs invalidate timings.",
        "Same native resource-OFF control does not qualify resource token parity.",
        "Tensor version availability does not detect foreign CUDA/Triton writes; exclusive owner invalidation remains required.",
    ],
    analyzer_sha256=sha(Path(__file__)),
)
(out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
