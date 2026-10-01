"""Independent raw HTTP metric/parity analysis; no controller-state promotion.

Source/model/Graph/metadata/profile/SASS gates remain mandatory. Every declared
raw artifact is SHA checked; every timing metric is recomputed from requests.
"""

import argparse
import gzip
import hashlib
import json
import math
import tarfile
from pathlib import Path

import numpy as np

from research.release_qualification.serving.workloads import prefix_tokens, work_specs
from research.selector_v4.public_qualification.gates import need, sha
from research.selector_v4.public_qualification.runtime.audit_main_sass import parse
from research.selector_v4.serving.http_stream import TokenStream
from research.selector_v4.serving.metric_gate import METRICS, WORKLOADS, evaluate_metrics
from research.selector_v4.serving.parity_gate import compare_blocks


def derived_metrics(block, expected):
    need(block["complete"] and not block["errors"], "No failed block or success-only metric")
    digest = hashlib.sha256(
        json.dumps(expected, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    need(block["workload_sha256"] == digest, "Exact frozen workload bytes required")
    rows = {r["id"]: r for r in block["requests"]}
    cells = {c["id"]: c for c in expected["cells"]}
    need(
        set(rows) == set(cells) and len(rows) == len(block["requests"]),
        "Every unique logical request required",
    )
    need(len({r["rid"] for r in rows.values()}) == len(rows), "Actual request IDs must be unique")
    for key, row in rows.items():
        cell = cells[key]
        need(row["complete"] and row["http_status"] == 200, "Actual complete HTTP response")
        need(row["payload"]["input_ids"] == cell["input_ids"], "Full HTTP input changed")
        need(
            row["payload"]["sampling_params"]
            == {"temperature": 0, "max_new_tokens": cell["output_tokens"], "ignore_eos": True},
            "Actual sampling contract changed",
        )
        need(
            len(row["tokens"]) == len(row["token_times"]) == cell["output_tokens"],
            "Every output token/timestamp required",
        )
        parsed = TokenStream(cell["output_tokens"])
        for event in row["raw_events"]:
            parsed.feed(
                b"data: " + event["data"].encode() + b"\n\n", event["client_arrival_seconds"]
            )
        raw = parsed.complete(expected_cached_tokens=cell["expect_cached"])
        need(
            all(
                row[k] == raw[k]
                for k in ("tokens", "token_times", "ttft", "tpot", "cached_tokens", "finish_reason")
            ),
            "Derived request differs from actual raw SSE events",
        )
        times = row["token_times"]
        need(
            all(math.isfinite(v) and v > 0 for v in times) and times == sorted(times),
            "Actual monotonic SSE arrivals",
        )
        need(
            row["ttft"] == times[0] and row["tpot"] == (times[-1] - times[0]) / (len(times) - 1),
            "Derived HTTP token timing mismatch",
        )
        need(
            row["tpot"] > 0 and row["cached_tokens"] >= cell["expect_cached"],
            "Valid TPOT and actual cache hit",
        )
        need(row["finish_reason"]["type"] == "length", "Length-complete sampled output")
    elapsed = block["elapsed"]
    need(math.isfinite(elapsed) and elapsed > 0, "Complete positive block elapsed time")
    need(
        all(0 <= r["request_start"] <= r["completion"] <= elapsed for r in rows.values()),
        "Every request lies in the complete block",
    )
    total = sum(len(r["tokens"]) for r in rows.values())
    derived = {
        "output_tokens_per_second": total / elapsed,
        "strict_slo_goodput": sum(r["ttft"] <= 2 and r["tpot"] <= 0.05 for r in rows.values())
        / elapsed,
    }
    for kind in ("TTFT", "TPOT"):
        for label, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
            derived[f"{kind}-{label}"] = float(
                np.quantile([r[kind.lower()] for r in rows.values()], q)
            )
    need(
        total == block["output_tokens"] and all(block[k] == v for k, v in derived.items()),
        "Producer metric differs from complete raw requests",
    )
    return derived


def load_arm(root, *, role, stage, allocation, model_id, binding_sha, model_sha):
    need(not (root / "failure.json").exists(), "Failed HTTP arm")
    complete = json.loads((root / "complete.json").read_text())
    need(
        complete["complete"]
        and all(
            complete[k] == v
            for k, v in {
                "role": role,
                "stage": stage,
                "allocation": allocation,
                "model_id": model_id,
            }.items()
        ),
        "Actual HTTP arm identity",
    )
    for name, digest in complete["files"].items():
        path = Path(name)
        need(not path.is_absolute() and ".." not in path.parts, "Safe HTTP raw path")
        need(sha(root / path) == digest, "HTTP raw evidence changed: " + name)
    environment = json.loads((root / "environment.json").read_text())
    need(
        environment["http_binding_sha256"] == binding_sha
        and environment["model_binding_sha256"] == model_sha,
        "Exact HTTP/model bindings",
    )
    data, metrics = {}, {w: {m: [] for m in METRICS} for w in WORKLOADS}
    actual = {
        f.name
        for f in (root / "observations").glob("*-b*.json")
        if not f.name.startswith((model_id + "-", "profile"))
    }
    expected = {f"{w}-b{b}.json" for w in WORKLOADS for b in range(4)}
    need(actual == expected, "Exactly all20 workload blocks required")
    for block in range(4):
        specs = work_specs(prefix_tokens(), block)
        for workload in WORKLOADS:
            key = f"{workload}-b{block}.json"
            value = json.loads((root / "observations" / key).read_text())
            result = derived_metrics(value, specs[workload])
            data[key] = value
            for metric in METRICS:
                metrics[workload][metric].append(result[metric])
    return environment, data, metrics


def profile_evidence(root, role):
    proof = json.loads((root / "observations/profile-timing-scope.json").read_text())
    need(proof["timing_excluded"] and proof["trace_files"], "Independent raw profiling required")
    kernels = []
    for name, digest in proof["trace_files"].items():
        path = root / name
        need(sha(path) == digest, "Profile bytes changed")
        if path.name.endswith((".json", ".json.gz")):
            if path.name.endswith(".gz"):
                with gzip.open(path, "rt") as stream:
                    text = stream.read()
            else:
                text = path.read_text()
            data = json.loads(text)
            kernels.extend(
                e
                for e in data.get("traceEvents", [])
                if e.get("cat") == "kernel" and "BatchPrefillWith" in e.get("name", "")
            )
    need(kernels, "Actual prefill kernel trace required")
    resource = [e for e in kernels if "ResourceKernel" in e["name"]]
    need(
        all(e["args"].get("shared memory") == 65536 for e in resource),
        "Actual Resource launch allocation required",
    )
    need(
        bool(resource) if role == "candidate" else not resource,
        "Actual candidate Resource/control Native launch required",
    )
    return {
        "prefill_launches": len(kernels),
        "actual_resource_launches": len(resource),
        "timing_excluded": True,
    }


def sass_evidence(root, *, job, binding_sha):
    """Reconstruct all three classes from original disassembly, never a PASS flag."""
    receipt = json.loads((root / "receipt.json").read_text())
    need(receipt["resource_scope"] == "ALL_PAGED_PREFIX_KERNELS", "Exact Resource HTTP scope")
    need(
        receipt["job"] == job and receipt["binding_sha256"] == binding_sha,
        "Independent SASS identity",
    )
    archive = root / "raw-sass.tar.gz"
    need(sha(archive) == receipt["raw_archive_sha256"], "Original SASS bytes changed")
    artifacts = receipt["artifacts"]
    names = {a["sass"] for a in artifacts}
    need(len(names) == len(artifacts), "Unique original SASS artifacts")
    tables = {kind: {} for kind in ("pristine", "native", "resource")}
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        need(
            len(members) == len(names) and {m.name for m in members} == names,
            "Every original SASS member required, no extras",
        )
        need(all(m.isfile() for m in members), "Regular original disassembly only")
        for artifact in artifacts:
            kind = artifact["kind"]
            need(kind in tables, "Known SASS class")
            raw = stream.extractfile(artifact["sass"]).read()
            need(
                hashlib.sha256(raw).hexdigest() == artifact["sass_sha256"],
                "Original disassembly changed",
            )
            records = parse(raw.decode())
            need(bool(records) and len(records) == artifact["symbols"], "Full SASS symbols")
            for key, value in records.items():
                need(
                    key not in tables[kind] or tables[kind][key] == value,
                    "Conflicting original SASS module",
                )
                tables[kind][key] = value
    expected = {
        k: v for k, v in tables["pristine"].items() if "BatchPrefillWithPagedKVCacheKernel" in k
    }
    need(
        bool(expected)
        and tables["pristine"] == tables["native"]
        and tables["resource"] == expected,
        "Exact full Native/pristine/Resource disassembly identity",
    )
    need(
        tables == receipt["tables"]
        and {k: len(v) for k, v in tables.items()} == receipt["kernels_per_class"],
        "Producer SASS tables differ from original instructions",
    )
    return {"kernels_per_class": {k: len(v) for k, v in tables.items()}}


def analyze(root, output, *, jobs, gpu, model_id, stage):
    need(
        not output.exists() and len(jobs) == 3 and len(set(jobs)) == 3,
        "Three independent complete allocations and new analysis required",
    )
    binding = json.loads((root / "binding.json").read_text())
    binding_sha = sha(root / "binding.json")
    model_sha = binding["models"][model_id]["binding_sha256"]
    allocations, parity, profiles, source = [], [], [], []
    for allocation, job in enumerate(jobs):
        path = root / "paired" / job
        need((path / "exit.txt").read_text().strip() == "0", "Complete paired HTTP allocation")
        arms = {}
        environments = []
        blocks = {}
        for role in ("pristine", "candidate"):
            env, data, metrics = load_arm(
                path / role,
                role=role,
                stage=stage,
                allocation=allocation,
                model_id=model_id,
                binding_sha=binding_sha,
                model_sha=model_sha,
            )
            need(
                env["gpu"] == gpu and env["actual_flashinfer_commit"] == binding[role + "_commit"],
                "Actual GPU family/package source",
            )
            environments.append(env)
            blocks[role] = data
            arms[role] = metrics
            profiles.append(profile_evidence(path / role, role))
        need(
            environments[0]["gpu_uuid"] == environments[1]["gpu_uuid"]
            and environments[0]["cpu_affinity"] == environments[1]["cpu_affinity"],
            "Paired actual GPU and CPU affinity",
        )
        sass_evidence(path / "independent-sass", job=job, binding_sha=binding_sha)
        for key in blocks["pristine"]:
            parity.append(
                compare_blocks(
                    blocks["pristine"][key],
                    blocks["candidate"][key],
                    require_logprobs=stage == "parity",
                )
            )
        snapshots = sorted((path / "candidate/training").glob("phase-*-3-*.json"))
        need(snapshots, "Actual scheduler metadata/input snapshot required")
        snapshot = json.loads(snapshots[-1].read_text())
        need(
            snapshot["phase"] == "FROZEN_SERVING"
            and snapshot["metadata_depth"] == 0
            and snapshot["metadata_boundaries"]["early_binds"] > 0,
            "Actual early-bound serving metadata required",
        )
        if stage != "parity":
            need(
                snapshot["metadata_boundaries"]["graph_updates"] > 0,
                "Actual Native Graph metadata/input updates required",
            )
        allocations.append(arms)
        source.append(
            {
                "job": job,
                "sass_receipt_sha256": sha(path / "independent-sass/receipt.json"),
                "metadata_snapshot_sha256": sha(snapshots[-1]),
            }
        )
    metric = evaluate_metrics(allocations) if stage == "performance" else None
    passed = all(v["parity_pass"] for v in parity) and (metric is None or metric["metric_pass"])
    output.mkdir(parents=True)
    result = {
        "pass": passed,
        "stage": stage,
        "gpu": gpu,
        "model_id": model_id,
        "jobs": jobs,
        "binding_sha256": binding_sha,
        "source": source,
        "profiles": profiles,
        "parity": parity,
        "metrics": metric,
        "full_http_qualified": False,
        "default_promotion": False,
        "historical_token_divergence_resolved": False,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", nargs=3, required=True)
    parser.add_argument("--gpu", choices=("gpu_4090", "gpu_5090"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--stage", choices=("functional", "performance", "parity"), required=True)
    args = parser.parse_args()
    result = analyze(
        args.root,
        args.output,
        jobs=args.jobs,
        gpu=args.gpu,
        model_id=args.model_id,
        stage=args.stage,
    )
    print(
        json.dumps(
            {
                "pass": result["pass"],
                "stage": args.stage,
                "gpu": args.gpu,
                "model_id": args.model_id,
            }
        )
    )
