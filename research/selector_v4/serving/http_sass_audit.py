"""Full Native identity and all paged-prefix Resource SASS identity."""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

from research.selector_v4.public_qualification.runtime.audit_main_sass import parse


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(campaign, job, out):
    out.mkdir()
    raw = out / "raw"
    raw.mkdir()
    toolchain = Path(
        "/ssd/scxi253/single-gpu-inference-distinguished-20260928/toolchain/nvidia/cu13/bin"
    )
    tables = {"pristine": {}, "native": {}, "resource": {}}
    artifacts = []
    for index, binary in enumerate(sorted((campaign / "cache" / job).rglob("*.so"))):
        relative = binary.relative_to(campaign / "cache" / job)
        phase = relative.parts[0]
        kind = (
            "resource"
            if binary.name.startswith("experimental_resource_")
            else ("pristine" if phase.startswith("pristine-") else "native")
        )
        result = subprocess.run(
            [str(toolchain / "cuobjdump"), "--dump-sass", str(binary)],
            capture_output=True,
            text=True,
            check=True,
            timeout=240,
            env={**os.environ, "PATH": str(toolchain) + ":" + os.environ["PATH"]},
        )
        records = parse(result.stdout)
        if not records:
            continue
        for key, value in records.items():
            if key in tables[kind]:
                assert tables[kind][key] == value, (kind, key, str(binary))
            tables[kind][key] = value
        target = raw / f"{kind}-{index}.sass"
        target.write_text(result.stdout)
        artifacts.append(
            {
                "kind": kind,
                "phase": phase,
                "binary": str(binary),
                "binary_sha256": sha(binary),
                "sass": target.name,
                "sass_sha256": sha(target),
                "symbols": len(records),
            }
        )
    expected = tables["pristine"]
    paged = {k: v for k, v in expected.items() if "BatchPrefillWithPagedKVCacheKernel" in k}
    differences = {}
    for kind, reference in (("native", expected), ("resource", paged)):
        actual = tables[kind]
        differences[kind] = {
            "missing": sorted(reference.keys() - actual.keys()),
            "extra": sorted(actual.keys() - reference.keys()),
            "mismatches": sorted(
                k for k in reference.keys() & actual.keys() if reference[k] != actual[k]
            ),
        }
    passed = bool(paged) and all(
        not values for difference in differences.values() for values in difference.values()
    )
    archive = out / "raw-sass.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for path in raw.iterdir():
            stream.add(path, arcname=path.name)
    receipt = {
        "resource_scope": "ALL_PAGED_PREFIX_KERNELS",
        "pass": passed,
        "job": job,
        "campaign": str(campaign),
        "binding_sha256": sha(campaign / "binding.json"),
        "kernels_per_class": {k: len(v) for k, v in tables.items()},
        "tables": tables,
        "differences": differences,
        "artifacts": artifacts,
        "raw_archive_sha256": sha(archive),
        "raw_archive_bytes": archive.stat().st_size,
        "normalization": "Reverse resource mangled-name length; normalize PC labels and whitespace only. All instruction operands and encodings retained.",
        "fresh_cases_consumed": 0,
        "serving_qualified": False,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in ["pass", "kernels_per_class", "raw_archive_sha256", "raw_archive_bytes"]
            }
        )
    )
    assert passed, differences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("job")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    audit(args.campaign, args.job, args.out)
