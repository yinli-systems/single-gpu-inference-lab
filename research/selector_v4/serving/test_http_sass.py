"""A forged PASS receipt cannot override different original CUDA instructions."""

import hashlib
import io
import json
import tarfile

import pytest

from research.selector_v4.public_qualification.runtime.audit_main_sass import parse
from research.selector_v4.serving.http_analysis import sass_evidence


def proof(root, fault=None):
    tables = {kind: {} for kind in ("pristine", "native", "resource")}
    artifacts, files = [], {}
    for kind in tables:
        operation = "BatchPrefillWithPagedKVCache"
        symbol = f"_Z{len(operation) + 6}{operation}Kernelv"
        if kind == "resource":
            symbol = f"_Z{len(operation) + 14}{operation}ResourceKernelv"
        instruction = "EXIT" if fault == "changed-instruction" and kind == "resource" else "RET"
        raw = f"arch = sm_120\nFunction : {symbol}\n/*0000*/ {instruction}; /* 0x1234 */\n".encode()
        records = parse(raw.decode())
        tables[kind] = records
        name = kind + ".sass"
        files[name] = raw
        artifacts.append(
            {
                "kind": kind,
                "sass": name,
                "sass_sha256": hashlib.sha256(raw).hexdigest(),
                "symbols": len(records),
            }
        )
    if fault == "missing-resource":
        artifacts.pop()
        files.pop("resource.sass")
        tables["resource"] = {}
    if fault == "extra-member":
        files["undeclared.sass"] = b"untracked instructions"
    archive = root / "raw-sass.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, raw in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            stream.addfile(member, io.BytesIO(raw))
    receipt = {
        "pass": True,
        "resource_scope": "ALL_PAGED_PREFIX_KERNELS",
        "job": "123",
        "binding_sha256": "binding",
        "raw_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "artifacts": artifacts,
        "tables": tables,
        "kernels_per_class": {k: len(v) for k, v in tables.items()},
    }
    if fault == "forged-tables":
        receipt["tables"]["resource"] = {"fake": "fake"}
    (root / "receipt.json").write_text(json.dumps(receipt))


def test_original_all_three_sass_classes_reconstruct_exactly(tmp_path):
    proof(tmp_path)
    assert sass_evidence(tmp_path, job="123", binding_sha="binding") == {
        "kernels_per_class": {"pristine": 1, "native": 1, "resource": 1}
    }


@pytest.mark.parametrize(
    "fault", ["changed-instruction", "missing-resource", "extra-member", "forged-tables"]
)
def test_rehashed_forged_pass_cannot_hide_original_sass_failure(tmp_path, fault):
    proof(tmp_path, fault)
    with pytest.raises(ValueError):
        sass_evidence(tmp_path, job="123", binding_sha="binding")
