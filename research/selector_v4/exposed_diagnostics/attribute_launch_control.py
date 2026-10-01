"""Isolated uncertified function-attribute versus launch-padding intervention.

Only private diagnostic source copies change. Native package/build and active
qualification caches remain untouched. No certificate/managed choice is used.
"""

import dataclasses
import hashlib
import json
import re
import shutil
from pathlib import Path

ARMS = ((49152, 49152), (65536, 49152), (65536, 65536))


def rewrite(source, attribute_bytes, launch_bytes):
    if (attribute_bytes, launch_bytes) not in ARMS:
        raise ValueError("Only the three preregistered legal arms are permitted")
    assignment = "smem_size = 65536;"
    attribute = (
        "cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)"
    )
    if source.count(assignment) != 2 or source.count(attribute) != 2:
        raise ValueError("Unreviewed final ragged/paged dispatch anchors")
    source = source.replace(
        assignment,
        f"smem_size = {launch_bytes}; const int diagnostic_optin_bytes = {attribute_bytes};",
    ).replace(attribute, attribute.replace("smem_size", "diagnostic_optin_bytes"))
    for layout in ("Ragged", "Paged"):
        prefix = f"BatchPrefillWith{layout}KVCacheResource"
        source = re.sub(r"\b" + prefix, prefix + f"A{attribute_bytes}L{launch_bytes}", source)
    return source


def rename_source(source, attribute_bytes, launch_bytes):
    for layout in ("Ragged", "Paged"):
        prefix = f"BatchPrefillWith{layout}KVCacheResource"
        source = re.sub(r"\b" + prefix, prefix + f"A{attribute_bytes}L{launch_bytes}", source)
    return source


def normalize_symbols(raw, attribute_bytes, launch_bytes):
    """Reverse only exact diagnostic identifier/mangled-length changes."""
    for layout in ("Ragged", "Paged"):
        native = f"BatchPrefillWith{layout}KVCacheKernel"
        arm = f"BatchPrefillWith{layout}KVCacheResourceA{attribute_bytes}L{launch_bytes}Kernel"
        raw = re.sub(
            r"(\d+)" + arm,
            lambda m, arm=arm, native=native: (
                str(int(m.group(1)) - len(arm) + len(native)) + native
            ),
            raw,
        )
    return raw


def build_modules(module_args, source_id, evidence):
    from flashinfer.experimental.prefill_resource import _jit, _ops
    from flashinfer.jit import env
    from flashinfer.jit.core import jit_spec_registry

    original = _jit.resource_spec(*module_args, expected_source_id=source_id)
    directory = original.sources[0].parent
    immutable = {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.rglob("*")
        if p.is_file() and p.name != "sources.lock"
    }
    result = {}
    for attribute_bytes, launch_bytes in ARMS:
        identity = hashlib.sha256(
            json.dumps(
                {
                    "source_id": source_id,
                    "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "attribute_bytes": attribute_bytes,
                    "launch_bytes": launch_bytes,
                    "original_spec": original.name,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        name = "sgi_attribute_launch_" + identity
        target = env.FLASHINFER_GEN_SRC_DIR / name
        if target.exists():
            raise FileExistsError("Keep first diagnostic compilation; no replacement")
        shutil.copytree(directory, target)
        header = target / "include/flashinfer/attention/prefill.cuh"
        header.write_text(rewrite(header.read_text(), attribute_bytes, launch_bytes))
        for path in target.iterdir():
            if path.is_file() and path.suffix in (".cu", ".cuh", ".inc"):
                path.write_text(rename_source(path.read_text(), attribute_bytes, launch_bytes))
        spec = dataclasses.replace(
            original,
            name=name,
            sources=[target / p.name for p in original.sources],
            extra_include_dirs=[target / "include", *(original.extra_include_dirs or [])[1:]],
        )
        jit_spec_registry.register(spec)
        module = _ops._register_resource_module(spec)
        label = f"attribute{attribute_bytes}_launch{launch_bytes}"
        ledger = {
            str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in target.rglob("*")
            if p.is_file() and p.name != "sources.lock"
        }
        proof = {
            "label": label,
            "attribute_bytes": attribute_bytes,
            "launch_bytes": launch_bytes,
            "spec_name": name,
            "binary": str(spec.get_library_path()),
            "binary_sha256": hashlib.sha256(spec.get_library_path().read_bytes()).hexdigest(),
            "source_files": ledger,
            "uncertified_diagnostic_only": True,
        }
        (evidence / (label + "-module.json")).write_text(json.dumps(proof, indent=2) + "\n")
        result[label] = (module, proof)
    for name, digest in immutable.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    return result
