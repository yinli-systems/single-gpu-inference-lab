from __future__ import annotations
from pathlib import Path
import hashlib, json


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def load() -> dict:
    data = json.loads(Path(__file__).with_name("manifest.json").read_text())
    if data.get("schema") != 1 or data.get("selector_version") != "4.0":
        raise ValueError("manifest schema")
    if digest(data["cases"]) != data["case_hash"]:
        raise ValueError("case hash")
    release = [c for c in data["cases"] if c["family"] == "release"]
    if digest(release) != data["release_hash"]:
        raise ValueError("release hash")
    if len(release) != 48:
        raise ValueError("release count")
    return data
