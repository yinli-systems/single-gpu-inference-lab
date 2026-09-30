from __future__ import annotations
import contextlib, fcntl, hashlib, json, os, tempfile
from pathlib import Path
from typing import Any, Callable
from .identity import TacticIdentity, canonical_json
from .schema import DEFAULT_THRESHOLDS, QUALIFICATION_REVISION, SCHEMA_VERSION, TACTIC_CAP, TACTIC_NATIVE, VALID_TACTICS

Validator = Callable[[TacticIdentity, dict[str, Any]], bool]

def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".v4_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise

def _entry_hash(identity: TacticIdentity) -> str:
    return hashlib.sha256(identity.key.encode()).hexdigest()[:32]

class SafeTacticCache:
    def __init__(self, root: str | os.PathLike[str], identity_template: TacticIdentity):
        identity_template.validate()
        self.root = Path(root)
        self.environment_hash = identity_template.environment_hash
        self.env_dir = self.root / f"v{SCHEMA_VERSION}" / self.environment_hash
        self.entries_dir = self.env_dir / "entries"
        self._memo: dict[str, dict[str, Any] | None] = {}
        manifest = {
            "schema": SCHEMA_VERSION,
            "qualification_revision": QUALIFICATION_REVISION,
            "environment": identity_template.to_dict()["environment"],
            "measurement_policy": identity_template.to_dict()["measurement_policy"],
        }
        manifest["manifest_sha256"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
        path = self.env_dir / "manifest.json"
        self.usable = True
        if not path.exists():
            _atomic_json(path, manifest)
        else:
            try:
                self.usable = json.loads(path.read_text()) == manifest
            except (OSError, json.JSONDecodeError, TypeError):
                self.usable = False

    def _path(self, identity: TacticIdentity) -> Path:
        if identity.environment_hash != self.environment_hash:
            raise ValueError("cache environment mismatch")
        return self.entries_dir / f"{_entry_hash(identity)}.json"

    @staticmethod
    def validate_entry(identity: TacticIdentity, entry: dict[str, Any]) -> bool:
        try:
            identity.validate()
            if entry.get("schema") != SCHEMA_VERSION or entry.get("qualification_revision") != QUALIFICATION_REVISION:
                return False
            if entry.get("identity_key") != identity.key or entry.get("identity") != identity.to_dict():
                return False
            tactic = entry.get("tactic")
            if tactic not in VALID_TACTICS:
                return False
            receipt = entry.get("receipt")
            if not isinstance(receipt, dict) or receipt.get("identity_key") != identity.key:
                return False
            if receipt.get("thresholds") != DEFAULT_THRESHOLDS.to_dict():
                return False
            if tactic == TACTIC_CAP:
                return receipt.get("passed") is True and receipt.get("tactic") == TACTIC_CAP and receipt.get("reason") == "qualified_cap"
            return receipt.get("tactic") == TACTIC_NATIVE
        except Exception:
            return False

    def lookup(self, identity: TacticIdentity, validator: Validator | None = None) -> dict[str, Any] | None:
        if not self.usable:
            return None
        key = identity.key
        if key in self._memo:
            return self._memo[key]
        path = self._path(identity)
        try:
            entry = json.loads(path.read_text())
            if path.stem != _entry_hash(identity) or not self.validate_entry(identity, entry):
                raise ValueError("entry contract rejected")
            if validator is not None and not validator(identity, entry):
                raise ValueError("runtime tactic revalidation rejected")
            self._memo[key] = entry
            return entry
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            self._memo[key] = None
            return None

    def publish(self, identity: TacticIdentity, receipt: dict[str, Any], *, provenance: dict[str, Any]) -> Path:
        tactic = receipt.get("tactic")
        entry = {
            "schema": SCHEMA_VERSION,
            "qualification_revision": QUALIFICATION_REVISION,
            "identity_key": identity.key,
            "identity": identity.to_dict(),
            "tactic": tactic,
            "receipt": receipt,
            "provenance": provenance,
        }
        if tactic not in VALID_TACTICS or not self.validate_entry(identity, entry):
            raise ValueError("refusing invalid tactic publication")
        if not self.usable:
            raise ValueError("cache manifest is unusable")
        path = self._path(identity); lock_path = path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.exists():
                try:
                    existing = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    raise ValueError("refusing to overwrite corrupt cache entry")
                if canonical_json(existing) != canonical_json(entry):
                    raise ValueError("conflicting tactic publication")
                self._memo[identity.key] = existing
                return path
            _atomic_json(path, entry)
        self._memo[identity.key] = entry
        return path

    def reload(self) -> None:
        self._memo.clear()
