"""Bind all actual model shards, tokenizer/config bytes and load geometry.

This is read-only provenance work. Hashes alone do not qualify numerical output,
serving performance, model completeness at runtime or Resource dispatch.
"""

import argparse
import hashlib
import json
from pathlib import Path


def hash_stable_file(path):
    initial = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    final = path.stat()
    if (initial.st_ino, initial.st_size, initial.st_mtime_ns) != (
        final.st_ino,
        final.st_size,
        final.st_mtime_ns,
    ):
        raise ValueError("Model file changed while hashing")
    return {"sha256": digest.hexdigest(), "bytes": initial.st_size}


def bind_model(root):
    root = Path(root).resolve(strict=True)
    config = json.loads((root / "config.json").read_text())
    heads = config["num_attention_heads"]
    dimension = config.get("head_dim") or config["hidden_size"] // heads
    if dimension != 128 or config["num_hidden_layers"] <= 0:
        raise ValueError("Unreviewed model geometry")
    index = root / "model.safetensors.index.json"
    if index.exists():
        index_data = json.loads(index.read_text())
        weights = set(index_data["weight_map"].values())
        if not weights:
            raise ValueError("Empty model shard index")
        names = weights | {index.name}
    else:
        weights = {"model.safetensors"}
        names = set(weights)
    if {f.name for f in root.glob("*.safetensors")} != weights:
        raise ValueError("Missing or extra safetensors shards")
    if list(root.glob("pytorch_model*.bin")):
        raise ValueError("Ambiguous alternative model weight files")
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    optional = {
        "generation_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer.model",
        "configuration.json",
    }
    names |= required | {n for n in optional if (root / n).exists()}
    files = {}
    for name in sorted(names):
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Model shard must be an immediate relative filename")
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError("Complete regular model/tokenizer files required")
        files[name] = hash_stable_file(path)
    # Re-read config/index after hashing all weights, so the earlier interpreted
    # shard map and dimensions must agree with the final source-bound bytes.
    if config != json.loads((root / "config.json").read_text()):
        raise ValueError("Model configuration changed during binding")
    if index.exists() and index_data != json.loads(index.read_text()):
        raise ValueError("Shard index changed during binding")
    return {
        "model_path": str(root),
        "files": files,
        "weight_files": sorted(weights),
        "weight_bytes": sum(files[n]["bytes"] for n in weights),
        "layers": config["num_hidden_layers"],
        "heads": heads,
        "kv_heads": config["num_key_value_heads"],
        "head_dim": dimension,
        "runtime_dtype": "bfloat16",
        "full_http_qualified": False,
        "qualification_authority": False,
    }


def verify_model(binding):
    actual = bind_model(binding["model_path"])
    if actual != binding:
        raise ValueError("Complete model/tokenizer source binding changed")
    return actual


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    result = bind_model(arguments.model)
    with arguments.out.open("x") as destination:
        destination.write(json.dumps(result, indent=2) + "\n")
