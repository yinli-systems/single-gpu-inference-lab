"""Synthetic files verify full provenance, not actual model loading."""

import json

import pytest

from research.selector_v4.serving.model_binding import bind_model, verify_model


def model(root, *, sharded):
    root.mkdir()
    config = {
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "num_hidden_layers": 36,
    }
    (root / "config.json").write_text(json.dumps(config))
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        (root / name).write_text("{}")
    names = (
        ["model-00001.safetensors", "model-00002.safetensors"] if sharded else ["model.safetensors"]
    )
    for name in names:
        (root / name).write_bytes(b"synthetic-unloaded-weight-data")
    if sharded:
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"layer0": names[0], "layer1": names[1]}})
        )
    return root


@pytest.mark.parametrize("sharded", [False, True])
def test_every_weight_tokenizer_and_config_byte_is_bound(tmp_path, sharded):
    root = model(tmp_path / "model", sharded=sharded)
    binding = bind_model(root)
    assert len(binding["weight_files"]) == (2 if sharded else 1)
    assert verify_model(binding) == binding
    (root / "vocab.json").write_text('{"changed":1}')
    with pytest.raises(ValueError, match="binding changed"):
        verify_model(binding)


@pytest.mark.parametrize(
    "fault",
    [
        "missing-shard",
        "extra-shard",
        "alternative",
        "missing-tokenizer",
        "symlink",
        "wrong-dimension",
    ],
)
def test_incomplete_or_ambiguous_model_cannot_get_a_binding(tmp_path, fault):
    root = model(tmp_path / "model", sharded=True)
    if fault == "missing-shard":
        (root / "model-00002.safetensors").unlink()
    elif fault == "extra-shard":
        (root / "extra.safetensors").write_bytes(b"extra")
    elif fault == "alternative":
        (root / "pytorch_model.bin").write_bytes(b"alternative")
    elif fault == "missing-tokenizer":
        (root / "tokenizer.json").unlink()
    elif fault == "symlink":
        (root / "tokenizer.json").unlink()
        (root / "tokenizer.json").symlink_to(root / "vocab.json")
    else:
        config = json.loads((root / "config.json").read_text())
        config["hidden_size"] = 2048
        (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        bind_model(root)
