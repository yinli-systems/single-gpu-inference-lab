"""_build_beam_sampling_params on real grammar states: main vs change.

Run once per tree (PYTHONPATH); `diff` dumps a canonical record per beam and
`bench` times the per-step grammar work for a batch of beams.

Canonical record per beam: dropped (None) or not, the exact allowed_token_ids
handed to the engine, and a digest of the membership answer for every token ID
in the vocabulary.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from types import SimpleNamespace

from transformers import AutoTokenizer

from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

MODEL = "/data/run01/scxi253/inference/models/Qwen1.5-MoE-A2.7B-Chat"
SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "role": {"enum": ["admin", "editor", "viewer"]},
        "tags": {"type": "array", "items": {"type": "string"}},
        "active": {"type": "boolean"},
    },
    "required": ["name", "age", "role", "tags", "active"],
}
INSTANCE = {
    "name": "Ada Lovelace, Countess of Lovelace, analyst of the Analytical Engine",
    "age": 36,
    "role": "editor",
    "tags": ["mathematics", "computing", "poetical science", "notes on Menabrea"],
    "active": True,
}



V = 151936
PROMPT = [151644, 872, 198]  # arbitrary prompt ids


def setup():
    tok = AutoTokenizer.from_pretrained(MODEL)
    backend = XgrammarBackend(
        vllm_config=SimpleNamespace(
            structured_outputs_config=SimpleNamespace(disable_any_whitespace=False),
            speculative_config=None,
        ),
        tokenizer=tok,
        vocab_size=V,
    )
    llm = LLM.__new__(LLM)
    llm.__dict__["model_config"] = SimpleNamespace(get_vocab_size=lambda: V)
    trajectory = tok.encode(
        json.dumps(INSTANCE, separators=(",", ":")), add_special_tokens=False
    )
    key = (StructuredOutputOptions.JSON, json.dumps(SCHEMA))
    return tok, backend, llm, trajectory, key


def beam(generated):
    return BeamSearchSequence(
        orig_prompt={"type": "token", "prompt_token_ids": PROMPT},
        tokens=PROMPT + list(generated),
        logprobs=[],
    )


def canonical(entry):
    if entry is None:
        return None
    params, allowed = entry
    if isinstance(allowed, list):  # main hands back the list; step makes a set
        allowed = set(allowed)
    membership = bytes(1 if t in allowed else 0 for t in range(V + 64))
    return {
        "allowed_token_ids": params.allowed_token_ids,
        "membership_sha256": hashlib.sha256(membership).hexdigest(),
        "num_allowed": sum(membership),
    }


def diff_mode(out):
    tok, backend, llm, trajectory, key = setup()
    beams = [beam(trajectory[:d]) for d in range(len(trajectory) + 1)]
    # terminated: the full instance plus EOS; off-grammar: a stray token
    beams.append(beam(trajectory + [tok.eos_token_id or 151643]))
    beams.append(beam(trajectory[:5] + [tok.encode("!!!", add_special_tokens=False)[0]]))
    bitmask = backend.allocate_token_bitmask(1)
    entries = llm._build_beam_sampling_params(
        beams, SamplingParams(logprobs=8, max_tokens=1), backend, key, bitmask
    )
    records = [canonical(e) for e in entries]
    json.dump(records, open(out, "w"), indent=0)
    dense = sum(1 for r in records if r and r["allowed_token_ids"] is None)
    print(f"{len(records)} beams: {sum(r is None for r in records)} dropped, "
          f"{dense} dense (no engine list), "
          f"{len(records) - dense - sum(r is None for r in records)} sparse")


def bench_mode(out):
    tok, backend, llm, trajectory, key = setup()
    width = 32
    # a step's 32 beams spread over the trajectory: the real mix of dense
    # (inside strings) and sparse (structural) positions
    depths = [round(i * (len(trajectory) - 1) / (width - 1)) for i in range(width)]
    beams = [beam(trajectory[:d]) for d in depths]
    bitmask = backend.allocate_token_bitmask(1)
    candidates = list(range(1000, 1000 + 2 * width))

    def one_step():
        entries = llm._build_beam_sampling_params(
            beams, SamplingParams(logprobs=2 * width, max_tokens=1), backend, key,
            bitmask,
        )
        # what _beam_search_step then does with each entry
        for e in entries:
            if e is None:
                continue
            allowed = e[1]
            if isinstance(allowed, list):
                allowed = set(allowed)
            for t in candidates:
                _ = t in allowed

    samples = []
    for _ in range(15):
        t0 = time.perf_counter()
        one_step()
        samples.append(time.perf_counter() - t0)
    ms = statistics.median(samples[3:]) * 1e3
    print(f"beam width {width}: grammar work per step {ms:.2f} ms "
          f"(samples {[round(x * 1e3, 1) for x in samples[3:]]})")
    json.dump({"width": width, "ms_per_step": ms}, open(out, "w"))


if __name__ == "__main__":
    {"diff": diff_mode, "bench": bench_mode}[sys.argv[1]](sys.argv[2])
