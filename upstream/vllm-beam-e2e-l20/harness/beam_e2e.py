"""End-to-end offline beam search on a real model: wall time and outputs.

  python beam_e2e.py MODEL OUT.json

Unconstrained: beam width 4/8/16/32 x prompt ~128/~2048 tokens, 32 new tokens.
JSON schema (xgrammar): beam width 4/8/16/32, up to 96 new tokens.
Each cell: one warm-up call, then 5 timed calls; outputs of the last call are
recorded so arms can be compared token for token.
"""

import json
import os
import statistics
import sys
import time

from vllm import LLM
from vllm.sampling_params import BeamSearchParams, StructuredOutputsParams

MODEL, OUT = sys.argv[1], sys.argv[2]

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
FILLER = (
    "The Analytical Engine was a proposed digital mechanical general-purpose "
    "computer designed by Charles Babbage. It incorporated an arithmetic logic "
    "unit, control flow in the form of conditional branching and loops, and "
    "integrated memory. "
)


def main():
    llm = LLM(
        model=MODEL,
        max_model_len=8192,
        max_logprobs=64,  # beam search asks for 2 * beam_width
        gpu_memory_utilization=0.6,
        seed=0,
        structured_outputs_config={"backend": "xgrammar"},
        enable_prefix_caching=os.environ.get("BEAM_PREFIX_CACHING", "1") == "1",
    )
    tok = llm.get_tokenizer()

    def prompt_of(n_tokens: int) -> dict:
        ids = tok.encode(FILLER * (n_tokens // 40 + 2), add_special_tokens=False)
        return {"prompt_token_ids": ids[:n_tokens]}

    chat_ids = tok.apply_chat_template(
        [{"role": "user", "content": "Describe a fictional person as JSON "
          "with fields name, age, role, tags and active."}],
        add_generation_prompt=True,
    )
    if hasattr(chat_ids, "keys"):  # newer transformers return a BatchEncoding
        chat_ids = chat_ids["input_ids"]
    json_prompt = {"prompt_token_ids": list(chat_ids)}

    cells = []
    equality_mode = os.environ.get("BEAM_EQUALITY_MODE") == "1"
    for kind, prompt_len, max_tokens, so in (
        ("plain", 128, 32, None),
        ("plain", 2048, 32, None),
        ("json", None, 96, StructuredOutputsParams(json=SCHEMA)),
    ):
        prompt = json_prompt if kind == "json" else prompt_of(prompt_len)
        widths = (4, 8, 16, 32)
        if equality_mode and kind == "plain" and prompt_len == 2048:
            widths = (8, 32)
        if equality_mode and kind == "json":
            widths = (4, 8, 16)
        for width in widths:
            params = BeamSearchParams(beam_width=width, max_tokens=max_tokens)
            if so is not None:
                params.structured_outputs = so
            llm.beam_search([prompt], params)  # warm-up
            times, out = [], None
            for _ in range(5):
                t0 = time.perf_counter()
                out = llm.beam_search([prompt], params)
                times.append(time.perf_counter() - t0)
            seqs = [
                {"tokens": s.tokens, "cum": float.hex(s.cum_logprob)}
                for s in out[0].sequences
            ]
            cell = dict(kind=kind, prompt_len=len(prompt["prompt_token_ids"]),
                        width=width, max_tokens=max_tokens,
                        wall_s=statistics.median(times), wall_s_all=times,
                        sequences=seqs)
            cells.append(cell)
            json.dump(cells, open(OUT, "w"))  # after every cell
            print(f"{kind:5s} prompt {cell['prompt_len']:5d} width {width:2d}: "
                  f"{cell['wall_s'] * 1e3:8.1f} ms  "
                  f"({', '.join(f'{t * 1e3:.0f}' for t in times)})", flush=True)
    json.dump(cells, open(OUT, "w"))


if __name__ == "__main__":
    main()
