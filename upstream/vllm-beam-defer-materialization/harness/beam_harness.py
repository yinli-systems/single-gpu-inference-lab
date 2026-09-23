"""Drive vLLM's real LLM.beam_search with a fake engine.

Run once with PYTHONPATH pointing at unmodified main and once at the change;
compare the dumped outputs and the timings.

  python beam_harness.py diff  OUT.json   # canonical outputs for many configs
  python beam_harness.py bench OUT.json   # per-step beam-logic CPU time

The fake engine is deterministic in the prompt tokens, so both runs see
identical candidates. Timing isolates the beam logic: total time inside
_beam_search_step minus the time spent inside the fake engine call.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

from vllm import CompletionOutput, RequestOutput
from vllm.entrypoints.generate.beam_search import offline as offline_mod
from vllm.entrypoints.llm import LLM
from vllm.logprobs import Logprob
from vllm.sampling_params import BeamSearchParams

EOS = 0
VOCAB = 50_000


class FakeEngine:
    def __init__(self, beam_width: int, eos_rate: float, tie_grid: float | None):
        self.k = 2 * beam_width
        self.eos_rate = eos_rate
        self.tie_grid = tie_grid
        self.seconds = 0.0
        self.calls = 0

    def logprobs_for(self, tokens: list[int]) -> dict[int, Logprob]:
        rng = random.Random(hash((len(tokens), tuple(tokens[-6:]))))
        ids = rng.sample(range(1, VOCAB), self.k)
        if rng.random() < self.eos_rate:
            ids[rng.randrange(self.k)] = EOS
        out: dict[int, Logprob] = {}
        for tid in ids:
            lp = -rng.expovariate(1.0)
            if self.tie_grid is not None:
                lp = round(lp / self.tie_grid) * self.tie_grid
            out[tid] = Logprob(lp)
        return out

    def __call__(self, prompts, **kwargs):
        t0 = time.perf_counter()
        results = []
        for prompt in prompts:
            tokens = prompt["prompt_token_ids"]
            results.append(
                RequestOutput(
                    request_id="inner",
                    prompt=None,
                    prompt_token_ids=tokens,
                    prompt_logprobs=None,
                    finished=True,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text="",
                            token_ids=[1],
                            cumulative_logprob=None,
                            logprobs=[self.logprobs_for(tokens)],
                            finish_reason="length",
                        )
                    ],
                )
            )
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return results


def make_llm(engine: FakeEngine) -> LLM:
    llm = LLM.__new__(LLM)
    llm.llm_engine = Mock()
    tokenizer = SimpleNamespace(
        eos_token_id=EOS,
        decode=lambda tokens, skip_special_tokens=False: "",
    )
    llm.renderer = Mock(get_tokenizer=Mock(return_value=tokenizer))
    llm._preprocess_cmpl = lambda prompts: prompts
    llm._render_and_run_requests = engine
    return llm


def prompts_for(num_prompts: int, prompt_len: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [
        {
            "type": "token",
            "prompt_token_ids": [rng.randrange(1, VOCAB) for _ in range(prompt_len)],
        }
        for _ in range(num_prompts)
    ]


def canonical(outputs) -> list:
    """Everything observable about the result, floats as exact bit patterns."""
    return [
        [
            {
                "tokens": seq.tokens,
                "cum": float.hex(seq.cum_logprob),
                "logprobs": [
                    sorted((k, float.hex(v.logprob)) for k, v in step.items())
                    for step in seq.logprobs
                ],
                "finish": seq.finish_reason,
            }
            for seq in out.sequences
        ]
        for out in outputs
    ]


def run(beam_width, prompt_len, max_tokens, *, num_prompts=2, seed=0,
        eos_rate=0.0, tie_grid=None, ignore_eos=False, length_penalty=1.0):
    engine = FakeEngine(beam_width, eos_rate, tie_grid)
    llm = make_llm(engine)
    step_seconds = []
    orig_step = offline_mod.BeamSearchOfflineMixin._beam_search_step

    def timed_step(self, *args, **kwargs):
        engine_before = engine.seconds
        t0 = time.perf_counter()
        r = orig_step(self, *args, **kwargs)
        step_seconds.append(
            (time.perf_counter() - t0) - (engine.seconds - engine_before)
        )
        return r

    offline_mod.BeamSearchOfflineMixin._beam_search_step = timed_step
    try:
        params = BeamSearchParams(
            beam_width=beam_width,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
            length_penalty=length_penalty,
        )
        outputs = llm.beam_search(prompts_for(num_prompts, prompt_len, seed), params)
    finally:
        offline_mod.BeamSearchOfflineMixin._beam_search_step = orig_step
    return outputs, step_seconds, engine.seconds / max(engine.calls, 1)


def diff_mode(out_path):
    results = {}
    cases = []
    for seed in range(40):
        for bw in (1, 2, 4, 8):
            cases.append(dict(beam_width=bw, prompt_len=5 + seed % 7, max_tokens=6,
                              seed=seed, eos_rate=0.3, tie_grid=None))
            cases.append(dict(beam_width=bw, prompt_len=4, max_tokens=5, seed=seed,
                              eos_rate=0.4, tie_grid=0.5))  # heavy ties
            cases.append(dict(beam_width=bw, prompt_len=4, max_tokens=5, seed=seed,
                              eos_rate=0.4, tie_grid=0.25, ignore_eos=True))
            cases.append(dict(beam_width=bw, prompt_len=3, max_tokens=5, seed=seed,
                              eos_rate=0.2, length_penalty=0.0 if seed % 2 else 2.0))
    for c in cases:
        key = json.dumps(c, sort_keys=True)
        outputs, _, _ = run(**c)
        results[key] = canonical(outputs)
    json.dump(results, open(out_path, "w"))
    print(f"{len(cases)} diff cases written to {out_path}")


def bench_mode(out_path):
    rows = []
    for bw in (4, 8, 16, 32, 64):
        for prompt_len in (128, 1024, 4096):
            if bw == 64 and prompt_len == 4096:
                continue
            per_rep = []
            engine_per_step = []
            for rep in range(5):
                _, steps, eng = run(bw, prompt_len, max_tokens=12, num_prompts=1,
                                    seed=100 + rep)
                per_rep.append(statistics.median(steps[2:]) * 1e3)  # skip warmup
                engine_per_step.append(eng * 1e3)
            rows.append(dict(beam_width=bw, prompt_len=prompt_len,
                             logic_ms=statistics.median(per_rep),
                             logic_ms_reps=per_rep,
                             fake_engine_ms=statistics.median(engine_per_step)))
            print(f"B={bw:3d} prompt={prompt_len:5d}  beam logic "
                  f"{statistics.median(per_rep):8.3f} ms/step  "
                  f"(fake engine {statistics.median(engine_per_step):7.3f} ms/step)",
                  flush=True)
    json.dump(rows, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    {"diff": diff_mode, "bench": bench_mode}[mode](out)
