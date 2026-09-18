#!/usr/bin/env python3
"""Install the experimental deadline-aware prefill budget controller into an
installed vLLM (0.29.0 + tracer v2). Anchor-based; idempotent.

Env (all read at import):
  VLLM_EXP_DEADLINE_MS    deadline for prefill-containing steps (enables the controller)
  VLLM_EXP_COST_MODEL     JSON with {"features": "m0"|"m2", "mu", "sd", "w", "b", "q95"}
  VLLM_EXP_CANDIDATES     comma list of prefill budgets (default 64,...,8192)
  VLLM_EXP_FIXED_BUDGET   if set, ignore the model and use this prefill budget
                          (equal split over active prefills) -- the fixed baseline
                          run through the same code path
"""
import re, sys
path = sys.argv[1]
src = open(path).read()
if "_exp_choose_budget" in src:
    print("already installed"); sys.exit(0)

MODULE = '''
import json as _json
import os as _os

# Experiment: deadline-aware prefill budget (aggregate M0 vs geometry M2 cost model).
_EXP_DEADLINE_MS = float(_os.environ.get("VLLM_EXP_DEADLINE_MS", "0") or 0)
_EXP_FIXED_BUDGET = int(_os.environ.get("VLLM_EXP_FIXED_BUDGET", "0") or 0)
_EXP_CANDIDATES = [int(x) for x in _os.environ.get("VLLM_EXP_CANDIDATES", "64,128,256,512,1024,2048,4096,8192").split(",")]
_EXP_MODEL = _json.load(open(_os.environ["VLLM_EXP_COST_MODEL"])) if _os.environ.get("VLLM_EXP_COST_MODEL") else None


def _exp_features(kind, chunks, depths, gen_reqs, gen_kv_sum):
    tok = sum(chunks)
    total = tok + gen_reqs
    if total > 128:
        padded = total
    else:
        padded = 1
        while padded < total:
            padded *= 2
    f = [gen_reqs, gen_kv_sum / 1e4, sum(depths) / 1e4, tok / 1e3]
    if kind == "m0":
        f.append(tok * sum(depths) / 1e7)
    else:
        proxy = sum(q * (kv + (q + 1) / 2) for q, kv in zip(chunks, depths)) + gen_kv_sum
        f += [len(chunks), 1.0 if total > 128 else 0.0, padded / 1e3, proxy / 1e6]
    return f


def _exp_predict(model, f):
    return sum((x - m) / s * w for x, m, s, w in zip(f, model["mu"], model["sd"], model["w"])) + model["b"]


def _exp_choose_budget(sched, token_budget):
    """Pick the prefill token budget for this step. Returns (token_budget, decision)."""
    pre, gen_reqs, gen_kv_sum = [], 0, 0
    for r in sched.running:
        remaining = r.num_tokens_with_spec + r.num_output_placeholders - r.num_computed_tokens
        if r.num_computed_tokens < r.num_prompt_tokens:
            pre.append((remaining, r.num_computed_tokens))
        else:
            gen_reqs += 1
            gen_kv_sum += r.num_computed_tokens
    for r in sched.waiting:
        pre.append((r.num_tokens - r.num_computed_tokens, r.num_computed_tokens))
    sched._exp_threshold = 0
    if not pre:
        return token_budget, None
    n = len(pre)
    best, best_pred = 0, None
    if _EXP_FIXED_BUDGET:
        best = _EXP_FIXED_BUDGET
    else:
        for c in _EXP_CANDIDATES:
            cap = max(c // n, 1)
            chunks = [min(rem, cap) for rem, _ in pre]
            depths = [d for _, d in pre]
            pred = _exp_predict(_EXP_MODEL, _exp_features(_EXP_MODEL["features"], chunks, depths, gen_reqs, gen_kv_sum))
            if pred + _EXP_MODEL["q95"] <= _EXP_DEADLINE_MS:
                best, best_pred = c, pred
    if best == 0:
        best = _EXP_CANDIDATES[0]  # never starve prefill: smallest candidate
    cap = max(best // n, 1)
    sched._exp_threshold = cap
    budget = min(token_budget, best + gen_reqs)
    return budget, {"budget": best, "cap": cap, "n_prefill": n, "pred_ms": best_pred}

'''
anchor = "logger = init_logger(__name__)\n"
assert src.count(anchor) == 1
src = src.replace(anchor, anchor + MODULE, 1)

a2 = "        token_budget = self.max_num_scheduled_tokens\n        spec = self.vllm_config.speculative_config\n"
assert src.count(a2) == 1
src = src.replace(a2, "        token_budget = self.max_num_scheduled_tokens\n        exp_decision = None\n        if _EXP_DEADLINE_MS or _EXP_FIXED_BUDGET:\n            token_budget, exp_decision = _exp_choose_budget(self, token_budget)\n        spec = self.vllm_config.speculative_config\n", 1)

a3 = "            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:\n                num_new_tokens = self.scheduler_config.long_prefill_token_threshold\n"
assert src.count(a3) == 1
src = src.replace(a3, "            _thr = getattr(self, '_exp_threshold', 0) or self.scheduler_config.long_prefill_token_threshold\n            if 0 < _thr < num_new_tokens:\n                num_new_tokens = _thr\n", 1)

a4 = "                    threshold = self.scheduler_config.long_prefill_token_threshold\n"
assert src.count(a4) == 1
src = src.replace(a4, "                    threshold = getattr(self, '_exp_threshold', 0) or self.scheduler_config.long_prefill_token_threshold\n", 1)

a5 = "        return scheduler_output\n"
assert src.count(a5) == 1
src = src.replace(a5, "        if exp_decision is not None:\n            scheduler_output.exp_decision = exp_decision\n        return scheduler_output\n", 1)
open(path, "w").write(src)
print("installed controller into", path)
