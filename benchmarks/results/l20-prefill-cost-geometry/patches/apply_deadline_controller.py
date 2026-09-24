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
  VLLM_EXP_PARTITION      equal (default; per-request cap = budget // n) or fcfs (vLLM's
                          fill-in-order chunking; the controller prices that partition and
                          sets no cap)
  VLLM_EXP_PPAS           1: P-PAS (arXiv 2608.15171) instead of a cost model: prefill tokens
                          capped at VLLM_EXP_PPAS_BCAP (2048) when N_p >= VLLM_EXP_PPAS_NTH (2)
                          running+waiting prefills and N_d > 0 running decodes, else at
                          VLLM_EXP_PPAS_BMAX (16384); no per-request cap
  VLLM_EXP_CACHE_AWARE    1: price the first VLLM_EXP_CACHE_AWARE_K (16) waiting requests at
                          their prefix-cache hit depth (pure coordinator lookup) instead of 0
  VLLM_EXP_ONLINE_MARGIN  1: replace the static q95 by the online (1-alpha) quantile of
                          realised residuals per prefill-count bucket (VLLM_EXP_ALPHA,
                          VLLM_EXP_WINDOW, VLLM_EXP_MIN_SAMPLES); the realised step time is
                          the result-ready gap attributed in update_from_output
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
# Online margin: per prefill-count bucket, the (1-alpha) quantile of realised residuals
# (result-ready gap minus prediction) over a sliding window replaces the static q95.
_EXP_ONLINE = bool(int(_os.environ.get("VLLM_EXP_ONLINE_MARGIN", "0") or 0))
_EXP_ALPHA = float(_os.environ.get("VLLM_EXP_ALPHA", "0.05"))
_EXP_WINDOW = int(_os.environ.get("VLLM_EXP_WINDOW", "64"))
_EXP_MIN_SAMPLES = int(_os.environ.get("VLLM_EXP_MIN_SAMPLES", "16"))
_EXP_RESID = {}  # bucket -> list of recent residuals (ms)
# Partition assumed when pricing a candidate budget: "equal" (per-request cap = budget // n, set
# via the long-prefill threshold) or "fcfs" (vLLM default fill-in-order; no cap is set).
_EXP_PARTITION = _os.environ.get("VLLM_EXP_PARTITION", "equal")
# P-PAS baseline (arXiv 2608.15171, Algorithm 1): cap on aggregate prefill tokens per iteration.
_EXP_PPAS = bool(int(_os.environ.get("VLLM_EXP_PPAS", "0") or 0))
_EXP_PPAS_NTH = int(_os.environ.get("VLLM_EXP_PPAS_NTH", "2"))
_EXP_PPAS_BMAX = int(_os.environ.get("VLLM_EXP_PPAS_BMAX", "16384"))
_EXP_PPAS_BCAP = int(_os.environ.get("VLLM_EXP_PPAS_BCAP", "2048"))
# Price waiting requests at their prefix-cache hit depth (the scheduler only learns it later).
_EXP_CACHE_AWARE = bool(int(_os.environ.get("VLLM_EXP_CACHE_AWARE", "0") or 0))
_EXP_CACHE_AWARE_K = int(_os.environ.get("VLLM_EXP_CACHE_AWARE_K", "16"))


def _exp_cached_depth(sched, r):
    """Prefix-cache hit length of a waiting request (0 if unknown); read-only lookup."""
    try:
        mgr = sched.kv_cache_manager
        if r.num_computed_tokens or not mgr.prefix_cache_lookup_enabled(r):
            return r.num_computed_tokens
        _, n, _ = mgr.coordinator.find_longest_cache_hit(r.block_hashes, r.num_tokens - 1)
        return int(n)
    except Exception:
        return r.num_computed_tokens


def _exp_chunks(c, pre):
    if _EXP_PARTITION == "fcfs":
        out, left = [], c
        for rem, _ in pre:
            take = min(rem, left); out.append(take); left -= take
        return out
    cap = max(c // len(pre), 1)
    return [min(rem, cap) for rem, _ in pre]


def _exp_bucket(n):
    return min(int(n), 8)


def _exp_margin(n):
    """Static q95 from the calibration fit, or the online bucket quantile once populated."""
    if _EXP_ONLINE:
        rs = _EXP_RESID.get(_exp_bucket(n))
        if rs and len(rs) >= _EXP_MIN_SAMPLES:
            xs = sorted(rs)
            k = min(len(xs) - 1, int(round((1 - _EXP_ALPHA) * (len(xs) - 1))))
            return xs[k], True
    return _EXP_MODEL["q95"], False


def _exp_observe(sched, scheduler_output):
    """Called from update_from_output: attribute the result-ready gap to the batch it belongs to."""
    import time as _time
    now = _time.monotonic()
    last = getattr(sched, "_exp_last_output_t", None)
    sched._exp_last_output_t = now
    dec = getattr(scheduler_output, "exp_decision", None)
    if last is None or dec is None or dec.get("pred_ms") is None:
        return
    gap_ms = (now - last) * 1e3
    if gap_ms > 4 * (dec["pred_ms"] + 50):  # idle gap, not a step time
        return
    b = _exp_bucket(dec["n_prefill"])
    rs = _EXP_RESID.setdefault(b, [])
    rs.append(gap_ms - dec["pred_ms"])
    if len(rs) > _EXP_WINDOW:
        del rs[0]
    dec["actual_gap_ms"] = gap_ms


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
        if kind == "m2n":  # M2 without the aggregate prefill-KV sum (pre-registered 2026-09-23)
            del f[2]
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
    for i, r in enumerate(sched.waiting):
        d = _exp_cached_depth(sched, r) if _EXP_CACHE_AWARE and i < _EXP_CACHE_AWARE_K else r.num_computed_tokens
        pre.append((r.num_tokens - d, d))
    sched._exp_threshold = 0
    if not pre:
        return token_budget, None
    if _EXP_PPAS:
        cap = _EXP_PPAS_BCAP if (len(pre) >= _EXP_PPAS_NTH and gen_reqs > 0) else _EXP_PPAS_BMAX
        return min(token_budget, cap + gen_reqs), {"budget": cap, "cap": 0, "n_prefill": len(pre), "ppas": True}
    n = len(pre)
    best, best_pred = 0, None
    margin, online = (0.0, False) if _EXP_FIXED_BUDGET else _exp_margin(n)
    if _EXP_FIXED_BUDGET:
        best = _EXP_FIXED_BUDGET
    else:
        for c in _EXP_CANDIDATES:
            chunks = [x for x in _exp_chunks(c, pre) if x > 0]
            depths = [d for (_, d), x in zip(pre, _exp_chunks(c, pre)) if x > 0]
            pred = _exp_predict(_EXP_MODEL, _exp_features(_EXP_MODEL["features"], chunks, depths, gen_reqs, gen_kv_sum))
            if pred + margin <= _EXP_DEADLINE_MS:
                best, best_pred = c, pred
    if best == 0:
        best = _EXP_CANDIDATES[0]  # never starve prefill: smallest candidate
        if _EXP_MODEL is not None:
            ch = _exp_chunks(best, pre)
            best_pred = _exp_predict(_EXP_MODEL, _exp_features(_EXP_MODEL["features"], [x for x in ch if x > 0], [d for (_, d), x in zip(pre, ch) if x > 0], gen_reqs, gen_kv_sum))
    cap = max(best // n, 1)
    sched._exp_threshold = 0 if _EXP_PARTITION == "fcfs" else cap
    budget = min(token_budget, best + gen_reqs)
    return budget, {"budget": best, "cap": sched._exp_threshold, "n_prefill": n, "pred_ms": best_pred, "margin_ms": margin, "online": online}

'''
anchor = "logger = init_logger(__name__)\n"
assert src.count(anchor) == 1
src = src.replace(anchor, anchor + MODULE, 1)

a2 = "        token_budget = self.max_num_scheduled_tokens\n        spec = self.vllm_config.speculative_config\n"
assert src.count(a2) == 1
src = src.replace(a2, "        token_budget = self.max_num_scheduled_tokens\n        exp_decision = None\n        if _EXP_DEADLINE_MS or _EXP_FIXED_BUDGET or _EXP_PPAS:\n            token_budget, exp_decision = _exp_choose_budget(self, token_budget)\n        spec = self.vllm_config.speculative_config\n", 1)

a3 = "            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:\n                num_new_tokens = self.scheduler_config.long_prefill_token_threshold\n"
assert src.count(a3) == 1
src = src.replace(a3, "            _thr = getattr(self, '_exp_threshold', 0) or self.scheduler_config.long_prefill_token_threshold\n            if 0 < _thr < num_new_tokens:\n                num_new_tokens = _thr\n", 1)

a4 = "                    threshold = self.scheduler_config.long_prefill_token_threshold\n"
assert src.count(a4) == 1
src = src.replace(a4, "                    threshold = getattr(self, '_exp_threshold', 0) or self.scheduler_config.long_prefill_token_threshold\n", 1)

a6 = "    ) -> dict[int, EngineCoreOutputs]:\n        sampled_token_ids = model_runner_output.sampled_token_ids\n"
assert src.count(a6) == 1
src = src.replace(a6, "    ) -> dict[int, EngineCoreOutputs]:\n        if _EXP_DEADLINE_MS or _EXP_FIXED_BUDGET:\n            _exp_observe(self, scheduler_output)\n        sampled_token_ids = model_runner_output.sampled_token_ids\n", 1)

a5 = "        return scheduler_output\n"
assert src.count(a5) == 1
src = src.replace(a5, "        if exp_decision is not None:\n            scheduler_output.exp_decision = exp_decision\n        return scheduler_output\n", 1)
open(path, "w").write(src)
print("installed controller into", path)
