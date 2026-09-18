"""Join engine iteration traces with runner step traces (tracer v2).

The runner records one CUDA-event pair per executed batch; the engine trace
records one row per scheduler iteration. Runner steps executed before the
engine trace started (server warm-up) shift the indices by a constant, so the
join is by *sequence*: pick the offset that maximises token-count equality
(``steps[k+o].num_tokens == iters[k].total_tokens``) and require near-total
agreement. Joining by nearest timestamp is wrong under async scheduling: the
engine stamps its iteration ~1 step after the runner starts the batch, and
around long prefill steps the nearest stamp belongs to the neighbouring batch
(that mis-join produced an apparent −25 ms gap-vs-CUDA discrepancy at
chunk 2048 that does not exist under the sequence join).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_joined(iter_path: Path, max_offset: int = 12, min_match: float = 0.98):
    it = [json.loads(l) for l in iter_path.read_text().splitlines() if l.strip()]
    sp = iter_path.with_name(iter_path.name.replace(".jsonl", ".steps.jsonl"))
    st = [json.loads(l) for l in sp.read_text().splitlines() if l.strip()] if sp.exists() else []
    ready = np.array([r["t"] + r["ms"] / 1e3 for r in it])
    gap = np.full(len(it), np.nan)
    gap[1:] = np.diff(ready) * 1e3  # result-ready gap = host-observed time of the batch iteration k waited on
    best = (None, -1)
    for o in range(max_offset + 1):
        m = sum(1 for k, r in enumerate(it) if k + o < len(st) and st[k + o]["num_tokens"] == r["total_tokens"])
        if m > best[1]:
            best = (o, m)
    o, m = best
    n_cmp = min(len(it), max(len(st) - o, 0))
    ok = st and n_cmp and m / n_cmp >= min_match
    rows = []
    for k, r in enumerate(it):
        s = st[k + o] if ok and k + o < len(st) and st[k + o]["num_tokens"] == r["total_tokens"] else None
        rows.append({**r, "gap_ms": gap[k], "cuda_ms": s["cuda_ms"] if s else np.nan,
                     "padded_tokens": s["padded_tokens"] if s else r["total_tokens"],
                     "cg_mode": s["cg_mode"] if s else "?", "runner_step": s["step"] if s else None,
                     "draft_ms": s.get("draft_ms") if s else None, "draft_rows": s.get("draft_rows") if s else None})
    info = {"iterations": len(it), "runner_steps": len(st), "offset": o, "matched": m, "compared": n_cmp,
            "match_frac": (m / n_cmp) if n_cmp else 0.0, "index_gaps": int(sum(1 for a, b in zip(it, it[1:]) if b["i"] != a["i"] + 1))}
    return rows, info
