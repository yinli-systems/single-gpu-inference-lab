# --- experiment-only SLO-aware waiting-queue order (VLLM_EXP_SLO_ORDER=1); appended to scheduler.py ---
# Online Moore-Hodgson on the waiting queue at every schedule(): requests are taken in deadline order
# (deadline = engine arrival + VLLM_EXP_SLO_TTFT_S - margin, i.e. arrival order), each costing its
# remaining prefill priced by geometry, a * L + b * (L * k0 + L (L + 1) / 2), at its exact prefix-cache
# depth k0; whenever the running total misses a deadline, the most expensive request accepted so far is
# moved to a "late" class. On-time requests get priority 0, late ones 1; the queue's tie-break is
# arrival time, so each class is served first-come first-served. Predicted prefill ms are converted
# to wall time by kappa, re-estimated online as wall time / predicted prefill ms over the last ~2 s of
# backlogged steps. Needs --scheduling-policy priority; clients send no priority. Never upstreamed.
import collections as _collections
import heapq as _heapq
import json as _json
import os as _os
import time as _time

_EXP_SLO = _os.environ.get("VLLM_EXP_SLO_ORDER") == "1"
_EXP_SLO_D = float(_os.environ.get("VLLM_EXP_SLO_TTFT_S", "5.0"))
_EXP_SLO_MARGIN = float(_os.environ.get("VLLM_EXP_SLO_MARGIN_S", "0.1"))
_EXP_SLO_A = float(_os.environ.get("VLLM_EXP_SLO_A", "10.1767"))
_EXP_SLO_B = float(_os.environ.get("VLLM_EXP_SLO_B", "0.9532"))
_EXP_SLO_LOG = _os.environ.get("VLLM_EXP_SLO_LOG")


def _exp_cost_ms(L, k0):
    return _EXP_SLO_A * L / 1e3 + _EXP_SLO_B * (L * k0 + L * (L + 1) / 2) / 1e6


class _ExpSloState:
    def __init__(self):
        self.k0 = {}
        self.kappa = 1.5
        self.win = _collections.deque()
        self.last_t = None
        self.last_pred = 0.0
        self.last_backlog = False
        self.log = open(_EXP_SLO_LOG, "a", buffering=1) if _EXP_SLO_LOG else None
        self.last_log = 0.0
        self.overhead = []


def _exp_slo_reorder(sched):
    st = sched.__dict__.get("_exp_slo")
    if st is None:
        st = sched.__dict__["_exp_slo"] = _ExpSloState()
    t0 = _time.perf_counter()
    now = _time.time()
    if st.last_t is not None and st.last_backlog:
        st.win.append((now - st.last_t, st.last_pred))
        while len(st.win) > 1 and sum(dt for dt, _ in st.win) > 2.0:
            st.win.popleft()
        pred = sum(p for _, p in st.win)
        if pred > 20.0:
            st.kappa = 0.8 * st.kappa + 0.2 * (sum(dt for dt, _ in st.win) * 1000.0 / pred)
    st.last_t = now
    queues = [q for q in (sched.waiting, sched.skipped_waiting) if hasattr(q, "_heap")]
    reqs = [r for q in queues for r in q._heap]
    live = {r.request_id for r in reqs}
    for rid in [k for k in st.k0 if k not in live]:
        del st.k0[rid]
    coord = sched.kv_cache_manager.coordinator
    t = now
    for r in sched.running:
        if r.num_computed_tokens < r.num_prompt_tokens:
            L = r.num_prompt_tokens - r.num_computed_tokens
            t += st.kappa * _exp_cost_ms(L, r.num_computed_tokens) / 1000.0
        r.priority = 0
    items = []
    for r in reqs:
        hit = st.k0.get(r.request_id)
        if hit is None or now - hit[0] > 1.0:
            try:
                k = coord.find_longest_cache_hit(r.block_hashes, r.num_tokens - 1)[1]
            except Exception:
                k = 0
            hit = st.k0[r.request_id] = (now, k)
        k0 = max(hit[1], r.num_computed_tokens)
        L = max(r.num_tokens - k0, 1)
        items.append((r.arrival_time + _EXP_SLO_D - _EXP_SLO_MARGIN, st.kappa * _exp_cost_ms(L, k0) / 1000.0, r))
    items.sort(key=lambda x: (x[0], x[2].arrival_time))
    acc, late = [], set()
    for d, p, r in items:
        if now > d:
            late.add(r.request_id)
            continue
        _heapq.heappush(acc, (-p, -r.arrival_time, r.request_id))
        t += p
        if t > d:
            negp, _, rid = _heapq.heappop(acc)
            t += negp
            late.add(rid)
    for r in reqs:
        r.priority = 1 if r.request_id in late else 0
    for q in queues:
        _heapq.heapify(q._heap)
    st.last_backlog = bool(reqs)
    st.overhead.append((_time.perf_counter() - t0) * 1000.0)
    if st.log and now - st.last_log >= 1.0:
        ov = sorted(st.overhead)
        st.log.write(_json.dumps({"t": now, "waiting": len(reqs), "late": len(late), "kappa": round(st.kappa, 3),
                                  "overhead_ms_p50": ov[len(ov) // 2], "overhead_ms_max": ov[-1], "steps": len(ov)}) + "\n")
        st.overhead = []
        st.last_log = now


def _exp_slo_account(sched, out):
    st = sched.__dict__.get("_exp_slo")
    if st is None:
        return
    pred = 0.0
    for rid, n in out.num_scheduled_tokens.items():
        r = sched.requests.get(rid)
        if r is None:
            continue
        depth = r.num_computed_tokens - n
        if depth < r.num_prompt_tokens:
            q = min(n, r.num_prompt_tokens - depth)
            pred += _exp_cost_ms(q, depth)
    st.last_pred = pred


if _EXP_SLO:
    _exp_orig_schedule = Scheduler.schedule

    def _exp_slo_schedule(self, *args, **kwargs):
        if self.policy == SchedulingPolicy.PRIORITY:
            _exp_slo_reorder(self)
        out = _exp_orig_schedule(self, *args, **kwargs)
        if self.policy == SchedulingPolicy.PRIORITY:
            _exp_slo_account(self, out)
        return out

    Scheduler.schedule = _exp_slo_schedule
