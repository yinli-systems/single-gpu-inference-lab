#!/usr/bin/env python3
"""Tracer v5 (DP/EP waves): per-rank host arrival stamps and within-rank CUDA
durations of the expert-parallel collectives in the allgather_reducescatter
all2all path (vLLM 0.29.0). Anchor-based, idempotent.

Env: VLLM_EXP_EP_TRACE=<prefix> -> <prefix>.<dp_rank>.jsonl with one row per
collective call:
  {"seq", "op": "dispatch"|"combine", "rank", "t_host_ns" (monotonic_ns before
   enqueue, shared timebase on one node), "sizes": tokens per DP rank in this
   wave, "local": this rank's tokens, "cuda_ms" (event pair around the
   collective, this rank's stream only)}
Cross-rank arrival skew = max_r t_host - min_r t_host for the same (seq, op);
never subtract CUDA event times across GPUs.
usage: apply_tracer_v5_ep.py <site-packages/vllm>
"""
import sys
sp = sys.argv[1]
p = f"{sp}/distributed/device_communicators/all2all.py"
s = open(p).read()
if "_EXP_EP_TRACE" in s:
    print("already"); sys.exit(0)

hdr = '''
import json as _ejson
import os as _eos
import time as _etime
from collections import deque as _edeque

_EXP_EP_TRACE = _eos.environ.get("VLLM_EXP_EP_TRACE")


class _ExpEpTracer:
    """Host arrival stamp + per-rank CUDA duration for each EP collective."""

    def __init__(self, prefix):
        self.prefix = prefix; self.fh = None; self.rank = None
        self.pending = _edeque(); self.seq = 0

    def begin(self, op, rank, sizes, local):
        import torch
        if self.fh is None:
            self.rank = rank
            self.fh = open(f"{self.prefix}.{rank}.jsonl", "a", buffering=1)
        t = _etime.monotonic_ns()
        ev = torch.cuda.Event(enable_timing=True); ev.record()
        rec = {"seq": self.seq, "op": op, "rank": rank, "t_host_ns": t, "sizes": list(sizes), "local": int(local), "_s": ev}
        self.seq += 1
        return rec

    def end(self, rec):
        import torch
        ev = torch.cuda.Event(enable_timing=True); ev.record()
        rec["_e"] = ev
        self.pending.append(rec)
        while self.pending and self.pending[0]["_e"].query():
            r = self.pending.popleft()
            r["cuda_ms"] = round(r["_s"].elapsed_time(r["_e"]), 3)
            del r["_s"], r["_e"]
            self.fh.write(_ejson.dumps(r) + "\\n")


_EXP_EP = _ExpEpTracer(_EXP_EP_TRACE) if _EXP_EP_TRACE else None
'''
anchor = "class AgRsAll2AllManager(All2AllManagerBase):\n"
assert s.count(anchor) == 1
s = s.replace(anchor, hdr + "\n\n" + anchor, 1)

# dispatch: wrap the all_gatherv in dispatch()
a1 = '''        tensors_to_gather = [hidden_states, topk_weights, topk_ids]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        gathered_tensors = dist_group.all_gatherv(
            tensors_to_gather,
            dim=0,
            sizes=sizes,
        )
'''
assert s.count(a1) == 1
s = s.replace(a1, '''        tensors_to_gather = [hidden_states, topk_weights, topk_ids]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        _rec = _EXP_EP.begin("dispatch", dist_group.rank_in_group, sizes, hidden_states.shape[0]) if _EXP_EP is not None else None
        gathered_tensors = dist_group.all_gatherv(
            tensors_to_gather,
            dim=0,
            sizes=sizes,
        )
        if _rec is not None:
            _EXP_EP.end(_rec)
''', 1)
a2 = '''        hidden_states = dist_group.reduce_scatterv(hidden_states, dim=0, sizes=sizes)
        return hidden_states
'''
assert s.count(a2) == 1
s = s.replace(a2, '''        _rec = _EXP_EP.begin("combine", dist_group.rank_in_group, sizes, sizes[dist_group.rank_in_group]) if _EXP_EP is not None else None
        hidden_states = dist_group.reduce_scatterv(hidden_states, dim=0, sizes=sizes)
        if _rec is not None:
            _EXP_EP.end(_rec)
        return hidden_states
''', 1)
open(p, "w").write(s)
print("ep tracer installed")
