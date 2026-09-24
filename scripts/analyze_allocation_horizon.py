#!/usr/bin/env python3
"""Whole-horizon check of per-step prefill allocation policies under a per-step deadline.

Supports section 9 of docs/when-token-budgets-lie.md. The clock is the A100 M2n controller model
(fit on one-prefill steps), with its q95 margin, as in the live controller. For three prefill
scenarios it runs the deadline controller to completion under five within-step allocation
policies and reports makespan and mean TTFT. It then reports the per-step saving a per-step
oracle would claim on the equal-split trajectory, which is not realizable over the horizon:
the attention work of finishing a set of prefills does not depend on how it is split across
steps. Run from the repository root.
"""
import json, sys
sys.path.insert(0, "scripts")
import replay_prefill_controller as R
M = json.load(open("benchmarks/results/a100-prefill-live-controller/models/m2n-one.json"))
pred = lambda chunks, depths, gk: sum((x-m)/s*w for x, m, s, w in zip(R.m2n_feats(R.state_row(chunks, depths, len(gk), gk)), M["mu"], M["sd"], M["w"])) + M["b"]
D, GRAN = 65.0, 16
GK = [4096 + 256*i for i in range(8)]  # 8 decoders

def alloc(policy, Q, rem, dep):
    act = [i for i in range(len(rem)) if rem[i] > 0]
    q = [0]*len(rem)
    if policy == "equal":            # current controller: equal split, capped by remaining, leftover redistributed
        left = Q
        while left > 0:
            live = [i for i in act if rem[i]-q[i] > 0]
            if not live: break
            share = max(left // len(live), 1)
            for i in live:
                g = min(share, rem[i]-q[i], left); q[i] += g; left -= g
                if left <= 0: break
    elif policy == "shallow":        # reviewer's greedy: next token to smallest k_i + q_i
        for _ in range(Q // GRAN):
            live = [i for i in act if rem[i]-q[i] > 0]
            if not live: break
            i = min(live, key=lambda i: dep[i] + q[i]); q[i] += min(GRAN, rem[i]-q[i])
    elif policy in ("fcfs", "srpt_tokens", "srpt_cost"):   # sequential: fill in priority order
        key = {"fcfs": lambda i: i, "srpt_tokens": lambda i: rem[i],
               "srpt_cost": lambda i: rem[i]*dep[i] + rem[i]*(rem[i]+1)/2}[policy]
        left = Q
        for i in sorted(act, key=key):
            g = min(rem[i], left); q[i] = g; left -= g
            if left <= 0: break
    return q

def run(policy, k0, r0):
    rem, dep = list(r0), list(k0); t = 0.0; steps = 0; fin = [None]*len(r0); toks = 0
    while any(rem):
        best = None
        for Q in range(GRAN, 8193, GRAN):      # outer controller: largest Q whose predicted step fits
            q = alloc(policy, Q, rem, dep)
            ch = [x for x in q if x > 0]
            if not ch: break
            if pred(ch, [dep[i] for i in range(len(q)) if q[i] > 0], GK) + M["q95"] <= D: best = q
            else: break
            if sum(q) < Q: break               # nothing more to add
        if best is None: best = alloc(policy, GRAN, rem, dep)
        ch = [x for x in best if x > 0]; ms = pred(ch, [dep[i] for i in range(len(best)) if best[i] > 0], GK)
        t += ms; steps += 1; toks += sum(best)
        for i, x in enumerate(best):
            if x: rem[i] -= x; dep[i] += x
            if rem[i] == 0 and fin[i] is None: fin[i] = t
    return t, steps, sum(fin)/len(fin), max(fin)

scen = {
  "4 x 16k fresh (live workload)":            ([0]*4, [16384]*4),
  "prefix-cached mix: depths 0/4k/12k/24k, 4k each": ([0, 4096, 12288, 24576], [4096]*4),
  "agent resume: 1 x 16k fresh + 3 x 1k at 20k depth": ([0, 20480, 20480, 20480], [16384, 1024, 1024, 1024]),
}
for name, (k0, r0) in scen.items():
    print(f"## {name}")
    base = None
    for p in ("equal", "shallow", "fcfs", "srpt_tokens", "srpt_cost"):
        t, n, mean_ttft, mk = run(p, k0, r0)
        base = base or (mk, mean_ttft)
        print(f"  {p:12s} makespan {mk/1000:6.2f} s ({mk/base[0]:.3f}x)  steps {n:4d}  mean TTFT {mean_ttft/1000:6.2f} s ({mean_ttft/base[1]:.3f}x)")

print("\n## what the proposed per-step fixed-Q oracle would report (on the equal-split trajectory)")
for name, (k0, r0) in scen.items():
    rem, dep = list(r0), list(k0); claimed = 0.0; total = 0.0
    while any(rem):
        best = None
        for Q in range(GRAN, 8193, GRAN):
            q = alloc("equal", Q, rem, dep); ch = [x for x in q if x > 0]
            if not ch: break
            if pred(ch, [dep[i] for i in range(len(q)) if q[i] > 0], GK) + M["q95"] <= D: best = q
            else: break
            if sum(q) < Q: break
        if best is None: best = alloc("equal", GRAN, rem, dep)
        Q = sum(best)
        cur = pred([x for x in best if x], [dep[i] for i in range(len(best)) if best[i]], GK)
        opt_q = alloc("shallow", Q, rem, dep)
        opt = pred([x for x in opt_q if x], [dep[i] for i in range(len(opt_q)) if opt_q[i]], GK)
        claimed += cur - opt; total += cur
        for i, x in enumerate(best):
            if x: rem[i] -= x; dep[i] += x
    print(f"  {name:52s} claimed per-step saving {claimed/total*100:5.1f}% of prefill time (real makespan change with the shallow policy: see above)")
