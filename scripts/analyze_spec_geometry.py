#!/usr/bin/env python3
"""Draft-side upper bound for per-request heterogeneous speculation.

From the spec-geometry traces, per (condition, class, B) over the *co-decoding
window* (steps with no prefill work and all B requests present):

  step_ms        model-step CUDA time (runner trace, sequence-joined)
  draft_ms       CUDA time of speculator.propose inside the step
  tokens/step    generated tokens per step = sum over rows of (1 + accepted)
  goodput        tokens/step / step_ms  (tok/s of the batch)
  acc/row        accepted draft tokens per row per step (from the scheduler's spec trace)

and the three-way comparison the go/no-go depends on:

  A  fixed K for every row                (measured: the K=7 condition)
  B  fixed K + adaptive verification      (measured: the adaptive condition)
  C  oracle per-request pre-draft K       (upper bound: B minus the draft time
     attributable to rows whose oracle K is 0, i.e. rows of a class whose
     per-row goodput is higher without speculation in this batch; draft time is
     taken as proportional to drafted rows)

Coverage: C is a composition, labelled as such; A and B are measured.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from step_trace_join import load_joined


def load_spec(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, nargs="+", required=True, help="spec_geometry JSON files")
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    table = {}
    print("| condition | class | B | co-decode steps | step ms p50 | draft ms p50 (share) | tokens/step | goodput tok/s | acc/row/step | rows classes |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for rf in args.results:
        rec = json.load(open(rf))
        for cond, c in rec["conditions"].items():
            for rep in sorted({b["repeat"] for b in c["batches"]}):
                stem = f"{rf.stem}-{cond}-r{rep}"
                it_path = args.trace_dir / f"{stem}.jsonl"
                if not it_path.exists():
                    continue
                rows, info = load_joined(it_path)
                spec = load_spec(args.trace_dir / f"{stem}.spec.jsonl")
                spec_t = np.array([s["t"] for s in spec]) if spec else np.array([])
                for b in c["batches"]:
                    if b["repeat"] != rep:
                        continue
                    B = b["B"]
                    t0 = b["t_all_first"]; t1 = b["t_end"]
                    dec = [r for r in rows if t0 <= r["t"] <= t1 and r["ctx_tokens"] == 0 and np.isfinite(r["cuda_ms"])]
                    win = [r for r in dec if r["gen_reqs"] == B]
                    B_eff = B
                    if len(win) < 10:
                        # rows finish at different times (high acceptance vs prefill stagger): use the
                        # largest co-decoding batch that has enough steps, and report it as B_eff
                        counts = defaultdict(int)
                        for r in dec:
                            counts[r["gen_reqs"]] += 1
                        cands = [g for g, n in counts.items() if n >= 10]
                        if not cands:
                            continue
                        B_eff = max(cands)
                        win = [r for r in dec if r["gen_reqs"] == B_eff]
                    step = np.array([r["cuda_ms"] for r in win])
                    draft = np.array([r.get("draft_ms", 0.0) or 0.0 for r in win])
                    # accepted tokens per step from the spec trace inside the window
                    if len(spec_t):
                        sel = [s for s, t in zip(spec, spec_t) if t0 <= t <= t1 and len(s["acc"]) == B_eff]
                    else:
                        sel = []
                    acc_rows_all = [a for s in sel for a in s["acc"]] if sel else []
                    if sel:
                        acc_per_step = np.array([sum(a[3] for a in s["acc"]) for s in sel])
                        tokens_per_step = float(np.mean(acc_per_step)) + B_eff
                        acc_row = float(np.mean(acc_per_step)) / B_eff
                        # per-row acceptance by prompt class (long vs short by depth at window start)
                        by_depth = defaultdict(list)
                        for s in sel:
                            for a in s["acc"]:
                                by_depth["long" if a[1] > 3000 else "short"].append(a[3])
                        cls_acc = {k: float(np.mean(v)) for k, v in by_depth.items()}
                    else:
                        tokens_per_step = float(B_eff); acc_row = 0.0; cls_acc = {}
                    good = tokens_per_step / (np.median(step) / 1e3)
                    key = (cond, b["class"], B, rep)
                    accs = np.array([a[3] for a in acc_rows_all]) if acc_rows_all else np.array([0.0])
                    depth_bins = defaultdict(list)
                    for a in acc_rows_all:
                        depth_bins[int(a[1] // 1024)].append(a[3])
                    table[key] = {"n_steps": len(win), "B_eff": B_eff, "step_ms_p50": float(np.median(step)), "step_ms_p95": float(np.quantile(step, .95)),
                                  "acc_quantiles": {q: float(np.quantile(accs, q / 100)) for q in (25, 50, 75, 90)}, "acc_frac_zero": float(np.mean(accs == 0)),
                                  "acc_by_depth_k": {str(k): [len(v), float(np.mean(v))] for k, v in sorted(depth_bins.items())},
                                  "draft_ms_p50": float(np.median(draft)), "draft_share": float(np.median(draft) / np.median(step)),
                                  "tokens_per_step": tokens_per_step, "goodput": good, "acc_per_row": acc_row, "class_acc": cls_acc,
                                  "wall_tok_per_s": b["tok_per_s"], "prompt_tokens_p50": float(np.median([r["prompt_tokens"] for r in b["requests"]]))}
                    print(f"| {cond} | {b['class']} | {B}{'' if B_eff == B else f' (eff {B_eff})'} | {len(win)} | {np.median(step):.1f} | {np.median(draft):.1f} ({np.median(draft)/np.median(step)*100:.0f}%) | "
                          f"{tokens_per_step:.1f} | {good:.0f} | {acc_row:.2f} | {' '.join(f'{k}:{v:.2f}' for k, v in cls_acc.items())} |")

    def cell(cond, cls, B):
        # only windows where the full batch co-decodes; reduced-batch fallbacks are excluded from the tables
        return [v for (c, k, b, r), v in table.items() if c == cond and k == cls and b == B and v["B_eff"] == B]

    print("\n## T1. draft share of the step (median over co-decode steps; mean over repeats)")
    conds_sorted = sorted({k[0] for k in table})
    classes_sorted = sorted({k[1] for k in table}); sizes_sorted = sorted({k[2] for k in table})
    for cond in conds_sorted:
        if not any(v["draft_ms_p50"] > 0 for (c, _, _, _), v in table.items() if c == cond):
            continue
        print(f"\n{cond}: draft ms / step ms (share)")
        print("| class | " + " | ".join(f"B={b}" for b in sizes_sorted) + " |"); print("| --- |" + " ---: |" * len(sizes_sorted))
        for cls in classes_sorted:
            row = []
            for b in sizes_sorted:
                vs = cell(cond, cls, b)
                row.append(f"{np.mean([v['draft_ms_p50'] for v in vs]):.1f} / {np.mean([v['step_ms_p50'] for v in vs]):.1f} ({np.mean([v['draft_share'] for v in vs])*100:.0f}%)" if vs else "—")
            print(f"| {cls} | " + " | ".join(row) + " |")

    print("\n## T2. accepted-prefix length per row per step: P25 / P50 / P75 / P90, and share of rows with 0 accepted")
    print("| condition | class | B | P25 | P50 | P75 | P90 | zero | mean |"); print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for (cond, cls, B, rep), v in sorted(table.items()):
        if rep != 0 or v["draft_ms_p50"] == 0 or v["B_eff"] != B:
            continue
        q = v["acc_quantiles"]
        print(f"| {cond} | {cls} | {B} | {q[25]:.0f} | {q[50]:.0f} | {q[75]:.0f} | {q[90]:.0f} | {v['acc_frac_zero']*100:.0f}% | {v['acc_per_row']:.2f} |")

    print("\n## T3. mixed batches: mean accepted per row per step by KV depth (1k bins; n rows)")
    for (cond, cls, B, rep), v in sorted(table.items()):
        if rep != 0 or not cls.startswith("mix") or v["draft_ms_p50"] == 0:
            continue
        print(f"{cond} {cls} B={B}: " + "; ".join(f"{k}k: {m:.2f} (n={n})" for k, (n, m) in v["acc_by_depth_k"].items()))

    print("\n## T4. fixed K7 vs adaptive verification (per class, B; mean over repeats)")
    print("| class | B | cond | goodput tok/s | step ms | draft ms | non-draft ms | tokens/step | acc/row |"); print("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for cls in classes_sorted:
        for b in sizes_sorted:
            for cond in conds_sorted:
                vs = cell(cond, cls, b)
                if not vs:
                    continue
                m = lambda k: float(np.mean([v[k] for v in vs]))
                print(f"| {cls} | {b} | {cond} | {m('goodput'):.0f} | {m('step_ms_p50'):.1f} | {m('draft_ms_p50'):.1f} | {m('step_ms_p50') - m('draft_ms_p50'):.1f} | {m('tokens_per_step'):.1f} | {m('acc_per_row'):.2f} |")

    # A / B / C composition per (class, B, rep)
    print("\n## A (fixed K) vs B (fixed K + adaptive verification) vs C (oracle per-request pre-draft K; composed upper bound)")
    print("| class | B | no-spec goodput | A goodput | B goodput | rows oracle K=0 | draft share (B) | C upper bound | C/B | C/best(A,B) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    comp = {}
    conds = {k[0] for k in table}
    nospec = next((c for c in conds if "nospec" in c), None)
    fixed = next((c for c in conds if "adaptive" not in c and "nospec" not in c and "eagle" not in c), None)
    adapt = next((c for c in conds if "adaptive" in c), None)
    for (cond, cls, B, rep), v in sorted(table.items()):
        if cond != (adapt or fixed):
            continue
        ns = table.get((nospec, cls, B, rep)) if nospec else None
        a = table.get((fixed, cls, B, rep)) if fixed else None
        bb = table.get((adapt, cls, B, rep)) if adapt else None
        ref = bb or a
        if not (ns and ref):
            continue
        # oracle K=0 rows: a class is better off without speculation when its per-row token rate
        # under no-spec (1 token per no-spec step) exceeds its per-row rate under spec
        # ((1+acc_class) per spec step); compared at the batch's step times
        rate_ns = 1.0 / ns["step_ms_p50"]
        k0_rows = 0
        for k, acc in ref["class_acc"].items():
            n_rows = B // 2 if cls.startswith("mix") else B
            if (1.0 + acc) / ref["step_ms_p50"] < rate_ns:
                k0_rows += n_rows
        frac0 = k0_rows / B
        # C: those rows skip the draft pass (draft time proportional to rows) and generate 1 token per step
        c_step = ref["step_ms_p50"] - ref["draft_ms_p50"] * frac0
        c_tokens = ref["tokens_per_step"] - sum(acc * (B // 2 if cls.startswith("mix") else B) for k, acc in ref["class_acc"].items()
                                                 if (1.0 + acc) / ref["step_ms_p50"] < rate_ns)
        c_good = c_tokens / (c_step / 1e3)
        best = max(x["goodput"] for x in (a, bb) if x)
        comp[f"{cls}-B{B}-r{rep}"] = {"nospec": ns["goodput"], "A": a["goodput"] if a else None, "B": bb["goodput"] if bb else None,
                                      "k0_rows_frac": frac0, "draft_share": ref["draft_share"], "C_upper": c_good, "C_over_B": c_good / ref["goodput"], "C_over_best": c_good / best}
        print(f"| {cls} | {B} | {ns['goodput']:.0f} | {a['goodput'] if a else float('nan'):.0f} | {bb['goodput'] if bb else float('nan'):.0f} | {k0_rows}/{B} | "
              f"{ref['draft_share']*100:.0f}% | {c_good:.0f} | {c_good/ref['goodput']:.3f} | {c_good/best:.3f} |")
    args.output.write_text(json.dumps({"cells": {"|".join(map(str, k)): v for k, v in table.items()}, "composition": comp}, indent=2) + "\n")


if __name__ == "__main__":
    main()
