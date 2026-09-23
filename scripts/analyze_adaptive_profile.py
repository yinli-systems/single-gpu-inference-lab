#!/usr/bin/env python3
"""Does the startup-static adaptive-verification cost profile mis-price real
context, and does the mispricing change the control decision?

Input: campaign25-style results — one spec_geometry JSON per (profile context,
repeat) with a single condition "adaptive", plus tracer-v4 step traces carrying
the per-step decision (av_budget, av_max_budget, av_pred_ms, av_est_accepted).

Per cell (profile_ctx, actual class, B), over the co-decoding window:
  goodput, step ms, draft ms
  budget: chosen draft-slot budget p50 / mean, share of steps at the max budget
  cost prediction error  = actual CUDA step ms - predicted ms   (signed p5 / p50 / p95)
  accepted-token prediction error = actual accepted/step - estimated
Then per (actual class, B):
  matched profile   = the profile whose context is closest to the actual context
  decision regret   = (goodput_matched - goodput_profile) / goodput_matched
  budget disagreement = (budget_profile - budget_matched) / max_budget  (signed: + over, - under)
  wrong-direction rate = share of steps where the static profile's budget is on the
                         other side of the matched profile's median budget by >= 1 tier
                         (tier = max_budget / 8, i.e. one draft slot per row on average)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from step_trace_join import load_joined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, nargs="+", required=True)
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cells = defaultdict(list)  # (profile_ctx, class, B) -> list of per-repeat dicts
    for rf in args.results:
        m = re.match(r"pc(\d+)-r(\d+)", rf.stem)
        if not m:
            continue
        pc, rep = int(m.group(1)), int(m.group(2))
        rec = json.load(open(rf))
        for cond, c in rec["conditions"].items():
            stem = f"{rf.stem}-{cond}-r0"
            rows, info = load_joined(args.trace_dir / f"{stem}.jsonl")
            sp = args.trace_dir / f"{stem}.spec.jsonl"
            spec = [json.loads(l) for l in sp.read_text().splitlines() if l.strip()] if sp.exists() else []
            spec_t = np.array([s["t"] for s in spec])
            # steps need the raw runner rows for av_* fields: re-read and index by runner step
            st = {}
            for l in (args.trace_dir / f"{stem}.steps.jsonl").read_text().splitlines():
                if l.strip():
                    d = json.loads(l); st[d["step"]] = d
            for b in c["batches"]:
                B = b["B"]; t0, t1 = b["t_all_first"], b["t_end"]
                dec = [r for r in rows if t0 <= r["t"] <= t1 and r["ctx_tokens"] == 0 and np.isfinite(r["cuda_ms"]) and r["runner_step"] in st]
                win = [r for r in dec if r["gen_reqs"] == B]
                B_eff = B
                if len(win) < 10:  # prefill stagger vs fast completion: use the largest co-decoding batch with enough steps
                    counts = defaultdict(int)
                    for r in dec:
                        counts[r["gen_reqs"]] += 1
                    cands = [g for g, n in counts.items() if n >= 10]
                    if not cands:
                        continue
                    B_eff = max(cands); win = [r for r in dec if r["gen_reqs"] == B_eff]
                srows = [st[r["runner_step"]] for r in win]
                if not all("av_budget" in s for s in srows):
                    continue
                cuda = np.array([r["cuda_ms"] for r in win]); draft = np.array([r["draft_ms"] or 0 for r in win])
                pred = np.array([s["av_pred_ms"] for s in srows]); budget = np.array([s["av_budget"] for s in srows]); maxb = np.array([s["av_max_budget"] for s in srows])
                est_acc = np.array([s["av_est_accepted"] for s in srows])
                sel = [s for s, t in zip(spec, spec_t) if t0 <= t <= t1 and len(s["acc"]) == B_eff]
                acc_step = np.array([sum(a[3] for a in s["acc"]) for s in sel]) if sel else np.array([np.nan])
                tokens_per_step = float(np.nanmean(acc_step)) + B_eff
                err = cuda - pred
                cells[(pc, b["class"], B)].append({
                    "rep": rep, "n": len(win), "B_eff": B_eff, "goodput": tokens_per_step / (np.median(cuda) / 1e3), "step_ms": float(np.median(cuda)), "draft_ms": float(np.median(draft)),
                    "tokens_per_step": tokens_per_step, "acc_per_row": (tokens_per_step - B_eff) / B_eff,
                    "budget_p50": float(np.median(budget)), "budget_mean": float(budget.mean()), "max_budget": float(np.median(maxb)), "at_max_frac": float(np.mean(budget >= maxb)),
                    "budget_steps": budget.tolist(),
                    "pred_ms_p50": float(np.median(pred)), "err_p5": float(np.quantile(err, .05)), "err_p50": float(np.median(err)), "err_p95": float(np.quantile(err, .95)),
                    "acc_est_err": float(np.nanmean(acc_step) - B * 0 - (est_acc.mean() - B)) if sel else None,
                    "prompt_tokens_p50": float(np.median([r["prompt_tokens"] for r in b["requests"]])),
                })
    ctx_of = lambda cls: int(re.search(r"ctx(\d+)", cls).group(1))
    profiles = sorted({k[0] for k in cells})
    print("## Per cell (mean over repeats): goodput, step/draft ms, chosen budget, cost-prediction error")
    print("| actual class | B | profile ctx | goodput | step ms | draft ms | acc/row | budget p50 / max (at max) | pred ms p50 | err p5 / p50 / p95 (actual − pred) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    summary = {}
    for (pc, cls, B), reps in sorted(cells.items(), key=lambda x: (ctx_of(x[0][1]), x[0][1], x[0][2], x[0][0])):
        m = lambda k: float(np.mean([r[k] for r in reps]))
        summary[f"{pc}|{cls}|{B}"] = {k: m(k) for k in reps[0] if isinstance(reps[0][k], (int, float)) and reps[0][k] is not None and k != "rep"}
        summary[f"{pc}|{cls}|{B}"]["reps"] = len(reps)
        beff = "" if m("B_eff") == B else f" (eff {m('B_eff'):.0f})"
        print(f"| {cls} | {B}{beff} | {pc} | {m('goodput'):.0f} | {m('step_ms'):.1f} | {m('draft_ms'):.1f} | {m('acc_per_row'):.2f} | {m('budget_p50'):.0f} / {m('max_budget'):.0f} ({m('at_max_frac')*100:.0f}%) | {m('pred_ms_p50'):.1f} | {m('err_p5'):+.1f} / {m('err_p50'):+.1f} / {m('err_p95'):+.1f} |")

    print("\n## Decision regret vs the matched profile (profile whose context is closest to the actual context)")
    print("| actual class | B | matched profile | profile | goodput | regret vs matched | budget disagreement (share of max, + over / − under) | wrong-direction steps (≥1 tier) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    regret = {}
    for cls in sorted({k[1] for k in cells}, key=ctx_of):
        for B in sorted({k[2] for k in cells if k[1] == cls}):
            avail = [pc for pc in profiles if (pc, cls, B) in cells]
            if len(avail) < 2:
                continue
            matched = min(avail, key=lambda pc: abs(pc - ctx_of(cls)))
            gm = np.mean([r["goodput"] for r in cells[(matched, cls, B)]])
            bm = np.median([b for r in cells[(matched, cls, B)] for b in r["budget_steps"]])
            maxb = np.mean([r["max_budget"] for r in cells[(matched, cls, B)]])
            tier = max(maxb / 8, 1)
            for pc in avail:
                g = np.mean([r["goodput"] for r in cells[(pc, cls, B)]])
                bs = np.array([b for r in cells[(pc, cls, B)] for b in r["budget_steps"]])
                dis = (np.median(bs) - bm) / maxb
                wrong = float(np.mean(np.abs(bs - bm) >= tier))
                regret[f"{cls}|{B}|{pc}"] = {"matched": matched, "goodput": g, "goodput_matched": gm, "regret": (gm - g) / gm, "budget_disagreement": dis, "wrong_direction_rate": wrong}
                print(f"| {cls} | {B} | {matched} | {pc}{' (matched)' if pc == matched else ''} | {g:.0f} | {(gm - g) / gm * 100:+.1f}% | {dis * 100:+.0f}% | {wrong * 100:.0f}% |")
    args.output.write_text(json.dumps({"cells": summary, "regret": regret}, indent=2) + "\n")


if __name__ == "__main__":
    main()
