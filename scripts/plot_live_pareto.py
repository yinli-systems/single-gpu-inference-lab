#!/usr/bin/env python3
"""Safe-throughput vs deadline-violation Pareto view of every live deadline-controller run.

One panel per (GPU, N): fixed prefill budgets (grey, joined in budget order = the fixed-budget
frontier) and the model-driven controllers overlaid. Each point is the mean over its block's
repeats with min-max whiskers; the marker shape says which campaign block it came from, because
some fixed budgets were only measured in one block (L20 fixed-384: campaign21). The shaded band is
the <= 5% violation target used by the pre-registered gates.

Inputs are the analyze_live_controller.py JSON files already in the repo.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path("benchmarks/results")
BLOCKS = {
    "L20": {"deadline": 100, "blocks": [
        ("campaign20", R / "l20-prefill-cost-geometry/live-controller.json", "o"),
        ("campaign21", R / "l20-prefill-cost-geometry/live-calibration.json", "s"),
        ("campaign30", R / "l20-prefill-live-controller-m2n/live.json", "D")]},
    "A100": {"deadline": 65, "blocks": [
        ("main", R / "a100-prefill-live-controller/live-main.json", "o"),
        ("fixed-hi", R / "a100-prefill-live-controller/live-fixedhi.json", "s")]},
}
CTRL = {"m0": ("M0 aggregate", "C3"), "m2": ("M2 published", "C1"), "m2n": ("M2n", "C0"),
        "m2online": ("M2 + online margin", "C2")}


def pts(path):
    d = json.load(open(path))
    out = {}
    for cond, v in d.items():
        runs = v["runs"]
        s = [r["safe_tok_per_s"] for r in runs]; x = [100 * r["violation_rate"] for r in runs]
        out[cond] = (float(np.mean(x)), float(np.mean(s)), min(x), max(x), min(s), max(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=R / "a100-prefill-live-controller/figures/live_pareto.png")
    args = ap.parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    summary = {}
    for row, (gpu, spec) in enumerate(BLOCKS.items()):
        data = [(name, pts(p), mk) for name, p, mk in spec["blocks"]]
        for col, N in enumerate((4, 8)):
            ax = axes[row][col]
            ax.axvspan(-1, 5, color="0.92", zorder=0)
            fixed = {}
            for name, d, mk in data:
                for cond, (x, y, x0, x1, y0, y1) in d.items():
                    m = re.match(rf"N{N}-fixed(\d+)$", cond)
                    if m:
                        b = int(m.group(1)); fixed.setdefault(b, []).append((x, y))
                        ax.errorbar(x, y, xerr=[[x - x0], [x1 - x]], yerr=[[y - y0], [y1 - y]], fmt=mk, color="0.45",
                                    mfc="white", ms=7, capsize=2, lw=1, zorder=3)
            bs = sorted(fixed)
            fx = [np.mean([p[0] for p in fixed[b]]) for b in bs]; fy = [np.mean([p[1] for p in fixed[b]]) for b in bs]
            ax.plot(fx, fy, color="0.6", lw=1, ls="--", zorder=2, label="fixed budgets, in budget order")
            for b, x, y in zip(bs, fx, fy):
                ax.annotate(f"fixed-{b}", (x, y), textcoords="offset points", xytext=(7, -11), fontsize=8, color="0.3")
            for name, d, mk in data:
                for key, (label, colr) in CTRL.items():
                    c = f"N{N}-{key}"
                    if c in d:
                        x, y, x0, x1, y0, y1 = d[c]
                        ax.errorbar(x, y, xerr=[[x - x0], [x1 - x]], yerr=[[y - y0], [y1 - y]], fmt=mk, color=colr, ms=9 if key == "m2n" else 7,
                                    capsize=2, lw=1.2, zorder=4, label=label)
                        summary.setdefault(f"{gpu}-N{N}", {})[f"{key}@{name}"] = {"violation_pct": x, "safe_tok_per_s": y}
            for b in bs:
                summary.setdefault(f"{gpu}-N{N}", {})[f"fixed-{b}"] = {"violation_pct": float(np.mean([p[0] for p in fixed[b]])),
                                                                     "safe_tok_per_s": float(np.mean([p[1] for p in fixed[b]]))}
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_xlim(-0.3, 80)
            ax.set_xlabel(f"prefill steps over the {spec['deadline']} ms deadline (%, symlog)")
            ax.set_ylabel("safe prefill tokens / s")
            ax.set_title(f"{gpu}, Qwen3-4B: 8 decoders + {N} × 16k prefills", fontsize=10)
            ax.grid(alpha=.3)
            for name, _, mk in data:
                ax.plot([], [], mk, color="0.3", mfc="white", label=f"block: {name}")
            h, lab = ax.get_legend_handles_labels()
            seen = {}
            for hh, ll in zip(h, lab):
                seen.setdefault(ll, hh)
            ax.legend(seen.values(), seen.keys(), fontsize=6.5, loc="lower right")
    fig.suptitle("Live deadline controller vs fixed prefill budgets. Shaded: ≤ 5% violations. Marker shape = campaign block;\n"
                 "points are means over 3 repeats with min–max whiskers", fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
