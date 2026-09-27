#!/usr/bin/env python3
"""Figures for docs/when-token-budgets-lie.md, generated from the checked-in artifacts.

  fig_pairing_swap.png   measured vs predicted step-time difference of the pairing swap
  fig_live_traces.png    live vs simulated prevalence of mispriced steps, and live goodput
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path("benchmarks/results")
OUT = Path("docs/figures")


H100 = R / "h100-prefill-cost-geometry"


def pairing_swap():
    P = R / "prefill-pairing-swap/raw"
    h_slope = json.load(open(H100 / "split-term.json"))["H100-Qwen3-4B"]["single"]["mean_slope_ms_per_M"]
    setups = [("L20, vLLM 0.29", P / "L20-Qwen3-4B", 6.5, "o", "C0"),
              ("L20, vLLM 0.30", P / "L20-Qwen3-4B-vllm030", 6.5, "x", "C2"),
              ("A100, vLLM 0.29", P / "A100-Qwen3-4B", 3.3, "s", "C3"),
              ("H100, vLLM 0.29", H100 / "raw/pairswap-Qwen3-4B", h_slope, "D", "C1"),
              ("H100, SGLang 0.5", H100 / "raw/sglang-pairswap-Qwen3-4B", h_slope, "^", "C4")]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6))
    ins = ax.inset_axes([0.46, 0.58, 0.25, 0.33])
    top = 0
    for label, d, slope, mk, col in setups:
        cfgs = [c for c in json.load(open(d / "pairswap.json"))["configs"] if "delta_A_minus_B_ms" in c]
        xs = [slope * c["dW_A_minus_B_M"] for c in cfgs]
        ys = [c["delta_A_minus_B_ms"] for c in cfgs]
        ax.scatter(xs, ys, marker=mk, color=col, s=45, label=label, zorder=3)
        ins.scatter(xs, ys, marker=mk, color=col, s=25, zorder=3)
        top = max(top, max(xs), max(ys))
        # per-setup swap: median A and B of the headline config
        c = cfgs[0]
        short = label.replace(", vLLM ", "\n").replace(", SGLang 0.5", "\nSGLang")
        bx.bar([f"{short}\nA", f"{short}\nB"], [c["median_A_ms"], c["median_B_ms"]],
               color=[col, "0.75"], edgecolor="black", linewidth=0.5)
    ax.plot([0, top * 1.05], [0, top * 1.05], "k--", lw=1, label="measured = predicted")
    ax.fill_between([0, top * 1.05], [0, 0.75 * top * 1.05], [0, 1.25 * top * 1.05], color="0.9", zorder=0,
                    label="±25% (pre-registered)")
    ins.plot([0, 12], [0, 12], "k--", lw=1)
    ins.fill_between([0, 12], [0, 9], [0, 15], color="0.9", zorder=0)
    ins.set_xlim(-0.5, 12); ins.set_ylim(-0.5, 12)
    ins.set_title("H100 (FA3), zoom", fontsize=7)
    ins.tick_params(labelsize=6)
    ins.grid(alpha=.3)
    ax.set_xlabel("predicted Δ = slope × (q_a − q_b)(k_a − k_b)  [ms]")
    ax.set_ylabel("measured Δ = median(A) − median(B)  [ms]")
    ax.set_title("Same aggregate state, different pairing (Qwen3-4B, 6 configs per setup)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    bx.set_ylabel("model-step CUDA time [ms]")
    bx.set_title("Headline config: k = (4k, 16k), q = (256, 768) vs (768, 256)", fontsize=10)
    bx.tick_params(axis="x", labelsize=7)
    bx.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pairing_swap.png", dpi=140)


def live_traces():
    sim = json.load(open(R / "live-trace-replay/sim-vs-live.json"))
    keep = ["mooncake ×0.1", "mooncake ×0.2", "mooncake ×0.2 window 60 s", "mooncake ×0.2 window 150 s",
            "azure code t180 ×0.25", "azure code t180 ×0.5", "azure code ×0.5 window 540 s", "azure code ×0.5 window 1050 s",
            "burstgpt t14h ×60", "burstgpt ×60 window 25740 s", "burstgpt ×60 window 64800 s"]
    rows = [r for k in keep for r in sim if r["cell"] == k]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    x = range(len(rows))
    ax.bar([i - 0.2 for i in x], [100 * r["live_p_geo"] for r in rows], width=0.4, label="live vLLM", color="C0")
    ax.bar([i + 0.2 for i in x], [100 * r["sim_p_geo"] for r in rows], width=0.4, label="simulator", color="0.7")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["cell"].replace("window", "w").replace(" (unsaturated)", "") for r in rows], rotation=55,
                       ha="right", fontsize=7)
    ax.set_ylabel("prefill steps mispriced by > 5 ms [%]")
    ax.set_title("Where the geometry occurs (default vLLM, A100, Qwen3-4B)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")
    shift = json.load(open(R / "live-trace-replay/replay-summary.json"))["shift"]["shift-a0538ee"]["regret"]
    order = ["b1024", "agentx", "b2048", "b512", "ppas-8k", "ppas", "ctl-m2n-fcfs-D100", "ctl-m0-fcfs-D100",
             "b4096", "b8192", "ctl-m2n-fcfs", "ctl-m0-fcfs"]
    vals = [100 * shift[k] for k in order]
    cols = ["C0" if k.startswith(("b", "agentx")) else "C1" if k.startswith("ppas") else "C3" for k in order]
    bars = bx.barh(order[::-1], vals[::-1], color=cols[::-1])
    for bar, v in zip(bars, vals[::-1]):
        bx.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=7)
    bx.set_xlabel("regret vs per-phase hindsight best [%]")
    bx.set_title("Workload shift chat → code → agent → chat (4 repeats)", fontsize=10)
    bx.tick_params(axis="y", labelsize=8)
    bx.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "fig_live_traces.png", dpi=140)


def hopper_waves():
    st = json.load(open(H100 / "staircase.json"))["depths"]
    kern = json.load(open(H100 / "kernel-qwen3-4b.json"))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.4))
    for d, col in zip(sorted(st, key=lambda d: d["depth"]), ("C0", "C1", "C3")):
        qs = sorted(int(q) for q in d["median_ms"])
        ax.plot(qs, [d["median_ms"][str(q)] for q in qs], "o-", color=col, label=f"cached depth {d['depth'] // 1024}k")
    for b in (512, 1024):
        ax.axvline(b + 64, color="0.6", ls=":", lw=1)
    ax.set_xlabel("prefill chunk q (tokens), one request, plus one decode row")
    ax.set_ylabel("model-step CUDA time [ms]")
    ax.set_title("H100 FA3: step time of one chunk (dotted: 132-SM wave boundaries at B_q = 128)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    labels = [r["label"].split(" (")[0] for r in next(iter(kern["kernels"].values()))["pair"] if "control" not in r["label"]]
    x = range(len(labels)); w = 0.8 / (len(kern["kernels"]) + 1)
    for i, (k, rep) in enumerate(kern["kernels"].items()):
        rows = [r for r in rep["pair"] if "control" not in r["label"]]
        bx.bar([j + i * w for j in x], [1000 * r["measured_ms"] for r in rows], width=w, label=f"{k} measured", color=f"C{i}")
        bx.scatter([j + i * w for j in x], [1000 * r["pred_W_ms"] for r in rows], marker="_", s=120, color="k", zorder=3)
        bx.scatter([j + i * w for j in x], [1000 * r["pred_TS_ms"] for r in rows], marker="x", s=30, color="r", zorder=3)
    bx.scatter([], [], marker="_", color="k", label="work model W")
    bx.scatter([], [], marker="x", color="r", label="tile-schedule model TS")
    bx.set_xticks([j + w for j in x]); bx.set_xticklabels(labels, rotation=20, fontsize=7)
    bx.set_ylabel("one attention layer, Δ = A − B [µs]")
    bx.set_title("Isolated kernel pairing swap (Qwen3-4B heads)", fontsize=9)
    bx.legend(fontsize=7)
    bx.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_hopper_waves.png", dpi=140)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pairing_swap()
    live_traces()
    hopper_waves()
    print("wrote", sorted(p.name for p in OUT.glob("fig_*.png")))
