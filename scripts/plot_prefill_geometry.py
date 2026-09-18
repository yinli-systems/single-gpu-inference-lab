#!/usr/bin/env python3
"""Figures for the prefill cost-geometry artifact.

  same_aggregate_geometry.png  measured prefill-step CUDA time vs aggregate KV
                               for equal-aggregate partitions (1x1024/2x512/4x256
                               and 1x2048/4x512/8x256), and the same steps vs the
                               per-request attention-work proxy (they collapse)
  ood_residuals.png            positive-residual quantiles of M0 / M1 / M2 on the
                               geometry-OOD splits (from analyze_step_cost_v2 JSON)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from step_trace_join import load_joined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--predictors", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    series = {"1x1024": "part-1x1024-chunk1024", "2x512": "part-2x512-chunk1024", "4x256": "part-4x256-chunk1024",
              "1x2048": "part2-1x2048-chunk2048", "4x512": "part2-4x512-chunk2048", "8x256": "part2-8x256-chunk2048"}
    data = {}
    for label, cell in series.items():
        for d in args.trace_dir:
            p = d / f"{cell}.jsonl"
            if p.exists():
                rows, _ = load_joined(p)
                n = int(label.split("x")[0])
                data[label] = [r for r in rows if np.isfinite(r["cuda_ms"]) and r["ctx_reqs"] == n and r["ctx_tokens"] >= 0.95 * int(label.split("x")[0]) * int(label.split("x")[1])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"1x1024": "C0", "2x512": "C1", "4x256": "C2", "1x2048": "C3", "4x512": "C4", "8x256": "C5"}
    for label, rows in data.items():
        agg = np.array([r["ctx_kv_sum"] + r["gen_kv_sum"] for r in rows]) / 1024
        proxy = np.array([r["attn_proxy"] for r in rows]) / 1e6
        y = np.array([r["cuda_ms"] for r in rows])
        mk = "o" if label.endswith(("1024", "512", "256")) and int(label.split("x")[1]) * int(label.split("x")[0]) <= 1024 else "^"
        axes[0].scatter(agg, y, s=10, alpha=.6, c=colors[label], marker=mk, label=f"{label} (n={len(rows)})")
        axes[1].scatter(proxy, y, s=10, alpha=.6, c=colors[label], marker=mk, label=label)
    axes[0].set_xlabel("aggregate KV read depth in the step (prefill + decode), K tokens")
    axes[0].set_ylabel("model-step CUDA time, ms")
    axes[0].set_title("Same aggregate coordinate (8 decoders, 1024 or 2048 prefill tokens/step)\n— cost depends on how the tokens are partitioned")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("attention-work proxy  Σ q_i·(kv_i + (q_i+1)/2), M")
    axes[1].set_ylabel("model-step CUDA time, ms")
    axes[1].set_title("The same steps against the per-request proxy — one line")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "same_aggregate_geometry.png", dpi=140)

    rep = json.load(open(args.predictors))
    splits = ["primary_geometry_one_to_multi", "reverse_geometry_multi_to_one", "ctx_le16k_to_32k", "batch_le16_to_32"]
    titles = {"primary_geometry_one_to_multi": "geometry OOD: one prefill → 2/4/8", "reverse_geometry_multi_to_one": "geometry OOD: 2/4/8 → one prefill",
              "ctx_le16k_to_32k": "context OOD: ≤16k → 32k", "batch_le16_to_32": "load OOD: B≤16 → 32"}
    fig, axes = plt.subplots(1, len(splits), figsize=(4.2 * len(splits), 4.2), sharey=False)
    for ax, sp in zip(axes, splits):
        s = rep["splits"].get(sp)
        if not s:
            continue
        names = [m for m in ("M0", "M0-lookupP99", "M1", "M2") if m in s["models"]]
        for j, q in enumerate(("pos_resid_p50", "pos_resid_p95", "pos_resid_p99")):
            vals = [s["models"][m][q] for m in names]
            ax.bar(np.arange(len(names)) + (j - 1) * 0.26, vals, width=0.26, label=q.replace("pos_resid_", "P").upper())
        ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names, fontsize=8)
        ax.set_title(titles[sp] + f"\n(test n={s['n_test']}, {s['test_in_m0_support_frac']*100:.0f}% in M0 support)", fontsize=9)
        ax.axhline(5, color="k", ls="--", lw=.8); ax.axhline(10, color="r", ls=":", lw=.8)
        ax.set_ylabel("positive residual (actual − predicted), ms"); ax.grid(alpha=.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "ood_residuals.png", dpi=140)


if __name__ == "__main__":
    main()
