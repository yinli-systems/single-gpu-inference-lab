#!/usr/bin/env python3
"""Task 26: per-cell PCIe rx/tx (nvidia-smi dmon -s ut, 1 s samples) for both GPUs.

`dmon.log` is wall-clock (HH:MM:SS); the harness windows are monotonic. Each cell's
`hog/start-NNN` file was touched at t_window0, so its mtime anchors that cell exactly
(offset = mtime - t_window0). Reports, per (spec, B), the median rxpci/txpci MB/s of each GPU
over the samples inside the 8 s window, pooled over repeats: GPU 1's numbers are the EP
collective traffic of the rank that moves no KV; GPU 0's show the hog + EP traffic and which
direction the hog saturates (h2d = GPU0 rx, d2h = GPU0 tx).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


def parse_dmon(path):
    rows = []  # (secs_of_day, gpu, sm, rx, tx)
    day = 0
    last = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        try:
            h, m, s = (int(x) for x in p[0].split(":"))
            gpu, sm, rx, tx = int(p[1]), int(p[2]), int(p[-2]), int(p[-1])
        except ValueError:
            continue
        t = h * 3600 + m * 60 + s
        if last is not None and t < last - 3600:
            day += 1  # midnight wrap
        last = t
        rows.append((t + day * 86400, gpu, sm, rx, tx))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    rec = json.loads((args.dir / "pcie.json").read_text())
    rows = parse_dmon(args.dir / "dmon.log")
    per = defaultdict(lambda: defaultdict(list))
    # dmon prints local time; convert the start-file mtime to seconds-of-day in local time (same day as the log's first row)
    first_day = None
    for cell in rec["cells"]:
        idx = int(Path(cell["hog_log"]).name.split("-")[1])
        sf = args.dir / "hog" / f"start-{idx:03d}"
        if not sf.exists():
            continue
        mt = dt.datetime.fromtimestamp(os.stat(sf).st_mtime)
        if first_day is None:
            first_day = mt.date()
        wall0 = (mt - dt.datetime.combine(first_day, dt.time())).total_seconds()
        wall1 = wall0 + (cell["t_window1"] - cell["t_window0"])
        key = (cell["spec"], cell["B"])
        for t, gpu, sm, rx, tx in rows:
            if wall0 + 1 <= t <= wall1:  # skip the first partial second
                per[key][f"gpu{gpu}_rx"].append(rx); per[key][f"gpu{gpu}_tx"].append(tx); per[key][f"gpu{gpu}_sm"].append(sm)
    out = {}
    print(f"{'spec':14s} {'B':>3s} {'n':>3s} | GPU0 rx/tx MB/s (sm%) | GPU1 rx/tx MB/s (sm%)")
    for key in sorted(per, key=lambda k: (k[1], k[0])):
        d = per[key]
        med = {k: statistics.median(v) for k, v in d.items()}
        mx = {k: max(v) for k, v in d.items()}
        out[f"{key[0]}|B{key[1]}"] = {"n_samples": len(d["gpu0_rx"]), "median": med, "max": mx}
        print(f"{key[0]:14s} {key[1]:3d} {len(d['gpu0_rx']):3d} | {med['gpu0_rx']:6.0f}/{med['gpu0_tx']:6.0f} ({med['gpu0_sm']:3.0f}) | {med['gpu1_rx']:6.0f}/{med['gpu1_tx']:6.0f} ({med['gpu1_sm']:3.0f})")
    if args.output:
        args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
