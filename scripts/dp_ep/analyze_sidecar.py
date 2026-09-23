#!/usr/bin/env python3
"""Task 26 slow-state diagnostic: join the CPU sidecar (cpusidecar.py) with the runner step trace.

Input: one arm dir of sbatch_c26kv3.sh (`<R>/<arm>/` with cpusidecar.jsonl, trace/step.jsonl[.dpN],
kvoffload.json; the GPU's NUMA node is read from `<R>/gpus.txt`). The sidecar and the step tracer share
CLOCK_MONOTONIC (`t_mono` / `t`), so every 0.5-s sidecar sample is labelled by the eager chunk steps
(cg NONE, >= chunk_min tokens) that ran inside it: SLOW / fast by the sustained-run rule of
modeseries.py (runs of consecutive eager steps with gaps < 0.5 s, run median cuda_ms > thresh), or
`nochunk` (decode-only / idle / startup).

Per state and per busiest thread (the launch-bound worker main thread is the top one): CPU
utilisation, schedstat run vs wait (wait = run-queue delay = contention), nonvoluntary switches,
CPU/NUMA placement vs the GPU's node, migrations; node-wide PSI cpu/mem, loadavg, MemAvailable, Shmem.
Pre-registered readings (README "Real mover, step-level re-analysis"):
  S-place   slow samples put the worker on a far NUMA node / different core than fast samples (B cures it)
  S-contend slow samples show wait/run >> fast samples, nv/s up, PSI cpu up (B does not cure it)
  S-mem     PSI memory > 0 or MemAvailable collapsing only during slow samples
Prints a run-by-run timeline (each eager run with the main thread's cpu/node/wait% and PSI) and the
per-state table; --json writes the numbers.
usage: analyze_sidecar.py <arm-dir> [--thresh 65] [--chunk-min 256] [--top 5] [--json out.json]
DP2 arms (c26kv5): --rank N keeps only trace/step.jsonl.dpN (the storing rank's chunk steps), --main-pid P
takes the busiest thread of that pid (EngineCore_DP<N>'s pid from pinner.txt / affinity-ready.txt) as the main thread.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def psi_avg10(lines):
    # "some avg10=0.00 avg60=0.00 avg300=0.00 total=0"
    for l in lines or []:
        m = re.search(r"avg10=([0-9.]+)", l)
        if m:
            return float(m.group(1))
    return float("nan")


def gpu_numa(arm_dir):
    p = os.path.join(os.path.dirname(arm_dir.rstrip("/")), "gpus.txt")
    nodes = []
    try:
        for l in open(p):
            m = re.search(r"numa (-?\d+)", l)
            if m:
                nodes.append(int(m.group(1)))
    except OSError:
        pass
    return nodes[0] if nodes else None


def eager_runs(steps, th, chunk_min):
    eager = [s for s in steps if str(s["cg_mode"]).endswith("NONE") and s["num_tokens"] >= chunk_min]
    runs = []
    for s in eager:
        if runs and s["t"] - runs[-1][-1]["t"] < 0.5:
            runs[-1].append(s)
        else:
            runs.append([s])
    out = []
    for r in runs:
        cm = np.array([s["cuda_ms"] for s in r])
        out.append(dict(t0=r[0]["t"], t1=r[-1]["t"], n=len(r), cuda_p50=float(np.median(cm)),
                        state="slow" if np.median(cm) > th else "fast", frac_slow=float((cm > th).mean())))
    return out


def cell_of(cells, t):
    """Cell whose measurement window contains t, else the next window starting within 60 s (its train)."""
    inside = [c for c in cells if c["t_window0"] <= t <= c["t_window1"]]
    if not inside:
        inside = [c for c in cells if c["t_window1"] < t <= c["t_window1"] + 5]
    if not inside:
        inside = sorted((c for c in cells if 0 < c["t_window0"] - t <= 60), key=lambda c: c["t_window0"])
    return f"{inside[0]['kind']} B={inside[0]['B']} r{inside[0]['repeat']}" if inside else "between"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir")
    ap.add_argument("--thresh", type=float, default=65.0)
    ap.add_argument("--chunk-min", type=int, default=256)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--json")
    ap.add_argument("--rank", type=int, default=None, help="use only trace/step.jsonl.dp<rank>")
    ap.add_argument("--main-pid", type=int, default=None, help="main thread = busiest thread of this pid")
    a = ap.parse_args()
    d = a.arm_dir.rstrip("/")

    side = load_jsonl(os.path.join(d, "cpusidecar.jsonl"))
    static = next((r for r in side if r.get("kind") == "static"), {})
    samples = [r for r in side if r.get("kind") == "sample"]
    step_files = sorted(glob.glob(os.path.join(d, "trace", "step.jsonl*")))
    if a.rank is not None:
        step_files = [p for p in step_files if p.endswith(f".dp{a.rank}")]
    steps = [s for p in step_files for s in load_jsonl(p)]
    steps.sort(key=lambda s: s["t"])
    gnode = gpu_numa(d)
    cells = []
    try:
        cells = json.load(open(os.path.join(d, "kvoffload.json")))["cells"]
    except (OSError, KeyError, ValueError):
        pass
    if not samples or not steps:
        print(f"{d}: {len(samples)} sidecar samples, {len(steps)} steps — nothing to join")
        return
    t_ref = steps[0]["t"]
    print(f"== {d}: {len(samples)} sidecar samples ({samples[0]['t_mono']-t_ref:+.0f}s .. {samples[-1]['t_mono']-t_ref:+.0f}s rel. first step), "
          f"{len(steps)} steps from {[os.path.basename(p) for p in step_files]}, GPU NUMA node {gnode}")
    print(f"   static: hz {static.get('hz')} ncpu {static.get('ncpu')} numa_balancing {static.get('numa_balancing')!r} thp {static.get('thp')!r} "
          f"shmem_thp {static.get('shmem_thp')!r} affinity_root n={len(static.get('affinity_root') or [])}")

    runs = eager_runs(steps, a.thresh, a.chunk_min)
    # label each sample interval (t_prev, t] by the run it overlaps (majority by overlap length)
    labels = []
    for i, r in enumerate(samples):
        t1 = r["t_mono"]
        t0 = samples[i - 1]["t_mono"] if i else t1 - 0.5
        best, best_ov = "nochunk", 0.0
        for run in runs:
            ov = min(t1, run["t1"] + 0.05) - max(t0, run["t0"] - 0.05)
            if ov > best_ov:
                best, best_ov = run["state"], ov
        n_steps = sum(1 for s in steps if t0 < s["t"] <= t1)  # activity of any kind
        labels.append((best if n_steps else "idle", t1 - t0))

    # SMT sibling map (static record, round-61 sidecar) and per-sample per-CPU busy fractions (/proc/stat)
    sib = {int(k): v for k, v in (static.get("siblings") or {}).items()}
    has_busy = any("cpu_busy" in r for r in samples)

    def sib_busy(sample, cpu):
        """busy fraction of the SMT sibling(s) of `cpu` in this sample (any process on the node), nan if unknown"""
        b = sample.get("cpu_busy") or {}
        vals = [b[str(s)] for s in sib.get(cpu, []) if str(s) in b]
        return float(np.mean(vals)) if vals else float("nan")

    # busiest threads over the whole arm
    tot = Counter()
    comm = {}
    for r in samples:
        for th in r["threads"]:
            tot[(th["pid"], th["tid"])] += th["cpu_ms"]
            comm[(th["pid"], th["tid"])] = th["comm"]
    top = [k for k, _ in tot.most_common(a.top)]
    print(f"\n== busiest threads (whole arm): " + "; ".join(f"{comm[k]}[{k[0]}/{k[1]}] {tot[k]/1e3:.0f} s" for k in top))
    main_key = top[0]
    if a.main_pid is not None:
        mine = [k for k, _ in tot.most_common() if k[0] == a.main_pid]
        if mine:
            main_key = mine[0]
            print(f"   main thread forced to pid {a.main_pid}: {comm[main_key]}[{main_key[0]}/{main_key[1]}] {tot[main_key]/1e3:.0f} s")

    # per-run timeline with the main thread's placement / wait and PSI
    print(f"\n== eager-run timeline (main thread {comm[main_key]}[{main_key[0]}/{main_key[1]}]; wait% = wait/(run+wait); cpu:node of the main thread, share of samples on its modal cpu)")
    prev_state = None
    for run in runs:
        idx = [i for i, r in enumerate(samples) if run["t0"] - 0.55 <= r["t_mono"] <= run["t1"] + 0.55]
        ths = [th for i in idx for th in samples[i]["threads"] if (th["pid"], th["tid"]) == main_key]
        cpus = Counter(th["cpu"] for th in ths)
        nodes = Counter(th["node"] for th in ths)
        run_ms = sum(th["run_ms"] for th in ths); wait_ms = sum(th["wait_ms"] for th in ths)
        util = sum(th["cpu_ms"] for th in ths) / max(1e-9, sum(samples[i]["t_mono"] - samples[i - 1]["t_mono"] for i in idx if i)) / 10.0 if idx else float("nan")
        psi_c = np.nanmean([psi_avg10(samples[i]["psi_cpu"]) for i in idx]) if idx else float("nan")
        psi_m = np.nanmean([psi_avg10(samples[i]["psi_mem"]) for i in idx]) if idx else float("nan")
        mem_av = min((samples[i]["mem"].get("MemAvailable", 0) for i in idx), default=0) / 2**20
        cell = cell_of(cells, run["t0"])
        flip = "  <-- FLIP" if prev_state is not None and run["state"] != prev_state else ""
        modal = cpus.most_common(1)[0] if cpus else (None, 0)
        sb = [sib_busy(samples[i], th["cpu"]) for i in idx for th in samples[i]["threads"] if (th["pid"], th["tid"]) == main_key]
        sb_txt = f" sib {100*np.nanmean(sb):3.0f}%" if has_busy and sb and not np.all(np.isnan(sb)) else ""
        print(f"  t {run['t0']-t_ref:7.1f}s n {run['n']:3d} cuda p50 {run['cuda_p50']:6.1f} {run['state'].upper():4s} | samples {len(idx):3d} util {util:5.0f}% "
              f"wait {100*wait_ms/max(1e-9,run_ms+wait_ms):5.1f}% nv/s {sum(th['nv'] for th in ths)/max(0.5,len(idx)*0.5):5.1f} "
              f"cpu {modal[0]}:{nodes.most_common(1)[0][0] if nodes else '?'} ({100*modal[1]/max(1,len(ths)):3.0f}%) ncpu {len(cpus)} "
              f"onGPUnode {100*sum(v for n,v in nodes.items() if n==gnode)/max(1,len(ths)):3.0f}%{sb_txt} psi cpu {psi_c:4.1f} mem {psi_m:4.1f} memav {mem_av:5.1f}G{flip}  [{cell}]")
        prev_state = run["state"]

    # per-state aggregate over samples
    out = {"arm": d, "gpu_numa": gnode, "static": static, "runs": runs, "threads": {}, "states": {}}
    states = ["slow", "fast", "nochunk", "idle"]
    print(f"\n== per-state aggregate (samples: " + ", ".join(f"{s} {sum(1 for l,_ in labels if l==s)}" for s in states) + ")")
    hdr = f"{'thread':30s} {'state':8s} {'n':>4s} {'util%':>6s} {'run/s':>7s} {'wait/s':>7s} {'wait%':>6s} {'nv/s':>6s} {'ncpu':>4s} {'modal cpu':>10s} {'onGPUnode%':>10s} {'migr/s':>6s} {'sib%':>7s} {'sib>=50':>7s}  allowed"
    print(hdr)
    for k in top:
        name = f"{comm[k]}[{k[0]}/{k[1]}]"[:30]
        out["threads"][name] = {}
        for st in states:
            idx = [i for i, (l, _) in enumerate(labels) if l == st]
            rows = [(i, th) for i in idx for th in samples[i]["threads"] if (th["pid"], th["tid"]) == k]
            if not rows:
                continue
            dur = sum(labels[i][1] for i, _ in rows)
            run_ms = sum(th["run_ms"] for _, th in rows); wait_ms = sum(th["wait_ms"] for _, th in rows)
            cpus = Counter(th["cpu"] for _, th in rows); nodes = Counter(th["node"] for _, th in rows)
            migr = sum(1 for (i1, a1), (i2, a2) in zip(rows, rows[1:]) if i2 == i1 + 1 and a1["cpu"] != a2["cpu"])
            sb = [sib_busy(samples[i], th["cpu"]) for i, th in rows] if has_busy else []
            sb_mean = float(np.nanmean(sb)) * 100 if sb and not np.all(np.isnan(sb)) else float("nan")
            sb_hi = float(np.nanmean(np.array(sb) >= 0.5)) * 100 if sb and not np.all(np.isnan(sb)) else float("nan")
            rec = dict(n=len(rows), util=100 * sum(th["cpu_ms"] for _, th in rows) / (dur * 1e3), run_per_s=run_ms / dur, wait_per_s=wait_ms / dur,
                       wait_pct=100 * wait_ms / max(1e-9, run_ms + wait_ms), nv_per_s=sum(th["nv"] for _, th in rows) / dur, ncpu=len(cpus),
                       modal_cpu=cpus.most_common(1)[0][0], modal_share=100 * cpus.most_common(1)[0][1] / len(rows),
                       on_gpu_node=100 * sum(v for n, v in nodes.items() if n == gnode) / len(rows), nodes=dict(nodes), migr_per_s=migr / dur,
                       sib_busy_pct=sb_mean, sib_busy_ge50_pct=sb_hi, allowed=Counter(th.get("allowed", "") for _, th in rows).most_common(1)[0][0])
            out["threads"][name][st] = rec
            print(f"{name:30s} {st:8s} {rec['n']:4d} {rec['util']:6.0f} {rec['run_per_s']:7.0f} {rec['wait_per_s']:7.1f} {rec['wait_pct']:6.1f} {rec['nv_per_s']:6.1f} "
                  f"{rec['ncpu']:4d} {str(rec['modal_cpu'])+':'+str(nodes.most_common(1)[0][0]):>10s} {rec['on_gpu_node']:10.0f} {rec['migr_per_s']:6.2f} "
                  f"{rec['sib_busy_pct']:7.0f} {rec['sib_busy_ge50_pct']:7.0f}  {rec['allowed']}")
    print(f"\n{'node-wide':30s} {'state':8s} {'n':>4s} {'psi_cpu':>8s} {'psi_mem':>8s} {'load1':>6s} {'memav_min':>9s} {'shmem_max':>9s} {'busy thr':>8s}")
    for st in states:
        idx = [i for i, (l, _) in enumerate(labels) if l == st]
        if not idx:
            continue
        pc = [psi_avg10(samples[i]["psi_cpu"]) for i in idx]; pm = [psi_avg10(samples[i]["psi_mem"]) for i in idx]
        rec = dict(n=len(idx), psi_cpu_mean=float(np.nanmean(pc)), psi_cpu_max=float(np.nanmax(pc)), psi_mem_mean=float(np.nanmean(pm)), psi_mem_max=float(np.nanmax(pm)),
                   load1=float(np.mean([float(samples[i]["loadavg"][0]) for i in idx if samples[i]["loadavg"]])),
                   memav_min_gb=min(samples[i]["mem"].get("MemAvailable", 0) for i in idx) / 2**20, shmem_max_gb=max(samples[i]["mem"].get("Shmem", 0) for i in idx) / 2**20,
                   busy_threads=float(np.mean([len(samples[i]["threads"]) for i in idx])))
        out["states"][st] = rec
        print(f"{'':30s} {st:8s} {rec['n']:4d} {rec['psi_cpu_mean']:4.1f}/{rec['psi_cpu_max']:3.0f} {rec['psi_mem_mean']:4.1f}/{rec['psi_mem_max']:3.0f} {rec['load1']:6.1f} {rec['memav_min_gb']:9.1f} {rec['shmem_max_gb']:9.1f} {rec['busy_threads']:8.1f}")

    # readings
    m = out["threads"].get(f"{comm[main_key]}[{main_key[0]}/{main_key[1]}]"[:30], {})
    if "slow" in m and "fast" in m:
        s, f = m["slow"], m["fast"]
        print("\n== readings (main thread, slow vs fast samples)")
        print(f"  S-place : on GPU node {s['on_gpu_node']:.0f}% vs {f['on_gpu_node']:.0f}%; modal cpu {s['modal_cpu']} ({s['modal_share']:.0f}%) vs {f['modal_cpu']} ({f['modal_share']:.0f}%); nodes {s['nodes']} vs {f['nodes']}")
        print(f"  S-smt   : sibling busy {s['sib_busy_pct']:.0f}% vs {f['sib_busy_pct']:.0f}% (samples with sibling >= 50 % busy: {s['sib_busy_ge50_pct']:.0f}% vs {f['sib_busy_ge50_pct']:.0f}%); allowed {s['allowed']} vs {f['allowed']}")
        print(f"  S-contend: wait {s['wait_pct']:.1f}% vs {f['wait_pct']:.1f}%; nv/s {s['nv_per_s']:.1f} vs {f['nv_per_s']:.1f}; util {s['util']:.0f}% vs {f['util']:.0f}%; "
              f"psi cpu {out['states']['slow']['psi_cpu_mean']:.1f} vs {out['states']['fast']['psi_cpu_mean']:.1f}")
        print(f"  S-mem   : psi mem {out['states']['slow']['psi_mem_mean']:.1f} vs {out['states']['fast']['psi_mem_mean']:.1f}; MemAvailable min {out['states']['slow']['memav_min_gb']:.1f} vs {out['states']['fast']['memav_min_gb']:.1f} GB")
    else:
        print(f"\n== readings: states present for the main thread: {sorted(m)} — no slow/fast contrast in this arm")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
