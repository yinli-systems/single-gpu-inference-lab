#!/usr/bin/env python3
"""Pin the processes of a `vllm serve` tree without numactl: every thread of the EngineCore process(es)
(cmdline/comm containing --match, default "EngineCore") to --engine-cpus, every other process of the tree
(frontend/API server, coordinator) to --other-cpus. Threads created later inherit their creator's mask.
Prints one line per process: pid, comm, first cmdline token, the mask before and after.
usage: pinner.py --root-pid <pid> --engine-cpus <list> --other-cpus <list> [--match EngineCore]
"""
import argparse, os, sys


def parse_list(s):
    out = []
    for part in s.strip().split(","):
        if part:
            a, _, b = part.partition("-")
            out += list(range(int(a), int(b or a) + 1))
    return set(out)


def read(p):
    try:
        with open(p) as f:
            return f.read()
    except OSError:
        return ""


def descendants(root):
    pids, todo = [], [root]
    while todo:
        p = todo.pop()
        if p in pids:
            continue
        pids.append(p)
        for t in os.listdir(f"/proc/{p}/task") if os.path.isdir(f"/proc/{p}/task") else []:
            todo += [int(x) for x in read(f"/proc/{p}/task/{t}/children").split()]
    return pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-pid", type=int, required=True)
    ap.add_argument("--engine-cpus", required=True)
    ap.add_argument("--other-cpus", required=True)
    ap.add_argument("--match", default="EngineCore")
    a = ap.parse_args()
    eng, oth = parse_list(a.engine_cpus), parse_list(a.other_cpus)
    for p in descendants(a.root_pid):
        comm = read(f"/proc/{p}/comm").strip()
        cmd = read(f"/proc/{p}/cmdline").replace("\0", " ").strip()
        is_engine = a.match in comm or a.match in cmd
        target = eng if is_engine else oth
        before = read(f"/proc/{p}/status")
        before = next((l.split(":", 1)[1].strip() for l in before.splitlines() if l.startswith("Cpus_allowed_list")), "?")
        n_ok = n_fail = 0
        for t in os.listdir(f"/proc/{p}/task") if os.path.isdir(f"/proc/{p}/task") else []:
            try:
                os.sched_setaffinity(int(t), target)
                n_ok += 1
            except OSError as e:
                n_fail += 1
                print(f"  tid {t}: {e}", file=sys.stderr)
        after = next((l.split(":", 1)[1].strip() for l in read(f"/proc/{p}/status").splitlines() if l.startswith("Cpus_allowed_list")), "?")
        print(f"pid {p:7d} {'ENGINE' if is_engine else 'other ':6s} comm {comm[:15]:15s} cmd {cmd[:60]!r} threads {n_ok} ok {n_fail} failed  allowed {before} -> {after}")


if __name__ == "__main__":
    main()
