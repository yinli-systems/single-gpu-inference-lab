#!/usr/bin/env python3
"""Split the calling job's CPU affinity (Slurm cpuset) into whole physical cores (SMT sibling groups) and
print shell assignments for the c26kv4 placement arms:
  ALL      every allowed CPU
  ENGINE   up to 2 cores (1 of 3) for the EngineCore process (its launch thread gets a core whose sibling only serves its own helpers)
  API      up to 2 cores (1 of 3) for the API server / frontend
  HARN     the remaining cores for the client harness + sidecar
  SQUEEZE  all but one core (3 of 6, 2 of 3): the provocation arm
With fewer than 6 cores the groups shrink proportionally (never empty). usage: eval $(python cpuplan.py)
"""
import os


def parse_list(s):
    out = []
    for part in s.strip().split(","):
        if part:
            a, _, b = part.partition("-")
            out += list(range(int(a), int(b or a) + 1))
    return out


def fmt(cpus):
    return ",".join(str(c) for c in sorted(cpus))


allowed = sorted(os.sched_getaffinity(0))
cores, seen = [], set()
for c in allowed:
    if c in seen:
        continue
    try:
        sib = [x for x in parse_list(open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read()) if x in allowed]
    except OSError:
        sib = [c]
    core = sorted(set(sib) | {c})
    seen |= set(core)
    cores.append(core)
n = len(cores)
e = max(1, min(2, n // 3)); a = max(1, min(2, (n - e) // 2)); h = max(1, n - e - a) if n > 2 else 1
groups = dict(ENGINE=cores[:e], API=cores[e:e + a] or cores[:1], HARN=cores[e + a:] or cores[-1:], SQUEEZE=cores[:max(1, min(3, n - 1))])
print(f"ALL={fmt(allowed)}")
for k, v in groups.items():
    print(f"{k}={fmt(c for core in v for c in core)}")
print(f"NCORES={n}")
