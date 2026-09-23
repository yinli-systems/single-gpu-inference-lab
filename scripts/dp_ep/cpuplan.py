#!/usr/bin/env python3
"""Split the calling job's CPU affinity (Slurm cpuset) into whole physical cores (SMT sibling groups) and
print shell assignments for the c26kv4 placement arms:
  ALL      every allowed CPU
  ENGINE   up to 2 cores (1 of 3) for the EngineCore process (its launch thread gets a core whose sibling only serves its own helpers)
  API      up to 2 cores (1 of 3) for the API server / frontend
  HARN     the remaining cores for the client harness + sidecar
  SQUEEZE  all but one core (3 of 6, 2 of 3): the provocation arm
With fewer than 6 cores the groups shrink proportionally (never empty). usage: eval $(python cpuplan.py)

DP form (round 62, c26kv5): `cpuplan.py --dp 2` prints, for a 12-thread = 6-core cpuset,
  ENGINE0 / ENGINE1        2 cores each (EngineCore_DP0 / _DP1 whole process)
  ENGINE0_MAIN / ENGINE1_MAIN  the first core of each pair (both SMT threads) for the main = model-launch thread only
  ENGINE0_REST / ENGINE1_REST  the second core of each pair for the engine's helper threads (NCCL proxies, ZMQ, connector)
  API      the remaining cores for the API server + DP coordinator + parent      HARN  = API (harness + sidecar share them)
  SQUEEZE  the first 2 cores for everything (provocation)
With fewer cores the engine pairs shrink to 1 core (MAIN = REST = that core) and API/HARN take what is left (never empty).
"""
import argparse, os


def parse_list(s):
    out = []
    for part in s.strip().split(","):
        if part:
            a, _, b = part.partition("-")
            out += list(range(int(a), int(b or a) + 1))
    return out


def fmt(cpus):
    return ",".join(str(c) for c in sorted(cpus))


def flat(cores):
    return [c for core in cores for c in core]


ap = argparse.ArgumentParser()
ap.add_argument("--dp", type=int, default=1)
args = ap.parse_args()

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
print(f"ALL={fmt(allowed)}")
if args.dp <= 1:
    e = max(1, min(2, n // 3)); a = max(1, min(2, (n - e) // 2)); h = max(1, n - e - a) if n > 2 else 1
    groups = dict(ENGINE=cores[:e], API=cores[e:e + a] or cores[:1], HARN=cores[e + a:] or cores[-1:], SQUEEZE=cores[:max(1, min(3, n - 1))])
    for k, v in groups.items():
        print(f"{k}={fmt(flat(v))}")
else:
    per = 2 if n >= 2 * args.dp + 1 else 1          # cores per EngineCore
    groups = {}
    for r in range(args.dp):
        mine = cores[r * per:(r + 1) * per] or cores[r % n:r % n + 1]
        groups[f"ENGINE{r}"] = flat(mine)
        groups[f"ENGINE{r}_MAIN"] = flat(mine[:1])
        groups[f"ENGINE{r}_REST"] = flat(mine[1:] or mine[:1])
    rest = cores[args.dp * per:] or cores[-1:]
    groups["API"] = flat(rest)
    groups["HARN"] = flat(rest)
    groups["SQUEEZE"] = flat(cores[:max(1, min(2, n - 1))])
    for k, v in groups.items():
        print(f"{k}={fmt(v)}")
print(f"NCORES={n}")
