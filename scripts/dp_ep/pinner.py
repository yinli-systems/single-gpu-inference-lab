#!/usr/bin/env python3
"""Pin the processes of a `vllm serve` tree without numactl: every thread of the EngineCore process(es)
(cmdline/comm containing --match, default "EngineCore") to --engine-cpus, every other process of the tree
(frontend/API server, DP coordinator, parent) to --other-cpus. Threads created later inherit their creator's mask.

DP form (round 62): --engine-cpus "0,1;2,3" gives the i-th list to the EngineCore whose title contains
"EngineCore_DP<i>" (vLLM sets `VLLM::EngineCore_DP<rank>` via setproctitle; fallback = pid order); a single list
(no ';') goes to every EngineCore (the c26kv4 usage). --engine-main-cpus "a;b" additionally pins only the *main*
thread (tid == pid: the scheduler + model-launch loop) of each EngineCore to its own list, all its other threads
(NCCL proxies, connector/copy helpers, watchdogs) to the corresponding --engine-cpus list.
Prints one line per process: pid, role, comm, first cmdline chars, the mask before and after.
usage: pinner.py --root-pid <pid> --engine-cpus <list[;list]> --other-cpus <list> [--engine-main-cpus <list[;list]>] [--match EngineCore]
"""
import argparse, os, sys


def parse_list(s):
    out = []
    for part in s.strip().split(","):
        if part:
            a, _, b = part.partition("-")
            out += list(range(int(a), int(b or a) + 1))
    return set(out)


def parse_lists(s):
    return [parse_list(x) for x in s.split(";") if x.strip()]


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
    return sorted(pids)


def allowed_of(p):
    return next((l.split(":", 1)[1].strip() for l in read(f"/proc/{p}/status").splitlines() if l.startswith("Cpus_allowed_list")), "?")


def pin_threads(p, target, main_target=None):
    """Return (ok, failed); main_target (if given) applies to tid == pid only."""
    n_ok = n_fail = 0
    for t in os.listdir(f"/proc/{p}/task") if os.path.isdir(f"/proc/{p}/task") else []:
        tgt = main_target if (main_target is not None and int(t) == p) else target
        try:
            os.sched_setaffinity(int(t), tgt)
            n_ok += 1
        except OSError as e:
            n_fail += 1
            print(f"  tid {t}: {e}", file=sys.stderr)
    return n_ok, n_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-pid", type=int, required=True)
    ap.add_argument("--engine-cpus", required=True, help="list, or ';'-separated lists per EngineCore_DP<i>")
    ap.add_argument("--other-cpus", required=True)
    ap.add_argument("--engine-main-cpus", default=None, help="list(s) for the main thread of each EngineCore only")
    ap.add_argument("--match", default="EngineCore")
    a = ap.parse_args()
    eng_lists, oth = parse_lists(a.engine_cpus), parse_list(a.other_cpus)
    main_lists = parse_lists(a.engine_main_cpus) if a.engine_main_cpus else None
    procs = []
    for p in descendants(a.root_pid):
        comm = read(f"/proc/{p}/comm").strip()
        cmd = read(f"/proc/{p}/cmdline").replace("\0", " ").strip()
        procs.append((p, comm, cmd, a.match in comm or a.match in cmd))
    engines = [x for x in procs if x[3]]
    # rank of an EngineCore: from its title (EngineCore_DP<i>), else its position in pid order
    def rank_of(idx, comm, cmd):
        for s in (cmd, comm):
            k = s.find("EngineCore_DP")
            if k >= 0:
                digits = "".join(ch for ch in s[k + 13:k + 16] if ch.isdigit())
                if digits:
                    return int(digits)
        return idx
    for idx, (p, comm, cmd, _) in enumerate(engines):
        r = rank_of(idx, comm, cmd)
        target = eng_lists[r] if r < len(eng_lists) else eng_lists[-1]
        main_t = None
        if main_lists:
            main_t = main_lists[r] if r < len(main_lists) else main_lists[-1]
        before = allowed_of(p)
        n_ok, n_fail = pin_threads(p, target, main_t)
        main_note = f" main->{sorted(main_t)}" if main_t else ""
        print(f"pid {p:7d} ENGINE r{r} comm {comm[:15]:15s} cmd {cmd[:60]!r} threads {n_ok} ok {n_fail} failed  allowed {before} -> {allowed_of(p)}{main_note}")
    for p, comm, cmd, is_engine in procs:
        if is_engine:
            continue
        before = allowed_of(p)
        n_ok, n_fail = pin_threads(p, oth)
        print(f"pid {p:7d} other    comm {comm[:15]:15s} cmd {cmd[:60]!r} threads {n_ok} ok {n_fail} failed  allowed {before} -> {allowed_of(p)}")


if __name__ == "__main__":
    main()
