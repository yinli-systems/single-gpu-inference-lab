#!/usr/bin/env python3
"""CPU-placement sidecar for the eager-step 'slow mode' (task 26, job 1602608: 512-token eager chunk
steps 48 -> 85 ms in sustained runs with the CPU-offload connector on, independent of copies).

Samples every --interval s, on the same time.monotonic() clock as the step tracer, every thread of
every process under the given root pid (the `vllm serve` tree): CPU ticks delta, last CPU
(processor) + its NUMA node, schedstat run/wait ns delta (wait = run-queue delay = CPU contention),
nonvoluntary context switches; plus node-wide PSI cpu/memory, loadavg, MemAvailable/Shmem.
Threads with zero CPU delta in the interval are dropped from the record. Writes JSONL.
usage: cpusidecar.py --root-pid <pid> --output <file> [--interval 0.5]
"""
import argparse, json, os, time

def cpu_to_node():
    m = {}
    base = "/sys/devices/system/node"
    for n in os.listdir(base):
        if n.startswith("node"):
            for c in os.listdir(f"{base}/{n}"):
                if c.startswith("cpu") and c[3:].isdigit():
                    m[int(c[3:])] = int(n[4:])
    return m

def descendants(root):
    pids, todo = set(), [root]
    while todo:
        p = todo.pop()
        if p in pids:
            continue
        pids.add(p)
        try:
            for t in os.listdir(f"/proc/{p}/task"):
                with open(f"/proc/{p}/task/{t}/children") as f:
                    todo += [int(x) for x in f.read().split()]
        except OSError:
            pass
    return pids

def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None

def sample_threads(pids, node_of):
    out = {}
    for p in pids:
        try:
            tids = os.listdir(f"/proc/{p}/task")
        except OSError:
            continue
        for t in tids:
            st = read(f"/proc/{p}/task/{t}/stat")
            if not st:
                continue
            comm = st[st.index("(") + 1 : st.rindex(")")]
            f = st[st.rindex(")") + 2 :].split()
            utime, stime, processor = int(f[11]), int(f[12]), int(f[36])
            ss = read(f"/proc/{p}/task/{t}/schedstat")
            run_ns, wait_ns, slices = (int(x) for x in ss.split()) if ss else (0, 0, 0)
            nv = 0
            for line in (read(f"/proc/{p}/task/{t}/status") or "").splitlines():
                if line.startswith("nonvoluntary_ctxt_switches"):
                    nv = int(line.split()[1])
            out[(p, int(t))] = dict(comm=comm, ticks=utime + stime, cpu=processor, node=node_of.get(processor, -1),
                                    run_ns=run_ns, wait_ns=wait_ns, slices=slices, nv=nv)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-pid", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--interval", type=float, default=0.5)
    a = ap.parse_args()
    node_of = cpu_to_node()
    hz = os.sysconf("SC_CLK_TCK")
    fh = open(a.output, "a", buffering=1)
    fh.write(json.dumps({"kind": "static", "t_mono": time.monotonic(), "hz": hz, "ncpu": os.cpu_count(),
                         "numa_balancing": (read("/proc/sys/kernel/numa_balancing") or "").strip(),
                         "thp": (read("/sys/kernel/mm/transparent_hugepage/enabled") or "").strip(),
                         "shmem_thp": (read("/sys/kernel/mm/transparent_hugepage/shmem_enabled") or "").strip(),
                         "affinity_root": sorted(os.sched_getaffinity(a.root_pid)) if hasattr(os, "sched_getaffinity") else None}) + "\n")
    prev = {}
    while True:
        try:
            pids = descendants(a.root_pid)
        except Exception:
            pids = set()
        if not pids or not os.path.exists(f"/proc/{a.root_pid}"):
            break
        cur = sample_threads(pids, node_of)
        t = time.monotonic()
        threads = []
        for key, v in cur.items():
            pv = prev.get(key)
            if pv is None:
                continue
            d_ticks = v["ticks"] - pv["ticks"]
            d_run = v["run_ns"] - pv["run_ns"]
            d_wait = v["wait_ns"] - pv["wait_ns"]
            if d_ticks == 0 and d_run < 1e6:
                continue
            threads.append(dict(pid=key[0], tid=key[1], comm=v["comm"], cpu=v["cpu"], node=v["node"], cpu_ms=round(d_ticks * 1000 / hz, 1),
                                run_ms=round(d_run / 1e6, 2), wait_ms=round(d_wait / 1e6, 2), slices=v["slices"] - pv["slices"], nv=v["nv"] - pv["nv"]))
        mem = {}
        for line in (read("/proc/meminfo") or "").splitlines():
            k = line.split(":")[0]
            if k in ("MemAvailable", "Shmem", "SwapFree", "Dirty"):
                mem[k] = int(line.split()[1])
        rec = dict(kind="sample", t_mono=t, threads=threads, loadavg=(read("/proc/loadavg") or "").split()[:3],
                   psi_cpu=(read("/proc/pressure/cpu") or "").splitlines()[:1], psi_mem=(read("/proc/pressure/memory") or "").splitlines()[:1], mem=mem)
        fh.write(json.dumps(rec) + "\n")
        prev = cur
        time.sleep(a.interval)

if __name__ == "__main__":
    main()
