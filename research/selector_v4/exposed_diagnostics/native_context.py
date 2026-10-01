"""Exposed-only diagnosis of prepared native calls around a private resource module.

No untouched geometry, release authority, full-plan or HTTP performance claim.
The native and resource calls share every input/output/workspace/metadata buffer.
"""

from __future__ import annotations
import argparse, gc, hashlib, itertools, json, math, os, time
from pathlib import Path

CELLS = [
    {
        "case": "exposed-canary-v4-07",
        "dtype": "bfloat16",
        "layout": "paged",
        "q": [48, 55, 114, 158, 186, 221, 264, 333, 381, 437],
        "cached": [24576, 13056, 6336, 3008, 2176, 512, 448, 192, 128, 64],
    },
    {
        "case": "exposed-canary-v4-07",
        "dtype": "float16",
        "layout": "ragged",
        "q": [48, 55, 114, 158, 186, 221, 264, 333, 381, 437],
        "cached": [24576, 13056, 6336, 3008, 2176, 512, 448, 192, 128, 64],
    },
    {
        "case": "exposed-canary-v4-03",
        "dtype": "float16",
        "layout": "paged",
        "q": [57, 81, 130, 214, 278, 359, 433],
        "cached": [24576, 9280, 3712, 3712, 256, 192, 64],
    },
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


def run(out, rep):
    import torch
    from flashinfer.prefill import (
        BatchPrefillWithRaggedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        make_prefill_resource_runner,
    )
    from flashinfer.jit import env

    if out.exists():
        raise FileExistsError("preserve all previous evidence")
    out.mkdir(parents=True)
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) > 12:
        raise RuntimeError("batch CPU binding not established; do not use unallocated cores")
    os.sched_setaffinity(0, set(affinity[:2]))
    torch.set_num_threads(1)
    props = torch.cuda.get_device_properties(0)
    save(
        out / "environment.json",
        {
            "gpu_name": props.name,
            "gpu_uuid": str(props.uuid),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "affinity": sorted(os.sched_getaffinity(0)),
            "script_sha256": sha(__file__),
            "source": "d7683dd8e86af2f37e8d25162c45b01f80824378",
            "rep": rep,
            "fresh_cases_consumed": 0,
            "prepared_run_only": True,
            "serving_qualified": False,
        },
    )
    rows = []
    numerics = []

    def libraries():
        return {
            str(f): sha(f)
            for f in Path(env.FLASHINFER_JIT_DIR).rglob("*.so")
            if not f.name.startswith("experimental_resource_")
        }

    def ip(xs):
        return torch.tensor([0, *itertools.accumulate(xs)], dtype=torch.int32, device="cuda")

    for cell_index, c in enumerate(CELLS):
        torch.manual_seed(70101 + cell_index)
        dtype = getattr(torch, c["dtype"])
        qs = c["q"]
        lengths = [a + b for a, b in zip(qs, c["cached"], strict=True)]
        q = torch.randn(sum(qs), 32, 128, dtype=dtype, device="cuda")
        ws = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
        if c["layout"] == "ragged":
            k = torch.randn(sum(lengths), 8, 128, dtype=dtype, device="cuda")
            v = torch.randn_like(k)
            w = BatchPrefillWithRaggedKVCacheWrapper(ws, backend="fa2")
            w.plan(
                ip(qs),
                ip(lengths),
                32,
                8,
                128,
                causal=True,
                q_data_type=dtype,
                disable_split_kv=True,
            )
        else:
            pages = [math.ceil(n / 16) for n in lengths]
            k = torch.randn(sum(pages), 16, 8, 128, dtype=dtype, device="cuda")
            v = torch.randn_like(k)
            w = BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
            w.plan(
                ip(qs),
                ip(pages),
                torch.randperm(sum(pages), device="cuda").to(torch.int32),
                torch.tensor([(n - 1) % 16 + 1 for n in lengths], dtype=torch.int32, device="cuda"),
                32,
                8,
                128,
                16,
                causal=True,
                q_data_type=dtype,
                disable_split_kv=True,
            )
        output = torch.empty_like(q)
        lse = torch.empty(q.shape[0], 32, dtype=torch.float32, device="cuda")
        kw = {"out": output, "lse": lse, "return_lse": True}
        inputs = [q, k, v]
        native = lambda: w.run(q, (k, v), **kw) if c["layout"] == "paged" else w.run(q, k, v, **kw)
        native()
        reference = [output.clone(), lse.clone()]
        before = libraries()
        module = w._cached_module
        plan = list(w._plan_info)
        runner = make_prefill_resource_runner(
            w, inputs, qo_lengths=qs, kv_lengths=lengths, run_kwargs=kw
        )
        if not runner._eligible or not runner._prepare(inputs):
            raise RuntimeError(
                "target not supported: " + str(getattr(runner, "_failure_reason", None))
            )
        proxy = runner._proxy
        if proxy._float_workspace_buffer.data_ptr() != w._float_workspace_buffer.data_ptr():
            raise RuntimeError("workspace not shared")
        resource = lambda: (
            proxy.run(q, (k, v), **kw) if c["layout"] == "paged" else proxy.run(q, k, v, **kw)
        )
        fallback = lambda: runner.forward(inputs, tactic=1)
        for call in (resource, native, fallback, native):
            call()
            torch.cuda.synchronize()
            assert torch.equal(output, reference[0]) and torch.equal(lse, reference[1])
        assert runner.receipt is None and w._cached_module is module and w._plan_info == plan
        assert all(libraries().get(n) == h for n, h in before.items())
        numerics.append(
            {
                "cell": c,
                "plan": plan,
                "exact": True,
                "native_module_unchanged": True,
                "native_libraries": before,
                "input_pointers": [t.data_ptr() for t in inputs],
                "output_pointers": [output.data_ptr(), lse.data_ptr()],
                "float_workspace_pointer": ws.data_ptr(),
                "int_workspace_pointer": w._int_workspace_buffer.data_ptr(),
                "prepared_metadata_versions": [
                    (n, t._version)
                    for n, t in vars(w).items()
                    if n.endswith("_buf") and isinstance(t, torch.Tensor)
                ],
            }
        )
        calls = {
            "native": native,
            "native_after_resource": native,
            "uncertified_fallback": fallback,
            "resource": resource,
        }
        graphs = {}
        for arm, call in calls.items():
            for n in (1, 16):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(n):
                        call()
                graphs[arm, n] = g
        for mode, n in [
            ("eager_prepared", 1),
            ("graph1_pure_replay", 1),
            ("graph16_pure_replay", 16),
        ]:
            actions = {
                arm: (call if mode == "eager_prepared" else graphs[arm, n].replay)
                for arm, call in calls.items()
            }
            for _ in range(3):
                actions["native"]()
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            for _ in range(16):
                actions["native"]()
            torch.cuda.synchronize()
            pilot = (time.perf_counter_ns() - start) / 1e6 / 16
            count = min(4096, 2 ** max(0, math.ceil(math.log2(max(1, 384 / pilot)))))
            for comparison, other in [
                ("native_context", "native_after_resource"),
                ("fallback_overhead", "uncertified_fallback"),
                ("resource_gain", "resource"),
            ]:
                for block in range(24):
                    sequence = (
                        ("native", other, other, "native")
                        if block % 2 == 0
                        else (other, "native", "native", other)
                    )
                    for position, arm in enumerate(sequence):
                        (resource if arm == "native_after_resource" else native)()
                        torch.cuda.synchronize()
                        a = torch.cuda.Event(enable_timing=True)
                        b = torch.cuda.Event(enable_timing=True)
                        enabled = gc.isenabled()
                        gc.disable()
                        try:
                            start = time.perf_counter_ns()
                            a.record()
                            for _ in range(count):
                                actions[arm]()
                            b.record()
                            b.synchronize()
                            wall = (time.perf_counter_ns() - start) / 1e6
                            device = a.elapsed_time(b)
                        finally:
                            if enabled:
                                gc.enable()
                        if wall < 120:
                            raise RuntimeError("scored window below minimum")
                        rows.append(
                            {
                                "cell_index": cell_index,
                                "comparison": comparison,
                                "rep": rep,
                                "execution_mode": mode,
                                "block": block,
                                "position": position,
                                "arm": arm,
                                "count": count,
                                "kernel_calls": count * n,
                                "window_wall_ms": wall,
                                "wall_us": wall * 1000 / (count * n),
                                "device_us": device * 1000 / (count * n),
                            }
                        )
                save(
                    out / "progress.json",
                    {
                        "cell_index": cell_index,
                        "mode": mode,
                        "comparison": comparison,
                        "rows": len(rows),
                    },
                )
                print(
                    "COMPLETE_CELL_MODE_PAIR", cell_index, mode, comparison, len(rows), flush=True
                )
        assert (
            all(libraries().get(n) == h for n, h in before.items()) and w._cached_module is module
        )
        save(out / "measurements.json", rows)
        save(out / "numerics.json", numerics)
        del graphs, runner, proxy, w, ws, q, k, v, output, lse, reference
        gc.collect()
        torch.cuda.empty_cache()
    save(
        out / "complete.json",
        {
            "complete": True,
            "rows": len(rows),
            "rep": rep,
            "files": {f.name: sha(f) for f in out.glob("*.json")},
            "fresh_cases_consumed": 0,
            "qualified_release": False,
            "qualified_http": False,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rep", type=int, required=True)
    a = p.parse_args()
    run(a.out, a.rep)
