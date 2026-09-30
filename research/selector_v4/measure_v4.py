"""V4.1 deployment-matched native-vs-cap measurement with symbol isolation."""
from __future__ import annotations
import argparse, gc, hashlib, itertools, json, math, os, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "selector_v32"
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))
import measure as base

from research.selector_v4.device_identity import device_uuid
from research.selector_v4.eligibility import evaluate_eligibility, effective_eligibility
from research.selector_v4.identity import TacticIdentity
from research.selector_v4.manifest_v4 import digest, load
from research.selector_v4.schema import QUALIFICATION_REVISION, TACTIC_CAP, TACTIC_NATIVE

ARMS = ("off", "cap")
EXECUTIONS = ("eager_full_call", "graph1_replay", "graph16_replay")
TARGET_WINDOW_US = 384000.0
MIN_WINDOW_US = 120000.0
_EXPECTED_CAP_ERRORS = ("invalid argument", "cudaerrorinvalidvalue")


def _power_two(value: int) -> int:
    return 1 << max(0, value - 1).bit_length()


def calibrate_eager(rt, q, torch):
    pilots = []
    for _ in range(2):
        rt.plan(inspect=False); rt.call(q); torch.cuda.synchronize()
        tick = time.perf_counter_ns()
        for __ in range(16): rt.plan(inspect=False); rt.call(q)
        torch.cuda.synchronize(); pilots.append((time.perf_counter_ns()-tick)/1000.0)
    calls = min(4096, _power_two(max(16, math.ceil(TARGET_WINDOW_US / (min(pilots)/16)))))
    return calls, pilots


def calibrate_graph(rt, count, torch) -> tuple[int, list[float]]:
    graph = rt.graphs[count]; pilots = []
    for _ in range(2):
        graph.replay(); torch.cuda.synchronize(); tick = time.perf_counter_ns()
        for __ in range(2): graph.replay()
        torch.cuda.synchronize(); pilots.append((time.perf_counter_ns() - tick) / 1000.0)
    per_replay = min(pilots) / 2.0
    repeats = min(4096, _power_two(max(1, math.ceil(TARGET_WINDOW_US / per_replay))))
    return repeats, pilots


def timed_execution(rt, execution, windows, q, torch):
    if execution == "eager_full_call":
        rt.eager_calls = int(windows[execution]["calls"])
        wall, device = base.timed_eager(rt, q, torch); kernel_calls = rt.eager_calls
    else:
        count = 1 if execution == "graph1_replay" else 16
        replays = int(windows[execution]["replays"])
        wall, device = base.timed_graph(rt, count, replays, q, torch); kernel_calls = count * replays
    elapsed = wall * kernel_calls
    if elapsed < MIN_WINDOW_US:
        raise RuntimeError(f"scored window below minimum: {execution} {elapsed:.3f}us")
    return wall, device, kernel_calls, elapsed


def operation_identity(env, case, dtype_name, layout, requested_split, info, execution, windows):
    actual_split = "split" if bool(info[14]) else "unsplit"
    operation = {
        "execution_mode": execution, "backend": "fa2", "causal": True,
        "layout": layout, "dtype": dtype_name, "requested_split": requested_split,
        "actual_split": actual_split, "num_qo_heads": 32, "num_kv_heads": 8,
        "head_dim_qk": 128, "head_dim_vo": 128,
        "page_size": 1 if layout == "ragged" else 16,
        "q": list(case["q"]), "cached": list(case["cached"]),
        "plan_signature": list(info),
        "window_config": {k: windows[execution][k] for k in ("calls", "replays") if k in windows[execution]},
    }
    environment = {
        "gpu_name": env["gpu_name"], "gpu_uuid": env["gpu_uuid"],
        "num_sms": env["num_sms"], "driver": env["driver"],
        "cuda": env["cuda"], "torch": env["torch"], "flashinfer": env["flashinfer"],
        "nvcc": env["nvcc"], "backend_source_sha256": env["backend_source_sha256"],
        "official_overlay_sha256": env["official_overlay_sha256"],
        "resource_binding_sha256": env["resource_binding_sha256"],
        "max_smem_per_sm": env["max_smem_per_sm"],
        "max_smem_per_block_optin": env["max_smem_per_block_optin"],
    }
    policy = {"execution_mode": execution, "timer": "deployment_wall", "qualification_revision": QUALIFICATION_REVISION}
    return TacticIdentity(environment, operation, policy)


def _runtime(base_module, flashinfer, torch, layout, qi, ks, vs, k, v, lengths, dtype,
             requested_split, arm, candidate, paged_bundle):
    # Always use the unmodified official planner; tactic is a separate run entry.
    rt = base_module.Runtime(flashinfer, torch, layout, qi, ks, vs, k, v, lengths,
                             dtype, requested_split, arm, False, paged_bundle)
    if candidate and arm == "cap":
        if layout == "ragged":
            rt._call = lambda q: rt.wrapper.run_resource(q, k, v, out=rt.out, lse=rt.lse, return_lse=True)
        else:
            kp, vp = paged_bundle[-2:]
            rt._call = lambda q: rt.wrapper.run_resource(q, (kp, vp), out=rt.out, lse=rt.lse, return_lse=True)
    return rt


def run(args):
    import torch, flashinfer
    if not torch.cuda.is_available() or flashinfer.__version__ != "0.7.0":
        raise RuntimeError("real CUDA and pinned FlashInfer 0.7.0 required")
    if args.out.exists(): raise FileExistsError("preserve prior evidence")
    if args.mode not in ("pristine", "paired") or not 0 <= args.rep < 3 or not 0 <= args.shard < args.shards <= 12:
        raise ValueError("bounded run identity")
    manifest = load(); all_cases = [c for c in manifest["cases"] if c["family"] == args.stage]
    cases = [c for i, c in enumerate(all_cases) if i % args.shards == args.shard]
    if not cases: raise ValueError("empty shard")

    package = Path(flashinfer.__file__).parent; root = package.parent; candidate = args.mode == "paired"
    binding_path = root / "RESOURCE_BINDING.json"; binding = None
    if candidate:
        binding = json.loads(binding_path.read_text())
        if (binding.get("selector_version") != QUALIFICATION_REVISION or binding.get("kernel_symbol_isolation") is not True or
                binding.get("resource_only_entry") is not True or binding.get("plan_vector_size") != 15 or binding.get("default_policy") != "native"):
            raise RuntimeError("candidate binding mismatch")
        for rel, expected in binding["modified_hashes"].items():
            if base.sha(root / rel) != expected: raise RuntimeError("candidate source drift " + rel)
    else:
        for rel, expected in base.EXPECTED.items():
            if base.sha(root / rel) != expected: raise RuntimeError("pristine source drift " + rel)

    args.out.mkdir(parents=True); args.refs.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
    affinity = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, set(affinity[:min(2, len(affinity))]))
    props = torch.cuda.get_device_properties(0)
    actual_uuid = device_uuid(torch)
    hardware = base.command(["nvidia-smi", "--id="+actual_uuid, "--query-gpu=name,uuid,driver_version,power.limit", "--format=csv,noheader"])
    if hardware["rc"] or not hardware["out"].strip(): raise RuntimeError("hardware identity unavailable")
    line = [x.strip() for x in hardware["out"].strip().splitlines()[0].split(",")]
    nvcc = base.command([os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "nvcc"), "--version"])
    backend_sha = base.backend_digest(root)
    resource_binding_sha = base.sha(binding_path) if candidate else backend_sha
    env = {
        "gpu_name": line[0], "gpu_uuid": line[1], "driver": line[2],
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "num_sms": props.multi_processor_count, "max_smem_per_sm": props.shared_memory_per_multiprocessor,
        "max_smem_per_block_optin": getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block),
        "torch": str(torch.__version__), "cuda": str(torch.version.cuda), "flashinfer": flashinfer.__version__,
        "nvcc": nvcc["out"].strip(), "mode": args.mode, "stage": args.stage,
        "rep": args.rep, "shard": args.shard, "shards": args.shards,
        "case_hash": manifest["case_hash"], "cases": cases,
        "source": {p.name: base.sha(p) for p in HERE.glob("*.py")},
        "official_overlay_sha256": os.environ.get("SGI_OFFICIAL_MANIFEST_SHA256"),
        "source_archive_sha256": os.environ.get("SGI_SOURCE_ARCHIVE_SHA256"),
        "backend_source_sha256": backend_sha, "resource_binding_sha256": resource_binding_sha,
        "profiled": False, "measurement_contract_revision": QUALIFICATION_REVISION,
        "kernel_symbol_isolation": bool(binding and binding.get("kernel_symbol_isolation")),
        "default_promotion": False, "serving_promotion": False,
    }
    expected_gpu = {"gpu_4090":"NVIDIA GeForce RTX 4090", "gpu_5090":"NVIDIA GeForce RTX 5090"}.get(os.environ.get("SLURM_JOB_PARTITION"))
    if env["gpu_uuid"] != actual_uuid or env["gpu_name"] != expected_gpu or not env["official_overlay_sha256"] or not env["source_archive_sha256"]:
        raise RuntimeError("environment/provenance mismatch")
    base.save(args.out / "environment.json", env)
    rows, qualifications, memory = [], [], []; started = time.monotonic()

    def process_case(case, dtype_name, layout, requested_split):
        key = {"case":case["id"], "family":case["family"], "dtype":dtype_name, "layout":layout, "split":requested_split}
        cell = digest(key); torch.manual_seed(int(cell[:8], 16)); dtype = getattr(torch, dtype_name)
        qs = list(case["q"]); lengths = [q + k for q, k in zip(qs, case["cached"])]
        q = torch.randn((sum(qs), 32, 128), device="cuda", dtype=dtype)
        k = torch.randn((sum(lengths), 8, 128), device="cuda", dtype=dtype); v = torch.randn_like(k)
        ks, vs = list(k.split(lengths)), list(v.split(lengths))
        qi = torch.tensor([0] + list(itertools.accumulate(qs)), device="cuda", dtype=torch.int32)
        paged_bundle = None
        if layout == "paged":
            page = 16; counts = [(length + page - 1) // page for length in lengths]; npages = sum(counts)
            pp = torch.tensor([0] + list(itertools.accumulate(counts)), device="cuda", dtype=torch.int32)
            order = torch.randperm(npages, device="cuda"); pi = order.to(torch.int32)
            last = torch.tensor([(length - 1) % page + 1 for length in lengths], device="cuda", dtype=torch.int32)
            kp = torch.zeros((npages, page, 8, 128), device="cuda", dtype=dtype); vp = torch.zeros_like(kp); offset = 0
            for length, count, kr, vr in zip(lengths, counts, ks, vs):
                kl = torch.zeros((count * page, 8, 128), device="cuda", dtype=dtype); vl = torch.zeros_like(kl)
                kl[:length].copy_(kr); vl[:length].copy_(vr)
                kp[order[offset:offset+count]] = kl.view(count, page, 8, 128)
                vp[order[offset:offset+count]] = vl.view(count, page, 8, 128); offset += count
            paged_bundle = (page, pp, pi, last, kp, vp)

        if not candidate:
            runtimes = {"pristine": _runtime(base, flashinfer, torch, layout, qi, ks, vs, k, v, lengths, dtype, requested_split, "pristine", False, paged_bundle)}
            actual_tactics = {"pristine": TACTIC_NATIVE}; cap_supported = None; cap_probe_error = None
        else:
            off = _runtime(base, flashinfer, torch, layout, qi, ks, vs, k, v, lengths, dtype, requested_split, "off", True, paged_bundle)
            runtimes = {"off": off}; actual_tactics = {"off": TACTIC_NATIVE}; cap_supported = False; cap_probe_error = None
        for runtime in runtimes.values(): runtime.attach(q)
        plans = {}
        for arm, runtime in list(runtimes.items()):
            info, _ = runtime.plan(); plans[arm] = info

        # Build pristine references before candidate support decisions.
        inputs = digest([base.thash(q), base.thash(k), base.thash(v)]); ref_path = args.refs / f"{cell}.pt"
        if not candidate:
            rt = runtimes["pristine"]
            for _ in range(3): rt.call(q)
            torch.cuda.synchronize(); rt.capture(q, 1); rt.capture(q, 16)
            rt.plan(); rt.call(q); torch.cuda.synchronize(); fp32 = base.oracle(q, ks, vs, qs, rt.out, rt.lse, dtype_name)
            if ref_path.exists():
                baseline = torch.load(ref_path, map_location="cpu", weights_only=True)
                if baseline["inputs"] != inputs or baseline.get("measurement_revision") != QUALIFICATION_REVISION: raise RuntimeError("stale pristine reference")
                if not torch.equal(rt.out.cpu(), baseline["out"]) or not torch.equal(rt.lse.cpu(), baseline["lse"]): raise RuntimeError("pristine repeat numerics changed")
            else:
                eager_calls, eager_pilots = calibrate_eager(rt, q, torch)
                graph1_replays, graph1_pilots = calibrate_graph(rt, 1, torch)
                graph16_replays, graph16_pilots = calibrate_graph(rt, 16, torch)
                windows = {
                    "eager_full_call":{"calls":eager_calls, "pilots_us":eager_pilots},
                    "graph1_replay":{"replays":graph1_replays, "pilots_us":graph1_pilots},
                    "graph16_replay":{"replays":graph16_replays, "pilots_us":graph16_pilots},
                }
                baseline = {"out":rt.out.cpu(), "lse":rt.lse.cpu(), "inputs":inputs, "FP32":fp32,
                    "windows":windows, "measurement_revision":QUALIFICATION_REVISION, "reference_hardware":hardware["out"]}
                torch.save(baseline, ref_path)
        else:
            if not ref_path.exists(): raise RuntimeError("missing independent pristine reference")
            baseline = torch.load(ref_path, map_location="cpu", weights_only=True); fp32 = baseline.get("FP32")
            if baseline["inputs"] != inputs or baseline["reference_hardware"] != hardware["out"]: raise RuntimeError("cross-mode reference mismatch")
        if baseline.get("measurement_revision") != QUALIFICATION_REVISION: raise RuntimeError("reference revision mismatch")
        windows = baseline["windows"]

        # Candidate support is decided before any scored timing. Ineligible or rejected cap paths
        # use a second independent native wrapper as a labeled null arm.
        if candidate:
            provisional = {ex: operation_identity(env, case, dtype_name, layout, requested_split, plans["off"], ex, windows) for ex in EXECUTIONS}
            static = {ex: evaluate_eligibility(identity) for ex, identity in provisional.items()}
            should_probe = all(item.eligible for item in static.values())
            cap = None
            if should_probe:
                cap = _runtime(base, flashinfer, torch, layout, qi, ks, vs, k, v, lengths, dtype, requested_split, "cap", True, paged_bundle)
                cap.attach(q); cap_info, _ = cap.plan(); plans["cap"] = cap_info
                if plans["off"] != cap_info or len(cap_info) != 15: raise RuntimeError("candidate plan pairing contract")
                try:
                    cap.call(q); torch.cuda.synchronize(); cap_supported = True
                except RuntimeError as exc:
                    message = str(exc)
                    if not any(token in message.lower() for token in _EXPECTED_CAP_ERRORS): raise
                    cap_probe_error = message; cap_supported = False; cap = None
            if cap is None:
                cap = _runtime(base, flashinfer, torch, layout, qi, ks, vs, k, v, lengths, dtype, requested_split, "off", True, paged_bundle)
                cap.attach(q); cap_info, _ = cap.plan(); plans["cap"] = cap_info
                if plans["off"] != cap_info or len(cap_info) != 15: raise RuntimeError("native fallback plan contract")
            runtimes["cap"] = cap; actual_tactics["cap"] = TACTIC_CAP if cap_supported else TACTIC_NATIVE

        # Warmup/capture in explicit native -> cap -> native order.
        for arm in ("pristine",) if not candidate else ("off", "cap"):
            rt = runtimes[arm]
            for _ in range(3): rt.call(q)
            torch.cuda.synchronize(); rt.capture(q, 1); rt.capture(q, 16)
        native_after_cap_exact = True
        if candidate:
            off = runtimes["off"]; off.out.fill_(float("nan")); off.lse.fill_(float("nan")); off.call(q); torch.cuda.synchronize()
            native_after_cap_exact = torch.equal(off.out.cpu(), baseline["out"]) and torch.equal(off.lse.cpu(), baseline["lse"])
            if not native_after_cap_exact: raise RuntimeError("native-after-cap numerical isolation failure")

        arm_quals = {}; identities = {}; eligibility = {}
        for arm, runtime in runtimes.items():
            runtime.out.fill_(float("nan")); runtime.lse.fill_(float("nan")); runtime.plan(); runtime.call(q); torch.cuda.synchronize()
            if not torch.equal(runtime.out.cpu(), baseline["out"]) or not torch.equal(runtime.lse.cpu(), baseline["lse"]): raise RuntimeError("full output/LSE not bit-exact " + arm)
            checks = {"eager_full_call":True}; checks.update(base.verify_graph_outputs(runtime, torch, baseline, arm))
            arm_quals[arm] = {"out_sha256":base.thash(runtime.out), "lse_sha256":base.thash(runtime.lse),
                "pristine_exact":True, "plan_info":plans[arm], "execution_checks":checks,
                "actual_tactic":actual_tactics[arm]}
            identities[arm] = {}
            for execution in EXECUTIONS:
                identity = operation_identity(env, case, dtype_name, layout, requested_split, plans[arm], execution, windows)
                identities[arm][execution] = {"key":identity.key, "payload":identity.to_dict()}
                if candidate:
                    runtime_reason = None if cap_supported else ("runtime_cap_unsupported" if cap_probe_error else "static_or_plan_ineligible")
                    eligibility[execution] = effective_eligibility(identity, cap_supported=cap_supported, runtime_reason=runtime_reason).to_dict()
                else:
                    eligibility[execution] = evaluate_eligibility(identity).to_dict()
        qualification = {
            **key, "inputs_sha256":inputs, "reference_sha256":base.sha(ref_path), "arms":arm_quals,
            "identities":identities, "eligibility":eligibility, "windows":windows, "FP32":fp32,
            "candidate_plan_core_equal": (plans["off"] == plans["cap"] if candidate else None),
            "cap_supported": cap_supported, "cap_probe_error": cap_probe_error,
            "native_after_cap_exact": native_after_cap_exact,
            "measurement_contract_revision":QUALIFICATION_REVISION,
        }
        qualifications.append(qualification)
        blocks = int(manifest["blocks"]); rng = random.Random(int(cell[:8], 16) + args.rep * 1009)
        pointers = {arm: runtime.pointers(q, k, v) for arm, runtime in runtimes.items()}
        for block in range(blocks):
            executions = list(EXECUTIONS); rng.shuffle(executions)
            for execution in executions:
                if not candidate:
                    roles = ("a","b","b","a") if block % 2 == 0 else ("b","a","a","b")
                    for position, role in enumerate(roles):
                        wall, device, calls, elapsed = timed_execution(runtimes["pristine"], execution, windows, q, torch)
                        rows.append({**key, "mode":"pristine", "arm":"pristine", "actual_tactic":TACTIC_NATIVE,
                            "comparison_group":"position", "rep":args.rep, "block":block, "execution_mode":execution,
                            "position":position, "role":role, "wall_us":wall, "device_us":device,
                            "kernel_calls":calls, "window_elapsed_us":elapsed,
                            "tactic_identity":identities["pristine"][execution]["key"]})
                else:
                    sequence = ("off","cap","cap","off") if block % 2 == 0 else ("cap","off","off","cap")
                    for position, arm in enumerate(sequence):
                        wall, device, calls, elapsed = timed_execution(runtimes[arm], execution, windows, q, torch)
                        rows.append({**key, "mode":"paired", "arm":arm, "actual_tactic":actual_tactics[arm],
                            "comparison_group":"cap", "rep":args.rep, "block":block, "execution_mode":execution,
                            "position":position, "role":"A" if arm == "off" else "B", "wall_us":wall,
                            "device_us":device, "kernel_calls":calls, "window_elapsed_us":elapsed,
                            "tactic_identity":identities[arm][execution]["key"]})
                for arm, runtime in runtimes.items():
                    if pointers[arm] != runtime.pointers(q, k, v): raise RuntimeError("storage changed " + arm)
        base.save(args.out / "progress.json", {**key, "rows":len(rows), "qualifications":len(qualifications), "elapsed":time.monotonic()-started})
        print("CELL", args.mode, args.rep, key, len(rows), flush=True)

    try:
        for case, dtype_name, layout, requested_split in itertools.product(cases, manifest["dtypes"], manifest["layouts"], manifest["requested_splits"]):
            process_case(case, dtype_name, layout, requested_split)
            allocated = int(torch.cuda.memory_allocated()); gc.collect(); torch.cuda.empty_cache(); free, total = torch.cuda.mem_get_info()
            memory.append({"case":case["id"], "dtype":dtype_name, "layout":layout, "split":requested_split,
                "allocated_before_gc":allocated, "allocated_after_gc":int(torch.cuda.memory_allocated()),
                "reserved_after_gc":int(torch.cuda.memory_reserved()), "free_device_bytes":int(free), "total_device_bytes":int(total)})
        cells = len(cases) * len(manifest["dtypes"]) * len(manifest["layouts"]) * len(manifest["requested_splits"])
        expected = cells * manifest["blocks"] * len(EXECUTIONS) * 4
        if len(rows) != expected: raise RuntimeError(f"incomplete matrix {len(rows)} != {expected}")
        base.save(args.out / "measurements.json", rows); base.save(args.out / "qualification.json", qualifications); base.save(args.out / "memory.json", memory)
        base.save(args.out / "complete.json", {"complete":True, "rows":len(rows), "expected":expected,
            "qualifications":len(qualifications), "files":{p.name:base.sha(p) for p in args.out.glob("*.json")},
            "performance_promotion":False, "serving_promotion":False})
        print("COMPLETE", args.mode, args.stage, args.rep, len(rows), flush=True)
    except BaseException as exc:
        base.save(args.out / "partial_measurements.json", rows); base.save(args.out / "qualification.json", qualifications); base.save(args.out / "partial_memory.json", memory)
        base.save(args.out / "failure.json", {"type":type(exc).__name__, "message":str(exc)})
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True); parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--mode", choices=["pristine","paired"], required=True)
    parser.add_argument("--stage", choices=["dev","canary","release","stress"], required=True)
    parser.add_argument("--rep", type=int, required=True); parser.add_argument("--shard", type=int, required=True); parser.add_argument("--shards", type=int, required=True)
    run(parser.parse_args())
