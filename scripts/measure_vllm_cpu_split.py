"""Per-process CPU-time split of a serving round (API server vs engine core)."""
import asyncio, json, os, subprocess, sys, time
sys.path.insert(0, "scripts")
from measure_vllm_feature_cost import run_round, build_prompts
from transformers import AutoTokenizer

MODEL = os.path.expanduser("~/inference/models/Qwen2.5-0.5B-Instruct")
tok = AutoTokenizer.from_pretrained(MODEL)
prompts = build_prompts(128, 96, 113); ids = [tok(p)["input_ids"] for p in prompts]
sampling = {"temperature": 1.0, "top_k": 50, "top_p": 0.95}
CLK = os.sysconf("SC_CLK_TCK")

def pids():
    api = subprocess.check_output(["pgrep", "-f", "bin/vllm serve"], text=True).split()[0]
    core = subprocess.check_output(["pgrep", "-f", "VLLM::EngineCore"], text=True).split()[0]
    return {"api": int(api), "core": int(core)}

def cpu_s(pid):
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    return (int(parts[11]) + int(parts[12])) / CLK  # utime + stime

def threads_cpu(pid):
    total = {}
    for tid in os.listdir(f"/proc/{pid}/task"):
        try:
            with open(f"/proc/{pid}/task/{tid}/stat") as f:
                raw = f.read()
            name = raw[raw.index("(") + 1: raw.rindex(")")]
            parts = raw.rsplit(")", 1)[1].split()
            total[(tid, name)] = (int(parts[11]) + int(parts[12])) / CLK
        except FileNotFoundError:
            pass
    return total

P = pids()
out = []
for lp in (None, 1, None, 1):
    before = {k: cpu_s(v) for k, v in P.items()}
    tb = {k: threads_cpu(v) for k, v in P.items()}
    t0 = time.perf_counter()
    res = asyncio.run(run_round("http://127.0.0.1:8123", "m", prompts, concurrency=64, max_tokens=256,
                                sampling=sampling, logprobs=lp, seed=113, prompt_token_ids=ids))
    wall = time.perf_counter() - t0
    after = {k: cpu_s(v) for k, v in P.items()}
    ta = {k: threads_cpu(v) for k, v in P.items()}
    threads = {}
    for k in P:
        d = {name: round(ta[k].get((tid, name), 0) - tb[k].get((tid, name), 0), 2) for (tid, name) in ta[k]}
        threads[k] = dict(sorted(((n, v) for n, v in d.items() if v > 0.05), key=lambda x: -x[1])[:6])
    row = {"logprobs": lp, "wall_s": round(wall, 2), "tok_s": round(res["output_tokens_per_s"]),
           "cpu_s": {k: round(after[k] - before[k], 2) for k in P},
           "cpu_util": {k: round((after[k] - before[k]) / wall, 2) for k in P},
           "threads": threads}
    print(json.dumps(row), flush=True); out.append(row)
