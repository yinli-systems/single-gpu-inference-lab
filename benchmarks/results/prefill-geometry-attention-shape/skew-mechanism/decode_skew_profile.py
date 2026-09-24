"""Locate the A100 decode-KV-skew penalty: 8 decoders, balanced 8x4096 vs skewed 4x7936+4x256 prompt KV,
same engine settings as the live cells; per-kernel CUDA time over decode-only steps with torch.profiler."""
import json, os, sys, time, collections
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
from vllm import LLM, SamplingParams
MODEL, OUT, EAGER = sys.argv[1], sys.argv[2], sys.argv[3] == "eager"
llm = LLM(model=MODEL, max_model_len=40960, gpu_memory_utilization=0.85, enforce_eager=EAGER, enable_prefix_caching=False, seed=0)
rng = torch.Generator().manual_seed(0)
def prompts(lens):
    return [{"prompt_token_ids": torch.randint(1000, 150000, (l,), generator=rng).tolist()} for l in lens]
cases = {"balanced": [4096] * 8, "skewed": [7936] * 4 + [256] * 4}
res = {"eager": EAGER, "cases": {}}
for name, lens in list(cases.items()) * 2:           # run each twice; keep the second (warm)
    sp = SamplingParams(max_tokens=600, ignore_eos=True, temperature=0)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter(); llm.generate(prompts(lens), sp, use_tqdm=False); wall = time.perf_counter() - t0
    k = collections.Counter()
    for e in prof.key_averages():
        if e.device_time_total > 0:
            k[e.key] += e.device_time_total / 1000.0
    res["cases"][name] = {"wall_s": wall, "kernels_ms": dict(k.most_common(40)), "total_cuda_ms": sum(k.values())}
json.dump(res, open(OUT, "w"), indent=1)
b, s = res["cases"]["balanced"], res["cases"]["skewed"]
print(f"eager={EAGER} wall balanced {b['wall_s']:.2f}s skewed {s['wall_s']:.2f}s | total CUDA {b['total_cuda_ms']:.0f} vs {s['total_cuda_ms']:.0f} ms")
diff = sorted(((s['kernels_ms'].get(n, 0) - b['kernels_ms'].get(n, 0), n) for n in set(b['kernels_ms']) | set(s['kernels_ms'])), reverse=True)
for d, n in diff[:8]:
    print(f"  {d:+8.1f} ms  bal {b['kernels_ms'].get(n,0):8.1f}  skew {s['kernels_ms'].get(n,0):8.1f}  {n[:110]}")
