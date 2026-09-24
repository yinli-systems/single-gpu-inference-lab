"""Does a CUDA graph captured once (split count fixed at capture) reproduce the decode-skew penalty?
Capture the 36-layer FA2 decode call with static buffers at capture length CAP, then replay with
balanced or skewed lengths copied into the same buffers. CAP in {actual max (adaptive), 8192, 40960}."""
import sys, json, torch
from vllm.vllm_flash_attn import flash_attn_varlen_func
torch.manual_seed(0)
dev = "cuda"; H, KVH, D, BS, GEN = 32, 8, 128, 16, 550
cases = {"balanced": [4096 + GEN] * 8, "skewed": [7936 + GEN] * 4 + [256 + GEN] * 4}
MAXB = (40960 + BS - 1) // BS
kc = torch.randn(8 * MAXB // 4 + 64, BS, KVH, D, dtype=torch.bfloat16, device=dev); vc = torch.randn_like(kc)
q = torch.randn(8, H, D, dtype=torch.bfloat16, device=dev); o = torch.empty_like(q)
cu_q = torch.arange(9, dtype=torch.int32, device=dev)
seqused = torch.zeros(8, dtype=torch.int32, device=dev); bt = torch.zeros(8, MAXB, dtype=torch.int32, device=dev)
PERM = torch.randperm(kc.shape[0], generator=torch.Generator().manual_seed(1)).to(dev)
SHUFFLE = len(sys.argv) > 2 and sys.argv[2] == "shuffled"
def load(lens):
    nxt = 0; bt.zero_()
    for i, l in enumerate(lens):
        nb = (l + BS - 1) // BS; ids = torch.arange(nxt, nxt + nb, device=dev); nxt += nb
        bt[i, :nb] = PERM[ids] if SHUFFLE else ids
    seqused.copy_(torch.tensor(lens, dtype=torch.int32))
def time_graph(cap_len, lens):
    load(lens)
    f = lambda: flash_attn_varlen_func(q=q, k=kc, v=vc, out=o, cu_seqlens_q=cu_q, max_seqlen_q=1, seqused_k=seqused,
                                       max_seqlen_k=cap_len, softmax_scale=D ** -0.5, causal=True, block_table=bt, fa_version=2)
    for _ in range(10): f()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(36): f()
    return g
out = {"gpu": torch.cuda.get_device_name(), "results": {}}
for cap in ("adaptive", 40960):
    for name, lens in cases.items():
        cap_len = max(lens) if cap == "adaptive" else cap
        g = time_graph(cap_len, cases["balanced"] if cap != "adaptive" else lens)   # fixed-capture: captured once on balanced shapes
        load(lens)
        for _ in range(20): g.replay()
        torch.cuda.synchronize(); ts = []
        for _ in range(200):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
        ts.sort(); out["results"][f"{cap}-{name}"] = ts[100]
    b, s = out["results"][f"{cap}-balanced"], out["results"][f"{cap}-skewed"]
    print(f"{out['gpu']}  {'shuffled' if SHUFFLE else 'contiguous'} blocks, capture max_seqlen_k={cap!s:8s}: balanced {b:.3f} ms  skewed {s:.3f} ms  skewed/balanced {s/b:.3f}  delta {s-b:+.3f} ms (36 layers)")
json.dump(out, open(sys.argv[1], "w"), indent=1)
