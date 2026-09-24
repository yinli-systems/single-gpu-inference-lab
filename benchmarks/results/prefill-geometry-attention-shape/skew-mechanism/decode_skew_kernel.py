"""Paged FlashAttention decode kernel, one layer of Qwen3-4B attention (32 q / 8 kv heads, dim 128, bf16),
8 sequences, balanced 8x4096(+~550 generated) vs skewed 4x7936+4x256 at equal KV sum. Times the kernel
exactly as vLLM 0.29 calls it for decode, for num_splits = the CUDA-graph value and 0 (FA heuristic)."""
import json, sys, torch
from vllm.vllm_flash_attn import flash_attn_varlen_func
try:
    from vllm.config import AttentionConfig
    cg_splits = AttentionConfig().flash_attn_max_num_splits_for_cuda_graph
except Exception as e:
    cg_splits = 32
torch.manual_seed(0)
dev = "cuda"; H, KVH, D, BS = 32, 8, 128, 16
GEN = 550   # ~ decoded tokens at the measured point (aggregate 35.2k)
cases = {"balanced": [4096 + GEN] * 8, "skewed": [7936 + GEN] * 4 + [256 + GEN] * 4}
nblocks = sum((l + BS - 1) // BS for l in cases["balanced"]) * 2 + 64
kc = torch.randn(nblocks, BS, KVH, D, dtype=torch.bfloat16, device=dev)
vc = torch.randn_like(kc)
out = {"gpu": torch.cuda.get_device_name(), "cg_num_splits": cg_splits, "results": {}}
for splits in (0,):
    for name, lens in cases.items():
        maxb = max((l + BS - 1) // BS for l in lens)
        bt = torch.zeros(8, maxb, dtype=torch.int32, device=dev); nxt = 0
        for i, l in enumerate(lens):
            nb = (l + BS - 1) // BS; bt[i, :nb] = torch.arange(nxt, nxt + nb, device=dev); nxt += nb
        q = torch.randn(8, H, D, dtype=torch.bfloat16, device=dev)
        cu_q = torch.arange(9, dtype=torch.int32, device=dev)
        seqused = torch.tensor(lens, dtype=torch.int32, device=dev)
        o = torch.empty_like(q)
        f = lambda: flash_attn_varlen_func(q=q, k=kc, v=vc, out=o, cu_seqlens_q=cu_q, max_seqlen_q=1, seqused_k=seqused,
                                           max_seqlen_k=max(lens), softmax_scale=D ** -0.5, causal=True, block_table=bt,
                                           num_splits=splits, fa_version=2)
        for _ in range(50): f()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(36): f()           # one decode step = 36 layers
        for _ in range(20): g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(200):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
        ts.sort()
        out["results"][f"splits{splits}-{name}"] = {"p50_ms_36_layers": ts[100], "p95_ms": ts[190]}
for splits in (0,):
    b, s = out["results"][f"splits{splits}-balanced"]["p50_ms_36_layers"], out["results"][f"splits{splits}-skewed"]["p50_ms_36_layers"]
    print(f"{out['gpu']}  num_splits={splits:2d}: attention over 36 layers balanced {b:.3f} ms, skewed {s:.3f} ms, skewed/balanced {s/b:.3f}, delta {s-b:+.3f} ms")
json.dump(out, open(sys.argv[1], "w"), indent=1)
