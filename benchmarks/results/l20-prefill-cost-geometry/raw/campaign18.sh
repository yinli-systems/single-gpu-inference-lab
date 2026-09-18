#!/bin/bash
# Discriminating experiments with tracer v2 (installed), 3 interleaved repeats per cell.
source ~/inference/vllm_env.sh
cd ~/inference/single-gpu-inference-lab
for i in 1 2 3 4; do git fetch -q origin stage2-compact-sampling-mask && break; sleep 10; done
git checkout -q -- . && git clean -fdq && git checkout -q -B stage2-compact-sampling-mask origin/stage2-compact-sampling-mask
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
R=~/inference/results/campaign18-$(git rev-parse --short HEAD); mkdir -p $R $R/trace
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 2
M=~/inference/models/Qwen3-4B
run() { # name, extra args...
  local name=$1; shift
  python scripts/measure_prefill_interference.py --model $M --vllm-bin vllm --repeats 3 --trace-dir $R/trace --output $R/$name.json "$@" > $R/$name.log 2>&1
}
set -x
# A. same decode batch (8) and same chunk (512 / 2048) across context 4k/8k/16k/32k  (each cell = one server, 3 repeats)
for L in 4096 8192 16384 32768; do run ctx-bg8-L$L --chunk-budgets 512,2048 --background 8 --background-tokens 4096 --inject 2 --long-tokens $L; done
# B. same prefill geometry (16k, chunk 512) while decode load changes 4/8/16/32
for B in 4 16 32; do run load-bg$B-L16384 --chunk-budgets 512 --background $B --background-tokens 4096 --inject 2 --long-tokens 16384; done
# C. constant total prefill query work 1024/step partitioned 1x1024, 2x512, 4x256 at ~16k depth (all prefills same length, injected together)
run part-1x1024 --chunk-budgets 1024 --background 8 --background-tokens 4096 --inject 1 --long-tokens 16384
run part-2x512  --chunk-budgets 1024 --background 8 --background-tokens 4096 --inject 2 --long-tokens 16384 --long-prefill-token-threshold 512
run part-4x256  --chunk-budgets 1024 --background 8 --background-tokens 4096 --inject 4 --long-tokens 16384 --long-prefill-token-threshold 256
# D. CUDA-graph boundaries: decode-only, C-1/C/C+1 around 8, 16, 24 (capture sizes 1,2,4,8,16,24,32,...)
for B in 7 8 9 15 16 17 23 24 25; do run graph-bg$B --chunk-budgets 2048 --background $B --background-tokens 2048 --inject 0 --decode-only-window-s 8 --repeats 3; done
set +x
echo CAMPAIGN18_DONE
