#!/bin/bash
# Measurement contract: (a) trace coverage + CUDA-vs-wait agreement, (b) trace on/off overhead, 3 interleaved repeats.
source ~/inference/vllm_env.sh
SP=~/inference/venv-vllm/lib/python3.12/site-packages
true
cd ~/inference/single-gpu-inference-lab
for i in 1 2 3 4; do git fetch -q origin stage2-compact-sampling-mask && break; sleep 10; done
git checkout -q -- . && git clean -fdq && git checkout -q -B stage2-compact-sampling-mask origin/stage2-compact-sampling-mask
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
R=~/inference/results/campaign17-$(git rev-parse --short HEAD); mkdir -p $R $R/trace
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 2
set -x
# interleave: off, on, off, on, off, on  (bg8, L16k, chunk 512 and 2048)
for rep in 1 2 3; do
  python scripts/measure_prefill_interference.py --model ~/inference/models/Qwen3-4B --vllm-bin vllm --chunk-budgets 512,2048 --background 8 --background-tokens 4096 --inject 2 --long-tokens 16384 --repeats 1 --output $R/off-rep$rep.json > $R/off-rep$rep.log 2>&1
  python scripts/measure_prefill_interference.py --model ~/inference/models/Qwen3-4B --vllm-bin vllm --chunk-budgets 512,2048 --background 8 --background-tokens 4096 --inject 2 --long-tokens 16384 --repeats 1 --trace-dir $R/trace --output $R/on-rep$rep.json > $R/on-rep$rep.log 2>&1
done
echo CAMPAIGN17_DONE
