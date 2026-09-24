#!/bin/bash
# vLLM 0.30.0 replication on the L20 (pre-registration addendum 9 H): every shape-campaign cell on
# Qwen3-4B, then the pairing swap. Refuses a busy GPU or a dirty tree; kills nothing.
set -u
source ~/inference/vllm030_env.sh
export VLLM_NO_USAGE_STATS=1
cd ~/inference/single-gpu-inference-lab
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
python -c "import vllm; assert vllm.__version__ == \"0.30.0\", vllm.__version__"
R=~/inference/results/vllm030-Qwen3-4B-$(git rev-parse --short HEAD); mkdir -p $R/trace
M=~/inference/models/Qwen3-4B; nvidia-smi > $R/nvidia-smi.txt; pip freeze > $R/pip-freeze.txt
run() { local name=$1; shift
  local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -gt 500 ] && { echo "GPU busy (${used} MiB) - stopping"; exit 2; }
  python scripts/measure_prefill_interference.py --model $M --vllm-bin vllm --repeats 3 --trace-dir $R/trace --output $R/$name.json --extra-server-args=--enable-scale-out "$@" > $R/$name.log 2>&1
  local rc=$?; echo "$(date +%H:%M:%S) $name exit=$rc"; [ $rc -ne 0 ] && { echo "CELL_FAILED $name"; exit 3; }; }
run part-1x1024  --chunk-budgets 1024 --background 8 --background-tokens 4096 --inject 1 --long-tokens 16384
run part-2x512   --chunk-budgets 1024 --background 8 --background-tokens 4096 --inject 2 --long-tokens 16384 --long-prefill-token-threshold 512
run part-4x256   --chunk-budgets 1024 --background 8 --background-tokens 4096 --inject 4 --long-tokens 16384 --long-prefill-token-threshold 256
run part2-1x2048 --chunk-budgets 2048 --background 8 --background-tokens 4096 --inject 1 --long-tokens 16384
run part2-4x512  --chunk-budgets 2048 --background 8 --background-tokens 4096 --inject 4 --long-tokens 16384 --long-prefill-token-threshold 512
run part2-8x256  --chunk-budgets 2048 --background 8 --background-tokens 4096 --inject 8 --long-tokens 16384 --long-prefill-token-threshold 256
for L in 4096 8192 16384 32768; do run ctx-bg8-L$L --chunk-budgets 512,2048 --background 8 --background-tokens 4096 --inject 2 --long-tokens $L; done
for B in 4 16 32; do run load-bg$B-L16384 --chunk-budgets 512 --background $B --background-tokens 4096 --inject 2 --long-tokens 16384; done
run skew-bal  --chunk-budgets 512 --background 8 --background-prompt-tokens-list 4096 --background-tokens 4096 --inject 2 --long-tokens 16384
run skew-skew --chunk-budgets 512 --background 8 --background-prompt-tokens-list 7936,256 --background-tokens 4096 --inject 2 --long-tokens 16384
RP=$R/pairswap; mkdir -p $RP
for i in 0 1 2 3 4 5; do python scripts/measure_pairing_swap.py --model $M --config-index $i --trace-dir $RP > $RP/config$i.log 2>&1; echo "$(date +%H:%M:%S) pairswap config $i exit=$?"; done
python scripts/measure_pairing_swap.py --analyze --trace-dir $RP --slope 6.5 --output $RP/pairswap.json
echo "VLLM030_DONE $R"
