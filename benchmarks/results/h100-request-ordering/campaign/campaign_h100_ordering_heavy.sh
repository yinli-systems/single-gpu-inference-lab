#!/bin/bash
# Addendum 5: fcfs load probe at Mooncake scale 1, 2, 4; S* = smallest with fcfs TTFT p50 >= 1 s (else 4);
# then all four arms at S* and S*/2, 2 repeats. Arg 1: the H100 Qwen3-4B shape results dir (cost coefficients).
set -u
SHAPE=$1
export CUDA_VISIBLE_DEVICES=0 VLLM_NO_USAGE_STATS=1 HF_HOME=/root/lab/hf HF_HUB_OFFLINE=1
export PATH=/root/lab/venv/bin:$PATH
cd /root/lab/single-gpu-inference-lab
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
[ "$(readlink -f /usr/local/cuda)" = /usr/local/cuda-12.8 ] || { echo "CUDA DEFAULT NOT 12.8"; exit 1; }
R=/root/lab/results/h100-ordering-heavy-$(git rev-parse --short HEAD); mkdir -p $R
nvidia-smi > $R/nvidia-smi.txt
TR=/root/lab/traces M=/root/lab/models/Qwen3-4B
COEF=$(cat $(ls -d /root/lab/results/h100-ordering-*/cost-coef.txt | grep -v heavy | head -1) | sed -E 's/.*a,b = ([0-9.]+,[0-9.]+).*/\1/')
echo "cost coefficients a,b = $COEF (same as the addendum-3 cells)" | tee $R/cost-coef.txt
run() { local name=$1; shift
  python scripts/replay_trace_serving.py --model $M --vllm-bin /root/lab/venv/bin/vllm --max-num-batched-tokens 2048 --window-s 240 --output $R/$name.json "$@" > $R/$name.log 2>&1
  echo "$(date +%H:%M:%S) $name exit=$? engine_failures=$(grep -c 'EngineCore failed' $R/$name/server.log 2>/dev/null)"; }
arm() { case $1 in fcfs) echo "";; cost) echo "--priority cost --cost-coef $COEF";; *) echo "--priority $1";; esac; }
SSTAR=""
for S in 1 2 4; do
  run probe-mooncake-x$S-fcfs --trace mooncake:$TR/toolagent_trace.jsonl@3 --rate-scale $S
  P50=$(python -c "import json;print(json.load(open('$R/probe-mooncake-x$S-fcfs.json'))['summary']['ttft_p50_p90_p99'][0])")
  echo "probe x$S fcfs TTFT p50 $P50 s"
  [ -z "$SSTAR" ] && python -c "import sys; sys.exit(0 if $P50 >= 1.0 else 1)" && SSTAR=$S
done
[ -z "$SSTAR" ] && SSTAR=4
HALF=$(python -c "print($SSTAR/2)")
echo "S* = $SSTAR, half = $HALF" | tee $R/sstar.txt
ARMS=(fcfs prompt uncached cost)
for rep in 0 1; do
  for S in $SSTAR $HALF; do
    for j in 0 1 2 3; do a=${ARMS[$(( (j + rep * 2) % 4 ))]}
      run mooncake-x$S-$a-r$rep --trace mooncake:$TR/toolagent_trace.jsonl@3 --rate-scale $S $(arm $a)
    done
  done
done
echo "H100_ORDERING_HEAVY_DONE $R"
