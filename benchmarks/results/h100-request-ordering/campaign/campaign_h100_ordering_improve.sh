#!/bin/bash
# Addendum 7: LRU cache-depth estimate and aging. Tuning window picks beta*, then the primary window.
set -u
export CUDA_VISIBLE_DEVICES=0 VLLM_NO_USAGE_STATS=1 HF_HOME=/root/lab/hf HF_HUB_OFFLINE=1
export PATH=/root/lab/venv/bin:$PATH
cd /root/lab/single-gpu-inference-lab
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
[ "$(readlink -f /usr/local/cuda)" = /usr/local/cuda-12.8 ] || { echo "CUDA DEFAULT NOT 12.8"; exit 1; }
USED=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits); [ "$USED" -gt 500 ] && { echo "GPU busy $USED"; exit 2; }
R=/root/lab/results/h100-ordering-improve-$(git rev-parse --short HEAD); mkdir -p $R
nvidia-smi > $R/nvidia-smi.txt
TR=/root/lab/traces M=/root/lab/models/Qwen3-4B COEF=10.1767,0.9532 CAP=455008
run() { local name=$1; shift
  python scripts/replay_trace_serving.py --model $M --vllm-bin /root/lab/venv/bin/vllm --max-num-batched-tokens 2048 --window-s 240 --output $R/$name.json "$@" > $R/$name.log 2>&1
  echo "$(date +%H:%M:%S) $name exit=$? engine_failures=$(grep -c 'EngineCore failed' $R/$name/server.log 2>/dev/null)"; }
TUNE="--phases tune=mooncake:$TR/toolagent_trace.jsonl@3:1:240:240"
PRIM="--trace mooncake:$TR/toolagent_trace.jsonl@3 --rate-scale 1"
run tune-fcfs $TUNE
for B in 0.01 0.05 0.25; do run tune-cost-lru-age$B $TUNE --priority cost --cost-coef $COEF --cache-capacity-tokens $CAP --age-beta $B; done
BSTAR=$(python - $R <<'PY'
import json, sys
R = sys.argv[1]
def st(n):
    d = json.load(open(f"{R}/{n}.json")); return d["summary"]["goodput_rps"]["ttft<=5s,tpot<=100ms"], d["summary"]["ttft_p50_p90_p99"][2]
g0, p0 = st("tune-fcfs")
c = {b: st(f"tune-cost-lru-age{b}") for b in ("0.01", "0.05", "0.25")}
ok = [b for b, (g, p) in c.items() if p <= p0]
print(max(ok, key=lambda b: c[b][0]) if ok else min(c, key=lambda b: c[b][1]))
PY
)
echo "beta* = $BSTAR" | tee $R/beta-star.txt
ARMS=("uncached-lru" "cost-lru" "cost-lru-age")
for rep in 0 1; do
  for j in 0 1 2; do a=${ARMS[$(( (j + rep) % 3 ))]}
    case $a in
      uncached-lru) run mooncake-x1-$a-r$rep $PRIM --priority uncached --cache-capacity-tokens $CAP;;
      cost-lru) run mooncake-x1-$a-r$rep $PRIM --priority cost --cost-coef $COEF --cache-capacity-tokens $CAP;;
      cost-lru-age) run mooncake-x1-$a-r$rep $PRIM --priority cost --cost-coef $COEF --cache-capacity-tokens $CAP --age-beta $BSTAR;;
    esac
  done
done
echo "H100_ORDERING_IMPROVE_DONE $R"
