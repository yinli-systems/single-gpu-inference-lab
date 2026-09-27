#!/bin/bash
# Addendum 8: engine-side SLO-aware ordering (online Moore-Hodgson) vs the client-side arms.
set -u
export CUDA_VISIBLE_DEVICES=0 VLLM_NO_USAGE_STATS=1 HF_HOME=/root/lab/hf HF_HUB_OFFLINE=1
export PATH=/root/lab/venv/bin:$PATH
cd /root/lab/single-gpu-inference-lab
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
[ "$(readlink -f /usr/local/cuda)" = /usr/local/cuda-12.8 ] || { echo "CUDA DEFAULT NOT 12.8"; exit 1; }
USED=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits); [ "$USED" -gt 500 ] && { echo "GPU busy $USED"; exit 2; }
SP=$(python -c "import site;print(site.getsitepackages()[0])")
python benchmarks/results/h100-request-ordering/patches/apply_slo_scheduler.py $SP
R=/root/lab/results/h100-ordering-slo-$(git rev-parse --short HEAD); mkdir -p $R
nvidia-smi > $R/nvidia-smi.txt
TR=/root/lab/traces M=/root/lab/models/Qwen3-4B COEF=10.1767,0.9532 CAP=455008
run() { local name=$1; shift
  python scripts/replay_trace_serving.py --model $M --vllm-bin /root/lab/venv/bin/vllm --max-num-batched-tokens 2048 --window-s 240 --output $R/$name.json "$@" > $R/$name.log 2>&1
  echo "$(date +%H:%M:%S) $name exit=$? engine_failures=$(grep -c 'EngineCore failed' $R/$name/server.log 2>/dev/null)"; }
slo() { local name=$1; shift
  VLLM_EXP_SLO_ORDER=1 VLLM_EXP_SLO_LOG=$R/$name.slo.jsonl run $name "$@" --server-priority-policy; }
P1="--trace mooncake:$TR/toolagent_trace.jsonl@3 --rate-scale 1"
P15="--trace mooncake:$TR/toolagent_trace.jsonl@3 --rate-scale 1.5"
slo smoke-slo-mh $P1 --window-s 30
[ -s $R/smoke-slo-mh.slo.jsonl ] && [ "$(grep -c 'EngineCore failed' $R/smoke-slo-mh/server.log)" = 0 ] || { echo "SMOKE_FAILED"; exit 4; }
echo "smoke ok: $(tail -1 $R/smoke-slo-mh.slo.jsonl)"
slo mooncake-x1-slo-mh-r0 $P1
slo mooncake-x1-slo-mh-r1 $P1
run mooncake-x1.5-fcfs-r0 $P15
slo mooncake-x1.5-slo-mh-r0 $P15
run mooncake-x1.5-prompt-r0 $P15 --priority prompt
run mooncake-x1.5-uncached-lru-r0 $P15 --priority uncached --cache-capacity-tokens $CAP
echo "H100_ORDERING_SLO_DONE $R"
