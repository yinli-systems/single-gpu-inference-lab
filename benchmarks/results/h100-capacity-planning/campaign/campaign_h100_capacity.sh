#!/bin/bash
# H100 capacity-planning campaign (docs/preregistration/2026-10-02-capacity-planning.md).
# 27 live replay cells: Azure code x {0.75,1,1.5,2,3} and Mooncake tool-agent x {0.5,0.6,0.7,0.8},
# each with b2048 (default), b8192 and b4096t1024. One vLLM server per cell; cells are spread over the
# GPUs given as arguments (default 0 1 2 3), one worker per GPU, each worker running its cells in order.
set -u
GPUS=${*:-0 1 2 3}
export VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 PATH=/root/lab/venv/bin:$PATH
cd /root/lab/single-gpu-inference-lab
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
TR=/root/lab/traces
(cd $TR && sha256sum -c --quiet) <<'SUMS' || { echo "TRACES DIFFER"; exit 1; }
48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71  toolagent_trace.jsonl
54e9a6d2a4bd06ba1e060304b900abbc74cbea53de96506e60fe5bb4f2277fb6  AzureLLMInferenceTrace_code.csv
SUMS
for g in $GPUS; do
  USED=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$USED" -gt 500 ] && { echo "GPU $g busy ($USED MiB)"; exit 2; }
done
R=/root/lab/results/h100-capacity-$(git rev-parse --short HEAD); mkdir -p $R
nvidia-smi > $R/nvidia-smi.txt

CELLS=()
for cfg in b2048:2048:0 b8192:8192:0 b4096t1024:4096:1024; do
  for r in 0.75 1 1.5 2 3; do CELLS+=("azure-code:azure:$TR/AzureLLMInferenceTrace_code.csv:$cfg:$r"); done
  for r in 0.5 0.6 0.7 0.8; do CELLS+=("mooncake-agent:mooncake:$TR/toolagent_trace.jsonl@3:$cfg:$r"); done
done

worker() {  # worker INDEX GPU NWORKERS
  local i=$1 gpu=$2 n=$3 k=0
  for cell in "${CELLS[@]}"; do
    if [ $((k % n)) -eq $i ]; then
      IFS=: read -r name kind path cname mbt thr rate <<< "$cell"
      out=$R/$name-$cname-x$rate
      CUDA_VISIBLE_DEVICES=$gpu python scripts/replay_trace_serving.py --model /root/lab/models/Qwen3-4B \
        --port $((8200 + i)) --trace "$kind:$path" --rate-scale $rate --window-s 240 \
        --max-num-batched-tokens $mbt --long-prefill-token-threshold $thr \
        --trace-dir $out-trace --output $out.json > $out.log 2>&1
      echo "$(date +%H:%M:%S) gpu$gpu $name-$cname-x$rate exit=$?"
    fi
    k=$((k + 1))
  done
}
GA=($GPUS); N=${#GA[@]}
for i in $(seq 0 $((N - 1))); do worker $i ${GA[$i]} $N & done
wait
python scripts/analyze_capacity_planning.py --live $R --pred benchmarks/results/h100-capacity-planning/predictions \
  --output $R/verdicts.json > $R/verdicts.log 2>&1
echo "H100_CAPACITY_DONE $R"
