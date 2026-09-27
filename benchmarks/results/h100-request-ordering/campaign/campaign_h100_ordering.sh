#!/bin/bash
# Addendum 3 of docs/preregistration/2026-09-27-h100-hopper-replication.md: request ordering arms on
# live traces. Arg 1: the H100 Qwen3-4B shape results dir (its steps.csv fixes the cost coefficients).
set -u
SHAPE=$1
export CUDA_VISIBLE_DEVICES=0 VLLM_NO_USAGE_STATS=1 HF_HOME=/root/lab/hf HF_HUB_OFFLINE=1
export PATH=/root/lab/venv/bin:$PATH
cd /root/lab/single-gpu-inference-lab
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
USED=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$USED" -gt 500 ] && { echo "GPU 0 busy (${USED} MiB) - refusing"; exit 2; }
R=/root/lab/results/h100-ordering-$(git rev-parse --short HEAD); mkdir -p $R
nvidia-smi > $R/nvidia-smi.txt
TR=/root/lab/traces M=/root/lab/models/Qwen3-4B
COEF=$(python - "$SHAPE/steps.csv" <<'PY'
import sys, numpy as np
sys.path.insert(0, "scripts")
import analyze_m2_variants as V
cells = V.load_cells(__import__("pathlib").Path(sys.argv[1]), True)
rows = [r for rs, _ in cells.values() for r in rs if r["ctx_reqs"] == 1]
X = np.array([[1.0, r["ctx_tokens"] / 1e3, (r["attn_proxy"] - r["gen_kv_sum"]) / 1e6] for r in rows])
y = np.array([r["cuda_ms"] for r in rows])
w, *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"{w[1]:.4f},{w[2]:.4f}")
PY
)
echo "cost coefficients a,b = $COEF (from $SHAPE/steps.csv, one-prefill steps)" | tee $R/cost-coef.txt
run() { local name=$1; shift
  python scripts/replay_trace_serving.py --model $M --vllm-bin /root/lab/venv/bin/vllm --max-num-batched-tokens 2048 --window-s 240 --output $R/$name.json "$@" > $R/$name.log 2>&1
  echo "$(date +%H:%M:%S) $name exit=$?"; }
arm() { case $1 in fcfs) echo "";; cost) echo "--priority cost --cost-coef $COEF";; *) echo "--priority $1";; esac; }
ARMS=(fcfs prompt uncached cost)
for rep in 0 1; do
  for S in 0.2 0.4; do
    for j in 0 1 2 3; do a=${ARMS[$(( (j + rep * 2) % 4 ))]}
      run mooncake-x$S-$a-r$rep --trace mooncake:$TR/toolagent_trace.jsonl@3 --rate-scale $S $(arm $a)
    done
  done
done
for a in "${ARMS[@]}"; do run azurecode-x0.5-$a-r0 --phases code=azure:$TR/AzureLLMInferenceTrace_code.csv:0.5:180:240 $(arm $a); done
echo "H100_ORDERING_DONE $R"
