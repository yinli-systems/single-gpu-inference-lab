#!/bin/bash
# Permutation campaign (docs/preregistration/2026-09-28-permutation-campaign.md). A100, vLLM 0.29 + tracer v2.
# Configs n2, n4, n8, n16 x layout seeds 0, 1, interleaved; 12 randomized blocks, 2 warm-up rounds each.
set -u
export CUDA_VISIBLE_DEVICES=0 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 PATH=/root/lab/venv/bin:$PATH
cd /root/lab/single-gpu-inference-lab
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
[ "$(readlink -f /usr/local/cuda)" = /usr/local/cuda-12.8 ] || { echo "CUDA DEFAULT NOT 12.8"; exit 1; }
USED=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits); [ "$USED" -gt 500 ] && { echo "GPU busy $USED"; exit 2; }
R=/root/lab/results/a100-permutations-$(git rev-parse --short HEAD); mkdir -p $R
nvidia-smi > $R/nvidia-smi.txt
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -lms 500 > $R/telemetry.csv &
TEL=$!
for cell in n2:0 n4:1 n8:0 n16:1 n2:1 n4:0 n8:1 n16:0; do
  c=${cell%:*}; s=${cell#*:}
  python scripts/measure_pairing_permutations.py --model /root/lab/models/Qwen3-4B --config $c --layout-seed $s --blocks 12 --warmup 2 --trace-dir $R > $R/$c-L$s.log 2>&1
  rc=$?; echo "$(date +%H:%M:%S) $c layout $s exit=$rc"
  [ $rc -ne 0 ] && { kill $TEL; echo "CELL_FAILED $c $s"; exit 3; }
done
kill $TEL
python scripts/measure_pairing_permutations.py --analyze --trace-dir $R --output $R/perm.json > $R/analyze.log 2>&1
python scripts/analyze_permutation_campaign.py --input $R/perm.json --output $R/verdicts.json > $R/verdicts.log 2>&1
echo "A100_PERMUTATIONS_DONE $R"
