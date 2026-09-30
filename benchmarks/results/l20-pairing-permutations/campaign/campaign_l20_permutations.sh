#!/bin/bash
# Permutation campaign on the L20 (docs/preregistration/2026-09-28-permutation-campaign.md, addendum 1).
# Same harness, configs, cell order and design as the A100 script; only the machine checks differ.
# Environment: ~/perm-l20/venv = the PyPI vLLM 0.29.0 wheel extracted unmodified plus tracer v2
# (apply_tracer_v2.py), other packages from ~/inference/venv-vllm (pip RECORD hashes verified).
set -u
B=$HOME/perm-l20
export CUDA_VISIBLE_DEVICES=0 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1
export CUDA_HOME=$HOME/inference/venv-vllm/lib/python3.12/site-packages/nvidia/cu13
export PATH=$B/venv/bin:$CUDA_HOME/bin:$PATH
MODEL=$HOME/inference/models/Qwen3-4B
cd $B/repo
git status --porcelain | grep -q . && { echo "DIRTY TREE"; exit 1; }
SP=$B/venv/lib/python3.12/site-packages
[ "$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null | tail -1)" = 0.29.0 ] || { echo "VLLM NOT 0.29.0"; exit 1; }
# tracer v2 applied to the pristine 0.29.0 wheel, nothing else
[ "$(md5sum < $SP/vllm/v1/engine/core.py | cut -c1-32)" = 0b151b41a22e311290b29237c0a1807c ] || { echo "core.py NOT TRACER V2"; exit 1; }
[ "$(md5sum < $SP/vllm/v1/worker/gpu/model_runner.py | cut -c1-32)" = 7d281bfa762680ac5cac54fbb9648ea9 ] || { echo "model_runner.py NOT TRACER V2"; exit 1; }
# Qwen/Qwen3-4B weights as published (Hugging Face LFS sha256)
(cd $MODEL && sha256sum -c --quiet) <<'SUMS' || { echo "MODEL WEIGHTS DIFFER"; exit 1; }
328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223  model-00001-of-00003.safetensors
6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5  model-00002-of-00003.safetensors
e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1  model-00003-of-00003.safetensors
SUMS
USED=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits); [ "$USED" -gt 500 ] && { echo "GPU busy $USED"; exit 2; }
R=$B/results/l20-permutations-$(git rev-parse --short HEAD); mkdir -p $R
nvidia-smi > $R/nvidia-smi.txt
pip freeze 2>/dev/null > $R/pip-freeze.txt
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -lms 500 > $R/telemetry.csv &
TEL=$!
for cell in n2:0 n4:1 n8:0 n16:1 n2:1 n4:0 n8:1 n16:0; do
  c=${cell%:*}; s=${cell#*:}
  python scripts/measure_pairing_permutations.py --model $MODEL --config $c --layout-seed $s --blocks 12 --warmup 2 --trace-dir $R > $R/$c-L$s.log 2>&1
  rc=$?; echo "$(date +%H:%M:%S) $c layout $s exit=$rc"
  [ $rc -ne 0 ] && { kill $TEL; echo "CELL_FAILED $c $s"; exit 3; }
done
kill $TEL
python scripts/measure_pairing_permutations.py --analyze --trace-dir $R --output $R/perm.json > $R/analyze.log 2>&1
echo "L20_PERMUTATIONS_DONE $R"
