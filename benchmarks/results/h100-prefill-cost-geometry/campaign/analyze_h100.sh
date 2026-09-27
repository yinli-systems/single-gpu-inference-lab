#!/bin/bash
# Regenerates every H100 summary in this artifact from raw/ (run from the repository root).
#   PY=python bash benchmarks/results/h100-prefill-cost-geometry/campaign/analyze_h100.sh
set -eu
PY=${PY:-python}
A=benchmarks/results/h100-prefill-cost-geometry
L=benchmarks/results
MODELS="Qwen3-4B Qwen3-8B Qwen2.5-1.5B-Instruct Qwen2.5-7B-Instruct"
STEPS=""; for m in $MODELS; do STEPS="$STEPS H100-$m=$A/raw/$m/steps.csv"; done
$PY scripts/analyze_partition_geometry.py --steps $STEPS --output $A/partition.json > $A/partition.md
$PY scripts/analyze_decode_kv_skew.py --steps $STEPS --output $A/decode-kv-skew.json > $A/decode-kv-skew.md
$PY scripts/analyze_m2_variants.py --steps $STEPS --output $A/m2-variants.json > $A/m2-variants.md
$PY scripts/analyze_learned_baseline.py --steps $STEPS --output $A/learned-baseline.json > $A/learned-baseline.md
$PY scripts/analyze_split_term.py --steps \
  H100-Qwen3-4B=$A/raw/Qwen3-4B/steps.csv:$A/raw/pairswap-Qwen3-4B/pairswap.json \
  H100-Qwen3-8B=$A/raw/Qwen3-8B/steps.csv:$A/raw/pairswap-Qwen3-8B/pairswap.json \
  H100-Qwen2.5-1.5B-Instruct=$A/raw/Qwen2.5-1.5B-Instruct/steps.csv \
  H100-Qwen2.5-7B-Instruct=$A/raw/Qwen2.5-7B-Instruct/steps.csv \
  L20-Qwen3-4B-posthoc=$L/l20-prefill-cost-geometry/steps.csv:$L/prefill-pairing-swap/raw/L20-Qwen3-4B/pairswap.json \
  A100-Qwen3-4B-posthoc=$L/a100-prefill-cost-geometry/raw/Qwen3-4B/steps.csv:$L/prefill-pairing-swap/raw/A100-Qwen3-4B/pairswap.json \
  A100-Qwen3-8B-posthoc=$L/a100-prefill-cost-geometry/raw/Qwen3-8B/steps.csv \
  --output $A/split-term.json > $A/split-term.md
SHAPE=""; for d in $L/prefill-geometry-attention-shape/raw/*; do SHAPE="$SHAPE $(basename $d)=$d/steps.csv"; done
$PY scripts/analyze_published_predictors.py --steps $STEPS \
  L20-Qwen3-4B=$L/l20-prefill-cost-geometry/steps.csv A100-Qwen3-4B=$L/a100-prefill-cost-geometry/raw/Qwen3-4B/steps.csv \
  A100-Qwen3-8B=$L/a100-prefill-cost-geometry/raw/Qwen3-8B/steps.csv $SHAPE \
  --pairswap L20-vllm029=$L/prefill-pairing-swap/raw/L20-Qwen3-4B/pairswap.json A100-vllm029=$L/prefill-pairing-swap/raw/A100-Qwen3-4B/pairswap.json \
  H100-vllm029-Qwen3-4B=$A/raw/pairswap-Qwen3-4B/pairswap.json H100-vllm029-Qwen3-8B=$A/raw/pairswap-Qwen3-8B/pairswap.json \
  H100-sglang-Qwen3-4B=$A/raw/sglang-pairswap-Qwen3-4B/pairswap.json \
  --output $A/published-predictors.json > $A/published-predictors.md
for S in qwen3-4b qwen25-7b; do
  EXTRA=""; [ $S = qwen3-4b ] && EXTRA="--model-pairswap $A/raw/pairswap-Qwen3-4B/pairswap.json --layers 36"
  $PY scripts/measure_kernel_pairing.py --analyze $A/raw/kernel/kernel-$S.json $EXTRA --output $A/kernel-$S.json > $A/kernel-$S.md
done
T=$(mktemp -d); cp $A/raw/staircase/* $T/; gunzip -f $T/*.gz
$PY scripts/measure_chunk_staircase.py --analyze --trace-dir $T --output $A/staircase.json > $A/staircase.md; rm -rf $T
echo done
