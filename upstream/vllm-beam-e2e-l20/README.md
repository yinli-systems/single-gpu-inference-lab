# Beam search end to end on the L20 (both beam patches, vLLM 0.29.0, Qwen2.5-0.5B-Instruct)

`harness/run_l20_ab.sh` alternates stock and patched arms (stock, patched, stock, patched) with
the node-local venv files swapped and restored (md5-checked); `harness/beam_e2e.py` runs one
warm-up and 5 timed `LLM.beam_search` calls per cell; `harness/compare_arms.py` makes
`results/compare.txt`. `harness/run_l20_eq.sh` repeats a subset under `VLLM_BATCH_INVARIANT=1`
for bit-exact comparison (`results/batch-invariant-equality.txt`).

| Cell | stock (s) | patched (s) | speedup |
| --- | ---: | ---: | ---: |
| plain p128 B=4 / 8 | 0.45 / 0.40–0.49 | 0.43 / 0.46 | ~1.0× (noise) |
| plain p128 B=32 | 1.28 | 0.69 | 1.85× |
| plain p2048 B=16 | 1.30 | 0.94 | 1.38× |
| plain p2048 B=32 | 3.20 | 1.64 | 1.95× |
| JSON B=8 | 2.34 | 1.53 | 1.53× |
| JSON B=16 | 5.55 | 2.43 | 2.29× |
| JSON B=32 | 15.59 | 4.15 | 3.76× |

Speedup = best of the two stock medians / best of the two patched medians. `sbatch_beam_ab.sh`
is the ParaCloud version of the same job (never ran: no GPU free on the permitted partitions).
