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

## Qwen3-4B (same L20, same harness, `MODEL=~/inference/models/Qwen3-4B`)

Results in [`results/qwen3-4b/`](results/qwen3-4b/). On the larger model the GPU takes a larger
share of each beam step, so the relative gain is smaller, as expected. The CPU saving per step is
the same:

| Cell | stock (s) | patched (s) | speedup |
| --- | ---: | ---: | ---: |
| plain p128 B=4 / 8 | 1.00 / 1.10 | 0.98 / 1.05 | 1.02× / 1.05× (noise) |
| plain p128 B=32 | 2.13 | 1.75 | 1.22× |
| plain p2048 B=8 | 1.64 | 1.52 | 1.08× |
| plain p2048 B=16 | 2.61 | 1.98 | 1.32× |
| plain p2048 B=32 | 4.71 | 3.21 | 1.47× |
| JSON B=4 / 8 | 2.70 / 4.99 | 2.23 / 4.08 | 1.21× / 1.22× |
| JSON B=16 | 6.87 | 4.95 | 1.39× |
| JSON B=32 | 11.65 | 6.84 | 1.70× |

Equivalence under `VLLM_BATCH_INVARIANT=1`
([`results/qwen3-4b/batch-invariant-equality.txt`](results/qwen3-4b/batch-invariant-equality.txt)).
Four processes ran in the order stock1, stock2, patched1, stock3; stock3 was an extra run started
by hand after the `run_l20_eq.sh` arms. stock2 = stock3 = patched1 token for token and bit for bit
in all 9 cells. stock1, the first process to load this model under batch invariance, differs from
all three (likely cold kernel caches; not investigated further). Stock
agreeing with stock (s2 = s3) is the control that makes patched = stock meaningful.
