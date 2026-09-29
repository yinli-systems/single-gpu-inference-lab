# Corrections preserved with the evidence

1. The old HTTP log's trailing215/750/etc. values were output tokens/second, not
milliseconds. Earlier latency/direction interpretations based on them were wrong.
Only named JSON fields and their units may be reported.
2. The formal HTTP total1296 is all three arms. Cap had432 requests; two differed
from pristine. The candidate parity denominator is2/432, not2/1296.
3. A successful Slurm exit or a complete HTTP batch is not token parity. New
uninstrumented replays with zero mismatches do not erase the old failure.
4. Pure kernel/numerical equality on fixed tensors does not establish full-model
batch invariance. Fixed RNG seeds alone are not a matched execution contract.
5. Most earlier locality_packet8 states retained identity; its old sentinel avoidance
was fallback, not proof of a faster or universally safe rearrangement.
6. Graph/batch logical histories match only part of the execution state. The actual
KV prefix, hidden states, graph bucket and earlier prefill shapes may still differ.
7. The confirmed delayed-cleanup truncation and the two old full128-token value
divergences are different failures. Fixing the former does not establish the latter's
cause or authorize performance promotion.
8. All old confirmatory/development data are exposed. No new cap/wide/shape-guard
performance claim is made, no favorable blocks are discarded, and no universal
no-regression guarantee follows from a conditional aggregate confidence interval.
