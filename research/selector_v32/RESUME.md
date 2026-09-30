# Latest recovery point: fallback safety diagnosis

Read `fallback_audit/RESULTS.md` and `STATUS.json` first. The source-a849 v3.2.1 canaries1641933/1641934 are complete and both HOLD; do not rerun them. Their raw data are archived under `fallback_audit/evidence/completed-v321-canaries.tgz`.

Existing diagnostic jobs:5090 **1641985**,4090 **1641986**. Campaign `/ssd/scxi253/sgi-fallback-diagnosis-20260930T104231Z`, frozen diagnostic source `80b6f5c5634e1ad41404213abc3e95c1d6544c3e`, archiveSHA `7bd093cab8b950097afc89c0f873f2fc0f078528de9293b44f3d5ce28fd49bd8`. First inspect their queue, logs and per-process complete/failure records. Both were pending at10:52:32UTC. Do not resubmit pending/running jobs or mutate their frozen source. `fallback_audit/SUBMISSION.json` is the actual submission receipt; `SUBMISSION_PLANNED.json` is historical only.

After all three processes on both GPUs finish, run `fallback_audit/analyze_diagnostic.py --root <campaign> --out <new path>`. The analyzer retains every non-injected sample, checks identity/completeness, and separates same-graph null controls from independent and shared-storage comparisons. Its output is diagnostic only, not a release pass.

The one-line native early-reject patch is generated and CPU-tested but not deployed. It skips only a provably false selector region and does not itself explain the5090 Graph1 host-span spike or4090 device-side differences. Preserve every old failure; do not replace wall-time with device-time scoring or delete slow samples to pass.

Thirty v3.2 release cases remain unexecuted. No fullHTTP run is authorized. The original2/432 token divergence remains unresolved, defaultOFF/HOLD, PR27 separate at672822f, Max GUI not verified. Report-only commits do not change the immutable GPU source snapshot.
