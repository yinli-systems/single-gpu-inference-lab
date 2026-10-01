# Main-branch experimental port: failed development attempts

Sourceeb1c024 over FlashInferc5bb61f is separate from the frozen research0.7b769e7c campaign. Jobs1644157/1644158 stopped at hardware identity (nvidia-smi required GPU- UUID prefix) before attention tests. Launcher fix, jobs1644162/1644163:35CPU tests passed;4GPU cases failed before resource compilation because the private JIT generator was passed an unsupported paged_kv_stride_mode keyword. Native baselines ran. Neither attempt qualifies the resource tactic or consumes fresh holdouts; only the exposed representative geometry was used.

All logs/receipts/source ledger are archived and individually SHA-bound in receipt.json. Native .so paths/SHA are retained remotely. Fix0e81e19 uses the reviewed _gen_batch_prefill_primary_module, plus fail-closed exception handling. Its new jobs1644229/1644230 are separate, pending validation at this snapshot. Research gains cannot be asserted for this port.
