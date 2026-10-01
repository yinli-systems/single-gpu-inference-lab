# Current-main HTTP ownership qualification

Source-bound Qwen3-4B-Instruct-2507 full-model diagnostic on ParaCloud, job 1644085 COMPLETED 0:0. Candidate ea34a9d has all 49 registered CPU tests passing with complete SGLang source imports.

Each arm issues 11 HTTP requests: 10 expected to complete and one intentional n=2 streaming disconnect. Original completes 8/10; both reused-RID replacement requests are aborted by stale cleanup. Candidate completes 10/10, with zero stale-owner aborts and no owned states left at cleanup. Batch, n=2 expansion and recovery after disconnect complete.

All raw SSE frames, cleanup events, server logs, source ledger, probe/launcher and allocation metadata are retained in the archive and individually SHA256-bound by receipt.json. Startup Slurm snapshot is preserved verbatim; terminal status is in the receipt. Resource cap is disabled. This diagnostic does not certify performance or resolve the earlier 2/432 token-value divergences.
