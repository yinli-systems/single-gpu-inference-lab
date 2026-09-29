# Delayed cleanup ownership: source-level reproducer and proposed fix

The installed SGLang create_abort_task sleeps two seconds then aborts any current state
with the old request ID. A request with the same RID registered after the old stream
finishes can be canceled. An identical method was inspected on upstream commit
211b1d9784d15845566081cc25c8103ce43cc4f2. PR38850 already discusses this hazard and
works around it inside scripted-runtime tests; this is related prior art, not our invention.

The minimal proposal snapshots the original request objects and ReqState identities
when cleanup begins, and rechecks the same state after the delay. A replacement before
or during the delay cannot be canceled, even when the same request object is reused.
Live original requests retain cleanup behavior. Batched requests use the existing cached
subobjects returned by GenerateReqInput.__getitem__.

CPU tests execute the exact original method and the replacement. Original behavior
reproduces the stale-abort path both before and during the delay. The proposed method
passes active, finished, removed, replaced, and batched-state cases. This is a controlled
CPU reproduction, not proof that it caused the historical incomplete stream, and not a
fix for the two complete-length token-value divergences. No shared runtime is patched.
