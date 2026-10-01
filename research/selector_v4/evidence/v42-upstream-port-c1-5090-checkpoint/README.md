# Current-main experimental port: RTX5090 checkpoint

Sourcec1c9537 over upstream85744da. Job1644287 COMPLETED0:0.37CPU+8GPU tests passed, covering both dtypes/layouts/causality modes, native/resource/native and repeated resource output/LSE exactness, and unchanged native libraries.32 balanced wall-time calibration blocks passed every confidence gate on the already-exposed BF16ragged representative: mean1.356818, lower95%1.356522, block minimum1.354459. Full raw logs/XML/calibration/source/telemetry are archived and SHA-bound. This is one development geometry and one GPU, not fresh release or HTTP performance.

The sample's final output remains exact, but actual managed profiling skipped both tactics and silently returned fallback. Consequently managed integration is explicitly unqualified at this checkpoint. Diagnostic job1644343 adds a valid-managed-tactic assertion and debug logs. RTX4090job1644286 is pending. Defaults and serving stayOFF.
