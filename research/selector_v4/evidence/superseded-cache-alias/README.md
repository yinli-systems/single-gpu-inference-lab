# Superseded smoke: shared JIT cache alias

This campaign is not v4 GPU evidence. It set `XDG_CACHE_HOME` but omitted `FLASHINFER_WORKSPACE_BASE`, so FlashInfer resolved its JIT cache under the pre-existing shared home workspace. Binary/generated-source inspection showed no v4 `ResourceKernel` symbols. The completed RTX 4090 output and the cancelled RTX 5090 attempt are retained to document the failure; no release case was consumed.
