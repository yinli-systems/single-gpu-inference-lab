# Selector-v3 analysis-only amendment

The measurement source, selector rule, manifest, case mapping, and submitted GPU jobs remain bound to commit `8edd481c9f43be76b82a4c77a37930f34df0a7d2`.

This later analysis revision adds two fail-closed checks without changing the experiment:

1. all scored shards must share the same GPU model, driver, SM count, power limit, CUDA, Torch, FlashInfer, immutable overlay, source archive, and profiling policy; GPU UUIDs may differ and are retained per shard;
2. the preregistered 30-case gate remains primary, while a separately labelled 28-case sensitivity excludes two release geometries that exactly duplicate canary boundary cases.

The sensitivity result cannot override a failed 30-case primary gate. Neither analysis may promote serving or default enablement while the historical 2/432 token divergence remains unresolved.
