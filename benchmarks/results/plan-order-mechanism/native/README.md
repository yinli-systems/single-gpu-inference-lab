# Native host-order prototype

This standalone C++ implementation mirrors the frozen Python descriptor-order policies exactly. It validates the complete descriptor bijection before returning an index permutation. It does not alter GPU buffers, install a global hook or claim production integration. CPU sort time must be measured on the actual GPU worker; Python diagnostic upload/validation time is not treated as native planning overhead.

Potential FlashInfer integration point: after generating the three work arrays inside PrefillSplitQOKVIndptr, apply the same permutation to all three arrays; keep o_indptr and merge_indptr unchanged. A public opt-in, API/JIT compatibility, graph padding masks and all actual callers need an upstream design review before this is deployed. Current ABI-gated runtime experiment deliberately rejects padded graph plans.
