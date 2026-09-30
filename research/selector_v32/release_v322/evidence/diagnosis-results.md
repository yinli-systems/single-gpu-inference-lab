# Fallback identity and host-tail diagnosis

Diagnostic only. Every non-injected sample is retained; old canary HOLD is unchanged.

| GPU | Scenario | Timer | Worst paired wall point across five exposed coordinates / two graph modes |
|---|---|---|---:|
| gpu_4090 | independent | legacy_events | 0.997334x |
| gpu_4090 | independent | pooled_events | 0.999898x |
| gpu_4090 | independent | wall_only | 0.999849x |
| gpu_4090 | same_storage | legacy_events | 0.995468x |
| gpu_4090 | same_storage | pooled_events | 0.993708x |
| gpu_4090 | same_storage | wall_only | 0.988266x |
| gpu_4090 | same_graph_null | legacy_events | 0.997543x |
| gpu_4090 | same_graph_null | pooled_events | 0.999689x |
| gpu_4090 | same_graph_null | wall_only | 0.996448x |
| gpu_5090 | independent | legacy_events | 0.999878x |
| gpu_5090 | independent | pooled_events | 0.999811x |
| gpu_5090 | independent | wall_only | 0.999910x |
| gpu_5090 | same_storage | legacy_events | 0.999898x |
| gpu_5090 | same_storage | pooled_events | 0.999766x |
| gpu_5090 | same_storage | wall_only | 0.999780x |
| gpu_5090 | same_graph_null | legacy_events | 0.999865x |
| gpu_5090 | same_graph_null | pooled_events | 0.999810x |
| gpu_5090 | same_graph_null | wall_only | 0.999797x |

Same-graph null comparisons are not candidate wins. Conditional bootstrap intervals use three process repeats and eight blocks; they do not establish population-wide safety. No fresh release or full HTTP result is produced.
