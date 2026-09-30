# Selector v3.2.2 final fresh-release verdict

Both GPU arrays completed all 96 process-mode runs with zero `failure.json` files. The frozen v3.2.2 gate is HOLD on both GPUs. These 30 release geometries are now exposed development evidence and must never be renamed as fresh for a revised selector.

| GPU | Selected paired geomean | Selected point worst | Selected joint-min 95% LCB | Paired unselected point worst | Paired unselected joint-min 95% LCB | Verdict |
|---|---:|---:|---:|---:|---:|---|
| 4090 | 1.285140x | 1.145044x | 1.143572x | 0.984377x | 0.972910x | HOLD |
| 5090 | 1.203699x | 0.987390x | 0.985577x | 0.986925x | 0.971911x | HOLD |

4090 selected-region evidence is strong, but fallback transparency failed. 5090 additionally contains a selected false positive at `holdout-v32-opp-04` (descriptor 47). Exact output/LSE qualification passed. Default and serving promotion remain disabled; historical 2/432 token divergence remains unresolved.
