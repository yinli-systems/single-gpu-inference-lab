# Resume v3.2.2 fresh release without duplication

GPU source commit: `2f14b21b79109b80ed4fdb8cad575d5c1ff61b83`. Active campaign: `/ssd/scxi253/single-gpu-inference-selector-v322-release-r2-20260930T133933Z`. Arrays: RTX4090 `1642865`, RTX5090 `1642866`. Read `squeue`/`sacct` and this campaign's raw run directories before any action; never resubmit pending/running/completed shards.

The first arrays `1642826/1642827` requested ten shards and were rejected by the frozen `measure.py` bound `shards<=8` before case selection or tensor creation. They were cancelled/superseded with zero release run files and zero consumed release cases. Receipts are retained under `evidence/failed-shards10/`.

Corrected r2 uses the identical source archive and selector, eight deterministic shards with case counts `[4,4,4,4,4,4,3,3]`, three process repeats, eight ABBA/BAAB blocks and max concurrency two per GPU. Do not alter rule, gate, manifest or measurement boundary after any r2 shard produces data while calling the evidence fresh.

After all sixteen array tasks complete, run `release_v322/analyze_v322.py` separately for each GPU with `--stage release --shards 8`. Preserve every failure. No full HTTP run is permitted unless both fresh GPU gates pass. Historical 2/432 token divergence remains unresolved and default remains OFF.
