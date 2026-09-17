#!/usr/bin/env python3
"""Compare per-request outputs dumped by measure_vllm_feature_cost.py.

Given two ``outputs-<server>-<rc>-r0.json`` files (same workload, same seeds),
report whether the generated token sequences and sampling masks agree. With
per-request seeds and identical engine configuration, the bit-packed and
compact sampling-mask layouts must produce identical tokens and identical
masks; any divergence is a correctness finding, not noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--json", type=Path, help="write the comparison summary here")
    args = ap.parse_args()
    A = json.loads(args.a.read_text())
    B = json.loads(args.b.read_text())
    assert len(A) == len(B), (len(A), len(B))
    tok_eq = mask_eq = mask_both = 0
    first_div = None
    prefix_matches = []
    for i, (x, y) in enumerate(zip(A, B)):
        ta, tb = x["token_ids"], y["token_ids"]
        if ta == tb:
            tok_eq += 1
        else:
            n = 0
            for u, v in zip(ta, tb):
                if u != v:
                    break
                n += 1
            prefix_matches.append(n)
            if first_div is None:
                first_div = {"request": i, "first_divergent_position": n, "len_a": len(ta), "len_b": len(tb)}
        ma, mb = x.get("sampling_mask"), y.get("sampling_mask")
        if ma is not None and mb is not None:
            mask_both += 1
            if [sorted(m) for m in ma] == [sorted(m) for m in mb]:
                mask_eq += 1
    summary = {
        "requests": len(A),
        "token_sequences_identical": tok_eq,
        "masks_compared": mask_both,
        "masks_identical": mask_eq,
        "first_divergence": first_div,
        "divergent_prefix_lengths": prefix_matches[:20],
    }
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
