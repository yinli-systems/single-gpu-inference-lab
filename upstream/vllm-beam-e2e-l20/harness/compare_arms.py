"""Compare alternating stock/patched arms: outputs and wall time."""

import json
import sys
from pathlib import Path

R = Path(sys.argv[1])
arms = {p.stem: json.load(open(p)) for p in sorted(R.glob("*.json"))}
stock = [a for a in arms if a.startswith("stock")]
patched = [a for a in arms if a.startswith("patched")]


def key(c):
    return (c["kind"], c["prompt_len"], c["width"])


def outputs(c):
    return [(s["tokens"], s["cum"]) for s in c["sequences"]]


cells = {key(c): c for c in arms[stock[0]]}
print(f"arms: {stock} vs {patched} (wall = median of 5 calls per arm)")
print(f"{'cell':22s} " + " ".join(f"{a:>9s}" for a in stock + patched)
      + f"  {'speedup':>8s}  stock=stock  stock=patched")
for k in cells:
    row = {a: next(c for c in arms[a] if key(c) == k) for a in stock + patched}
    best_stock = min(row[a]["wall_s"] for a in stock)
    best_patched = min(row[a]["wall_s"] for a in patched)
    s_eq = all(outputs(row[a]) == outputs(row[stock[0]]) for a in stock)
    p_eq = all(outputs(row[a]) == outputs(row[stock[0]]) for a in patched)
    name = f"{k[0]} p{k[1]} w{k[2]}"
    print(f"{name:22s} " + " ".join(f"{row[a]['wall_s'] * 1e3:8.1f} " for a in stock + patched)
          + f" {best_stock / best_patched:7.2f}x  {str(s_eq):>11s}  {str(p_eq):>13s}")
