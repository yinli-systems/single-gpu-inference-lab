"""Dense grammar masks: what does main pay per beam, end to end, to answer
"is candidate token t allowed?" for the ~2 * beam_width candidates?

main:  list = _bitmask_to_token_ids(row)  ->  set(list)  ->  t in set
alt:   test each candidate's bit directly in the packed row
"""

import random
import statistics
import time

import torch

from bitmask_harness import reference

V = 151936
rng = random.Random(0)
row = torch.full(((V + 31) // 32,), -1, dtype=torch.int32)
for _ in range(V - 147123):  # ~the real dense-mask density
    i = rng.randrange(V)
    row[i // 32] &= ~torch.tensor(1 << (i % 32)).to(torch.int32)
candidates = [rng.randrange(V) for _ in range(64)]  # 2 * beam_width, B = 32


def med(fn, reps=31):
    s = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        s.append(time.perf_counter() - t0)
    return statistics.median(s) * 1e3


ids = reference(row, V)
t_unpack = med(lambda: reference(row, V))
t_set = med(lambda: set(ids))
allowed = set(ids)
t_member = med(lambda: [t in allowed for t in candidates])

words = row.tolist()  # packed row as Python ints, once per beam
t_words = med(lambda: row.tolist())
t_bits = med(lambda: [(words[t >> 5] >> (t & 31)) & 1 for t in candidates])
assert [t in allowed for t in candidates] == [
    bool((words[t >> 5] >> (t & 31)) & 1) for t in candidates
]
print(f"allowed {len(ids)} of {V}")
print(f"main: unpack {t_unpack:.3f} + set() {t_set:.3f} + 64 lookups "
      f"{t_member:.4f} = {t_unpack + t_set + t_member:.3f} ms per beam")
print(f"alt:  row.tolist() {t_words:.3f} + 64 bit tests {t_bits:.4f} "
      f"= {t_words + t_bits:.3f} ms per beam")
