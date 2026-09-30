"""Equivalent early rejection for guarded native planning, not graph timing."""

def apply(source: str) -> str:
    old='  } else {\n    // Selector v3:'
    new='  } else if (!plan_info.split_kv && batch_size >= 5) {\n    // Selector v3:'
    if source.count(old)!=1:raise ValueError('unreviewed guarded entry')
    return source.replace(old,new)

def equivalence_witness(pairing_gate,split_kv,descriptor_gate,batch_size):
    # The actual pairing predicate entails batch_size >= 5.
    pairing=pairing_gate and batch_size>=5
    original=pairing and not split_kv and descriptor_gate
    early=False if split_kv or batch_size<5 else pairing and descriptor_gate
    return original==early
