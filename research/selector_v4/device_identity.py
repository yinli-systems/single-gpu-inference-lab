"""Resolve the actual CUDA device UUID, independent of NVML enumeration order."""
import re


def device_uuid(torch):
    raw=str(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()),'uuid',''))
    value=raw if raw.startswith('GPU-') else 'GPU-'+raw
    if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',value):
        raise RuntimeError('actual CUDA device UUID is unavailable')
    return value

if __name__=='__main__':
    import torch
    print(device_uuid(torch))
