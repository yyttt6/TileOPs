"""Full bytewise input fingerprint, including BF16 and noncontiguous tensors."""
import hashlib

PROTOCOL = 'full-input-bytes-v1'

def tensor_digest(inputs):
    digest = hashlib.sha256()
    for value in inputs:
        if not hasattr(value, 'detach'):
            digest.update(repr(value).encode())
            continue
        import torch
        digest.update(repr((tuple(value.shape), tuple(value.stride()), value.storage_offset(), str(value.dtype))).encode())
        data = value.detach().to('cpu').contiguous().reshape(-1).view(torch.uint8)
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()
