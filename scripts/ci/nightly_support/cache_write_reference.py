"""FP8 cache-write reference on CPU with byte-preserving scatter.

Ascend IndexPut does not implement FP8. Keep the workload's arithmetic and full
verification, and scatter the converted FP8 storage bytes instead.
"""

import hashlib
from pathlib import Path

from reference_placement import ReferencePlan, cpu_tree


def reference(op, workload, inputs):
    import torch

    if op == "PagedKVCacheWriteFwdOp":
        k, v, kc, vc, slots, ks, vs = inputs
        kc, vc = kc.clone(), vc.clone()
        written = (slots >= 0).nonzero().squeeze(1)
        rows = slots[written]
        for pages, new, scale in ((kc, k, ks), (vc, v, vs)):
            stored = new[written]
            if scale is not None:
                limit = torch.finfo(pages.dtype).max
                stored = (stored.float() / scale[0]).clamp(-limit, limit)
            flat = pages.view(-1, *pages.shape[2:])
            flat.view(torch.uint8)[rows] = stored.to(pages.dtype).view(torch.uint8)
        return kc, vc
    if op != "MLAKVCacheWriteFwdOp":
        raise ValueError(op)
    from workloads.attention.mla_cache_write import _join_pairs, _split_pairs

    kv, pe, cache, slots, scale, positions, table = inputs
    cache = cache.clone()
    written = (slots >= 0).nonzero().squeeze(1)
    stored_pe = pe[written]
    if positions is not None:
        cos, sin = table[positions[written]].float().chunk(2, -1)
        a, b = _split_pairs(stored_pe.float(), workload.rope_layout)
        stored_pe = _join_pairs(a * cos - b * sin, b * cos + a * sin, workload.rope_layout).to(
            pe.dtype
        )
    stored = torch.cat([kv[written], stored_pe], -1)
    if scale is not None:
        limit = torch.finfo(cache.dtype).max
        stored = (stored.float() / scale[0]).clamp(-limit, limit)
    cache.view(-1, cache.shape[-1]).view(torch.uint8)[slots[written]] = stored.to(cache.dtype).view(
        torch.uint8
    )
    return cache


class CacheWriteCase:
    def __init__(self, op, original):
        self.original = original
        self.reference = lambda *inputs: reference(op, original.workload, cpu_tree(inputs))
        # Rebuild the SAME workload validator on the SAME actual CPU input bytes,
        # so MLA rotation intervals/untouched-row checks remain intact on CPU.
        self.verification = original.workload.verification(*cpu_tree(original.inputs))

    def __getattr__(self, name):
        return getattr(self.original, name)


def prepare_cache_write_reference(op, case):
    import torch

    if op not in {"MLAKVCacheWriteFwdOp", "PagedKVCacheWriteFwdOp"}:
        return None
    if case.inputs[0].device.type != "npu" or case.inputs[2].dtype != torch.float8_e4m3fn:
        return None
    return ReferencePlan(
        CacheWriteCase(op, case),
        cpu_tree,
        dict(
            reference_cache_write="fp8-byte-scatter-original-workload-cpu-v1",
            reference_cache_write_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )
