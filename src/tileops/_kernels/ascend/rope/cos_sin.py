"""Rotate dense or packed heads using the caller's cos/sin tables in their original precision."""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.rope import RoPECosSinFwdInterface
from tileops.backend._registry import register_kernel_type


@lru_cache(maxsize=128)
def _rotate(tokens, heads, dim, half, positions, dtype, cos_type, sin_type, interleaved):
    total = tokens * heads * dim
    block = min(1024, total)
    tiles = (total + block - 1) // block
    blocks = min(64, tiles)

    @T.prim_func
    def main(
        x: T.Tensor((total,), dtype),
        cos: T.Tensor((positions, half), cos_type),
        sin: T.Tensor((positions, half), sin_type),
        rows: T.Tensor((tokens,), "int64"),
        out: T.Tensor((total,), dtype),
    ):
        with T.Kernel(blocks) as bx:
            result = T.alloc_shared((block,), dtype)
            for step in T.serial(T.ceildiv(tiles, blocks)):
                start = (step * blocks + bx) * block
                if start < total:
                    with T.SimtVF(threads=min(256, block)):
                        for j in T.Parallel(block):
                            idx = start + j
                            if idx < total:
                                col = idx % dim
                                if col < 2 * half:
                                    row = rows[idx // (heads * dim)]
                                    k = col // 2 if interleaved else col % half
                                    left = col % 2 == 0 if interleaved else col < half
                                    peer = (
                                        (col ^ 1)
                                        if interleaved
                                        else T.if_then_else(left, col + half, col - half)
                                    )
                                    value = T.cast(x[idx], "float32")
                                    other = T.cast(x[idx - col + peer], "float32")
                                    c = T.cast(cos[row, k], "float32")
                                    s = T.cast(sin[row, k], "float32")
                                    result[j] = T.cast(
                                        value * c + T.if_then_else(left, -other, other) * s, dtype
                                    )
                                else:
                                    result[j] = x[idx]
                    count = T.min(block, total - start)
                    T.copy(result[:count], out[start : start + count])

    return tilelang.compile(main, target="ascend", out_idx=[4])


class AscendRoPECosSinKernel(Kernel, RoPECosSinFwdInterface):
    devices = frozenset({"npu"})
    supported_archs = None

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call

    def forward(self, x, cos, sin, cu_seqlens, seqlen_offsets):
        if not x.numel() or not cos.shape[-1]:
            return x if self.call.inplace else x.clone(memory_format=torch.contiguous_format)
        heads, dim = x.shape[-2:]
        tokens = x.numel() // (heads * dim)
        index = torch.arange(tokens, dtype=torch.int64, device=x.device)
        if cu_seqlens is None:
            length = x.shape[1]
            seq = index // length
            within = index % length
        else:
            length = tokens
            starts = cu_seqlens.to(torch.int64)
            seq = torch.searchsorted(starts[1:], index, right=True)
            within = index - starts[seq]
        if isinstance(seqlen_offsets, torch.Tensor):
            offset = seqlen_offsets.to(torch.int64).clamp(-length, cos.shape[0])[seq]
        else:
            offset = max(-length, min(cos.shape[0], seqlen_offsets))
        rows = (within + offset).clamp(0, cos.shape[0] - 1)
        program = _rotate(
            tokens,
            heads,
            dim,
            cos.shape[1],
            cos.shape[0],
            self.dtype_to_str(x.dtype),
            self.dtype_to_str(cos.dtype),
            self.dtype_to_str(sin.dtype),
            self.call.interleaved,
        )
        out = program(x.contiguous().view(-1), cos, sin, rows).view_as(x)
        return x.copy_(out) if self.call.inplace else out


register_kernel_type("RoPECosSinFwdOp", "ascend", AscendRoPECosSinKernel)
