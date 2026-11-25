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


@lru_cache(maxsize=128)
def _staged(
    tokens, heads, dim, half, positions, dtype, cos_type, sin_type, interleaved, dense_period
):
    # A block loads both members of each pair before storing the output; the
    # contiguous in-place specialization can therefore use the input as its destination.
    group = min(heads, 8192 // dim)
    while heads % group:
        group -= 1
    groups = heads // group
    tasks = tokens * groups
    blocks = min(64, tasks)
    row_count = 1 if dense_period else tokens

    @T.prim_func
    def main(
        x: T.Tensor((tokens, heads, dim), dtype),
        cos: T.Tensor((positions, half), cos_type),
        sin: T.Tensor((positions, half), sin_type),
        rows: T.Tensor((row_count,), "int64"),
        output: T.Tensor((tokens, heads, dim), dtype),
    ):
        with T.Kernel(blocks) as bx:
            data = T.alloc_shared((group, dim), dtype)
            ctable = T.alloc_shared((half,), cos_type)
            stable = T.alloc_shared((half,), sin_type)
            result = T.alloc_shared((group, dim), dtype)
            for step in T.serial(T.ceildiv(tasks, blocks)):
                task = step * blocks + bx
                if task < tasks:
                    token = task // groups
                    first = (task % groups) * group
                    if dense_period:
                        row = T.min(token % max(1, dense_period), positions - 1)
                    else:
                        row = rows[token]
                    T.copy(x[token, first : first + group, :], data)
                    T.copy(cos[row, :], ctable)
                    T.copy(sin[row, :], stable)
                    with T.SimtVF(threads=256):
                        for h, col in T.Parallel(group, dim):
                            if col < half * 2:
                                k = col // 2 if interleaved else col % half
                                left = col % 2 == 0 if interleaved else col < half
                                peer = (
                                    (col ^ 1)
                                    if interleaved
                                    else T.if_then_else(left, col + half, col - half)
                                )
                                v = T.cast(data[h, col], "float32")
                                other = T.cast(data[h, peer], "float32")
                                c = T.cast(ctable[k], "float32")
                                t = T.cast(stable[k], "float32")
                                result[h, col] = T.cast(
                                    v * c + T.if_then_else(left, -other, other) * t, dtype
                                )
                            else:
                                result[h, col] = data[h, col]
                    T.copy(result, output[token, first : first + group, :], l2_cache_ctrl=0)

    return tilelang.compile(main, target="ascend", out_idx=[])


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
        staged = (
            tokens >= 512
            and 4 <= heads
            and dim <= 1024
            and not (self.call.inplace and not x.is_contiguous())
        )
        dense_period = (
            x.shape[1]
            if (
                staged
                and cu_seqlens is None
                and isinstance(seqlen_offsets, int)
                and seqlen_offsets == 0
            )
            else 0
        )
        if dense_period:
            rows = torch.empty((1,), dtype=torch.int64, device=x.device)
        else:
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
        if staged:
            program = _staged(
                tokens,
                heads,
                dim,
                cos.shape[1],
                cos.shape[0],
                self.dtype_to_str(x.dtype),
                self.dtype_to_str(cos.dtype),
                self.dtype_to_str(sin.dtype),
                self.call.interleaved,
                dense_period,
            )
            out = (
                x
                if self.call.inplace
                else torch.empty_like(x, memory_format=torch.contiguous_format)
            )
            program(
                x.contiguous().view(tokens, heads, dim),
                cos.contiguous(),
                sin.contiguous(),
                rows,
                out.view(tokens, heads, dim),
            )
            return out
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
