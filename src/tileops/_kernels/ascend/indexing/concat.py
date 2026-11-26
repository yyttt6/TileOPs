"""Stream each input into its last contiguous segment of the concatenated output."""

from functools import lru_cache
from math import prod

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.indexing import ConcatFwdInterface
from tileops._kernels.kernel_base import Kernel
from tileops.backend._registry import register_kernel_type


@lru_cache(maxsize=128)
def _append(rows, width, total, offset, src_dtype, out_dtype):
    chunk = min(8192, width)
    parts = (width + chunk - 1) // chunk
    blocks = min(64, rows * parts)

    @T.prim_func
    def main(src: T.Tensor((rows, width), src_dtype), out: T.Tensor((rows, total), out_dtype)):
        T.func_attr({"tilelang_callee_allocated_outputs": 0})
        with T.Kernel(blocks) as bx:
            data = T.alloc_shared((chunk,), src_dtype)
            result = T.alloc_shared((chunk,), out_dtype)
            T.annotate_buffer_versions({data: 2, result: 2})
            for step in T.Pipelined(T.ceildiv(rows * parts, blocks), num_stages=2):
                task = step * blocks + bx
                if task < rows * parts:
                    row = task // parts
                    col = (task % parts) * chunk
                    count = T.min(width - col, chunk)
                    T.copy(src[row, col : col + count], data[:count])
                    if src_dtype == out_dtype:
                        T.copy(
                            data[:count],
                            out[row, offset + col : offset + col + count],
                            l2_cache_ctrl=0,
                        )
                    else:
                        with T.SimtVF(threads=256):
                            for i in T.Parallel(chunk):
                                result[i] = T.cast(data[i], out_dtype)
                        T.copy(
                            result[:count],
                            out[row, offset + col : offset + col + count],
                            l2_cache_ctrl=0,
                        )

    return tilelang.compile(main, target="ascend", out_idx=[])


@lru_cache(maxsize=32)
def _narrow_pair(rows, width, dtype):
    # Each core loads two dense row batches and writes one dense interleaved batch.
    batch = min(512, 8192 // (2 * width))
    groups = (rows + batch - 1) // batch
    blocks = min(64, groups)

    @T.prim_func
    def main(
        a: T.Tensor((rows, width), dtype),
        b: T.Tensor((rows, width), dtype),
        y: T.Tensor((rows, 2 * width), dtype),
    ):
        with T.Kernel(blocks) as bx:
            left = T.alloc_shared((batch, width), dtype)
            right = T.alloc_shared((batch, width), dtype)
            joined = T.alloc_shared((batch, 2 * width), dtype)
            for step in T.serial(T.ceildiv(groups, blocks)):
                row = (step * blocks + bx) * batch
                if row < rows:
                    count = T.min(batch, rows - row)
                    T.copy(a[row : row + count, :], left[:count, :])
                    T.copy(b[row : row + count, :], right[:count, :])
                    with T.SimtVF(threads=256):
                        for i, j in T.Parallel(batch, width):
                            joined[i, j] = left[i, j]
                            joined[i, j + width] = right[i, j]
                    T.copy(joined[:count, :], y[row : row + count, :], l2_cache_ctrl=0)

    return tilelang.compile(main, target="ascend", out_idx=[2])


@lru_cache(maxsize=32)
def _row_concat(rows, first, second, third, dtype):
    """Assemble each complete row in the UB, then store it with a single DMA."""
    total = first + second + third
    blocks = min(64, rows)
    last_width = third or first

    @T.prim_func
    def main(
        a: T.Tensor((rows, first), dtype),
        b: T.Tensor((rows, second), dtype),
        c: T.Tensor((rows, last_width), dtype),
        out: T.Tensor((rows, total), dtype),
    ):
        with T.Kernel(blocks) as bx:
            joined = T.alloc_shared((total,), dtype)
            T.annotate_buffer_versions({joined: 2})
            for step in T.Pipelined(T.ceildiv(rows, blocks), num_stages=2):
                row = step * blocks + bx
                if row < rows:
                    T.copy(a[row, :], joined[:first])
                    T.copy(b[row, :], joined[first : first + second])
                    if third:
                        T.copy(c[row, :], joined[first + second :])
                    T.copy(joined, out[row, :], l2_cache_ctrl=0)

    return tilelang.compile(main, target="ascend", out_idx=[3])


@lru_cache(maxsize=32)
def _flat_four(first, second, third, fourth, tile, dtype):
    # A leading-axis concat has four contiguous regions. One launch copies all four.
    total = first + second + third + fourth
    blocks = min(64, total // tile)

    @T.prim_func
    def main(
        a: T.Tensor((first,), dtype),
        b: T.Tensor((second,), dtype),
        c: T.Tensor((third,), dtype),
        d: T.Tensor((fourth,), dtype),
        out: T.Tensor((total,), dtype),
    ):
        with T.Kernel(blocks) as bx:
            buf = T.alloc_shared((tile,), dtype)
            T.annotate_buffer_versions({buf: 2})
            for step in T.Pipelined(T.ceildiv(total // tile, blocks), num_stages=2):
                at = (step * blocks + bx) * tile
                if at < total:
                    if at < first:
                        T.copy(a[at : at + tile], buf)
                    elif at < first + second:
                        off = at - first
                        T.copy(b[off : off + tile], buf)
                    elif at < first + second + third:
                        off = at - first - second
                        T.copy(c[off : off + tile], buf)
                    else:
                        off = at - first - second - third
                        T.copy(d[off : off + tile], buf)
                    T.copy(buf, out[at : at + tile], l2_cache_ctrl=0)

    return tilelang.compile(main, target="ascend", out_idx=[4])


@lru_cache(maxsize=8)
def _pair_eight(rows, dtype):
    batch = 512
    tiles = (rows + batch - 1) // batch
    blocks = min(64, tiles)

    @T.prim_func
    def main(
        a: T.Tensor((rows, 8), dtype),
        b: T.Tensor((rows, 8), dtype),
        out: T.Tensor((rows, 16), dtype),
    ):
        with T.Kernel(blocks) as bx:
            left = T.alloc_shared((batch, 8), dtype)
            right = T.alloc_shared((batch, 8), dtype)
            joined = T.alloc_shared((batch, 16), dtype)
            T.annotate_buffer_versions({left: 2, right: 2, joined: 2})
            for step in T.Pipelined(T.ceildiv(tiles, blocks), num_stages=2):
                row = (step * blocks + bx) * batch
                if row < rows:
                    count = T.min(batch, rows - row)
                    T.copy(a[row : row + count, :], left[:count, :])
                    T.copy(b[row : row + count, :], right[:count, :])
                    with T.SimdVF():
                        full = T.simd.pset(16)
                        lane = T.simd.vci(T.int16(0), "int16")
                        inner = T.simd.vand(lane, T.simd.vdup(T.int16(7), "int16"), full)
                        half = T.simd.vand(lane, T.simd.vdup(T.int16(8), "int16"), full)
                        pick = T.simd.vcmps(half, T.int16(0), full, "eq")
                        low = T.simd.vadd(
                            T.simd.vshls(T.simd.vshrs(lane, 4, full), 3, full), inner, full
                        )
                        high = T.simd.vadds(low, T.int16(64), full)
                        for v in range(batch // 16):
                            x = T.simd.vld(left[v * 16, 0])
                            y = T.simd.vld(right[v * 16, 0])
                            z0 = T.simd.vsel(T.simd.vselr(x, low), T.simd.vselr(y, low), pick)
                            z1 = T.simd.vsel(T.simd.vselr(x, high), T.simd.vselr(y, high), pick)
                            T.simd.vsts(joined[v * 16, 0], z0, full)
                            T.simd.vsts(joined[v * 16 + 8, 0], z1, full)
                    T.copy(joined[:count, :], out[row : row + count, :], l2_cache_ctrl=0)

    return tilelang.compile(main, target="ascend", out_idx=[2])


class AscendConcatKernel(Kernel, ConcatFwdInterface):
    devices = frozenset({"npu"})
    supported_archs = None

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.dim = call.dim
        self.dtype = call.out_dtype

    def forward(self, tensors):
        shape = list(tensors[0].shape)
        shape[self.dim] = sum(t.shape[self.dim] for t in tensors)
        if not prod(shape):
            return torch.empty(shape, dtype=self.dtype, device=tensors[0].device)
        rows = prod(shape[: self.dim])
        if (
            len(tensors) == 2
            and tensors[0].dtype == tensors[1].dtype
            and tensors[0].shape == tensors[1].shape
        ):
            width = tensors[0].numel() // rows
            if (
                width == 8
                and rows >= 512
                and rows % 16 == 0
                and self.dtype in (torch.float16, torch.bfloat16)
            ):
                # Interleave the raw 16-bit lanes. No numeric conversion is involved.
                packed = _pair_eight(rows, "uint16")(
                    tensors[0].view(torch.uint16).view(rows, 8),
                    tensors[1].view(torch.uint16).view(rows, 8),
                )
                return packed.view(self.dtype).view(shape)
            if width <= 32 and rows >= 256:
                return _narrow_pair(rows, width, self.dtype_str)(
                    tensors[0].view(rows, width), tensors[1].view(rows, width)
                ).view(shape)
        if rows >= 256 and len(tensors) in (2, 3) and all(t.dtype == self.dtype for t in tensors):
            widths = [t.numel() // rows for t in tensors]
            if min(widths) >= 256 and sum(widths) <= 16384:
                a, b = tensors[:2]
                third = widths[2] if len(tensors) == 3 else 0
                c = tensors[2] if third else a
                program = _row_concat(rows, widths[0], widths[1], third, self.dtype_str)
                return program(
                    a.view(rows, widths[0]),
                    b.view(rows, widths[1]),
                    c.view(rows, third or widths[0]),
                ).view(shape)
        if rows == 1 and len(tensors) == 4 and all(t.dtype == self.dtype for t in tensors):
            sizes = [t.numel() for t in tensors]
            if min(sizes) > 0 and all(size % 16384 == 0 for size in sizes):
                return _flat_four(*sizes, 16384, self.dtype_str)(
                    *(t.view(-1) for t in tensors)
                ).view(shape)
        out = torch.empty(shape, dtype=self.dtype, device=tensors[0].device)
        total = out.numel() // rows
        offset = 0
        for tensor in tensors:
            width = tensor.numel() // rows
            if width:
                program = _append(
                    rows, width, total, offset, self.dtype_to_str(tensor.dtype), self.dtype_str
                )
                program(tensor.view(rows, width), out.view(rows, total))
                offset += width
        return out


register_kernel_type("ConcatFwdOp", "ascend", AscendConcatKernel)
