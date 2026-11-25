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
            for step in T.serial(T.ceildiv(rows * parts, blocks)):
                task = step * blocks + bx
                if task < rows * parts:
                    row = task // parts
                    col = (task % parts) * chunk
                    count = T.min(width - col, chunk)
                    T.copy(src[row, col : col + count], data[:count])
                    if src_dtype == out_dtype:
                        T.copy(data[:count], out[row, offset + col : offset + col + count])
                    else:
                        with T.SimtVF(threads=256):
                            for i in T.Parallel(chunk):
                                result[i] = T.cast(data[i], out_dtype)
                        T.copy(result[:count], out[row, offset + col : offset + col + count])

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
            if width <= 32 and rows >= 256:
                return _narrow_pair(rows, width, self.dtype_str)(
                    tensors[0].view(rows, width), tensors[1].view(rows, width)
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
