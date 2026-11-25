"""Compute prefix ranks per broadcast run and gather contiguous source rows."""

from functools import lru_cache
from math import prod

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.indexing import MaskedScatterFwdInterface, mask_runs
from tileops._kernels.kernel_base import Kernel
from tileops.backend._registry import register_kernel_type


@lru_cache(maxsize=64)
def _scatter(rows, width, source_size, dtype):
    tile = min(4096, width)
    parts = (width + tile - 1) // tile
    tasks = rows * parts
    blocks = min(64, tasks)

    @T.prim_func
    def main(
        original: T.Tensor((rows, width), dtype),
        ranks: T.Tensor((rows,), "int64"),
        source: T.Tensor((source_size,), dtype),
        out: T.Tensor((rows, width), dtype),
    ):
        with T.Kernel(blocks) as bx:
            buf = T.alloc_shared((tile,), dtype)
            for step in T.serial(T.ceildiv(tasks, blocks)):
                task = step * blocks + bx
                if task < tasks:
                    row = task // parts
                    column = (task % parts) * tile
                    count = T.min(width - column, tile)
                    rank = ranks[row]
                    previous = T.alloc_var("int64")
                    previous = 0
                    if row > 0:
                        previous = ranks[row - 1]
                    if rank != previous:
                        start = previous * width + column
                        T.copy(source[start : start + count], buf[:count])
                    else:
                        T.copy(original[row, column : column + count], buf[:count])
                    T.copy(buf[:count], out[row, column : column + count])

    return tilelang.compile(main, target="ascend", out_idx=[3])


class AscendMaskedScatterKernel(Kernel, MaskedScatterFwdInterface):
    devices = frozenset({"npu"})
    supported_archs = None

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.dtype = call.dtype

    def forward(self, input, mask, source):
        shape = torch.broadcast_shapes(input.shape, mask.shape)
        original = input.expand(shape)
        if not original.numel() or not source.numel():
            return original.clone(memory_format=torch.contiguous_format)
        leading, width = mask_runs(shape, mask.shape)
        rows = prod(shape[:leading])
        mask = mask.expand(shape)[(slice(None),) * leading + (0,) * (len(shape) - leading)]
        ranks = torch.cumsum(mask.reshape(-1), dim=0, dtype=torch.int64)
        program = _scatter(rows, width, source.numel(), self.dtype_str)
        return program(original.contiguous().view(rows, width), ranks, source.reshape(-1)).view(
            shape
        )


register_kernel_type("MaskedScatterFwdOp", "ascend", AscendMaskedScatterKernel)
