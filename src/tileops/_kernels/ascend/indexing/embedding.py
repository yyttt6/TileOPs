"""Gather embedding rows with one block DMA per row segment."""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T

from tileops._kernels.indexing import EmbeddingFwdInterface
from tileops._kernels.kernel_base import Kernel
from tileops.backend._registry import register_kernel_type


@lru_cache(maxsize=64)
def _lookup(rows, vocab, width, dtype, index_dtype):
    chunk = min(width, 8192)
    parts = (width + chunk - 1) // chunk
    tasks = rows * parts
    blocks = min(tasks, 64)

    @T.prim_func
    def main(
        ids: T.Tensor((rows,), index_dtype),
        weight: T.Tensor((vocab, width), dtype),
        output: T.Tensor((rows, width), dtype),
    ):
        with T.Kernel(blocks) as bx:
            data = T.alloc_shared((chunk,), dtype)
            for step in T.serial(T.ceildiv(tasks, blocks)):
                task = step * blocks + bx
                if task < tasks:
                    row = task // parts
                    column = (task % parts) * chunk
                    count = T.min(width - column, chunk)
                    index = ids[row]
                    T.copy(weight[index, column : column + count], data[:count])
                    T.copy(data[:count], output[row, column : column + count])

    return tilelang.compile(main, target="ascend", out_idx=[2])


class AscendEmbeddingKernel(Kernel, EmbeddingFwdInterface):
    devices = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call):
        if call.renorm:
            return "row renormalization by max_norm is unsupported"
        return super().refusal(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.dtype = call.dtype

    def forward(self, input, weight):
        shape = (*input.shape, weight.shape[1])
        if not input.numel() or not weight.shape[1]:
            return weight.new_empty(shape)
        program = _lookup(
            input.numel(), *weight.shape, self.dtype_str, self.dtype_to_str(input.dtype)
        )
        return program(input.reshape(-1), weight).view(shape)


register_kernel_type("EmbeddingFwdOp", "ascend", AscendEmbeddingKernel)
