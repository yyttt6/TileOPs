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
            T.annotate_buffer_versions({data: 2})
            for step in T.Pipelined(T.ceildiv(tasks, blocks), num_stages=2):
                task = step * blocks + bx
                if task < tasks:
                    row = task // parts
                    column = (task % parts) * chunk
                    count = T.min(width - column, chunk)
                    index = ids[row]
                    T.copy(weight[index, column : column + count], data[:count])
                    T.copy(data[:count], output[row, column : column + count], l2_cache_ctrl=0)

    return tilelang.compile(main, target="ascend", out_idx=[2])


@lru_cache(maxsize=32)
def _lookup_narrow(rows, vocab, width, dtype, index_dtype):
    # Gather short rows by lane, then write one contiguous output tile.
    batch = min(rows, max(1, 2048 // width))
    tiles = (rows + batch - 1) // batch
    blocks = min(64, tiles)

    @T.prim_func
    def main(
        ids: T.Tensor((rows,), index_dtype),
        weight: T.Tensor((vocab, width), dtype),
        output: T.Tensor((rows, width), dtype),
    ):
        with T.Kernel(blocks) as bx:
            data = T.alloc_shared((batch, width), dtype)
            for step in T.serial(T.ceildiv(tiles, blocks)):
                first = (step * blocks + bx) * batch
                if first < rows:
                    count = T.min(batch, rows - first)
                    with T.SimtVF(threads=256):
                        for i, j in T.Parallel(batch, width):
                            if i < count:
                                data[i, j] = weight[ids[first + i], j]
                    T.copy(data[:count, :], output[first : first + count, :], l2_cache_ctrl=0)

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
        builder = _lookup_narrow if weight.shape[1] <= 32 else _lookup
        program = builder(
            input.numel(), *weight.shape, self.dtype_str, self.dtype_to_str(input.dtype)
        )
        return program(input.reshape(-1), weight).view(shape)


register_kernel_type("EmbeddingFwdOp", "ascend", AscendEmbeddingKernel)
