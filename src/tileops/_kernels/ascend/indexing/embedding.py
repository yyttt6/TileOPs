"""Gather embedding rows with one block DMA per row segment."""

from functools import lru_cache

import tilelang
import torch
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


@lru_cache(maxsize=32)
def _prefetched_ids(rows, vocab, width, dtype, index_dtype):
    """Fetch a block of consecutive IDs, keeping each gathered table row as its own DMA."""
    batch = 32
    groups = rows // batch
    blocks = min(groups, 64)

    @T.prim_func
    def main(
        ids: T.Tensor((rows,), index_dtype),
        weight: T.Tensor((vocab, width), dtype),
        out: T.Tensor((rows, width), dtype),
    ):
        with T.Kernel(blocks) as bx:
            indices = T.alloc_shared((batch,), index_dtype)
            values = T.alloc_shared((width,), dtype)
            T.annotate_buffer_versions({values: 2})
            for outer in T.serial(T.ceildiv(groups, blocks)):
                group = outer * blocks + bx
                if group < groups:
                    first = group * batch
                    T.copy(ids[first : first + batch], indices)
                    for item in T.Pipelined(batch, num_stages=2):
                        idx = indices[item]
                        T.copy(weight[idx, :], values)
                        T.copy(values, out[first + item, :], l2_cache_ctrl=0)

    return tilelang.compile(main, target="ascend", out_idx=[2])


@lru_cache(maxsize=32)
def _packed_lookup(rows, vocab, words, index_dtype):
    """Move whole 64-bit words from the indexed table to contiguous output."""
    total = rows * words
    blocks = min(64, (total + 255) // 256)
    extent = (total + blocks - 1) // blocks
    wide = vocab * words > 2**31 - 1

    @T.prim_func
    def main(
        ids: T.Tensor((rows,), index_dtype),
        table: T.Tensor((vocab, words), "int64"),
        out: T.Tensor((rows, words), "int64"),
    ):
        with T.Kernel(blocks) as bx:
            with T.SimtVF(threads=1024):
                for lane in T.Parallel(extent):
                    index = bx * extent + lane
                    if index < total:
                        row = index // words
                        address = T.cast(ids[row], "int64" if wide else "int32")
                        out[row, index % words] = table[address, index % words]

    return tilelang.compile(main, target="ascend", out_idx=[2])


@lru_cache(maxsize=32)
def _row_batch(rows, vocab, width, dtype, index_dtype, bytes_per_element):
    # Give each DMA a static destination within the tile. The scalar core
    # must see all the freshly fetched indices before it issues the row reads.
    batch = 8 if width * bytes_per_element >= 8192 else 16
    blocks = min(64, rows // batch)
    rows_per_core = rows // blocks

    @T.prim_func
    def main(
        ids: T.Tensor((rows,), index_dtype),
        table: T.Tensor((vocab * width,), dtype),
        out: T.Tensor((rows * width,), dtype),
    ):
        with T.Kernel(blocks) as bx:
            indices = T.alloc_shared((rows_per_core,), index_dtype)
            gathered = T.alloc_shared((batch * width,), dtype)
            T.annotate_buffer_versions({gathered: 2})
            first = bx * rows_per_core
            T.copy(ids[first : first + rows_per_core], indices)
            T.ascend_set_flag("MTE2_S", 6)
            T.ascend_wait_flag("MTE2_S", 6)
            for step in T.Pipelined(rows_per_core // batch, num_stages=2):
                for j in T.unroll(batch):
                    offset = T.cast(indices[step * batch + j], "int64") * width
                    T.copy(table[offset : offset + width], gathered[j * width : (j + 1) * width])
                start = (first + step * batch) * width
                T.copy(gathered, out[start : start + batch * width], l2_cache_ctrl=0)

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
        row_bytes = weight.shape[1] * weight.element_size()
        if (
            input.numel() >= 256
            and weight.shape[1] <= 160
            and row_bytes % 8 == 0
            and weight.data_ptr() % 8 == 0
        ):
            words = row_bytes // 8
            program = _packed_lookup(
                input.numel(), weight.shape[0], words, self.dtype_to_str(input.dtype)
            )
            packed = program(
                input.reshape(-1), weight.view(torch.int64).view(weight.shape[0], words)
            )
            return packed.view(weight.dtype).view(shape)
        if (
            input.numel() >= 1024
            and input.numel() % 1024 == 0
            and 160 < weight.shape[1] <= 8192
            and row_bytes % 32 == 0
        ):
            program = _row_batch(
                input.numel(),
                *weight.shape,
                self.dtype_str,
                self.dtype_to_str(input.dtype),
                weight.element_size(),
            )
            return program(input.reshape(-1), weight.view(-1)).view(shape)
        if weight.shape[1] <= 32:
            builder = _lookup_narrow
        elif 1024 <= input.numel() and input.numel() % 32 == 0 and weight.shape[1] <= 8192:
            builder = _prefetched_ids
        else:
            builder = _lookup
        program = builder(
            input.numel(), *weight.shape, self.dtype_str, self.dtype_to_str(input.dtype)
        )
        return program(input.reshape(-1), weight).view(shape)


register_kernel_type("EmbeddingFwdOp", "ascend", AscendEmbeddingKernel)
