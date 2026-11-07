"""Ascend paged cache gather: 32-row regions, up to 8-row reads, store mode 0.

FP16/BF16 and scaled FP8 share the same two-region schedule. Request lengths,
page IDs, starts and scale remain live device data on every invocation.
"""

import functools
import math
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.attention.paged_cache_gather import (
    PagedKVCacheGatherCall,
    PagedKVCacheGatherFwdInterface,
)
from tileops.kernels.kernel_base import Kernel


def _vector_cores(device_index):
    from tilelang.carver.arch.driver import get_num_vector_cores

    count = get_num_vector_cores(device_id=device_index)
    if not count:
        raise RuntimeError("cannot determine Ascend vector core count")
    return int(count)


@functools.lru_cache(maxsize=64)
def _compile(batch, pages, page_size, table_width, width, dtype, out_dtype, starts, blocks):
    group_rows, tile_rows = 32, 8
    buffers = 2
    quantized = dtype == "float8_e4m3fn"
    ready_event = "V_MTE3" if quantized else "MTE2_MTE3"
    store_l2_cache_ctrl = 0
    total = T.dynamic("total")

    @T.prim_func
    def main(
        dst: T.Tensor((total, width), out_dtype),
        cache: T.Tensor((pages * page_size, width), dtype),
        table: T.Tensor((batch, table_width), "int32"),
        cu: T.Tensor((batch + 1,), "int32"),
        start: T.Tensor((batch,) if starts else (batch + 1,), "int32"),
        scale: T.Tensor((1,) if quantized else (pages * page_size, width),
                        "float32" if quantized else dtype),
    ):
        with T.Kernel(blocks) as bx:
            # Two disjoint 32-row payload regions alternate across groups.
            src = T.alloc_shared((buffers, group_rows, width), dtype)
            converted = T.alloc_shared((buffers, group_rows, width) if quantized else (1, 1, 1), out_dtype)
            scale_ub = T.alloc_shared((1,), "float32")
            offsets = T.alloc_shared((batch + 1,), "int32")
            beginnings = T.alloc_shared((batch,), "int32")
            # At worst every row is a separate page/request segment. These
            # descriptors are scalar-written/read; payload DMA never reads them.
            addresses = T.alloc_shared((group_rows,), "int64")
            lengths = T.alloc_shared((group_rows,), "int32")
            destinations = T.alloc_shared((group_rows,), "int32")
            if quantized:
                T.copy(scale, scale_ub)
            T.copy(cu, offsets)
            if starts:
                T.copy(start, beginnings)
            T.ascend_set_flag("MTE2_S", 0)
            T.ascend_wait_flag("MTE2_S", 0)
            # Partition contiguous tile intervals across cores, including tails.
            tiles_per_core = T.ceildiv(T.ceildiv(total, 16), blocks)
            first = bx * tiles_per_core * 16
            end = T.min(total, (bx + 1) * tiles_per_core * 16)
            request = T.alloc_var("int32", init=0)
            lo = T.alloc_var("int32", init=0)
            hi = T.alloc_var("int32", init=batch)
            if first < end:
                while lo < hi:
                    mid = (lo + hi) // 2
                    if offsets[mid + 1] <= first:
                        lo = mid + 1
                    else:
                        hi = mid
                request = lo
                groups = T.ceildiv(end - first, group_rows)
                for group in T.serial(groups):
                    row = first + group * group_rows
                    valid = T.min(group_rows, end - row)
                    # Issue the previous output BEFORE preparing current
                    # metadata (which contains dependent GM scalar reads).
                    # Otherwise metadata preparation can delay MTE3 launch.
                    if group > 0:
                        previous_row = row - group_rows
                        for slot in T.unroll(buffers):
                            if (group - 1) % buffers == slot:
                                T.ascend_wait_flag(ready_event, slot)
                                if quantized:
                                    T.copy(converted[slot, :, :],
                                           dst[previous_row:previous_row + group_rows, :], l2_cache_ctrl=store_l2_cache_ctrl)
                                else:
                                    T.copy(src[slot, :, :],
                                           dst[previous_row:previous_row + group_rows, :], l2_cache_ctrl=store_l2_cache_ctrl)
                                T.ascend_set_flag("MTE3_MTE2", slot)
                                if quantized:
                                    T.ascend_set_flag("MTE3_V", slot)
                    prepared = T.alloc_var("int32", init=0)
                    segments = T.alloc_var("int32", init=0)
                    # Prepare all addresses before issuing any payload read.
                    while prepared < valid:
                        while offsets[request + 1] <= row + prepared:
                            request = request + 1
                        pos = row + prepared - offsets[request]
                        if starts:
                            pos = pos + beginnings[request]
                        count = T.min(
                            tile_rows - prepared % tile_rows,
                            T.min(valid - prepared, T.min(
                                offsets[request + 1] - row - prepared,
                                page_size - pos % page_size)),
                        )
                        addresses[segments] = T.cast(
                            table[request, pos // page_size], "int64"
                        ) * page_size + pos % page_size
                        lengths[segments] = count
                        destinations[segments] = prepared
                        prepared = prepared + count
                        segments = segments + 1
                    # Constant event IDs and buffer slices in each unrolled branch.
                    for slot in T.unroll(buffers):
                        if group % buffers == slot:
                            if group >= buffers:
                                T.ascend_wait_flag("MTE3_MTE2", slot)
                            for segment in T.serial(segments):
                                physical = addresses[segment]
                                target = destinations[segment]
                                count = lengths[segment]
                                if count == tile_rows:
                                    T.copy(cache[physical:physical + tile_rows, :],
                                           src[slot, target:target + tile_rows, :])
                                else:
                                    T.copy(cache[physical:physical + count, :],
                                           src[slot, target:target + count, :])
                            if quantized:
                                if group >= buffers:
                                    T.ascend_wait_flag("MTE3_V", slot)
                                T.ascend_set_flag("MTE2_V", slot)
                                T.ascend_wait_flag("MTE2_V", slot)
                                with T.SimtVF(threads=256):
                                    for rr, cc in T.Parallel(group_rows, width):
                                        if rr < valid:
                                            converted[slot, rr, cc] = T.cast(src[slot, rr, cc], "float32") * scale_ub[0]
                            T.ascend_set_flag(ready_event, slot)
                # Drain the final group; only this group may be partial.
                final_row = first + (groups - 1) * group_rows
                final_valid = end - final_row
                for slot in T.unroll(buffers):
                    if (groups - 1) % buffers == slot:
                        T.ascend_wait_flag(ready_event, slot)
                        if quantized:
                            if final_valid == group_rows:
                                T.copy(converted[slot, :, :], dst[final_row:final_row + group_rows, :], l2_cache_ctrl=store_l2_cache_ctrl)
                            else:
                                T.copy(converted[slot, 0:final_valid, :], dst[final_row:end, :], l2_cache_ctrl=store_l2_cache_ctrl)
                        else:
                            if final_valid == group_rows:
                                T.copy(src[slot, :, :], dst[final_row:final_row + group_rows, :], l2_cache_ctrl=store_l2_cache_ctrl)
                            else:
                                T.copy(src[slot, 0:final_valid, :], dst[final_row:end, :], l2_cache_ctrl=store_l2_cache_ctrl)
                        T.ascend_set_flag("MTE3_MTE2", slot)
                        if quantized:
                            T.ascend_set_flag("MTE3_V", slot)
                # Consume the last release event for each used buffer.
                for slot in T.unroll(buffers):
                    if groups > slot:
                        T.ascend_wait_flag("MTE3_MTE2", slot)
                        if quantized:
                            T.ascend_wait_flag("MTE3_V", slot)
            T.ascend_set_flag("MTE3_S", 0)
            T.ascend_wait_flag("MTE3_S", 0)

    return tilelang.compile(main, target="ascend", out_idx=[], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    })


class AscendPagedKVCacheGatherKernel(Kernel, PagedKVCacheGatherFwdInterface):
    """Gather supported row geometries without copying metadata to the host."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: PagedKVCacheGatherCall):
        if call.page_size not in (16, 64) or call.row_width not in (576, 1024):
            return "requires page size 16/64 and row width 576/1024"
        if not 0 < call.batch <= 256 or min(call.num_pages, call.table_width) <= 0:
            return "requires 1..256 requests and nonempty page pool/table"
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
            return "requires FP16, BF16 or FP8 E4M3FN cache"
        if call.dtype == torch.float8_e4m3fn:
            if call.out_dtype not in (torch.float16, torch.bfloat16):
                return "FP8 requires FP16/BF16 destination"
        elif call.out_dtype != call.dtype:
            return "unquantized gather preserves dtype"
        return super().refusal(call)

    def __init__(self, call):
        reason = self.refusal(call)
        if reason:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        self._compiled = _compile(
            call.batch, call.num_pages, call.page_size, call.table_width, call.row_width,
            self.dtype_str, self.dtype_to_str(call.out_dtype), call.has_starts,
            _vector_cores(self.device_index),
        )

    def forward(self, dst, cache, block_table, cu_seq_lens, seq_starts=None, scale=None):
        tensors = dict(dst=dst, cache=cache, block_table=block_table, cu_seq_lens=cu_seq_lens)
        if seq_starts is not None:
            tensors["seq_starts"] = seq_starts
        if scale is not None:
            tensors["scale"] = scale
        self._require_devices(**tensors)
        if any(t.device != dst.device for t in tensors.values()):
            raise ValueError("every gather tensor must be on the same NPU")
        c = self.call
        if (seq_starts is not None) != c.has_starts or (scale is not None) != (c.dtype == torch.float8_e4m3fn):
            raise ValueError("optional inputs do not match gather specialization")
        if (cache.shape[:2] != (c.num_pages, c.page_size)
                or tuple(dst.shape[1:]) != tuple(cache.shape[2:])
                or math.prod(dst.shape[1:]) != c.row_width
                or cache.dtype != c.dtype or dst.dtype != c.out_dtype):
            raise ValueError("cache/destination does not match gather specialization")
        metadata = [(block_table, (c.batch, c.table_width), torch.int32),
                    (cu_seq_lens, (c.batch + 1,), torch.int32)]
        if seq_starts is not None:
            metadata.append((seq_starts, (c.batch,), torch.int32))
        if scale is not None:
            metadata.append((scale, (1,), torch.float32))
        if any(tuple(t.shape) != shape or t.dtype != dtype for t, shape, dtype in metadata):
            raise ValueError("gather metadata shape/dtype does not match specialization")
        if any(not t.is_contiguous() for t in tensors.values()):
            raise ValueError("kernel inputs must be contiguous; use the public Op for views")
        if not dst.numel():
            return None
        flat_cache = cache.view(c.num_pages * c.page_size, c.row_width)
        self._compiled(dst.view(dst.shape[0], c.row_width), flat_cache, block_table,
                       cu_seq_lens, seq_starts if c.has_starts else cu_seq_lens,
                       scale if scale is not None else flat_cache)
        return None




register_kernel_type("PagedKVCacheGatherFwdOp", "ascend", AscendPagedKVCacheGatherKernel)
