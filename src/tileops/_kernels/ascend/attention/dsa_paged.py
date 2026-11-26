"""Sparse paged MLA: decode just the selected FP8 cache rows into a bounded working set."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch
import torch.nn.functional as F

from tileops._kernels.ascend.attention.dsa_dense import _attend, device_core_counts
from tileops._kernels.attention.call_spec import DSAPagedFwdInterface
from tileops._kernels.kernel_base import Kernel
from tileops.backend._registry import register_kernel_type


@functools.lru_cache(maxsize=32)
def _gather(batch, seq, k, pages, page_size, table_width):
    kt = 16
    slots = (k + 63) // 64 * 64

    @T.prim_func
    def main(
        raw: T.Tensor((pages, page_size, 656), "uint8"),
        table: T.Tensor((batch, table_width), "int32"),
        lengths: T.Tensor((batch,), "int32"),
        ids: T.Tensor((batch, seq, k), "int32"),
        keys: T.Tensor((batch * seq, slots, 1, 576), "bfloat16"),
        positions: T.Tensor((batch * seq, 1, 1, slots), "int32"),
    ):
        with T.Kernel(batch * seq * (slots // kt)) as bx:
            row = bx // (slots // kt)
            b = row // seq
            s = row % seq
            first = bx % (slots // kt) * kt
            phys = T.alloc_shared((kt,), "int32")
            packed = T.alloc_shared((kt, 576), "bfloat16")
            # A single DMA brings in the exact 656-byte ABI row. The rounded UB
            # stride keeps scales and the BF16 rope lane aligned between slots.
            raw_rows = T.alloc_shared((kt, 672), "uint8")
            codes = T.view(raw_rows, (kt, 672), dtype="float8_e4m3fn")
            words = T.view(raw_rows, (kt, 168), dtype="uint32")
            factors = T.view(raw_rows, (kt, 168), dtype="float32")
            rope = T.view(raw_rows, (kt, 336), dtype="bfloat16")
            with T.SimtVF(threads=32):
                for j in T.Parallel(kt):
                    slot = first + j
                    address = T.alloc_local((1,), "int32")
                    address[0] = -1
                    if slot < k:
                        pos = ids[b, s, slot]
                        if pos >= 0 and pos < lengths[b]:
                            page = table[b, pos // page_size]
                            address[0] = page * page_size + pos % page_size
                    phys[j] = address[0]
                    positions[row, 0, 0, slot] = T.if_then_else(address[0] >= 0, slot, -1)
            for j in T.serial(kt):
                addr = phys[j]
                if addr >= 0:
                    page = addr // page_size
                    offset = addr % page_size
                    T.copy(raw[page, offset, :], raw_rows[j, :656])
                else:
                    T.fill(words[j, :], T.uint32(0))
            with T.SimtVF(threads=128):
                for rr, dd in T.Parallel(kt, 64):
                    packed[rr, 512 + dd] = rope[rr, 264 + dd]
            with T.SimdVF():
                pred = T.simd.pset(32)
                for j in T.serial(kt):
                    for part in T.serial(8):
                        raw_vec = T.simd.vld(codes[j, part * 64], dist="UNPK4_B8")
                        values = T.simd.vcvt(raw_vec, "float32")
                        sf = T.simd.vld(factors[j, 128 + part // 2], dist="BRC_B32")
                        value = T.simd.vcvt(
                            T.simd.vmul(values, sf, pred),
                            "bfloat16",
                            round="ROUND_R",
                            sat=False,
                            part=0,
                        )
                        T.simd.vsts(packed[j, part * 64], value, pred, dist="PK_B32")
            T.copy(packed, keys[row, first : first + kt, 0, :])

    return tilelang.compile(main, target="ascend", out_idx=[4, 5])


def choose_split_count(rows, topk, cube_cores, table_width):
    """Balance KV tiles across Cube waves and prefer fewer merges for ties."""
    from tileops._kernels.ascend.attention.dsa_paged_indices import metadata_row_tile

    tiles = topk // 128
    candidates = (
        splits
        for splits in (2, 4)
        if tiles % splits == 0 and metadata_row_tile(rows * splits, topk // splits, table_width)
    )

    def critical_tiles(splits):
        waves = (rows * splits + cube_cores - 1) // cube_cores
        return waves * (tiles // splits), splits

    return min(candidates, key=critical_tiles, default=0)


def optimized_route(q, cache, table, indices, cube_cores):
    from tileops._kernels.ascend.attention.dsa_paged_indices import metadata_row_tile

    batch, seq, heads, depth = q.shape
    topk = indices.shape[-1]
    if (
        q.dtype != torch.bfloat16
        or cache.dtype != torch.uint8
        or cache.ndim != 3
        or cache.shape[-1] != 656
        or depth != 576
        or not 0 < topk <= 4096
        or topk % 128
        or cube_cores <= 0
    ):
        return 0
    rows, table_width = batch * seq, table.shape[-1]
    if heads == 64:
        return int(bool(metadata_row_tile(rows, topk, table_width)))
    if heads == 128:
        return choose_split_count(rows, topk, cube_cores, table_width)
    return 0


class AscendDSAPagedKernel(Kernel, DSAPagedFwdInterface):
    devices = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call

    def forward(self, q, kv_cache, block_table, cache_seqlens, indices):
        b, s, h, d = q.shape
        topk = indices.shape[-1]
        if not b or not s or not h or not topk:
            return q.new_zeros((b, s, h, 512)), q.new_full(
                (b, h, s), -float("inf"), dtype=torch.float32
            )
        cube_cores, vector_cores = device_core_counts(q.device)
        splits = optimized_route(q, kv_cache, block_table, indices, cube_cores)
        scale = d**-0.5 if self.call.sm_scale is None else self.call.sm_scale
        if splits == 1:
            from tileops._kernels.ascend.attention.dsa_paged_h64 import forward_h64

            return forward_h64(
                q, kv_cache, block_table, cache_seqlens, indices, scale, cube_cores, vector_cores
            )
        if splits > 1:
            from tileops._kernels.ascend.attention.dsa_paged_h128 import forward_h128

            return forward_h128(
                q,
                kv_cache,
                block_table,
                cache_seqlens,
                indices,
                scale,
                splits,
                cube_cores,
                vector_cores,
                first_split_init=(splits == 2),
            )
        bm = 32 if h <= 32 else 64
        q = q.reshape(b * s, 1, 1, h, 576)
        if h % bm:
            q = F.pad(q, (0, 0, 0, (-h) % bm))
        gathered, selected = _gather(b, s, topk, *kv_cache.shape[:2], block_table.shape[1])(
            kv_cache,
            block_table,
            cache_seqlens,
            indices,
        )
        slots = gathered.shape[1]
        o, lse = _attend(b * s, 1, 1, h, slots, slots, 576, 512, "bfloat16", scale, True, True)(
            q, gathered, selected
        )
        return o[:, :, :, :h, :].reshape(b, s, h, 512), lse[:, :, :, :h].reshape(b, s, h).transpose(
            1, 2
        ).contiguous()


register_kernel_type("DSAPagedFwdOp", "ascend", AscendDSAPagedKernel)
