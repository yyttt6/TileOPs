"""Ascend Native Sparse Attention (NSA) compressed-block selection."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.layout import make_ascend_compact_nz_layout

from tileops.backend import register_kernel_type
from tileops.kernels.ascend.attention.nsa_common import (
    calibrated_cube_blocks,
    check_inputs,
    reduce_max_rows,
    reduce_sum_rows,
)
from tileops.kernels.attention.call_spec import NSACall, NSATopKFwdInterface
from tileops.kernels.kernel_base import Kernel, vector_aligned

__all__ = ["AscendNSATopKVarlenKernel"]

# The five manifest call configurations this implementation serves, in both dtypes.
# ``refusal`` reads the same table.
_CONFIGURATIONS = (
    (8, 8192, 32, 2, 128, 256, 16, 32, 1.0),
    (4, 4096, 32, 2, 128, 128, 16, 32, 1.0),
    (3, 5153, 32, 2, 128, 163, 16, 32, 0.08838834764831845),
    (1, 16384, 32, 2, 128, 256, 32, 64, 0.08838834764831845),
    (4, 8192, 64, 2, 64, 256, 8, 32, 0.125),
)


def _make_nsa_topk_varlen_kernel(
    batch: int,
    tokens: int,
    heads: int,
    heads_kv: int,
    dim: int,
    chunk_num: int,
    selected_blocks: int,
    block_size: int,
    scale: float,
    dtype: str,
    num_blocks: int,
):
    """Pair queries within a request; each AIV ranks one complete importance row."""
    assert (
        batch,
        tokens,
        heads,
        heads_kv,
        dim,
        chunk_num,
        selected_blocks,
        block_size,
        scale,
    ) in _CONFIGURATIONS, "Ascend NSA TopK supports the five declared workload configurations"
    assert dtype in ("float16", "bfloat16")
    assert num_blocks > 0
    group = heads // heads_kv
    chunk_pad = max(64, 1 << (chunk_num - 1).bit_length())
    # Compact request-local pairs keep both vectors busy without assigning one
    # permanently inactive parity of token tasks to half of the physical grid.
    # The bound adds at most batch // 2 empty tasks when all lengths are even.
    pair_bound = (tokens + batch) // 2
    tasks = pair_bound * heads_kv
    grid = min(num_blocks, tasks)
    steps = (tasks + grid - 1) // grid

    def make_compute(width):
        max_rows = reduce_max_rows(group, width)
        sum_rows = reduce_sum_rows(group, width)

        @T.macro
        def compute(
            q,
            k_cmp,
            block_indices,
            sid,
            token,
            end,
            position,
            kv_head,
            chunk_begin,
            request_chunks,
            reload_keys,
            q_l1,
            k_l1,
            q_l0,
            k_l0,
            scores_l0c,
            queries,
            scores,
            exponentials,
            row_maximum,
            row_sum,
            importance,
            best_id,
            selected,
        ):
            # Both rows have the same candidate window: block_size is even,
            # and each pair starts at an even request-local position.
            current = position // block_size
            candidates = T.min(current + 1, request_chunks)
            closed = T.min((position + sid + 1) // block_size, request_chunks)
            with T.SimtVF(threads=128):
                for g, d in T.Parallel(group, dim):
                    queries[g, d] = T.if_then_else(
                        token + sid < end,
                        T.cast(
                            T.cast(
                                q[T.min(token + sid, tokens - 1), kv_head * group + g, d], "float32"
                            )
                            * T.cast(scale, "float32"),
                            dtype,
                        ),
                        T.cast(0, dtype),
                    )
                for rank in T.Parallel(selected_blocks):
                    selected[rank] = -1
            T.dual_copy(queries[:group, :], q_l1)
            key_tile = T.alloc_shared((width // 2 + 1, dim), dtype)
            T.annotate_layout({key_tile: make_ascend_compact_nz_layout(key_tile)})
            T.annotate_buffer_versions({key_tile: 1})
            if reload_keys:
                with T.SimtVF(threads=128):
                    for c, d in T.Parallel(width // 2, dim):
                        column = c + sid * (width // 2)
                        key_tile[c, d] = T.if_then_else(
                            column < request_chunks,
                            k_cmp[T.min(chunk_begin + column, chunk_num - 1), kv_head, d],
                            T.cast(0, dtype),
                        )
                T.dual_copy(key_tile[: width // 2, :], k_l1[:width, :])
            T.copy(q_l1, q_l0)
            T.copy(k_l1[:width, :], k_l0[:width, :])
            T.gemm(
                q_l0,
                k_l0[:width, :],
                scores_l0c[:, :width],
                transpose_B=True,
                clear_accum=True,
            )
            T.dual_copy(scores_l0c[:, :width], scores[:, :width])
            with T.SimtVF(threads=128):
                for g, c in T.Parallel(group, width):
                    exponentials[g, c] = T.if_then_else(
                        c < closed, scores[g, c], -T.infinity("float32")
                    )
            max_rows(exponentials, row_maximum)
            with T.SimtVF(threads=128):
                for g, c in T.Parallel(group, width):
                    exponentials[g, c] = T.if_then_else(
                        c < closed, T.exp(exponentials[g, c] - row_maximum[g]), 0.0
                    )
            sum_rows(exponentials, row_sum)
            with T.SimtVF(threads=128):
                for g in T.Parallel(group):
                    row_maximum[g] = row_maximum[g] + T.log(row_sum[g])
                for c in T.Parallel(width):
                    value = T.alloc_var("float32", init=0.0)
                    for g in T.serial(group):
                        value += T.if_then_else(
                            c < current and c < candidates,
                            T.exp(scores[g, c] - row_maximum[g]),
                            0.0,
                        )
                    priority = c == 0 or c == current - 1 or c == current
                    importance[0, c] = T.if_then_else(
                        c < candidates,
                        T.if_then_else(priority, T.cast(group, "float32"), value),
                        -T.infinity("float32"),
                    )

            for rank in T.serial(T.min(selected_blocks, candidates)):
                # Keep the maximum and its smallest tied ID in vector registers.
                # One SIMD call replaces two reductions and an intervening SIMT
                # call materializing a whole candidate-ID tensor in UB.
                with T.SimdVF():
                    mask = T.simd.pset(32, "PAT_ALL")
                    one = T.simd.pset(32, "PAT_VL1")
                    top = T.simd.vcmax(T.simd.vld(importance[0, 0]), mask)
                    if width >= 128:
                        top = T.simd.vmax(
                            top, T.simd.vcmax(T.simd.vld(importance[0, 64]), mask), mask
                        )
                    if width >= 256:
                        top = T.simd.vmax(
                            top, T.simd.vcmax(T.simd.vld(importance[0, 128]), mask), mask
                        )
                        top = T.simd.vmax(
                            top, T.simd.vcmax(T.simd.vld(importance[0, 192]), mask), mask
                        )
                    top = T.simd.vdupv(top, mask)
                    sentinel = T.simd.vdup(T.int32(width), "int32", mask)
                    ids = T.simd.vci(T.int32(0), "int32")
                    tied = T.simd.vcmp(T.simd.vld(importance[0, 0]), top, mask, op="eq")
                    lowest = T.simd.vcmin(T.simd.vsel(ids, sentinel, tied), mask)
                    if width >= 128:
                        ids = T.simd.vci(T.int32(64), "int32")
                        tied = T.simd.vcmp(T.simd.vld(importance[0, 64]), top, mask, op="eq")
                        lowest = T.simd.vmin(
                            lowest, T.simd.vcmin(T.simd.vsel(ids, sentinel, tied), mask), mask
                        )
                    if width >= 256:
                        ids = T.simd.vci(T.int32(128), "int32")
                        tied = T.simd.vcmp(T.simd.vld(importance[0, 128]), top, mask, op="eq")
                        lowest = T.simd.vmin(
                            lowest, T.simd.vcmin(T.simd.vsel(ids, sentinel, tied), mask), mask
                        )
                        ids = T.simd.vci(T.int32(192), "int32")
                        tied = T.simd.vcmp(T.simd.vld(importance[0, 192]), top, mask, op="eq")
                        lowest = T.simd.vmin(
                            lowest, T.simd.vcmin(T.simd.vsel(ids, sentinel, tied), mask), mask
                        )
                    T.simd.vsts(best_id[0], lowest, one, dist="ONEPT_B32", extent=1)
                with T.SimtVF(threads=128):
                    for c in T.Parallel(width):
                        importance[0, c] = T.if_then_else(
                            c == best_id[0], -T.infinity("float32"), importance[0, c]
                        )
                    for _ in T.Parallel(1):
                        selected[rank] = best_id[0]
            with T.SimtVF(threads=128):
                for rank in T.Parallel(selected_blocks):
                    if token + sid < end:
                        block_indices[token + sid, kv_head, rank] = selected[rank]

        return compute

    compute64 = make_compute(64)
    compute128 = make_compute(128)
    compute256 = make_compute(256)

    @T.prim_func
    def main(
        q: T.Tensor((tokens, heads, dim), dtype),
        k_cmp: T.Tensor((chunk_num, heads_kv, dim), dtype),
        offsets: T.Tensor((batch + 1,), "int32"),
        chunk_offsets: T.Tensor((batch + 1,), "int32"),
        token_indices: T.Tensor((tokens, 2), "int32"),
        block_indices: T.Tensor((tokens, heads_kv, selected_blocks), "int32"),
    ):
        with T.MixedKernel(grid, sids=2) as (bx, sid):
            q_l1 = T.alloc_l1((2 * group, dim), dtype)
            k_l1 = T.alloc_l1((chunk_pad, dim), dtype)
            q_l0 = T.alloc_l0a((2 * group, dim), dtype)
            k_l0 = T.alloc_l0b((chunk_pad, dim), dtype)
            scores_l0c = T.alloc_l0c((2 * group, chunk_pad), "float32")
            # Each AIV stages one query; the shared K tile is split by columns.
            # Ghost rows separate compact-NZ column tiles in UB.
            queries = T.alloc_shared((group + 1, dim), dtype)
            scores = T.alloc_shared((group, chunk_pad), "float32")
            exponentials = T.alloc_shared((group, chunk_pad), "float32")
            row_maximum = T.alloc_shared((group,), "float32")
            row_sum = T.alloc_shared((group,), "float32")
            importance = T.alloc_shared((1, chunk_pad), "float32")
            best_id = T.alloc_shared((1,), "int32")
            selected = T.alloc_shared((selected_blocks,), "int32")
            T.annotate_layout(
                {
                    queries: make_ascend_compact_nz_layout(queries),
                }
            )
            T.annotate_buffer_versions(
                {
                    q_l1: 1,
                    k_l1: 1,
                    q_l0: 1,
                    k_l0: 1,
                    scores_l0c: 1,
                    queries: 1,
                    scores: 1,
                    exponentials: 1,
                    row_maximum: 1,
                    row_sum: 1,
                    importance: 1,
                    best_id: 1,
                    selected: 1,
                }
            )
            cached_request = T.alloc_var("int32", init=-1)
            cached_head = T.alloc_var("int32", init=-1)
            cached_width = T.alloc_var("int32", init=0)
            for step in T.serial(steps):
                task = step * grid + bx
                if task < tasks:
                    pair = task // heads_kv
                    kv_head = task % heads_kv
                    request = T.alloc_var("int32", init=0)
                    position = T.alloc_var("int32", init=0)
                    begin = T.alloc_var("int32", init=0)
                    end = T.alloc_var("int32", init=0)
                    prefix = T.alloc_var("int32", init=0)
                    active = T.alloc_var("bool", init=False)
                    # The number of odd request lengths is runtime metadata.
                    # Locate its compact pair prefix on device rather than
                    # assuming global even tokens begin request-local pairs.
                    for r in T.serial(batch):
                        length = offsets[r + 1] - offsets[r]
                        pair_end = prefix + T.ceildiv(length, 2)
                        if pair >= prefix and pair < pair_end:
                            request = r
                            position = 2 * (pair - prefix)
                            begin = offsets[r]
                            end = offsets[r + 1]
                            active = True
                        prefix = pair_end
                    if active:
                        token = begin + position
                        # Retain the public metadata ABI and use its canonical
                        # query identity after locating the compact pair.
                        request = token_indices[token, 0]
                        position = token_indices[token, 1]
                        chunk_begin = chunk_offsets[request]
                        request_chunks = T.min(
                            chunk_offsets[request + 1] - chunk_begin,
                            T.ceildiv(end - begin, block_size),
                        )
                        candidates = T.min(position // block_size + 1, request_chunks)
                        if candidates <= 3:
                            # All visible blocks are mandatory priority blocks.
                            # Their importance is exactly group, independent of QK.
                            with T.SimtVF(threads=128):
                                for rank in T.Parallel(selected_blocks):
                                    if token + sid < end:
                                        block_indices[token + sid, kv_head, rank] = T.if_then_else(
                                            rank < candidates, rank, -1
                                        )
                        else:
                            width = T.if_then_else(
                                candidates <= 64, 64, T.if_then_else(candidates <= 128, 128, 256)
                            )
                            reload_keys = (
                                cached_request != request
                                or cached_head != kv_head
                                or cached_width != width
                            )
                            if candidates <= 64:
                                compute64(
                                    q,
                                    k_cmp,
                                    block_indices,
                                    sid,
                                    token,
                                    end,
                                    position,
                                    kv_head,
                                    chunk_begin,
                                    request_chunks,
                                    reload_keys,
                                    q_l1,
                                    k_l1,
                                    q_l0,
                                    k_l0,
                                    scores_l0c,
                                    queries,
                                    scores,
                                    exponentials,
                                    row_maximum,
                                    row_sum,
                                    importance,
                                    best_id,
                                    selected,
                                )
                            elif candidates <= 128:
                                if chunk_pad >= 128:
                                    compute128(
                                        q,
                                        k_cmp,
                                        block_indices,
                                        sid,
                                        token,
                                        end,
                                        position,
                                        kv_head,
                                        chunk_begin,
                                        request_chunks,
                                        reload_keys,
                                        q_l1,
                                        k_l1,
                                        q_l0,
                                        k_l0,
                                        scores_l0c,
                                        queries,
                                        scores,
                                        exponentials,
                                        row_maximum,
                                        row_sum,
                                        importance,
                                        best_id,
                                        selected,
                                    )
                            else:
                                if chunk_pad >= 256:
                                    compute256(
                                        q,
                                        k_cmp,
                                        block_indices,
                                        sid,
                                        token,
                                        end,
                                        position,
                                        kv_head,
                                        chunk_begin,
                                        request_chunks,
                                        reload_keys,
                                        q_l1,
                                        k_l1,
                                        q_l0,
                                        k_l0,
                                        scores_l0c,
                                        queries,
                                        scores,
                                        exponentials,
                                        row_maximum,
                                        row_sum,
                                        importance,
                                        best_id,
                                        selected,
                                    )
                            cached_request = request
                            cached_head = kv_head
                            cached_width = width

    return main


@functools.lru_cache(maxsize=64)
def _compile_nsa_topk_varlen_kernel(
    batch: int,
    tokens: int,
    heads: int,
    heads_kv: int,
    dim: int,
    chunk_num: int,
    selected_blocks: int,
    block_size: int,
    scale: float,
    dtype: str,
    num_blocks: int,
):
    return tilelang.compile(
        _make_nsa_topk_varlen_kernel(
            batch,
            tokens,
            heads,
            heads_kv,
            dim,
            chunk_num,
            selected_blocks,
            block_size,
            scale,
            dtype,
            num_blocks,
        ),
        target="ascend",
        out_idx=-1,
    )


class AscendNSATopKVarlenKernel(Kernel, NSATopKFwdInterface):
    """NSA TopK for the five manifest configurations in FP16 and BF16."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: NSACall) -> str | None:
        configuration = (
            call.batch,
            call.c_seq_len,
            call.heads,
            call.heads_kv,
            call.dim,
            call.chunk_num,
            call.selected_blocks,
            call.block_size,
            call.scale,
        )
        if configuration not in _CONFIGURATIONS or call.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            return "requires one of the ten declared NSATopKVarlenFwdOp manifest calls"
        return super().refusal(call)

    def __init__(self, call: NSACall) -> None:
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        self._compiled = _compile_nsa_topk_varlen_kernel(
            call.batch,
            call.c_seq_len,
            call.heads,
            call.heads_kv,
            call.dim,
            call.chunk_num,
            call.selected_blocks,
            call.block_size,
            call.scale,
            self.dtype_str,
            calibrated_cube_blocks(),
        )

    def forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        expected = (
            ("q", q, (self.call.c_seq_len, self.call.heads, self.call.dim), self.dtype),
            ("k_cmp", k_cmp, (self.call.chunk_num, self.call.heads_kv, self.call.dim), self.dtype),
            ("offsets", offsets, (self.call.batch + 1,), torch.int32),
            ("chunk_offsets", chunk_offsets, (self.call.batch + 1,), torch.int32),
            ("token_indices", token_indices, (self.call.c_seq_len, 2), torch.int32),
        )
        check_inputs(self, expected)
        normalized = tuple(tensor.contiguous() for _, tensor, _shape, _dtype in expected)
        return self._compiled(
            *(vector_aligned(tensor) if i < 2 else tensor for i, tensor in enumerate(normalized))
        )


register_kernel_type("NSATopKVarlenFwdOp", "ascend", AscendNSATopKVarlenKernel)
