"""Ascend Native Sparse Attention (NSA) compressed-block selection."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.layout import make_ascend_compact_nz_layout

from tileops.backend import register_kernel_type
from tileops.kernels.attention.call_spec import NSACall, NSATopKFwdInterface
from tileops.kernels.kernel_base import Kernel, vector_aligned

__all__ = ["AscendNSATopKVarlenKernel"]


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
    num_blocks: int = 36,
    projection_first_task: int = 0,
    projection_steps: int = 0,
    projection_stride: int = 0,
):
    """Cache one FP32 QK matrix and rank its complete GQA importance rows.

    Projection arguments select diagnostic global task streams. Their zero
    production defaults write the complete output through the same template.
    """
    configurations = (
        (8, 8192, 32, 2, 128, 256, 16, 32, 1.0),
        (4, 4096, 32, 2, 128, 128, 16, 32, 1.0),
        (3, 5153, 32, 2, 128, 163, 16, 32, 0.08838834764831845),
        (1, 16384, 32, 2, 128, 256, 32, 64, 0.08838834764831845),
        (4, 8192, 64, 2, 64, 256, 8, 32, 0.125),
    )
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
    ) in configurations, "Ascend NSA TopK supports the five declared workload configurations"
    assert dtype in ("float16", "bfloat16")
    assert num_blocks > 0
    group = heads // heads_kv
    # All requests fit this bound without reading metadata on the host. Padding
    # to a power of two also gives the SIMD reductions whole 64-float vectors.
    chunk_pad = 1 << (chunk_num - 1).bit_length()
    tasks = tokens * heads_kv
    grid = min(num_blocks, tasks)
    projected = projection_steps > 0
    if projected:
        assert projection_first_task >= 0 and projection_stride >= grid
        assert projection_first_task + (projection_steps - 1) * projection_stride + grid <= tasks
        steps = projection_steps
    else:
        assert projection_first_task == projection_steps == projection_stride == 0
        steps = (tasks + grid - 1) // grid

    @T.macro
    def reduce_max_rows(src, dst, rows: int):
        # A reduction produces one scalar per row. Spell out its address and
        # store width instead of relying on provider-specific row lowering.
        with T.SimdVF():
            mask = T.simd.pset(32, "PAT_ALL")
            one = T.simd.pset(32, "PAT_VL1")
            for row in T.serial(rows):
                identity = T.simd.vdup(T.float32(-3.402823e38), "float32", mask)
                first = T.simd.vcmax(T.simd.vld(src[row, 0]), mask)
                second = T.simd.vcmax(T.simd.vld(src[row, 64]), mask)
                acc0 = T.simd.vmax(identity, first, mask)
                acc1 = T.simd.vmax(acc0, second, mask)
                if chunk_pad == 256:
                    third = T.simd.vcmax(T.simd.vld(src[row, 128]), mask)
                    fourth = T.simd.vcmax(T.simd.vld(src[row, 192]), mask)
                    acc2 = T.simd.vmax(acc1, third, mask)
                    acc3 = T.simd.vmax(acc2, fourth, mask)
                    T.simd.vsts(dst[row], acc3, one, dist="ONEPT_B32", extent=1)
                else:
                    T.simd.vsts(dst[row], acc1, one, dist="ONEPT_B32", extent=1)

    @T.macro
    def reduce_sum_rows(src, dst, rows: int):
        with T.SimdVF():
            mask = T.simd.pset(32, "PAT_ALL")
            one = T.simd.pset(32, "PAT_VL1")
            for row in T.serial(rows):
                identity = T.simd.vdup(T.float32(0.0), "float32", mask)
                first = T.simd.vcadd(T.simd.vld(src[row, 0]), mask)
                second = T.simd.vcadd(T.simd.vld(src[row, 64]), mask)
                acc0 = T.simd.vadd(identity, first, mask)
                acc1 = T.simd.vadd(acc0, second, mask)
                if chunk_pad == 256:
                    third = T.simd.vcadd(T.simd.vld(src[row, 128]), mask)
                    fourth = T.simd.vcadd(T.simd.vld(src[row, 192]), mask)
                    acc2 = T.simd.vadd(acc1, third, mask)
                    acc3 = T.simd.vadd(acc2, fourth, mask)
                    T.simd.vsts(dst[row], acc3, one, dist="ONEPT_B32", extent=1)
                else:
                    T.simd.vsts(dst[row], acc1, one, dist="ONEPT_B32", extent=1)

    @T.prim_func
    def main(
        q: T.Tensor((tokens, heads, dim), dtype),
        k_cmp: T.Tensor((chunk_num, heads_kv, dim), dtype),
        offsets: T.Tensor((batch + 1,), "int32"),
        chunk_offsets: T.Tensor((batch + 1,), "int32"),
        token_indices: T.Tensor((tokens, 2), "int32"),
        block_indices: T.Tensor((tokens, heads_kv, selected_blocks), "int32"),
    ):
        # A single AIV owns the entire GQA group, its reductions and every
        # output slot; no importance accumulator is shared between AIVs.
        with T.MixedKernel(grid, sids=1) as (bx, _sid):
            q_l1 = T.alloc_l1((group, dim), dtype)
            k_l1 = T.alloc_l1((chunk_pad, dim), dtype)
            q_l0 = T.alloc_l0a((group, dim), dtype)
            k_l0 = T.alloc_l0b((chunk_pad, dim), dtype)
            scores_l0c = T.alloc_l0c((group, chunk_pad), "float32")
            # One ghost row separates compact-NZ column tiles in UB. It is
            # physical padding, excluded from every logical copy and GEMM.
            queries = T.alloc_shared((group + 1, dim), dtype)
            keys = T.alloc_shared((chunk_pad + 1, dim), dtype)
            scores = T.alloc_shared((group, chunk_pad), "float32")
            exponentials = T.alloc_shared((group, chunk_pad), "float32")
            row_maximum = T.alloc_shared((group,), "float32")
            row_sum = T.alloc_shared((group,), "float32")
            importance = T.alloc_shared((1, chunk_pad), "float32")
            # Negated IDs and the padding sentinel are exactly representable
            # in FP32. Max on negative IDs selects the smallest tied block id.
            candidate_ids = T.alloc_shared((1, chunk_pad), "float32")
            best_score = T.alloc_shared((1,), "float32")
            best_id = T.alloc_shared((1,), "float32")
            selected = T.alloc_shared((selected_blocks,), "int32")
            T.annotate_layout(
                {
                    queries: make_ascend_compact_nz_layout(queries),
                    keys: make_ascend_compact_nz_layout(keys),
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
                    keys: 1,
                    scores: 1,
                    exponentials: 1,
                    row_maximum: 1,
                    row_sum: 1,
                    importance: 1,
                    candidate_ids: 1,
                    best_score: 1,
                    best_id: 1,
                    selected: 1,
                }
            )
            for step in T.serial(steps):
                if projected:
                    task = projection_first_task + step * projection_stride + bx
                else:
                    task = step * grid + bx
                if task < tasks:
                    token = task // heads_kv
                    kv_head = task % heads_kv
                    request = token_indices[token, 0]
                    position = token_indices[token, 1]
                    current = position // block_size
                    chunk_begin = chunk_offsets[request]
                    request_length = offsets[request + 1] - offsets[request]
                    request_chunks = T.min(
                        chunk_offsets[request + 1] - chunk_begin,
                        T.ceildiv(request_length, block_size),
                    )
                    candidates = T.min(current + 1, request_chunks)
                    closed = T.min((position + 1) // block_size, request_chunks)
                    with T.SimtVF(threads=128):
                        for g, d in T.Parallel(group, dim):
                            # The dtype store is intentional: FLA rounds scaled
                            # Q before its FP32 dot, rather than scaling QK.
                            queries[g, d] = T.cast(
                                T.cast(q[token, kv_head * group + g, d], "float32")
                                * T.cast(scale, "float32"),
                                dtype,
                            )
                        for c, d in T.Parallel(chunk_pad, dim):
                            if c < candidates:
                                keys[c, d] = k_cmp[chunk_begin + c, kv_head, d]
                            else:
                                keys[c, d] = T.cast(0, dtype)
                    T.copy(queries[:group, :], q_l1)
                    T.copy(keys[:chunk_pad, :], k_l1)
                    T.copy(q_l1, q_l0)
                    T.copy(k_l1, k_l0)
                    T.gemm(q_l0, k_l0, scores_l0c, transpose_B=True, clear_accum=True)
                    T.copy(scores_l0c, scores)

                    with T.SimtVF(threads=128):
                        for g, c in T.Parallel(group, chunk_pad):
                            exponentials[g, c] = T.if_then_else(
                                c < closed, scores[g, c], -T.infinity("float32")
                            )
                    reduce_max_rows(exponentials, row_maximum, group)
                    with T.SimtVF(threads=128):
                        for g, c in T.Parallel(group, chunk_pad):
                            exponentials[g, c] = T.if_then_else(
                                c < closed,
                                T.exp(exponentials[g, c] - row_maximum[g]),
                                0.0,
                            )
                    reduce_sum_rows(exponentials, row_sum, group)
                    with T.SimtVF(threads=128):
                        for g in T.Parallel(group):
                            row_maximum[g] = T.if_then_else(
                                closed > 0, row_maximum[g] + T.log(row_sum[g]), 0.0
                            )
                        for c in T.Parallel(chunk_pad):
                            importance[0, c] = 0.0
                        # Every column has one thread owner throughout the
                        # serial GQA reduction, including the 32-head group.
                        for g in T.serial(group):
                            for c in T.Parallel(chunk_pad):
                                importance[0, c] += T.if_then_else(
                                    c < current and c < candidates,
                                    T.exp(scores[g, c] - row_maximum[g]),
                                    0.0,
                                )
                        for c in T.Parallel(chunk_pad):
                            priority = c == 0 or c == current - 1 or c == current
                            importance[0, c] = T.if_then_else(
                                c < candidates,
                                T.if_then_else(
                                    priority, T.cast(group, "float32"), importance[0, c]
                                ),
                                -T.infinity("float32"),
                            )

                    # Rank unquantized FP32 scores. Min-id resolves equal scores
                    # deterministically; the public contract permits any tie.
                    for rank in T.serial(selected_blocks):
                        reduce_max_rows(importance, best_score, 1)
                        with T.SimtVF(threads=128):
                            for c in T.Parallel(chunk_pad):
                                candidate_ids[0, c] = T.if_then_else(
                                    c < candidates and importance[0, c] == best_score[0],
                                    -T.cast(c, "float32"),
                                    -T.cast(chunk_pad, "float32"),
                                )
                        reduce_max_rows(candidate_ids, best_id, 1)
                        with T.SimtVF(threads=128):
                            for _ in T.Parallel(1):
                                if rank < candidates:
                                    picked = -T.cast(best_id[0], "int32")
                                    if picked >= 0 and picked < candidates and picked < chunk_pad:
                                        importance[0, picked] = -T.infinity("float32")
                                        selected[rank] = picked
                                    else:
                                        selected[rank] = -1
                                else:
                                    selected[rank] = -1
                    T.copy(selected, block_indices[token, kv_head, :])

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
    num_blocks: int = 36,
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
        configurations = (
            (8, 8192, 32, 2, 128, 256, 16, 32, 1.0),
            (4, 4096, 32, 2, 128, 128, 16, 32, 1.0),
            (3, 5153, 32, 2, 128, 163, 16, 32, 0.08838834764831845),
            (1, 16384, 32, 2, 128, 256, 32, 64, 0.08838834764831845),
            (4, 8192, 64, 2, 64, 256, 8, 32, 0.125),
        )
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
        if configuration not in configurations or call.dtype not in (torch.float16, torch.bfloat16):
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
        )

    def forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        inputs = (q, k_cmp, offsets, chunk_offsets, token_indices)
        for tensor in inputs:
            if tensor.device.type != "npu" or tensor.device != q.device:
                raise ValueError("NSA TopK inputs must be on one NPU device")
        normalized = tuple(tensor.contiguous() for tensor in inputs)
        return self._compiled(
            *(vector_aligned(tensor) if i < 2 else tensor for i, tensor in enumerate(normalized))
        )


register_kernel_type("NSATopKVarlenFwdOp", "ascend", AscendNSATopKVarlenKernel)
