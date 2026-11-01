"""Ascend attention over the closed compressed chunks of each packed request."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.layout import make_ascend_compact_nz_layout

from tileops.backend import register_kernel_type
from tileops.kernels.attention.call_spec import NSACall, NSACompressedFwdInterface
from tileops.kernels.kernel_base import Kernel, vector_aligned

__all__ = ["AscendNSACompressedFwdVarlenKernel"]


def _make_nsa_compressed_varlen_kernel(
    batch: int,
    tokens: int,
    heads: int,
    heads_kv: int,
    dim_k: int,
    dim_v: int,
    chunk_num: int,
    block_size: int,
    scale: float,
    dtype: str,
    num_blocks: int = 36,
    projection_first_task: int = 0,
    projection_steps: int = 0,
    projection_stride: int = 0,
):
    """Build one persistent token/head-tile program, retaining full GM addresses.

    Projection parameters select diagnostic persistent-task streams. Their zero
    production defaults cover all tokens and never change the output shapes.
    The runtime closed-chunk count can exceed one tile even at a manifest shape:
    request boundaries are inputs, not constants of a compiled specialization.
    """
    assert batch > 0 and tokens > 0 and chunk_num > 0
    assert heads_kv > 0 and heads % heads_kv == 0
    assert heads // heads_kv % 16 == 0
    assert dim_k in (64, 128) and dim_v in (64, 128)
    assert block_size > 0 and num_blocks > 0
    assert dtype in ("float16", "bfloat16")
    head_tile = 16
    vector_rows = 8
    key_tile = 32
    softmax_tile = 64
    head_groups = heads // heads_kv // head_tile
    token_tasks = heads_kv * head_groups
    tasks = tokens * token_tasks
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
    def reduce_max_rows(source, destination):
        # Keep each row's result at its contiguous four-byte scalar address.
        # The explicit store also states its one-element memory footprint.
        with T.SimdVF():
            full = T.simd.pset(32, "PAT_ALL")
            one = T.simd.pset(32, "PAT_VL1")
            for row in T.serial(vector_rows):
                identity = T.simd.vdup(T.float32(-3.4028234663852886e38), "float32", full)
                reduced = T.simd.vcmax(T.simd.vld(source[row, 0]), full)
                value = T.simd.vmax(identity, reduced, full)
                T.simd.vsts(destination[row], value, one, dist="ONEPT_B32", extent=1)

    @T.macro
    def reduce_sum_rows(source, destination):
        with T.SimdVF():
            full = T.simd.pset(32, "PAT_ALL")
            one = T.simd.pset(32, "PAT_VL1")
            for row in T.serial(vector_rows):
                identity = T.simd.vdup(T.float32(0), "float32", full)
                reduced = T.simd.vcadd(T.simd.vld(source[row, 0]), full)
                value = T.simd.vadd(identity, reduced, full)
                T.simd.vsts(destination[row], value, one, dist="ONEPT_B32", extent=1)

    @T.prim_func
    def main(
        q: T.Tensor((tokens, heads, dim_k), dtype),
        k_cmp: T.Tensor((chunk_num, heads_kv, dim_k), dtype),
        v_cmp: T.Tensor((chunk_num, heads_kv, dim_v), dtype),
        offsets: T.Tensor((batch + 1,), "int32"),
        chunk_offsets: T.Tensor((batch + 1,), "int32"),
        token_indices: T.Tensor((tokens, 2), "int32"),
        output: T.Tensor((tokens, heads, dim_v), dtype),
        lse: T.Tensor((tokens, heads), dtype),
    ):
        with T.MixedKernel(grid, sids=2) as (bx, sid):
            q_l1 = T.alloc_l1((head_tile, dim_k), dtype)
            k_l1 = T.alloc_l1((key_tile, dim_k), dtype)
            v_l1 = T.alloc_l1((dim_v, key_tile), dtype)
            p_l1 = T.alloc_l1((head_tile, key_tile), dtype)
            q_l0 = T.alloc_l0a((head_tile, dim_k), dtype)
            k_l0 = T.alloc_l0b((key_tile, dim_k), dtype)
            p_l0 = T.alloc_l0a((head_tile, key_tile), dtype)
            v_l0 = T.alloc_l0b((dim_v, key_tile), dtype)
            scores_l0c = T.alloc_l0c((head_tile, key_tile), "float32")
            value_l0c = T.alloc_l0c((head_tile, dim_v), "float32")
            keys = T.alloc_shared((key_tile // 2 + 1, dim_k), dtype)
            values = T.alloc_shared((dim_v // 2 + 1, key_tile), dtype)
            scores = T.alloc_shared((vector_rows, softmax_tile), "float32")
            probabilities = T.alloc_shared((vector_rows + 1, key_tile), dtype)
            value_tile = T.alloc_shared((vector_rows, dim_v), "float32")
            result = T.alloc_shared((vector_rows, dim_v), dtype)
            block_maximum = T.alloc_shared((vector_rows,), "float32")
            block_sum = T.alloc_shared((vector_rows,), "float32")
            maximum = T.alloc_shared((vector_rows,), "float32")
            denominator = T.alloc_shared((vector_rows,), "float32")
            numerator = T.alloc_shared((vector_rows, dim_v), "float32")
            alpha = T.alloc_shared((vector_rows,), "float32")
            T.annotate_layout(
                {
                    keys: make_ascend_compact_nz_layout(keys),
                    values: make_ascend_compact_nz_layout(values),
                    probabilities: make_ascend_compact_nz_layout(probabilities),
                }
            )
            T.annotate_buffer_versions(
                {
                    q_l1: 1,
                    k_l1: 1,
                    v_l1: 1,
                    p_l1: 1,
                    q_l0: 1,
                    k_l0: 1,
                    p_l0: 1,
                    v_l0: 1,
                    scores_l0c: 1,
                    value_l0c: 1,
                    keys: 1,
                    values: 1,
                    scores: 1,
                    probabilities: 1,
                    value_tile: 1,
                    result: 1,
                    block_maximum: 1,
                    block_sum: 1,
                    maximum: 1,
                    denominator: 1,
                    numerator: 1,
                    alpha: 1,
                }
            )

            for step in T.serial(steps):
                if projected:
                    task = projection_first_task + step * projection_stride + bx
                else:
                    task = step * grid + bx
                if task < tasks:
                    token_row = task // token_tasks
                    kv_head = task % token_tasks // head_groups
                    head_group = task % head_groups
                    head_start = kv_head * (head_groups * head_tile) + head_group * head_tile
                    request = token_indices[token_row, 0]
                    position = token_indices[token_row, 1]
                    token = offsets[request] + position
                    chunk_begin = chunk_offsets[request]
                    closed = (position + 1) // block_size

                    if closed > 0:
                        T.copy(q[token, head_start : head_start + head_tile, :], q_l1)
                        T.copy(q_l1, q_l0)
                        for tile in T.serial(T.ceildiv(closed, key_tile)):
                            start = tile * key_tile
                            valid = T.min(key_tile, closed - start)
                            # V is read only for valid keys. Zeroing the remaining
                            # tile also prevents 0 * NaN from future/tail summaries.
                            with T.SimtVF(threads=128):
                                for s, d in T.Parallel(key_tile // 2, dim_k):
                                    key_position = s + sid * (key_tile // 2)
                                    keys[s, d] = T.if_then_else(
                                        key_position < valid,
                                        k_cmp[chunk_begin + start + key_position, kv_head, d],
                                        T.cast(0, dtype),
                                    )
                                for d, s in T.Parallel(dim_v // 2, key_tile):
                                    value_dimension = d + sid * (dim_v // 2)
                                    values[d, s] = T.if_then_else(
                                        s < valid,
                                        v_cmp[chunk_begin + start + s, kv_head, value_dimension],
                                        T.cast(0, dtype),
                                    )
                            T.dual_copy(keys[: key_tile // 2, :], k_l1)
                            T.dual_copy(values[: dim_v // 2, :], v_l1)
                            T.copy(k_l1, k_l0)
                            T.copy(v_l1, v_l0)
                            T.gemm(
                                q_l0,
                                k_l0,
                                scores_l0c,
                                transpose_B=True,
                                clear_accum=True,
                            )
                            T.dual_copy(scores_l0c, scores[:, :key_tile])
                            with T.SimtVF(threads=128):
                                for h, s in T.Parallel(vector_rows, softmax_tile):
                                    scores[h, s] = T.if_then_else(
                                        s < valid,
                                        scores[h, s] * scale,
                                        -T.infinity("float32"),
                                    )
                            reduce_max_rows(scores, block_maximum)
                            if tile == 0:
                                with T.SimtVF(threads=128):
                                    for h in T.Parallel(vector_rows):
                                        maximum[h] = block_maximum[h]
                            else:
                                with T.SimtVF(threads=128):
                                    for h in T.Parallel(vector_rows):
                                        next_maximum = T.max(maximum[h], block_maximum[h])
                                        alpha[h] = T.exp(maximum[h] - next_maximum)
                                        maximum[h] = next_maximum
                            with T.SimtVF(threads=128):
                                for h, s in T.Parallel(vector_rows, softmax_tile):
                                    scores[h, s] = T.exp(scores[h, s] - maximum[h])
                                    if s < key_tile:
                                        probabilities[h, s] = T.cast(scores[h, s], dtype)
                            reduce_sum_rows(scores, block_sum)
                            T.dual_copy(probabilities[:vector_rows, :], p_l1)
                            T.copy(p_l1, p_l0)
                            T.gemm(
                                p_l0,
                                v_l0,
                                value_l0c,
                                transpose_B=True,
                                clear_accum=True,
                            )
                            T.dual_copy(value_l0c, value_tile)
                            if tile == 0:
                                with T.SimtVF(threads=128):
                                    for h in T.Parallel(vector_rows):
                                        denominator[h] = block_sum[h]
                                    for h, d in T.Parallel(vector_rows, dim_v):
                                        numerator[h, d] = value_tile[h, d]
                            else:
                                with T.SimtVF(threads=128):
                                    for h in T.Parallel(vector_rows):
                                        denominator[h] = denominator[h] * alpha[h] + block_sum[h]
                                    for h, d in T.Parallel(vector_rows, dim_v):
                                        numerator[h, d] = (
                                            numerator[h, d] * alpha[h] + value_tile[h, d]
                                        )

                        with T.SimtVF(threads=128):
                            for h, d in T.Parallel(vector_rows, dim_v):
                                numerator[h, d] = T.if_then_else(
                                    denominator[h] > 0, numerator[h, d] / denominator[h], 0
                                )
                        with T.SimtVF(threads=128):
                            T.copy(numerator, result)
                        # Each SID owns eight half/bfloat16 LSE elements (16 B).
                        # Exact SIMT GM stores avoid a rounded-up 32 B DMA write.
                        with T.SimtVF(threads=128):
                            for h in T.Parallel(vector_rows):
                                lse[token, head_start + sid * vector_rows + h] = T.cast(
                                    T.if_then_else(
                                        denominator[h] > 0,
                                        maximum[h] + T.log(denominator[h]),
                                        0,
                                    ),
                                    dtype,
                                )
                    else:
                        with T.SimtVF(threads=128):
                            for h, d in T.Parallel(vector_rows, dim_v):
                                result[h, d] = T.cast(0, dtype)
                            for h in T.Parallel(vector_rows):
                                lse[token, head_start + sid * vector_rows + h] = T.cast(0, dtype)
                    T.dual_copy(result, output[token, head_start : head_start + head_tile, :])

    return main


@functools.lru_cache(maxsize=64)
def _compile_nsa_compressed_varlen_kernel(
    batch: int,
    tokens: int,
    heads: int,
    heads_kv: int,
    dim_k: int,
    dim_v: int,
    chunk_num: int,
    block_size: int,
    scale: float,
    dtype: str,
    num_blocks: int = 36,
):
    return tilelang.compile(
        _make_nsa_compressed_varlen_kernel(
            batch,
            tokens,
            heads,
            heads_kv,
            dim_k,
            dim_v,
            chunk_num,
            block_size,
            scale,
            dtype,
            num_blocks,
        ),
        target="ascend",
        out_idx=[-2, -1],
    )


class AscendNSACompressedFwdVarlenKernel(Kernel, NSACompressedFwdInterface):
    """Closed-chunk attention for the three compressed manifest workloads.

    The underlying builder keeps DK and DV independent for diagnostics. Public
    dispatch is restricted to the declared metadata shapes and parameters; the
    contents of offsets and chunk_offsets remain runtime inputs.
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: NSACall) -> str | None:
        configurations = (
            (8, 8192, 256, 32, 2, 128, 128, 32, 0.08838834764831845, torch.float16),
            (4, 4096, 128, 32, 2, 128, 128, 32, 0.08838834764831845, torch.float16),
            (2, 1024, 33, 32, 2, 128, 128, 32, 0.08838834764831845, torch.bfloat16),
        )
        configuration = (
            call.batch,
            call.c_seq_len,
            call.chunk_num,
            call.heads,
            call.heads_kv,
            call.dim,
            call.dim_v,
            call.block_size,
            call.scale,
            call.dtype,
        )
        if configuration not in configurations:
            return "requires one of the three NSACompressedVarlenFwdOp manifest configurations"
        return super().refusal(call)

    def __init__(self, call: NSACall) -> None:
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        self._compiled = _compile_nsa_compressed_varlen_kernel(
            call.batch,
            call.c_seq_len,
            call.heads,
            call.heads_kv,
            call.dim,
            call.dim_v,
            call.chunk_num,
            call.block_size,
            call.scale,
            self.dtype_str,
        )

    def forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        v_cmp: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = (q, k_cmp, v_cmp, offsets, chunk_offsets, token_indices)
        for tensor in inputs:
            if tensor.device.type != "npu" or tensor.device != q.device:
                raise ValueError("NSA compressed inputs must be on one NPU device")
        # The manifest accepts strided inputs. Normalize reads without changing
        # the caller's tensors; vectorized payload loads also need aligned bases.
        normalized = tuple(tensor.contiguous() for tensor in inputs)
        normalized = tuple(
            vector_aligned(tensor) if i < 3 else tensor for i, tensor in enumerate(normalized)
        )
        return self._compiled(*normalized)


register_kernel_type("NSACompressedVarlenFwdOp", "ascend", AscendNSACompressedFwdVarlenKernel)
