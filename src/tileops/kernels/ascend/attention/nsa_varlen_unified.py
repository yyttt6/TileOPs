"""One Ascend NSA forward template for the four declared manifest workloads."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.layout import make_ascend_compact_nz_layout

from tileops.backend import register_kernel_type
from tileops.kernels.attention.call_spec import NSACall, NSAFwdInterface
from tileops.kernels.kernel_base import Kernel

__all__ = ["AscendNSAFwdVarlenKernel"]


def _make_nsa_varlen_kernel(
    batch: int,
    tokens: int,
    heads: int,
    heads_kv: int,
    dim: int,
    selected_blocks: int,
    block_size: int,
    is_causal: bool,
    scale: float,
    dtype: str,
    num_blocks: int = 36,
    projection_first_task: int = 0,
    projection_steps: int = 0,
    projection_stride: int = 0,
):
    """Consume each selected-slot prefix once, initializing at its first valid block.

    The three optional projection arguments are for diagnostic task streams only.
    Their all-zero production default covers every task and retains the full ABI.
    """
    from contextlib import nullcontext

    configurations = (
        (4, 8192, 16, 1, 64, 1, 32, True, 0.1, "float16"),
        (2, 8192, 16, 1, 64, 4, 32, True, 0.1, "float16"),
        (2, 8190, 16, 1, 64, 4, 32, False, 0.1, "float16"),
        (2, 1024, 32, 2, 128, 8, 32, True, 0.08838834764831845, "bfloat16"),
    )
    assert (
        batch,
        tokens,
        heads,
        heads_kv,
        dim,
        selected_blocks,
        block_size,
        is_causal,
        scale,
        dtype,
    ) in configurations, "unified NSA supports only the four manifest call configurations"
    assert num_blocks > 0, "num_blocks must be positive"
    head_tile = 16
    vector_rows = 8
    key_tile = 32
    softmax_tile = 64
    kv_versions = 2 if selected_blocks > 1 else 1
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
    def reduce_max_rows(src, dst):
        # Each 64-score row reduces to one float at its own scalar address.
        with T.SimdVF():
            full = T.simd.pset(32, "PAT_ALL")
            one = T.simd.pset(32, "PAT_VL1")
            for row in T.serial(vector_rows):
                identity = T.simd.vdup(T.float32(-3.402823e38), "float32", full)
                reduced = T.simd.vcmax(T.simd.vld(src[row, 0]), full)
                result = T.simd.vmax(identity, reduced, full)
                T.simd.vsts(dst[row], result, one, dist="ONEPT_B32", extent=1)

    @T.macro
    def reduce_sum_rows(src, dst):
        with T.SimdVF():
            full = T.simd.pset(32, "PAT_ALL")
            one = T.simd.pset(32, "PAT_VL1")
            for row in T.serial(vector_rows):
                identity = T.simd.vdup(T.float32(0.0), "float32", full)
                reduced = T.simd.vcadd(T.simd.vld(src[row, 0]), full)
                result = T.simd.vadd(identity, reduced, full)
                T.simd.vsts(dst[row], result, one, dist="ONEPT_B32", extent=1)

    @T.prim_func
    def main(
        q: T.Tensor((tokens, heads, dim), dtype),
        k: T.Tensor((tokens, heads_kv, dim), dtype),
        v: T.Tensor((tokens, heads_kv, dim), dtype),
        block_indices: T.Tensor((tokens, heads_kv, selected_blocks), "int32"),
        block_counts: T.Tensor((tokens, heads_kv), "int32"),
        offsets: T.Tensor((batch + 1,), "int32"),
        token_indices: T.Tensor((tokens, 2), "int32"),
        output: T.Tensor((tokens, heads, dim), dtype),
    ):
        with T.MixedKernel(grid, sids=2) as (bx, sid):
            q_l1 = T.alloc_l1((head_tile, dim), dtype)
            k_l1 = T.alloc_l1((key_tile, dim), dtype)
            v_l1 = T.alloc_l1((dim, key_tile), dtype)
            p_l1 = T.alloc_l1((head_tile, key_tile), dtype)
            q_l0 = T.alloc_l0a((head_tile, dim), dtype)
            k_l0 = T.alloc_l0b((key_tile, dim), dtype)
            p_l0 = T.alloc_l0a((head_tile, key_tile), dtype)
            v_l0 = T.alloc_l0b((dim, key_tile), dtype)
            scores_l0c = T.alloc_l0c((head_tile, key_tile), "float32")
            value_l0c = T.alloc_l0c((head_tile, dim), "float32")
            keys = T.alloc_shared((key_tile // 2 + 1, dim), dtype)
            values = T.alloc_shared((dim // 2 + 1, key_tile), dtype)
            scores = T.alloc_shared((vector_rows, softmax_tile), "float32")
            probabilities = T.alloc_shared((vector_rows + 1, key_tile), dtype)
            value_tile = T.alloc_shared((vector_rows, dim), "float32")
            result = T.alloc_shared((vector_rows, dim), dtype)
            block_maximum = T.alloc_shared((vector_rows,), "float32")
            block_sum = T.alloc_shared((vector_rows,), "float32")
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
                    k_l1: kv_versions,
                    v_l1: kv_versions,
                    p_l1: 1,
                    q_l0: 1,
                    k_l0: 1,
                    p_l0: 1,
                    v_l0: 1,
                    scores_l0c: 1,
                    value_l0c: 1,
                    keys: kv_versions,
                    values: kv_versions,
                    scores: 1,
                    probabilities: 1,
                    value_tile: 1,
                    result: 1,
                    block_maximum: 1,
                    block_sum: 1,
                }
            )
            if selected_blocks > 1:
                maximum = T.alloc_shared((vector_rows,), "float32")
                denominator = T.alloc_shared((vector_rows,), "float32")
                numerator = T.alloc_shared((vector_rows, dim), "float32")
                alpha = T.alloc_shared((vector_rows,), "float32")
                T.annotate_buffer_versions({maximum: 1, denominator: 1, numerator: 1, alpha: 1})

            # Each core maintains its own scalar from the same immutable metadata.
            # Only SPU control writes it; no VF result is shared through a scalar.
            seen_epoch = T.alloc_var("int32", init=0)
            for step in T.serial(steps):
                if projected:
                    task = projection_first_task + step * projection_stride + bx
                else:
                    task = step * grid + bx
                if task < tasks:
                    token = task // heads_kv
                    kv_head = task % heads_kv
                    head_start = kv_head * head_tile
                    request = token_indices[token, 0]
                    position = token_indices[token, 1]
                    begin = offsets[request]
                    limit = position + 1 if is_causal else offsets[request + 1] - begin
                    count = block_counts[token, kv_head]
                    seen_epoch = 0
                    T.copy(q[token, head_start : head_start + head_tile, :], q_l1)
                    T.copy(q_l1, q_l0)

                    for slot in (
                        T.Pipelined(
                            selected_blocks,
                            num_stages=2,
                            annotations={"enable_offset": True},
                        )
                        if selected_blocks > 1
                        else T.serial(selected_blocks)
                    ):
                        if slot < count:
                            block = block_indices[token, kv_head, slot]
                            start = T.cast(block, "int64") * block_size
                            if block >= 0 and start < limit:
                                valid = T.min(key_tile, limit - start)
                                # Mask inactive K rows and zero inactive V rows before
                                # QK/PV. Physical K loads may precede the semantic mask.
                                with T.Stage(0) if selected_blocks > 1 else nullcontext():
                                    with T.SimtVF(threads=128):
                                        for s, d in T.Parallel(key_tile // 2, dim):
                                            key_position = s + sid * (key_tile // 2)
                                            keys[s, d] = T.if_then_else(
                                                key_position < valid,
                                                k[begin + start + key_position, kv_head, d],
                                                T.cast(0, dtype),
                                            )
                                        for d, s in T.Parallel(dim // 2, key_tile):
                                            value_dimension = d + sid * (dim // 2)
                                            values[d, s] = T.if_then_else(
                                                s < valid,
                                                v[begin + start + s, kv_head, value_dimension],
                                                T.cast(0, dtype),
                                            )
                                    T.dual_copy(keys[: key_tile // 2, :], k_l1)
                                    T.dual_copy(values[: dim // 2, :], v_l1)
                                with T.Stage(1) if selected_blocks > 1 else nullcontext():
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
                                    if selected_blocks > 1:
                                        if seen_epoch == 0:
                                            with T.SimtVF(threads=128):
                                                for h in T.Parallel(vector_rows):
                                                    maximum[h] = block_maximum[h]
                                        else:
                                            with T.SimtVF(threads=128):
                                                for h in T.Parallel(vector_rows):
                                                    next_maximum = T.max(
                                                        maximum[h], block_maximum[h]
                                                    )
                                                    alpha[h] = T.exp(maximum[h] - next_maximum)
                                                    maximum[h] = next_maximum
                                    with T.SimtVF(threads=128):
                                        for h, s in T.Parallel(vector_rows, softmax_tile):
                                            if selected_blocks == 1:
                                                scores[h, s] = T.exp(
                                                    scores[h, s] - block_maximum[h]
                                                )
                                            else:
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
                                    if selected_blocks > 1:
                                        if seen_epoch == 0:
                                            with T.SimtVF(threads=128):
                                                for h in T.Parallel(vector_rows):
                                                    denominator[h] = block_sum[h]
                                                for h, d in T.Parallel(vector_rows, dim):
                                                    numerator[h, d] = value_tile[h, d]
                                        else:
                                            with T.SimtVF(threads=128):
                                                for h in T.Parallel(vector_rows):
                                                    denominator[h] = (
                                                        denominator[h] * alpha[h] + block_sum[h]
                                                    )
                                                for h, d in T.Parallel(vector_rows, dim):
                                                    numerator[h, d] = (
                                                        numerator[h, d] * alpha[h]
                                                        + value_tile[h, d]
                                                    )
                                    seen_epoch = 1

                    if seen_epoch == 0:
                        with T.SimtVF(threads=128):
                            for h, d in T.Parallel(vector_rows, dim):
                                result[h, d] = T.cast(0, dtype)
                    elif selected_blocks == 1:
                        with T.SimtVF(threads=128):
                            for h, d in T.Parallel(vector_rows, dim):
                                value_tile[h, d] = T.if_then_else(
                                    block_sum[h] > 0, value_tile[h, d] / block_sum[h], 0
                                )
                        with T.SimtVF(threads=128):
                            T.copy(value_tile, result)
                    else:
                        with T.SimtVF(threads=128):
                            for h, d in T.Parallel(vector_rows, dim):
                                numerator[h, d] = T.if_then_else(
                                    denominator[h] > 0, numerator[h, d] / denominator[h], 0
                                )
                        with T.SimtVF(threads=128):
                            T.copy(numerator, result)
                    T.dual_copy(result, output[token, head_start : head_start + head_tile, :])

    return main


@functools.lru_cache(maxsize=64)
def _compile_nsa_varlen_kernel(
    batch: int,
    tokens: int,
    heads: int,
    heads_kv: int,
    dim: int,
    selected_blocks: int,
    block_size: int,
    is_causal: bool,
    scale: float,
    dtype: str,
    num_blocks: int = 36,
):
    return tilelang.compile(
        _make_nsa_varlen_kernel(
            batch,
            tokens,
            heads,
            heads_kv,
            dim,
            selected_blocks,
            block_size,
            is_causal,
            scale,
            dtype,
            num_blocks,
        ),
        target="ascend",
        out_idx=-1,
    )


class AscendNSAFwdVarlenKernel(Kernel, NSAFwdInterface):
    """One NSA implementation restricted to the four manifest call configurations."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: NSACall) -> str | None:
        configurations = (
            (4, 8192, 16, 1, 64, 1, 32, True, 0.1, torch.float16),
            (2, 8192, 16, 1, 64, 4, 32, True, 0.1, torch.float16),
            (2, 8190, 16, 1, 64, 4, 32, False, 0.1, torch.float16),
            (2, 1024, 32, 2, 128, 8, 32, True, 0.08838834764831845, torch.bfloat16),
        )
        configuration = (
            call.batch,
            call.c_seq_len,
            call.heads,
            call.heads_kv,
            call.dim,
            call.selected_blocks,
            call.block_size,
            call.is_causal,
            call.scale,
            call.dtype,
        )
        if configuration not in configurations:
            return "requires one of the four declared NSAVarlenFwdOp manifest call configurations"
        return super().refusal(call)

    def __init__(self, call: NSACall) -> None:
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        self._compiled = _compile_nsa_varlen_kernel(
            call.batch,
            call.c_seq_len,
            call.heads,
            call.heads_kv,
            call.dim,
            call.selected_blocks,
            call.block_size,
            call.is_causal,
            call.scale,
            self.dtype_str,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indices: torch.Tensor,
        block_counts: torch.Tensor,
        offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        inputs = (q, k, v, block_indices, block_counts, offsets, token_indices)
        for tensor in inputs:
            if tensor.device.type != "npu" or tensor.device != q.device:
                raise ValueError("NSA inputs must be on one NPU device")
            if not tensor.is_contiguous():
                raise ValueError("NSA inputs must be contiguous")
        return self._compiled(*inputs)


register_kernel_type("NSAVarlenFwdOp", "ascend", AscendNSAFwdVarlenKernel)
