"""One Ascend NSA forward template for the four declared manifest workloads."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.layout import make_ascend_compact_nz_layout

from tileops.backend import register_kernel_type
from tileops.kernels.ascend.attention.nsa_common import (
    VECTOR_LANES,
    calibrated_cube_blocks,
    check_inputs,
    normalize_cast_rows,
)
from tileops.kernels.attention.call_spec import NSACall, NSAFwdInterface
from tileops.kernels.kernel_base import Kernel

__all__ = ["AscendNSAFwdVarlenKernel"]

# The four manifest call configurations this implementation serves, keyed by the
# TileLang dtype string the builder takes. ``refusal`` reads the same table.
_CONFIGURATIONS = (
    (4, 8192, 16, 1, 64, 1, 32, True, 0.1, "float16"),
    (2, 8192, 16, 1, 64, 4, 32, True, 0.1, "float16"),
    (2, 8190, 16, 1, 64, 4, 32, False, 0.1, "float16"),
    (2, 1024, 32, 2, 128, 8, 32, True, 0.08838834764831845, "bfloat16"),
)


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
    num_blocks: int,
):
    """Consume each selected-slot prefix once, initializing at its first valid block."""
    from contextlib import nullcontext

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
    ) in _CONFIGURATIONS, "unified NSA supports only the four manifest call configurations"
    assert num_blocks > 0, "num_blocks must be positive"
    head_tile = 16
    vector_rows = 8
    key_tile = 32
    softmax_tile = VECTOR_LANES
    kv_versions = 2 if selected_blocks > 1 else 1
    tasks = tokens * heads_kv
    grid = min(num_blocks, tasks)
    steps = (tasks + grid - 1) // grid
    normalize = normalize_cast_rows(vector_rows, dim, dtype)

    @T.macro
    def softmax_simd(scores, maximum, alpha, block_sum, probabilities, valid, first):
        with T.SimdVF():
            full = T.simd.pset(32, "PAT_ALL")
            one = T.simd.pset(32, "PAT_VL1")
            lanes = T.simd.vci(0, "int32")
            valid_mask = T.simd.vcmps(lanes, T.cast(valid, "int32"), full, op="lt")
            negative_infinity = T.simd.vdup(-T.infinity("float32"), "float32", full)
            for row in T.serial(vector_rows):
                score = T.simd.vld(scores[row, 0])
                scaled = T.simd.vmuls(score, scale, full)
                masked = T.simd.vsel(scaled, negative_infinity, valid_mask)
                block_max = T.simd.vdupv(T.simd.vcmax(masked, full), full)
                row_max = T.simd.alloc_var("float32")
                row_max = block_max
                if selected_blocks > 1:
                    if not first:
                        previous_max = T.simd.vld(maximum[row], dist="BRC_B32")
                        row_max = T.simd.vmax(previous_max, block_max, full)
                        factor = T.simd.vexpdif(previous_max, row_max, full)
                        T.simd.vsts(alpha[row], factor, one, dist="ONEPT_B32", extent=1)
                    T.simd.vsts(maximum[row], row_max, one, dist="ONEPT_B32", extent=1)
                probability = T.simd.vexpdif(masked, row_max, full)
                total = T.simd.vcadd(probability, full)
                T.simd.vsts(block_sum[row], total, one, dist="ONEPT_B32", extent=1)
                converted = T.simd.vcvt(
                    probability, dtype, full, round="ROUND_R", sat=False, part=0
                )
                store_mask = T.simd.pset(32, "PAT_VL16")
                T.simd.vsts(
                    probabilities[row, 0],
                    converted,
                    store_mask,
                    dist="PK_B32",
                    extent=16,
                )
                second_indices = T.simd.vci(16, "int32")
                second_indices = T.simd.vand(
                    T.reinterpret(second_indices, "uint32x64"),
                    T.simd.vdup(T.uint32(63), "uint32", full),
                    full,
                )
                second = T.simd.vselr(probability, second_indices)
                converted_second = T.simd.vcvt(
                    second, dtype, full, round="ROUND_R", sat=False, part=0
                )
                T.simd.vsts(
                    probabilities[row, 16],
                    converted_second,
                    store_mask,
                    dist="PK_B32",
                    extent=16,
                )

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
            v_l1 = T.alloc_l1((key_tile, dim), dtype)
            p_l1 = T.alloc_l1((head_tile, key_tile), dtype)
            q_l0 = T.alloc_l0a((head_tile, dim), dtype)
            k_l0 = T.alloc_l0b((key_tile, dim), dtype)
            p_l0 = T.alloc_l0a((head_tile, key_tile), dtype)
            v_l0 = T.alloc_l0b((key_tile, dim), dtype)
            scores_l0c = T.alloc_l0c((head_tile, key_tile), "float32")
            value_l0c = T.alloc_l0c((head_tile, dim), "float32")
            keys = T.alloc_shared((key_tile // 2 + 1, dim), dtype)
            values = T.alloc_shared((key_tile // 2 + 1, dim), dtype)
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
                            T.min(count, selected_blocks),
                            num_stages=2,
                            annotations={"enable_offset": True},
                        )
                        if selected_blocks > 1
                        else T.serial(T.min(count, selected_blocks))
                    ):
                        if slot < count:
                            block = block_indices[token, kv_head, slot]
                            start = T.cast(block, "int64") * block_size
                            if block >= 0 and start < limit:
                                valid = T.min(key_tile, limit - start)
                                # Mask inactive K rows and zero inactive V rows before
                                # QK/PV. Physical K loads may precede the semantic mask.
                                with T.Stage(0) if selected_blocks > 1 else nullcontext():
                                    if valid == key_tile:
                                        load_start = begin + T.cast(start, "int32")
                                        # DMA complete rows directly; use the guarded
                                        # vector path only for causal/sequence tails.
                                        T.copy(
                                            k[load_start : load_start + key_tile, kv_head, :],
                                            k_l1,
                                        )
                                        T.copy(
                                            v[load_start : load_start + key_tile, kv_head, :],
                                            v_l1,
                                        )
                                    else:
                                        with T.SimtVF(threads=128):
                                            for s, d in T.Parallel(key_tile // 2, dim):
                                                key_position = s + sid * (key_tile // 2)
                                                keys[s, d] = T.if_then_else(
                                                    key_position < valid,
                                                    k[begin + start + key_position, kv_head, d],
                                                    T.cast(0, dtype),
                                                )
                                            # Match the source's contiguous dimension. The
                                            # L1-to-L0B copy converts this row-major V tile
                                            # to the layout consumed by the PV GEMM.
                                            for s, d in T.Parallel(key_tile // 2, dim):
                                                value_position = s + sid * (key_tile // 2)
                                                values[s, d] = T.if_then_else(
                                                    value_position < valid,
                                                    v[begin + start + value_position, kv_head, d],
                                                    T.cast(0, dtype),
                                                )
                                        T.dual_copy(keys[: key_tile // 2, :], k_l1)
                                        T.dual_copy(values[: key_tile // 2, :], v_l1)
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
                                    if selected_blocks > 1:
                                        if seen_epoch == 0:
                                            softmax_simd(
                                                scores,
                                                maximum,
                                                alpha,
                                                block_sum,
                                                probabilities,
                                                valid,
                                                True,
                                            )
                                        else:
                                            softmax_simd(
                                                scores,
                                                maximum,
                                                alpha,
                                                block_sum,
                                                probabilities,
                                                valid,
                                                False,
                                            )
                                    else:
                                        softmax_simd(
                                            scores,
                                            block_maximum,
                                            block_maximum,
                                            block_sum,
                                            probabilities,
                                            valid,
                                            True,
                                        )
                                    T.dual_copy(probabilities[:vector_rows, :], p_l1)
                                    T.copy(p_l1, p_l0)
                                    T.gemm(
                                        p_l0,
                                        v_l0,
                                        value_l0c,
                                        transpose_B=False,
                                        clear_accum=True,
                                    )
                                    T.dual_copy(value_l0c, value_tile)
                                    if selected_blocks > 1:
                                        if seen_epoch == 0:
                                            with T.SimdVF():
                                                full = T.simd.pset(32, "PAT_ALL")
                                                one = T.simd.pset(32, "PAT_VL1")
                                                for h in T.serial(vector_rows):
                                                    block_total = T.simd.vld(
                                                        block_sum[h], dist="BRC_B32"
                                                    )
                                                    T.simd.vsts(
                                                        denominator[h],
                                                        block_total,
                                                        one,
                                                        dist="ONEPT_B32",
                                                        extent=1,
                                                    )
                                                    for part in T.serial(dim // VECTOR_LANES):
                                                        value = T.simd.vld(
                                                            value_tile[h, part * VECTOR_LANES]
                                                        )
                                                        T.simd.vsts(
                                                            numerator[h, part * VECTOR_LANES],
                                                            value,
                                                            full,
                                                        )
                                        else:
                                            with T.SimdVF():
                                                full = T.simd.pset(32, "PAT_ALL")
                                                one = T.simd.pset(32, "PAT_VL1")
                                                for h in T.serial(vector_rows):
                                                    rescale = T.simd.vld(alpha[h], dist="BRC_B32")
                                                    block_total = T.simd.vld(
                                                        block_sum[h], dist="BRC_B32"
                                                    )
                                                    total = T.simd.alloc_var("float32")
                                                    total = T.simd.vld(
                                                        denominator[h], dist="BRC_B32"
                                                    )
                                                    T.simd.vmadd(total, rescale, block_total, full)
                                                    T.simd.vsts(
                                                        denominator[h],
                                                        total,
                                                        one,
                                                        dist="ONEPT_B32",
                                                        extent=1,
                                                    )
                                                    for part in T.serial(dim // VECTOR_LANES):
                                                        value = T.simd.vld(
                                                            value_tile[h, part * VECTOR_LANES]
                                                        )
                                                        accumulated = T.simd.alloc_var("float32")
                                                        accumulated = T.simd.vld(
                                                            numerator[h, part * VECTOR_LANES]
                                                        )
                                                        T.simd.vmadd(
                                                            accumulated, rescale, value, full
                                                        )
                                                        T.simd.vsts(
                                                            numerator[h, part * VECTOR_LANES],
                                                            accumulated,
                                                            full,
                                                        )
                                    seen_epoch = 1

                    if seen_epoch == 0:
                        with T.SimtVF(threads=128):
                            for h, d in T.Parallel(vector_rows, dim):
                                result[h, d] = T.cast(0, dtype)
                    elif selected_blocks == 1:
                        normalize(value_tile, block_sum, result)
                    else:
                        normalize(numerator, denominator, result)
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
    num_blocks: int,
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
            Kernel.dtype_to_str(call.dtype),
        )
        if configuration not in _CONFIGURATIONS:
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
            calibrated_cube_blocks(),
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
        expected = (
            ("q", q, (self.call.c_seq_len, self.call.heads, self.call.dim), self.dtype),
            ("k", k, (self.call.c_seq_len, self.call.heads_kv, self.call.dim), self.dtype),
            ("v", v, (self.call.c_seq_len, self.call.heads_kv, self.call.dim), self.dtype),
            (
                "block_indices",
                block_indices,
                (self.call.c_seq_len, self.call.heads_kv, self.call.selected_blocks),
                torch.int32,
            ),
            ("block_counts", block_counts, (self.call.c_seq_len, self.call.heads_kv), torch.int32),
            ("offsets", offsets, (self.call.batch + 1,), torch.int32),
            ("token_indices", token_indices, (self.call.c_seq_len, 2), torch.int32),
        )
        check_inputs(self, expected)
        return self._compiled(*(tensor.contiguous() for _, tensor, _shape, _dtype in expected))


register_kernel_type("NSAVarlenFwdOp", "ascend", AscendNSAFwdVarlenKernel)
