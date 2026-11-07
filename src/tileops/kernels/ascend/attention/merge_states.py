"""SIMD attention-state merging with dynamic token and head counts."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.attention.call_spec import (
    MergeAttentionStatesCall,
    MergeAttentionStatesFwdInterface,
)
from tileops.kernels.kernel_base import Entry, Kernel, vector_aligned

__all__ = ["AscendMergeAttentionStatesKernel", "merge_states_program"]


def merge_states_program(
    dim: int, dtype: str, row_tile: int = 64, col_tile: int = 128, stages: int = 2
):
    """Return the dynamic-row SIMD program, without compilation or device access.

    A 64-row tile computes one vector of partition weights. Its 128-column
    value tiles reuse those weights, keeping the head width out of the UB
    capacity calculation. The first value tile shares its VF with statistics.
    """
    # One FP32 vector covers all statistics. Two 64-lane vectors cover the
    # common 128-wide head; two versions of three value tiles use 96 KiB UB.
    if row_tile not in (32, 64, 128) or col_tile <= 0 or col_tile % 64 or stages not in (1, 2):
        raise ValueError(
            "merge tiling requires 32/64/128 rows, whole SIMD column vectors and 1/2 stages"
        )
    # Statistics always occupy whole registers, including the 32-row variant.
    stats_rows = (row_tile + 63) // 64 * 64
    max_blocks = 64
    rows = T.dynamic("rows")
    jobs = T.ceildiv(rows, row_tile)
    blocks = T.min(max_blocks, jobs)

    @T.macro
    def statistics(sa, sb, wa, wb, lse, active_rows):
        full = T.simd.pset(32)
        zero = T.simd.vdup(T.float32(0), "float32", full)
        one = T.simd.vdup(T.float32(1), "float32", full)
        negative_infinity = T.simd.vdup(-T.infinity("float32"), "float32", full)
        # DMA only initializes active rows; sanitize the padded UB lanes before
        # using them in any arithmetic, including the unmasked maximum.
        for chunk in T.serial(stats_rows // 64):
            valid = T.simd.update_mask(T.max(0, T.min(64, active_rows - chunk * 64)), 32)
            a = T.simd.vsel(T.simd.vld(sa[chunk * 64]), negative_infinity, valid)
            b = T.simd.vsel(T.simd.vld(sb[chunk * 64]), negative_infinity, valid)
            empty_a = T.simd.vcmps(a, -T.infinity("float32"), full, "eq")
            empty_b = T.simd.vcmps(b, -T.infinity("float32"), full, "eq")
            both_empty = T.simd.pand(empty_a, empty_b, full)
            maximum = T.simd.vmax(a, b, full)
            # Ascend maximum may choose the non-NaN operand. torch.maximum, the
            # shared reference, propagates either NaN.
            maximum = T.simd.vsel(a, maximum, T.simd.vcmp(a, a, full, "ne"))
            maximum = T.simd.vsel(b, maximum, T.simd.vcmp(b, b, full, "ne"))
            safe_max = T.simd.vsel(zero, maximum, both_empty)
            ea = T.simd.vexp(T.simd.vsub(a, safe_max, full), full, precision="ftz_false")
            eb = T.simd.vexp(T.simd.vsub(b, safe_max, full), full, precision="ftz_false")
            denominator = T.simd.vadd(ea, eb, full)
            safe_denominator = T.simd.vmax(denominator, one, full)
            safe_denominator = T.simd.vsel(
                denominator,
                safe_denominator,
                T.simd.vcmp(denominator, denominator, full, "ne"),
            )
            if dim > 0:
                # Divide normal numbers, then multiply exponential weights. A
                # subnormal exponential sent directly to FTZ division loses mass.
                inverse = T.simd.vdiv(one, safe_denominator, full, precision="exact")
                T.simd.vsts(wa[chunk * 64], T.simd.vmul(ea, inverse, full), full)
                T.simd.vsts(wb[chunk * 64], T.simd.vmul(eb, inverse, full), full)
            merged_lse = T.simd.vadd(T.simd.vln(safe_denominator, full), safe_max, full)
            T.simd.vsts(
                lse[chunk * 64], T.simd.vsel(negative_infinity, merged_lse, both_empty), full
            )

    @T.macro
    def values(va, vb, output, sa, sb, wa, wb, active_rows, active_cols):
        full = T.simd.pset(32)
        zero = T.simd.vdup(T.float32(0), "float32", full)
        for r in T.serial(active_rows):
            weight_a = T.simd.vld(wa[r], "BRC_B32")
            weight_b = T.simd.vld(wb[r], "BRC_B32")
            empty_a = T.simd.vcmps(T.simd.vld(sa[r], "BRC_B32"), -T.infinity("float32"), full, "eq")
            empty_b = T.simd.vcmps(T.simd.vld(sb[r], "BRC_B32"), -T.infinity("float32"), full, "eq")
            for c in T.serial(T.ceildiv(active_cols, 64)):
                valid = T.simd.update_mask(T.min(64, active_cols - c * 64), 32)
                a = T.simd.vcvt(T.simd.vld(va[r, c * 64], "UNPK_B16"), "float32", valid, part=0)
                b = T.simd.vcvt(T.simd.vld(vb[r, c * 64], "UNPK_B16"), "float32", valid, part=0)
                # Only LSE=-inf denotes an empty state. A finite state's NaN
                # must survive even when its computed weight underflows to zero.
                a = T.simd.vsel(zero, a, empty_a)
                b = T.simd.vsel(zero, b, empty_b)
                merged = T.simd.vadd(
                    T.simd.vmul(a, weight_a, valid),
                    T.simd.vmul(b, weight_b, valid),
                    valid,
                )
                narrowed = T.simd.vcvt(merged, dtype, valid, round="ROUND_R", sat=False)
                T.simd.vsts(output[r, c * 64], narrowed, valid, "PK_B32")

    if dim == 0:

        @T.prim_func
        def lse_only(
            Sa: T.Tensor((rows,), "float32"),
            Sb: T.Tensor((rows,), "float32"),
            S: T.Tensor((rows,), "float32"),
        ):
            with T.Kernel(blocks) as core:
                sa = T.alloc_shared((stats_rows,), "float32")
                sb = T.alloc_shared((stats_rows,), "float32")
                lse = T.alloc_shared((stats_rows,), "float32")
                T.annotate_buffer_versions({sa: stages, sb: stages, lse: stages})
                for iteration in T.Pipelined(T.ceildiv(jobs, blocks), num_stages=stages):
                    job = iteration * blocks + core
                    if job < jobs:
                        begin = job * row_tile
                        count = T.min(row_tile, rows - begin)
                        T.copy(Sa[begin : begin + count], sa[:count])
                        T.copy(Sb[begin : begin + count], sb[:count])
                        with T.SimdVF():
                            statistics(sa, sb, lse, lse, lse, count)
                        T.copy(lse[:count], S[begin : begin + count])

        return lse_only.with_attr("tilelang_callee_allocated_outputs", 0)

    @T.prim_func
    def main(
        Va: T.Tensor((rows, dim), dtype),
        Sa: T.Tensor((rows,), "float32"),
        Vb: T.Tensor((rows, dim), dtype),
        Sb: T.Tensor((rows,), "float32"),
        V: T.Tensor((rows, dim), dtype),
        S: T.Tensor((rows,), "float32"),
    ):
        with T.Kernel(blocks) as core:
            va = T.alloc_shared((row_tile, col_tile), dtype)
            vb = T.alloc_shared((row_tile, col_tile), dtype)
            output = T.alloc_shared((row_tile, col_tile), dtype)
            sa = T.alloc_shared((stats_rows,), "float32")
            sb = T.alloc_shared((stats_rows,), "float32")
            wa = T.alloc_shared((stats_rows,), "float32")
            wb = T.alloc_shared((stats_rows,), "float32")
            lse = T.alloc_shared((stats_rows,), "float32")
            T.annotate_buffer_versions(
                {
                    va: stages,
                    vb: stages,
                    output: stages,
                    sa: stages,
                    sb: stages,
                    wa: stages,
                    wb: stages,
                    lse: stages,
                }
            )
            for iteration in T.Pipelined(T.ceildiv(jobs, blocks), num_stages=stages):
                job = iteration * blocks + core
                if job < jobs:
                    begin = job * row_tile
                    count = T.min(row_tile, rows - begin)
                    first_cols = min(dim, col_tile)
                    T.copy(Sa[begin : begin + count], sa[:count])
                    T.copy(Sb[begin : begin + count], sb[:count])
                    T.copy(Va[begin : begin + count, :first_cols], va[:count, :first_cols])
                    T.copy(Vb[begin : begin + count, :first_cols], vb[:count, :first_cols])
                    with T.SimdVF():
                        statistics(sa, sb, wa, wb, lse, count)
                        # The next instructions broadcast the just-written UB
                        # weights; the scheduler cannot insert a barrier inside VF.
                        T.simd.mem_bar("VST_VLD")
                        values(va, vb, output, sa, sb, wa, wb, count, first_cols)
                    T.copy(output[:count, :first_cols], V[begin : begin + count, :first_cols])
                    T.copy(lse[:count], S[begin : begin + count])
                    for column in T.serial(1, T.ceildiv(dim, col_tile)):
                        col = column * col_tile
                        width = T.min(col_tile, dim - col)
                        T.copy(Va[begin : begin + count, col : col + width], va[:count, :width])
                        T.copy(Vb[begin : begin + count, col : col + width], vb[:count, :width])
                        with T.SimdVF():
                            values(va, vb, output, sa, sb, wa, wb, count, width)
                        T.copy(output[:count, :width], V[begin : begin + count, col : col + width])

    return main.with_attr("tilelang_callee_allocated_outputs", 0)


@functools.lru_cache(maxsize=64)
def _merge_states_kernel(dim: int, dtype: str):
    @tilelang.jit(
        out_idx=[],
        target="ascend",
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False},
    )
    def build():
        return merge_states_program(dim, dtype)

    return build


class AscendMergeAttentionStatesKernel(Kernel, MergeAttentionStatesFwdInterface):
    """One fused SIMD pass with FP32 weights and a dynamic flattened row count."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: MergeAttentionStatesCall) -> Entry:
        index = call.device.index if call.device is not None else None
        return (call.dim, call.dtype, index), lambda: cls(call.dim, call.dtype, device_index=index)

    def __init__(self, dim: int, dtype: torch.dtype, device_index=None):
        super().__init__(device_index=device_index)
        self.dim = dim
        self.dtype = dtype
        self.config = {"row_tile": 64, "col_tile": 128, "stages": 2, "max_blocks": 64}
        self.kernel = _merge_states_kernel(dim, self.dtype_str)

    def forward(self, v_a, s_a, v_b, s_b):
        self._require_devices(v_a=v_a, s_a=s_a, v_b=v_b, s_b=s_b)
        tokens, heads, dim = v_a.shape
        rows = tokens * heads
        v = torch.empty((tokens, heads, dim), dtype=v_a.dtype, device=v_a.device)
        s = torch.empty((tokens, heads), dtype=s_a.dtype, device=s_a.device)
        if rows == 0:
            return v, s
        a = vector_aligned(s_a.contiguous()).view(rows)
        b = vector_aligned(s_b.contiguous()).view(rows)
        if dim == 0:
            self.kernel()(a, b, s.view(rows))
        else:
            self.kernel()(
                vector_aligned(v_a.contiguous()).view(rows, dim),
                a,
                vector_aligned(v_b.contiguous()).view(rows, dim),
                b,
                v.view(rows, dim),
                s.view(rows),
            )
        return v, s


register_kernel_type("MergeAttentionStatesFwdOp", "ascend", AscendMergeAttentionStatesKernel)
