"""Staged SIMD AIV backward for the post-projection Engram GateConv fusion.

Each token owns its activation gradients and FP32 weight-gradient partials.
A separate ordered reduction combines the partials without floating atomics.
The saved tensors are the actual forward outputs, including rounded ``vhat``.
"""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.sequence_modeling.engram.call_spec import (
    EngramGateConvBwdInterface,
    EngramGateConvCall,
)

__all__ = ["AscendEngramGateConvBwdKernel"]

# The value-gradient stage holds 42 bytes per SIMD-padded channel in the 216 KiB UB.
_MAX_HIDDEN_DIM = 4096


@T.macro
def _zero_simd(buffer, words):
    """Clear complete words with SIMD, including buffers shorter than one VReg."""
    bits = T.view(buffer, (words,), dtype="uint32")
    full = T.simd.pset(32)
    zero = T.simd.vdup(T.uint32(0), "uint32", full)
    for chunk in T.serial(T.ceildiv(words, 64)):
        count = T.min(64, words - chunk * 64)
        valid = T.simd.update_mask(count, 32)
        T.simd.vsts(bits[chunk * 64], zero, valid, extent=count)


@functools.lru_cache(maxsize=32)
def _build_silu_grad(rows: int, seq_len: int, d: int, dp: int, dtype: str):
    """Return ``dsilu[rows,dp]`` and ``dconv_partial[4,rows,dp]``."""
    vd = (dp + 63) // 64 * 64

    @T.prim_func
    def main(
        DY: T.Tensor((rows, d), dtype),
        VH: T.Tensor((rows, d), dtype),
        WV: T.Tensor((d,), dtype),
        CW: T.Tensor((4, d), dtype),
        RV: T.Tensor((rows,), "float32"),
        DS: T.Tensor((rows, dp), "float32"),
        PC: T.Tensor((4, rows, dp), "float32"),
    ):
        with T.Kernel(rows) as row:
            vh = T.alloc_shared((4, vd), dtype)
            wv = T.alloc_shared((vd,), dtype)
            cw = T.alloc_shared((4, vd), dtype)
            rv = T.alloc_shared((4, 8), "float32")
            dy = T.alloc_shared((vd,), dtype)
            ds = T.alloc_shared((vd,), "float32")
            pc = T.alloc_shared((4, vd), "float32")
            with T.SimdVF():
                _zero_simd(vh, 2 * vd)
                _zero_simd(rv, 32)
                if d < vd:
                    _zero_simd(wv, vd // 2)
                    _zero_simd(cw, 2 * vd)
                    _zero_simd(dy, vd // 2)
            T.copy(WV, wv[:d])
            T.copy(CW, cw[:, :d])
            T.copy(DY[row, :], dy[:d])
            for p in T.unroll(4):
                src = row - 3 + p
                if row % seq_len >= 3 - p:
                    T.copy(VH[src, :], vh[p, :d])
                    T.copy(RV[src : src + 1], rv[p, :1])
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.vdup(1.0, "float32", full)
                r = T.simd.alloc_local((4,), "float32")
                for p in T.unroll(4):
                    r[p] = T.simd.vld(rv[p, 0], "BRC_B32")
                for chunk in T.serial(vd // 64):
                    col = chunk * 64
                    valid = T.simd.update_mask(T.min(64, T.max(0, d - col)), 32)
                    weight = T.simd.vcvt(T.simd.vld(wv[col], "UNPK_B16"), "float32")
                    values = T.simd.alloc_local((4,), "float32")
                    conv = T.simd.alloc_var("float32")
                    conv = T.simd.vdup(0.0, "float32", full)
                    for p in T.unroll(4):
                        values[p] = T.simd.vcvt(T.simd.vld(vh[p, col], "UNPK_B16"), "float32")
                        coeff = T.simd.vcvt(T.simd.vld(cw[p, col], "UNPK_B16"), "float32")
                        normalized = T.simd.vmul(T.simd.vmul(values[p], r[p], valid), weight, valid)
                        conv = T.simd.vadd(conv, T.simd.vmul(normalized, coeff, valid), full)
                    sig = T.simd.vdiv(
                        one,
                        T.simd.vadds(
                            T.simd.vexp(T.simd.vmuls(conv, -1.0, valid), valid), 1.0, valid
                        ),
                        valid,
                    )
                    deriv = T.simd.vadd(
                        sig,
                        T.simd.vmul(
                            T.simd.vmul(conv, sig, valid), T.simd.vsub(one, sig, valid), valid
                        ),
                        valid,
                    )
                    upstream = T.simd.vcvt(T.simd.vld(dy[col], "UNPK_B16"), "float32")
                    gradient = T.simd.vmul(upstream, deriv, valid)
                    T.simd.vsts(ds[col], gradient, full)
                    for p in T.unroll(4):
                        partial = T.simd.vmul(
                            T.simd.vmul(T.simd.vmul(gradient, values[p], valid), r[p], valid),
                            weight,
                            valid,
                        )
                        T.simd.vsts(pc[p, col], partial, full)
            T.copy(ds[:dp], DS[row, :])
            T.copy(pc[:, :dp], PC[:, row, :])

    return tilelang.compile(main, target="ascend", out_idx=[5, 6])


@functools.lru_cache(maxsize=32)
def _build_value_gate_grad(rows: int, seq_len: int, d: int, dp: int, dtype: str):
    """Keep the value-to-gate gradient handoff in UB, with ordered arithmetic."""
    vd = (dp + 63) // 64 * 64

    @T.prim_func
    def main(
        DY: T.Tensor((rows, d), dtype),
        H: T.Tensor((rows, d), dtype),
        K: T.Tensor((rows, d), dtype),
        V: T.Tensor((rows, d), dtype),
        VH: T.Tensor((rows, d), dtype),
        WH: T.Tensor((d,), dtype),
        WV: T.Tensor((d,), dtype),
        CW: T.Tensor((4, d), dtype),
        A: T.Tensor((rows,), "float32"),
        RH: T.Tensor((rows,), "float32"),
        RK: T.Tensor((rows,), "float32"),
        RV: T.Tensor((rows,), "float32"),
        DS: T.Tensor((rows, dp), "float32"),
        DH: T.Tensor((rows, dp), dtype),
        DK: T.Tensor((rows, dp), dtype),
        DV: T.Tensor((rows, dp), dtype),
        PH: T.Tensor((rows, dp), "float32"),
        PV: T.Tensor((rows, dp), "float32"),
    ):
        with T.Kernel(rows) as row:
            vh = T.alloc_shared((vd,), dtype)
            wv = T.alloc_shared((vd,), dtype)
            cw = T.alloc_shared((4, vd), dtype)
            rv = T.alloc_shared((8,), "float32")
            dy = T.alloc_shared((vd,), dtype)
            ds = T.alloc_shared((4, vd), "float32")
            gn = T.alloc_shared((vd,), "float32")
            gz = T.alloc_shared((vd,), "float32")
            pv = T.alloc_shared((vd,), "float32")
            with T.SimdVF():
                _zero_simd(ds, 4 * vd)
                if d < vd:
                    _zero_simd(vh, vd // 2)
                    _zero_simd(wv, vd // 2)
                    _zero_simd(cw, 2 * vd)
                    _zero_simd(dy, vd // 2)
            T.copy(VH[row, :], vh[:d])
            T.copy(WV, wv[:d])
            T.copy(CW, cw[:, :d])
            T.copy(RV[row : row + 1], rv[:1])
            T.copy(DY[row, :], dy[:d])
            for p in T.unroll(4):
                dst = row + 3 - p
                if row % seq_len + 3 - p < seq_len:
                    T.copy(DS[dst, :], ds[p, :dp])
            with T.SimdVF():
                full = T.simd.pset(32)
                r = T.simd.vld(rv[0], "BRC_B32")
                r3 = T.simd.vmul(T.simd.vmul(r, r, full), r, full)
                denom = T.simd.vdup(T.float32(d), "float32", full)
                dot = T.simd.alloc_var("float32")
                dot = T.simd.vdup(0.0, "float32", full)
                for chunk in T.serial(vd // 64):
                    col = chunk * 64
                    valid = T.simd.update_mask(T.min(64, T.max(0, d - col)), 32)
                    grad_norm = T.simd.alloc_var("float32")
                    grad_norm = T.simd.vdup(0.0, "float32", full)
                    for p in T.unroll(4):
                        upstream = T.simd.vld(ds[p, col])
                        coeff = T.simd.vcvt(T.simd.vld(cw[p, col], "UNPK_B16"), "float32")
                        grad_norm = T.simd.vadd(
                            grad_norm, T.simd.vmul(upstream, coeff, valid), full
                        )
                    z = T.simd.vcvt(T.simd.vld(vh[col], "UNPK_B16"), "float32")
                    weight = T.simd.vcvt(T.simd.vld(wv[col], "UNPK_B16"), "float32")
                    dot = T.simd.vadd(
                        dot, T.simd.vmul(T.simd.vmul(z, weight, valid), grad_norm, valid), full
                    )
                    T.simd.vsts(gn[col], grad_norm, full)
                summed = T.simd.vdupv(T.simd.vcadd(dot, full), full)
                T.simd.mem_bar("VST_VLD")
                for chunk in T.serial(vd // 64):
                    col = chunk * 64
                    valid = T.simd.update_mask(T.min(64, T.max(0, d - col)), 32)
                    grad_norm = T.simd.vld(gn[col])
                    z = T.simd.vcvt(T.simd.vld(vh[col], "UNPK_B16"), "float32")
                    weight = T.simd.vcvt(T.simd.vld(wv[col], "UNPK_B16"), "float32")
                    upstream = T.simd.vcvt(T.simd.vld(dy[col], "UNPK_B16"), "float32")
                    direct = T.simd.vmul(T.simd.vmul(r, weight, valid), grad_norm, valid)
                    correction = T.simd.vdiv(
                        T.simd.vmul(T.simd.vmul(r3, z, valid), summed, valid), denom, valid
                    )
                    result = T.simd.vadd(T.simd.vsub(direct, correction, valid), upstream, valid)
                    partial = T.simd.vmul(T.simd.vmul(grad_norm, z, valid), r, valid)
                    T.simd.vsts(gz[col], result, full)
                    T.simd.vsts(pv[col], partial, full)
            T.copy(pv[:dp], PV[row, :])
            # The transpose-convolution inputs are dead. Reuse their storage for
            # gate inputs/outputs; CW becomes H/K norms and GN becomes dWH partials.
            # PV has its own storage so its DMA can finish independently.
            gate_low = T.view(ds, (8, vd), dtype=dtype)
            gate_norm = T.view(cw, (2, vd), dtype="float32")
            gate_a = T.alloc_shared((8,), "float32")
            gate_rh = T.alloc_shared((8,), "float32")
            gate_rk = T.alloc_shared((8,), "float32")
            if d < vd:
                with T.SimdVF():
                    _zero_simd(gate_low, 4 * vd)
            T.copy(H[row, :], gate_low[0, :d])
            T.copy(K[row, :], gate_low[1, :d])
            T.copy(V[row, :], gate_low[2, :d])
            T.copy(WH, gate_low[3, :d])
            T.copy(A[row : row + 1], gate_a[:1])
            T.copy(RH[row : row + 1], gate_rh[:1])
            T.copy(RK[row : row + 1], gate_rk[:1])
            with T.SimdVF():
                T.simd.mem_bar("VST_VLD")
                gate_full = T.simd.pset(32)
                gate_one = T.simd.vdup(1.0, "float32", gate_full)
                gate_alpha = T.simd.vld(gate_a[0], "BRC_B32")
                gate_r_h = T.simd.vld(gate_rh[0], "BRC_B32")
                gate_r_k = T.simd.vld(gate_rk[0], "BRC_B32")
                gate_ga = T.simd.alloc_var("float32")
                gate_ga = T.simd.vdup(0.0, "float32", gate_full)
                for gate_chunk in T.serial(vd // 64):
                    gate_col = gate_chunk * 64
                    gate_valid = T.simd.update_mask(T.min(64, T.max(0, d - gate_col)), 32)
                    gate_upstream = T.simd.vld(gz[gate_col])
                    gate_value = T.simd.vcvt(
                        T.simd.vld(gate_low[2, gate_col], "UNPK_B16"), "float32"
                    )
                    gate_ga = T.simd.vadd(
                        gate_ga, T.simd.vmul(gate_upstream, gate_value, gate_valid), gate_full
                    )
                gate_gate = T.simd.vdiv(
                    T.simd.vmul(
                        T.simd.vmul(
                            T.simd.vdupv(T.simd.vcadd(gate_ga, gate_full), gate_full),
                            gate_alpha,
                            gate_full,
                        ),
                        T.simd.vsub(gate_one, gate_alpha, gate_full),
                        gate_full,
                    ),
                    T.simd.vdup(d**0.5, "float32", gate_full),
                    gate_full,
                )
                gate_dot_h = T.simd.alloc_var("float32")
                gate_dot_k = T.simd.alloc_var("float32")
                gate_dot_h = T.simd.vdup(0.0, "float32", gate_full)
                gate_dot_k = T.simd.vdup(0.0, "float32", gate_full)
                for gate_chunk in T.serial(vd // 64):
                    gate_col = gate_chunk * 64
                    gate_valid = T.simd.update_mask(T.min(64, T.max(0, d - gate_col)), 32)
                    gate_hidden = T.simd.vcvt(
                        T.simd.vld(gate_low[0, gate_col], "UNPK_B16"), "float32"
                    )
                    gate_key = T.simd.vcvt(T.simd.vld(gate_low[1, gate_col], "UNPK_B16"), "float32")
                    gate_weight = T.simd.vcvt(
                        T.simd.vld(gate_low[3, gate_col], "UNPK_B16"), "float32"
                    )
                    gate_norm_h = T.simd.vmul(
                        T.simd.vmul(gate_hidden, gate_r_h, gate_valid), gate_weight, gate_valid
                    )
                    gate_norm_k = T.simd.vmul(
                        T.simd.vmul(gate_key, gate_r_k, gate_valid), gate_weight, gate_valid
                    )
                    gate_dot_h = T.simd.vadd(
                        gate_dot_h,
                        T.simd.vmul(
                            T.simd.vmul(
                                T.simd.vmul(gate_hidden, gate_weight, gate_valid),
                                gate_gate,
                                gate_valid,
                            ),
                            gate_norm_k,
                            gate_valid,
                        ),
                        gate_full,
                    )
                    gate_dot_k = T.simd.vadd(
                        gate_dot_k,
                        T.simd.vmul(
                            T.simd.vmul(
                                T.simd.vmul(gate_key, gate_weight, gate_valid),
                                gate_gate,
                                gate_valid,
                            ),
                            gate_norm_h,
                            gate_valid,
                        ),
                        gate_full,
                    )
                    T.simd.vsts(gate_norm[0, gate_col], gate_norm_h, gate_full)
                    T.simd.vsts(gate_norm[1, gate_col], gate_norm_k, gate_full)
                gate_sum_h = T.simd.vdupv(T.simd.vcadd(gate_dot_h, gate_full), gate_full)
                gate_sum_k = T.simd.vdupv(T.simd.vcadd(gate_dot_k, gate_full), gate_full)
                gate_rh3 = T.simd.vmul(
                    T.simd.vmul(gate_r_h, gate_r_h, gate_full), gate_r_h, gate_full
                )
                gate_rk3 = T.simd.vmul(
                    T.simd.vmul(gate_r_k, gate_r_k, gate_full), gate_r_k, gate_full
                )
                gate_denom = T.simd.vdup(T.float32(d), "float32", gate_full)
                T.simd.mem_bar("VST_VLD")
                for gate_chunk in T.serial(vd // 64):
                    gate_col = gate_chunk * 64
                    gate_valid = T.simd.update_mask(T.min(64, T.max(0, d - gate_col)), 32)
                    gate_hidden = T.simd.vcvt(
                        T.simd.vld(gate_low[0, gate_col], "UNPK_B16"), "float32"
                    )
                    gate_key = T.simd.vcvt(T.simd.vld(gate_low[1, gate_col], "UNPK_B16"), "float32")
                    gate_weight = T.simd.vcvt(
                        T.simd.vld(gate_low[3, gate_col], "UNPK_B16"), "float32"
                    )
                    gate_norm_h = T.simd.vld(gate_norm[0, gate_col])
                    gate_norm_k = T.simd.vld(gate_norm[1, gate_col])
                    gate_upstream = T.simd.vld(gz[gate_col])
                    gate_direct_h = T.simd.vmul(
                        T.simd.vmul(
                            T.simd.vmul(gate_r_h, gate_weight, gate_valid), gate_gate, gate_valid
                        ),
                        gate_norm_k,
                        gate_valid,
                    )
                    gate_direct_k = T.simd.vmul(
                        T.simd.vmul(
                            T.simd.vmul(gate_r_k, gate_weight, gate_valid), gate_gate, gate_valid
                        ),
                        gate_norm_h,
                        gate_valid,
                    )
                    gate_correction_h = T.simd.vdiv(
                        T.simd.vmul(
                            T.simd.vmul(gate_rh3, gate_hidden, gate_valid), gate_sum_h, gate_valid
                        ),
                        gate_denom,
                        gate_valid,
                    )
                    gate_correction_k = T.simd.vdiv(
                        T.simd.vmul(
                            T.simd.vmul(gate_rk3, gate_key, gate_valid), gate_sum_k, gate_valid
                        ),
                        gate_denom,
                        gate_valid,
                    )
                    gate_grad_h = T.simd.vsub(gate_direct_h, gate_correction_h, gate_valid)
                    gate_grad_k = T.simd.vsub(gate_direct_k, gate_correction_k, gate_valid)
                    gate_grad_v = T.simd.vmul(gate_alpha, gate_upstream, gate_valid)
                    gate_partial = T.simd.vmul(
                        gate_gate,
                        T.simd.vadd(
                            T.simd.vmul(
                                T.simd.vmul(gate_norm_k, gate_hidden, gate_valid),
                                gate_r_h,
                                gate_valid,
                            ),
                            T.simd.vmul(
                                T.simd.vmul(gate_norm_h, gate_key, gate_valid), gate_r_k, gate_valid
                            ),
                            gate_valid,
                        ),
                        gate_valid,
                    )
                    T.simd.vsts(
                        gate_low[4, gate_col],
                        T.simd.vcvt(gate_grad_h, dtype, round="ROUND_R", sat=False),
                        gate_full,
                        dist="PK_B32",
                    )
                    T.simd.vsts(
                        gate_low[5, gate_col],
                        T.simd.vcvt(gate_grad_k, dtype, round="ROUND_R", sat=False),
                        gate_full,
                        dist="PK_B32",
                    )
                    T.simd.vsts(
                        gate_low[6, gate_col],
                        T.simd.vcvt(gate_grad_v, dtype, round="ROUND_R", sat=False),
                        gate_full,
                        dist="PK_B32",
                    )
                    T.simd.vsts(gn[gate_col], gate_partial, gate_full)
            T.copy(gate_low[4, :dp], DH[row, :])
            T.copy(gate_low[5, :dp], DK[row, :])
            T.copy(gate_low[6, :dp], DV[row, :])
            T.copy(gn[:dp], PH[row, :])

    return tilelang.compile(main, target="ascend", out_idx=[13, 14, 15, 16, 17])


def _reduce_layout(rows: int, dp: int):
    target_tiles = min(8, (dp + 63) // 64)
    width = ((dp + target_tiles - 1) // target_tiles + 63) // 64 * 64
    tiles = (dp + width - 1) // width
    row_cap = min(rows, 512, max(1, (184 * 1024) // (width * 4) - 1))
    row_block = max(n for n in range(1, row_cap + 1) if rows % n == 0)
    return width, tiles, row_block


@functools.lru_cache(maxsize=32)
def _build_weight_reduce(rows: int, dp: int):
    width, tiles, row_block = _reduce_layout(rows, dp)

    @T.prim_func
    def main(
        PC: T.Tensor((4, rows, dp), "float32"),
        PV: T.Tensor((rows, dp), "float32"),
        PH: T.Tensor((rows, dp), "float32"),
        DWH: T.Tensor((dp,), "float32"),
        DWV: T.Tensor((dp,), "float32"),
        DCW: T.Tensor((4, dp), "float32"),
    ):
        with T.Kernel(6 * tiles) as bx:
            group = bx // tiles
            offset = bx % tiles * width
            columns = T.min(width, dp - offset)
            partial = T.alloc_shared((row_block, width), "float32")
            total = T.alloc_shared((width,), "float32")
            T.annotate_buffer_versions({partial: 1, total: 1})
            for block in T.serial(rows // row_block):
                start = block * row_block
                if group == 0:
                    T.copy(
                        PH[start : start + row_block, offset : offset + columns],
                        partial[:, :columns],
                    )
                elif group == 1:
                    T.copy(
                        PV[start : start + row_block, offset : offset + columns],
                        partial[:, :columns],
                    )
                else:
                    T.copy(
                        PC[group - 2, start : start + row_block, offset : offset + columns],
                        partial[:, :columns],
                    )
                with T.SimdVF():
                    full = T.simd.pset(32)
                    T.simd.mem_bar("VST_VLD")
                    for chunk in T.serial(width // 64):
                        col = chunk * 64
                        valid = T.simd.update_mask(T.min(64, T.max(0, columns - col)), 32)
                        running = T.simd.alloc_var("float32")
                        if block == 0:
                            running = T.simd.vdup(0.0, "float32", full)
                        else:
                            running = T.simd.vld(total[col])
                        for row in T.serial(row_block):
                            values = T.simd.vld(partial[row, col])
                            running = T.simd.vadd(running, values, valid)
                        T.simd.vsts(total[col], running, full)
            if group == 0:
                T.copy(total[:columns], DWH[offset : offset + columns])
            elif group == 1:
                T.copy(total[:columns], DWV[offset : offset + columns])
            else:
                T.copy(total[:columns], DCW[group - 2, offset : offset + columns])

    return tilelang.compile(main, target="ascend", out_idx=[3, 4, 5])


class AscendEngramGateConvBwdKernel(Kernel, EngramGateConvBwdInterface):
    """Three-stage deterministic SIMD backward using the forward saved tensors."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: EngramGateConvCall) -> Entry:
        return call, lambda: cls(call)

    @classmethod
    def refusal(cls, call: EngramGateConvCall) -> str | None:
        if call.dtype not in (torch.float16, torch.bfloat16):
            return "requires float16 or bfloat16"
        if min(call.m, call.seq_len, call.d) <= 0:
            return "requires positive batch, sequence and hidden dimensions"
        if call.d > _MAX_HIDDEN_DIM:
            return "requires hidden dimensions up to 4096 to fit the AIV UB budget"
        return super().refusal(call)

    def __init__(self, call: EngramGateConvCall) -> None:
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        self.rows = call.m * call.seq_len
        # Each low-precision gradient row must own complete 32-byte DMA blocks.
        self.dp = (call.d + 15) // 16 * 16
        self._silu = _build_silu_grad(self.rows, call.seq_len, call.d, self.dp, self.dtype_str)
        self._value_gate = _build_value_gate_grad(
            self.rows, call.seq_len, call.d, self.dp, self.dtype_str
        )
        self._reduce = _build_weight_reduce(self.rows, self.dp)

    def forward(
        self,
        dY: torch.Tensor,
        H: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
        vhat: torch.Tensor,
        alpha: torch.Tensor,
        rrms_h: torch.Tensor,
        rrms_k: torch.Tensor,
        rrms_v: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Read the forward's state; allocate new activation and weight gradients."""
        self._require_devices(
            dY=dY,
            H=H,
            k=k,
            v=v,
            rms_w_h=rms_w_h,
            rms_w_v=rms_w_v,
            conv_w=conv_w,
            vhat=vhat,
            alpha=alpha,
            rrms_h=rrms_h,
            rrms_k=rrms_k,
            rrms_v=rrms_v,
        )
        c = self.call
        dy, h, key, value, z = (t.reshape(self.rows, c.d).contiguous() for t in (dY, H, k, v, vhat))
        a, rh, rk, rv = (t.reshape(self.rows).contiguous() for t in (alpha, rrms_h, rrms_k, rrms_v))
        ds, pc = self._silu(dy, z, rms_w_v, conv_w, rv)
        dh, dk, dv, ph, pv = self._value_gate(
            dy, h, key, value, z, rms_w_h, rms_w_v, conv_w, a, rh, rk, rv, ds
        )
        dwh, dwv, dcw = self._reduce(pc, pv, ph)
        gradients = [t.reshape(c.m, c.seq_len, self.dp)[:, :, : c.d] for t in (dh, dk, dv)]
        return gradients + [dwh[: c.d], dwv[: c.d], dcw[:, : c.d]]


register_kernel_type("EngramGateConvBwdOp", "ascend", AscendEngramGateConvBwdKernel)
