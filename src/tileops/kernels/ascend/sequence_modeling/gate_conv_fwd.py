"""Ascend Engram post-projection gating and causal depthwise convolution."""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.sequence_modeling.engram.call_spec import (
    CONV_KERNEL_SIZE,
    EngramGateConvCall,
    EngramGateConvFwdInterface,
)

__all__ = ["AscendEngramGateConvFwdKernel"]

_MAX_HIDDEN_DIM = 4096


@functools.lru_cache(maxsize=64)
def _build_gate_conv_fwd_gate(M: int, seq_len: int, d: int, eps: float, dtype: str):
    """Gate and save per-token statistics using explicit FP32 SIMD reductions."""
    dp = (d + 63) // 64 * 64

    @T.prim_func
    def main(
        H: T.Tensor((M, seq_len, d), dtype),
        K: T.Tensor((M, seq_len, d), dtype),
        V: T.Tensor((M, seq_len, d), dtype),
        WH: T.Tensor((d,), dtype),
        VHat: T.Tensor((M, seq_len, d), dtype),
        Alpha: T.Tensor((M, seq_len), "float32"),
        RH: T.Tensor((M, seq_len), "float32"),
        RK: T.Tensor((M, seq_len), "float32"),
        RV: T.Tensor((M, seq_len), "float32"),
    ):
        with T.Kernel(M * seq_len) as row:
            batch = row // seq_len
            token = row % seq_len
            h = T.alloc_shared((dp,), dtype)
            k = T.alloc_shared((dp,), dtype)
            value = T.alloc_shared((dp,), dtype)
            weight = T.alloc_shared((dp,), dtype)
            vhat = T.alloc_shared((dp,), dtype)
            stats = T.alloc_shared((4, 8), "float32")
            T.copy(H[batch, token, :], h[:d])
            T.copy(K[batch, token, :], k[:d])
            T.copy(V[batch, token, :], value[:d])
            T.copy(WH, weight[:d])
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.vdup(T.float32(1), "float32", full)
                dimension = T.simd.vdup(T.float32(d), "float32", full)
                epsilon = T.simd.vdup(T.float32(eps), "float32", full)
                hs = T.simd.alloc_local((1,), "float32")
                ks = T.simd.alloc_local((1,), "float32")
                hs[0] = T.simd.vdup(T.float32(0), "float32", full)
                ks[0] = T.simd.vdup(T.float32(0), "float32", full)
                for chunk in T.serial(dp // 64):
                    mask = T.simd.update_mask(T.min(64, d - chunk * 64), 32)
                    hv = T.simd.vcvt(T.simd.vld(h[chunk * 64], dist="UNPK_B16"), "float32", part=0)
                    kv = T.simd.vcvt(T.simd.vld(k[chunk * 64], dist="UNPK_B16"), "float32", part=0)
                    # MODE_ZEROING explicitly removes the uninitialized UB tail.
                    hs[0] = T.simd.vadd(hs[0], T.simd.vmul(hv, hv, mask), full)
                    ks[0] = T.simd.vadd(ks[0], T.simd.vmul(kv, kv, mask), full)
                hsum = T.simd.vdupv(T.simd.vcadd(hs[0], full), full)
                ksum = T.simd.vdupv(T.simd.vcadd(ks[0], full), full)
                hmean = T.simd.vadd(T.simd.vdiv(hsum, dimension, full), epsilon, full)
                kmean = T.simd.vadd(T.simd.vdiv(ksum, dimension, full), epsilon, full)
                rh = T.simd.vdiv(one, T.simd.vsqrt(hmean, full), full)
                rk = T.simd.vdiv(one, T.simd.vsqrt(kmean, full), full)
                T.simd.vsts(stats[0, 0], rh, full, dist="ONEPT_B32", extent=1)
                T.simd.vsts(stats[1, 0], rk, full, dist="ONEPT_B32", extent=1)
                dot = T.simd.alloc_local((1,), "float32")
                dot[0] = T.simd.vdup(T.float32(0), "float32", full)
                for chunk in T.serial(dp // 64):
                    mask = T.simd.update_mask(T.min(64, d - chunk * 64), 32)
                    hv = T.simd.vcvt(T.simd.vld(h[chunk * 64], dist="UNPK_B16"), "float32", part=0)
                    kv = T.simd.vcvt(T.simd.vld(k[chunk * 64], dist="UNPK_B16"), "float32", part=0)
                    wv = T.simd.vcvt(
                        T.simd.vld(weight[chunk * 64], dist="UNPK_B16"), "float32", part=0
                    )
                    hn = T.simd.vmul(T.simd.vmul(hv, rh, mask), wv, mask)
                    kn = T.simd.vmul(T.simd.vmul(kv, rk, mask), wv, mask)
                    dot[0] = T.simd.vadd(dot[0], T.simd.vmul(hn, kn, mask), full)
                logit_sum = T.simd.vdupv(T.simd.vcadd(dot[0], full), full)
                divisor = T.simd.vdup(T.float32(d**0.5), "float32", full)
                logit = T.simd.vdiv(logit_sum, divisor, full)
                exponential = T.simd.vexp(T.simd.vneg(logit, full), full)
                alpha = T.simd.vdiv(one, T.simd.vadd(one, exponential, full), full)
                T.simd.vsts(stats[2, 0], alpha, full, dist="ONEPT_B32", extent=1)
                vs = T.simd.alloc_local((1,), "float32")
                vs[0] = T.simd.vdup(T.float32(0), "float32", full)
                for chunk in T.serial(dp // 64):
                    mask = T.simd.update_mask(T.min(64, d - chunk * 64), 32)
                    vv = T.simd.vcvt(
                        T.simd.vld(value[chunk * 64], dist="UNPK_B16"), "float32", part=0
                    )
                    gated = T.simd.vmul(alpha, vv, mask)
                    narrowed = T.simd.vcvt(gated, dtype, round="ROUND_R", sat=False, part=0)
                    T.simd.vsts(vhat[chunk * 64], narrowed, full, dist="PK_B32")
                    # Register round-trip preserves dtype rounding without a UB hazard.
                    rounded = T.simd.vcvt(narrowed, "float32", part=0)
                    vs[0] = T.simd.vadd(vs[0], T.simd.vmul(rounded, rounded, mask), full)
                vsum = T.simd.vdupv(T.simd.vcadd(vs[0], full), full)
                vmean = T.simd.vadd(T.simd.vdiv(vsum, dimension, full), epsilon, full)
                rv = T.simd.vdiv(one, T.simd.vsqrt(vmean, full), full)
                T.simd.vsts(stats[3, 0], rv, full, dist="ONEPT_B32", extent=1)
            # The selected Ascend align-v2 DMA takes byte lengths: these writes
            # preserve both arbitrary dtype row widths and four-byte scalar rows.
            T.copy(vhat[:d], VHat[batch, token, :])
            T.copy(stats[2, :1], Alpha[batch, token : token + 1])
            T.copy(stats[0, :1], RH[batch, token : token + 1])
            T.copy(stats[1, :1], RK[batch, token : token + 1])
            T.copy(stats[3, :1], RV[batch, token : token + 1])

    return tilelang.compile(main, target="ascend", out_idx=[4, 5, 6, 7, 8])


@functools.lru_cache(maxsize=64)
def _build_gate_conv_fwd_conv(M: int, seq_len: int, d: int, dtype: str):
    """Apply four causal taps, SiLU and the unrounded gated-value residual."""
    dp = (d + 63) // 64 * 64

    @T.prim_func
    def main(
        V: T.Tensor((M, seq_len, d), dtype),
        VHat: T.Tensor((M, seq_len, d), dtype),
        Alpha: T.Tensor((M, seq_len), "float32"),
        RV: T.Tensor((M, seq_len), "float32"),
        WV: T.Tensor((d,), dtype),
        CW: T.Tensor((CONV_KERNEL_SIZE, d), dtype),
        Y: T.Tensor((M, seq_len, d), dtype),
    ):
        with T.Kernel(M * seq_len) as row:
            batch = row // seq_len
            token = row % seq_len
            value = T.alloc_shared((dp,), dtype)
            vh = T.alloc_shared((CONV_KERNEL_SIZE, dp), dtype)
            wv = T.alloc_shared((dp,), dtype)
            cw = T.alloc_shared((CONV_KERNEL_SIZE, dp), dtype)
            alpha = T.alloc_shared((8,), "float32")
            rv = T.alloc_shared((CONV_KERNEL_SIZE, 8), "float32")
            result = T.alloc_shared((dp,), dtype)
            T.copy(V[batch, token, :], value[:d])
            T.copy(WV, wv[:d])
            T.copy(CW, cw[:, :d])
            T.copy(Alpha[batch, token : token + 1], alpha[:1])
            for p in T.unroll(CONV_KERNEL_SIZE):
                if token >= CONV_KERNEL_SIZE - 1 - p:
                    src = token - (CONV_KERNEL_SIZE - 1) + p
                    T.copy(VHat[batch, src, :], vh[p, :d])
                    T.copy(RV[batch, src : src + 1], rv[p, :1])
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.vdup(T.float32(1), "float32", full)
                gate = T.simd.vld(alpha[0], dist="BRC_B32")
                for chunk in T.serial(dp // 64):
                    mask = T.simd.update_mask(T.min(64, d - chunk * 64), 32)
                    weight = T.simd.vcvt(
                        T.simd.vld(wv[chunk * 64], dist="UNPK_B16"), "float32", part=0
                    )
                    conv = T.simd.alloc_local((1,), "float32")
                    conv[0] = T.simd.vdup(T.float32(0), "float32", full)
                    for p in T.unroll(CONV_KERNEL_SIZE):
                        if token >= CONV_KERNEL_SIZE - 1 - p:
                            raw = T.simd.vcvt(
                                T.simd.vld(vh[p, chunk * 64], dist="UNPK_B16"), "float32", part=0
                            )
                            reciprocal = T.simd.vld(rv[p, 0], dist="BRC_B32")
                            tap = T.simd.vcvt(
                                T.simd.vld(cw[p, chunk * 64], dist="UNPK_B16"), "float32", part=0
                            )
                            normalized = T.simd.vmul(
                                T.simd.vmul(raw, reciprocal, mask), weight, mask
                            )
                            conv[0] = T.simd.vadd(conv[0], T.simd.vmul(normalized, tap, mask), full)
                    denominator = T.simd.vadd(
                        one, T.simd.vexp(T.simd.vneg(conv[0], full), full), full
                    )
                    silu = T.simd.vdiv(conv[0], denominator, full)
                    raw_value = T.simd.vcvt(
                        T.simd.vld(value[chunk * 64], dist="UNPK_B16"), "float32", part=0
                    )
                    residual = T.simd.vmul(gate, raw_value, mask)
                    y = T.simd.vadd(silu, residual, mask)
                    narrowed = T.simd.vcvt(y, dtype, round="ROUND_R", sat=False, part=0)
                    T.simd.vsts(result[chunk * 64], narrowed, full, dist="PK_B32")
            T.copy(result[:d], Y[batch, token, :])

    return tilelang.compile(main, target="ascend", out_idx=[6])


class AscendEngramGateConvFwdKernel(Kernel, EngramGateConvFwdInterface):
    """Two AIV launches implementing the manifest's Engram forward reference."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: EngramGateConvCall) -> Entry:
        return call, lambda: cls(call)

    @classmethod
    def refusal(cls, call: EngramGateConvCall):
        if call.dtype not in (torch.float16, torch.bfloat16):
            return "Engram GateConv requires float16 or bfloat16 inputs"
        if call.m <= 0 or call.seq_len <= 0 or call.d <= 0:
            return "Engram GateConv requires positive batch, sequence and hidden dimensions"
        if call.d > _MAX_HIDDEN_DIM:
            return "Engram GateConv supports hidden dimensions up to 4096"
        return super().refusal(call)

    def __init__(self, call: EngramGateConvCall):
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.dtype = call.dtype
        self.gate_kernel = _build_gate_conv_fwd_gate(
            call.m, call.seq_len, call.d, call.eps, self.dtype_str
        )
        self.conv_kernel = _build_gate_conv_fwd_conv(call.m, call.seq_len, call.d, self.dtype_str)

    def forward(self, H, k, v, rms_w_h, rms_w_v, conv_w) -> list[torch.Tensor]:
        self._require_devices(H=H, k=k, v=v, rms_w_h=rms_w_h, rms_w_v=rms_w_v, conv_w=conv_w)
        vhat, alpha, rrms_h, rrms_k, rrms_v = self.gate_kernel(H, k, v, rms_w_h)
        Y = self.conv_kernel(v, vhat, alpha, rrms_v, rms_w_v, conv_w)
        return [Y, vhat, alpha, rrms_h, rrms_k, rrms_v]


register_kernel_type("EngramGateConvFwdOp", "ascend", AscendEngramGateConvFwdKernel)
