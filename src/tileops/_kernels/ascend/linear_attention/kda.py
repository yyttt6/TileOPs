"""Kimi Delta Attention forward for the Ascend backend."""

import math
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.linear_attention.call_spec import KDACall, KDAFwdInterface
from tileops.backend._registry import register_kernel_type

__all__ = ["AscendKDAFwdKernel"]


@tilelang.jit(target="ascend")
def _kda_kernel(
    B,
    S,
    H,
    HV,
    DK,
    DV,
    N,
    dtype,
    packed,
    has_state,
    vfirst,
    norm,
    raw_gate,
    has_bias,
    bound,
    beta_sigmoid,
    neg_eigval,
    scale,
):
    BK = (DK + 31) // 32 * 32
    RD = BK // 32
    if S == 1:
        VPW = 8 if N * HV >= 56 else 4
    else:
        # Estimate the slowest block's work, including one shared token preparation.
        VPW = min(
            (1, 2, 4),
            key=lambda g: (math.ceil(N * HV * math.ceil(DV / (8 * g)) / 56) * (2 + g), -g),
        )
    VECTOR = (
        norm
        and abs(scale) <= 1.0
        and not raw_gate
        and not beta_sigmoid
        and not has_state
        and S > 1
        and DK % 64 == 0
        and DK <= 128
        and DV % 16 == 0
    )
    if VECTOR:
        VPW = 1
    BV = 8 * VPW
    PV = max(BV, 16)
    VT = (DV + BV - 1) // BV
    TASKS = N * HV * VT
    NB = min(56, TASKS)
    SP = BV + 8
    available = 216 * 1024 - BK * SP * 4 - 4096
    max_tokens = available // (18 * BK + (8 if VECTOR else 4) * PV + 18)
    CT = 1 if S == 1 or max_tokens < 8 else min(64, max_tokens // 8 * 8)
    PREP_WARPS = 1 if CT == 1 else 8
    CHECK_RANGE = norm and abs(scale) <= 1.0
    STATE_DMA = not vfirst and DV % 8 == 0
    tile_shape = (BV, BK) if VECTOR else (BK, SP)
    state_shape = (N, HV, DV, DK) if vfirst else (N, HV, DK, DV)

    @T.macro
    def prepare(
        qu, ku, gu, bu, nq, nk, ng, strength_buf, qf, kf, safe_token, valid_t, a_log, bias, hv
    ):
        with T.SimtVF(threads=PREP_WARPS * 32):
            tx = T.get_thread_binding()
            lane = tx % 32
            for group in T.serial(T.ceildiv(valid_t, PREP_WARPS)):
                token = group * PREP_WARPS + tx // 32
                # Every warp executes the same control flow. Tail warps use the
                # last valid source token and write only their own padded rows.
                source_token = T.min(token, valid_t - 1)
                query = T.alloc_local((RD,), "float32")
                key = T.alloc_local((RD,), "float32")
                decay = T.alloc_local((RD,), "float32")
                q2 = T.alloc_var("float32", init=0)
                k2 = T.alloc_var("float32", init=0)
                for j in T.unroll(RD):
                    channel = lane + j * 32
                    query[j] = T.float32(0)
                    key[j] = T.float32(0)
                    decay[j] = T.float32(1)
                    if channel < DK:
                        query[j] = T.cast(qu[source_token, channel], "float32")
                        key[j] = T.cast(ku[source_token, channel], "float32")
                        gate = T.alloc_var("float32")
                        gate = T.cast(gu[source_token, channel], "float32")
                        if raw_gate:
                            if has_bias:
                                gate = gate + bias[hv * DK + channel]
                            if bound is None:
                                # log(softplus(x)) avoids exp(A)*exp(x)
                                # losing a finite product at A=88,x=-88.
                                logsp = T.alloc_var("float32")
                                if gate < -8.0:
                                    # softplus(x)/exp(x) = 1-y/2+y*y/3...
                                    # This also avoids log(1+tiny) rounding
                                    # to zero before a large rate is applied.
                                    eg = T.exp(gate)
                                    logsp = gate + T.log(1.0 - 0.5 * eg + eg * eg / 3.0)
                                elif gate > 20.0:
                                    logsp = T.log(gate)
                                else:
                                    logsp = T.log(T.log(1.0 + T.exp(gate)))
                                gate = -T.exp(T.min(a_log[hv] + logsp, 80.0))
                            else:
                                z = T.alloc_var("float32", init=0)
                                if gate != 0.0:
                                    z = T.exp(
                                        T.min(a_log[hv] + T.log(T.abs(gate)), 4.382026634673881)
                                    )
                                    if gate < 0.0:
                                        z = -z
                                if z >= 0.0:
                                    gate = T.float32(bound) / (1.0 + T.exp(-z))
                                else:
                                    ez = T.exp(z)
                                    gate = T.float32(bound) * ez / (1.0 + ez)
                        decay[j] = T.exp(gate)
                    q2 += query[j] * query[j]
                    k2 += key[j] * key[j]
                rq = T.alloc_var("float32", init=1)
                rk = T.alloc_var("float32", init=1)
                if norm:
                    rq = T.rsqrt(T.warp_reduce_sum(q2) + 1e-6)
                    rk = T.rsqrt(T.warp_reduce_sum(k2) + 1e-6)
                strength = T.alloc_var("float32")
                strength = T.cast(bu[source_token], "float32")
                if beta_sigmoid:
                    if strength >= 0:
                        strength = 1.0 / (1.0 + T.exp(-strength))
                    else:
                        eb = T.exp(strength)
                        strength = eb / (1.0 + eb)
                    if neg_eigval:
                        strength = 2.0 * strength
                max_query = T.alloc_var("float32", init=0)
                max_key = T.alloc_var("float32", init=0)
                unsafe = T.alloc_var("float32", init=0)
                for j in T.unroll(RD):
                    channel = lane + j * 32
                    qj = query[j] * rq * T.float32(scale)
                    kj = key[j] * rk
                    nq[token, channel] = qj
                    nk[token, channel] = kj
                    ng[token, channel] = decay[j]
                    if CHECK_RANGE:
                        if not (decay[j] <= 1.0 and T.abs(qj) <= 1.001 and T.abs(kj) <= 1.001):
                            unsafe = T.float32(1)
                    else:
                        max_query = T.max(max_query, T.abs(qj))
                        max_key = T.max(max_key, T.abs(kj))
                if CHECK_RANGE:
                    unsafe = T.warp_reduce_max(unsafe)
                else:
                    max_query = T.warp_reduce_max(max_query)
                    max_key = T.warp_reduce_max(max_key)
                if lane == 0:
                    strength_buf[token] = strength
                    if CHECK_RANGE:
                        safe_token[token] = T.int32(
                            unsafe == 0 and strength >= 0.0 and strength <= 1.0
                        )
                        qf[token] = T.float32(1)
                        kf[token] = T.float32(1)
                    else:
                        qf[token] = T.min(1.0, 72057594037927936.0 / T.max(max_query, 1.0))
                        kf[token] = T.min(1.0, 72057594037927936.0 / T.max(max_key, 1.0))

    @T.macro
    def advance(
        state, nq, nk, ng, strength_buf, qf, kf, vu, yu, valid_t, valid_v, lane, column, FAST
    ):
        for token in T.serial(valid_t):
            query = T.alloc_local((RD,), "float32")
            key = T.alloc_local((RD,), "float32")
            decay = T.alloc_local((RD,), "float32")
            for j in T.unroll(RD):
                channel = lane + j * 32
                query[j] = nq[token, channel]
                key[j] = nk[token, channel]
                decay[j] = ng[token, channel]
            strength = strength_buf[token]
            for u in T.unroll(VPW):
                if column + u < valid_v:
                    max_state = T.alloc_var("float32", init=0)
                    for j in T.unroll(RD):
                        state[u, j] = state[u, j] * decay[j]
                        if not FAST:
                            max_state = T.max(max_state, T.abs(state[u, j]))
                    if not FAST:
                        state_factor = T.min(
                            1.0, 72057594037927936.0 / T.max(T.warp_reduce_max(max_state), 1.0)
                        )
                        key_factor = kf[token]
                    prediction = T.alloc_var("float32", init=0)
                    for j in T.unroll(RD):
                        if FAST:
                            prediction += key[j] * state[u, j]
                        else:
                            prediction += (key[j] * key_factor) * (state[u, j] * state_factor)
                    prediction = T.warp_reduce_sum(prediction)
                    if not FAST:
                        prediction = (prediction / key_factor) / state_factor
                    correction = (
                        strength * T.cast(vu[token, column + u], "float32") - strength * prediction
                    )
                    max_updated = T.alloc_var("float32", init=0)
                    for j in T.unroll(RD):
                        state[u, j] = state[u, j] + key[j] * correction
                        if not FAST:
                            max_updated = T.max(max_updated, T.abs(state[u, j]))
                    if not FAST:
                        state_out_factor = T.min(
                            1.0, 72057594037927936.0 / T.max(T.warp_reduce_max(max_updated), 1.0)
                        )
                        query_factor = qf[token]
                    out = T.alloc_var("float32", init=0)
                    for j in T.unroll(RD):
                        if FAST:
                            out += query[j] * state[u, j]
                        else:
                            out += (query[j] * query_factor) * (state[u, j] * state_out_factor)
                    out = T.warp_reduce_sum(out)
                    if not FAST:
                        out = (out / query_factor) / state_out_factor
                    if lane == 0:
                        yu[token, column + u] = T.cast(out, dtype)

    @T.macro
    def check_vector_range(su, vu, safe_token, warp_flags, tile_flag, valid_t):
        with T.SimtVF(threads=256):
            tx = T.get_thread_binding()
            unsafe = T.alloc_var("float32", init=0)
            for i in T.unroll(T.ceildiv(BV * BK, 256)):
                pos = i * 256 + tx
                # Keep TileLang bounds checks separate from the guarded reads.
                if pos < BV * BK:  # noqa: SIM102
                    if not (T.abs(su[pos // BK, pos % BK]) <= 1099511627776.0):
                        unsafe = T.float32(1)
            for i in T.unroll(T.ceildiv(CT * BV, 256)):
                pos = i * 256 + tx
                if pos < valid_t * BV:  # noqa: SIM102
                    if not (T.abs(T.cast(vu[pos // BV, pos % BV], "float32")) <= 1099511627776.0):
                        unsafe = T.float32(1)
            if tx < valid_t:  # noqa: SIM102
                if safe_token[tx] == 0:
                    unsafe = T.float32(1)
            unsafe = T.warp_reduce_max(unsafe)
            if tx % 32 == 0:
                warp_flags[tx // 32] = unsafe
            T.sync_threads()
            if tx == 0:
                combined = T.alloc_var("float32", init=0)
                for warp in T.unroll(8):
                    combined = T.max(combined, warp_flags[warp])
                tile_flag[0] = combined

    @T.macro
    def advance_vector(su, nq, nk, ng, strength, vu, yf, yu, valid_t):
        # The explicit names keep the state rows live across the whole token tile.
        with T.SimdVF():
            full = T.simd.pset(32)
            first_lane = T.simd.update_mask(1, 32)
            state_0_lo = T.simd.alloc_var("float32")
            state_0_lo = T.simd.vld(su[0, 0])
            if DK > 64:
                state_0_hi = T.simd.alloc_var("float32")
                state_0_hi = T.simd.vld(su[0, 64])
            state_1_lo = T.simd.alloc_var("float32")
            state_1_lo = T.simd.vld(su[1, 0])
            if DK > 64:
                state_1_hi = T.simd.alloc_var("float32")
                state_1_hi = T.simd.vld(su[1, 64])
            state_2_lo = T.simd.alloc_var("float32")
            state_2_lo = T.simd.vld(su[2, 0])
            if DK > 64:
                state_2_hi = T.simd.alloc_var("float32")
                state_2_hi = T.simd.vld(su[2, 64])
            state_3_lo = T.simd.alloc_var("float32")
            state_3_lo = T.simd.vld(su[3, 0])
            if DK > 64:
                state_3_hi = T.simd.alloc_var("float32")
                state_3_hi = T.simd.vld(su[3, 64])
            state_4_lo = T.simd.alloc_var("float32")
            state_4_lo = T.simd.vld(su[4, 0])
            if DK > 64:
                state_4_hi = T.simd.alloc_var("float32")
                state_4_hi = T.simd.vld(su[4, 64])
            state_5_lo = T.simd.alloc_var("float32")
            state_5_lo = T.simd.vld(su[5, 0])
            if DK > 64:
                state_5_hi = T.simd.alloc_var("float32")
                state_5_hi = T.simd.vld(su[5, 64])
            state_6_lo = T.simd.alloc_var("float32")
            state_6_lo = T.simd.vld(su[6, 0])
            if DK > 64:
                state_6_hi = T.simd.alloc_var("float32")
                state_6_hi = T.simd.vld(su[6, 64])
            state_7_lo = T.simd.alloc_var("float32")
            state_7_lo = T.simd.vld(su[7, 0])
            if DK > 64:
                state_7_hi = T.simd.alloc_var("float32")
                state_7_hi = T.simd.vld(su[7, 64])
            for token in T.serial(valid_t):
                qlo = T.simd.vld(nq[token, 0])
                klo = T.simd.vld(nk[token, 0])
                dlo = T.simd.vld(ng[token, 0])
                if DK > 64:
                    qhi = T.simd.vld(nq[token, 64])
                    khi = T.simd.vld(nk[token, 64])
                    dhi = T.simd.vld(ng[token, 64])
                step = T.simd.vld(strength[token], dist="BRC_B32")
                state_0_lo = T.simd.vmul(state_0_lo, dlo)
                projection = T.simd.vcadd(T.simd.vmul(state_0_lo, klo), full)
                if DK > 64:
                    state_0_hi = T.simd.vmul(state_0_hi, dhi)
                    projection = T.simd.vadd(
                        projection, T.simd.vcadd(T.simd.vmul(state_0_hi, khi), full)
                    )
                value = T.simd.vcvt(T.simd.vld(vu[token, 0], dist="BRC_B16"), "float32", part=0)
                correction = T.simd.vsub(
                    T.simd.vmul(value, step), T.simd.vmul(T.simd.vdupv(projection), step)
                )
                T.simd.vmula(state_0_lo, klo, correction, full)
                answer = T.simd.vcadd(T.simd.vmul(state_0_lo, qlo), full)
                if DK > 64:
                    T.simd.vmula(state_0_hi, khi, correction, full)
                    answer = T.simd.vadd(answer, T.simd.vcadd(T.simd.vmul(state_0_hi, qhi), full))
                T.simd.vsts(yf[token, 0], answer, first_lane, dist="ONEPT_B32")
                state_1_lo = T.simd.vmul(state_1_lo, dlo)
                projection = T.simd.vcadd(T.simd.vmul(state_1_lo, klo), full)
                if DK > 64:
                    state_1_hi = T.simd.vmul(state_1_hi, dhi)
                    projection = T.simd.vadd(
                        projection, T.simd.vcadd(T.simd.vmul(state_1_hi, khi), full)
                    )
                value = T.simd.vcvt(T.simd.vld(vu[token, 1], dist="BRC_B16"), "float32", part=0)
                correction = T.simd.vsub(
                    T.simd.vmul(value, step), T.simd.vmul(T.simd.vdupv(projection), step)
                )
                T.simd.vmula(state_1_lo, klo, correction, full)
                answer = T.simd.vcadd(T.simd.vmul(state_1_lo, qlo), full)
                if DK > 64:
                    T.simd.vmula(state_1_hi, khi, correction, full)
                    answer = T.simd.vadd(answer, T.simd.vcadd(T.simd.vmul(state_1_hi, qhi), full))
                T.simd.vsts(yf[token, 1], answer, first_lane, dist="ONEPT_B32")
                state_2_lo = T.simd.vmul(state_2_lo, dlo)
                projection = T.simd.vcadd(T.simd.vmul(state_2_lo, klo), full)
                if DK > 64:
                    state_2_hi = T.simd.vmul(state_2_hi, dhi)
                    projection = T.simd.vadd(
                        projection, T.simd.vcadd(T.simd.vmul(state_2_hi, khi), full)
                    )
                value = T.simd.vcvt(T.simd.vld(vu[token, 2], dist="BRC_B16"), "float32", part=0)
                correction = T.simd.vsub(
                    T.simd.vmul(value, step), T.simd.vmul(T.simd.vdupv(projection), step)
                )
                T.simd.vmula(state_2_lo, klo, correction, full)
                answer = T.simd.vcadd(T.simd.vmul(state_2_lo, qlo), full)
                if DK > 64:
                    T.simd.vmula(state_2_hi, khi, correction, full)
                    answer = T.simd.vadd(answer, T.simd.vcadd(T.simd.vmul(state_2_hi, qhi), full))
                T.simd.vsts(yf[token, 2], answer, first_lane, dist="ONEPT_B32")
                state_3_lo = T.simd.vmul(state_3_lo, dlo)
                projection = T.simd.vcadd(T.simd.vmul(state_3_lo, klo), full)
                if DK > 64:
                    state_3_hi = T.simd.vmul(state_3_hi, dhi)
                    projection = T.simd.vadd(
                        projection, T.simd.vcadd(T.simd.vmul(state_3_hi, khi), full)
                    )
                value = T.simd.vcvt(T.simd.vld(vu[token, 3], dist="BRC_B16"), "float32", part=0)
                correction = T.simd.vsub(
                    T.simd.vmul(value, step), T.simd.vmul(T.simd.vdupv(projection), step)
                )
                T.simd.vmula(state_3_lo, klo, correction, full)
                answer = T.simd.vcadd(T.simd.vmul(state_3_lo, qlo), full)
                if DK > 64:
                    T.simd.vmula(state_3_hi, khi, correction, full)
                    answer = T.simd.vadd(answer, T.simd.vcadd(T.simd.vmul(state_3_hi, qhi), full))
                T.simd.vsts(yf[token, 3], answer, first_lane, dist="ONEPT_B32")
                state_4_lo = T.simd.vmul(state_4_lo, dlo)
                projection = T.simd.vcadd(T.simd.vmul(state_4_lo, klo), full)
                if DK > 64:
                    state_4_hi = T.simd.vmul(state_4_hi, dhi)
                    projection = T.simd.vadd(
                        projection, T.simd.vcadd(T.simd.vmul(state_4_hi, khi), full)
                    )
                value = T.simd.vcvt(T.simd.vld(vu[token, 4], dist="BRC_B16"), "float32", part=0)
                correction = T.simd.vsub(
                    T.simd.vmul(value, step), T.simd.vmul(T.simd.vdupv(projection), step)
                )
                T.simd.vmula(state_4_lo, klo, correction, full)
                answer = T.simd.vcadd(T.simd.vmul(state_4_lo, qlo), full)
                if DK > 64:
                    T.simd.vmula(state_4_hi, khi, correction, full)
                    answer = T.simd.vadd(answer, T.simd.vcadd(T.simd.vmul(state_4_hi, qhi), full))
                T.simd.vsts(yf[token, 4], answer, first_lane, dist="ONEPT_B32")
                state_5_lo = T.simd.vmul(state_5_lo, dlo)
                projection = T.simd.vcadd(T.simd.vmul(state_5_lo, klo), full)
                if DK > 64:
                    state_5_hi = T.simd.vmul(state_5_hi, dhi)
                    projection = T.simd.vadd(
                        projection, T.simd.vcadd(T.simd.vmul(state_5_hi, khi), full)
                    )
                value = T.simd.vcvt(T.simd.vld(vu[token, 5], dist="BRC_B16"), "float32", part=0)
                correction = T.simd.vsub(
                    T.simd.vmul(value, step), T.simd.vmul(T.simd.vdupv(projection), step)
                )
                T.simd.vmula(state_5_lo, klo, correction, full)
                answer = T.simd.vcadd(T.simd.vmul(state_5_lo, qlo), full)
                if DK > 64:
                    T.simd.vmula(state_5_hi, khi, correction, full)
                    answer = T.simd.vadd(answer, T.simd.vcadd(T.simd.vmul(state_5_hi, qhi), full))
                T.simd.vsts(yf[token, 5], answer, first_lane, dist="ONEPT_B32")
                state_6_lo = T.simd.vmul(state_6_lo, dlo)
                projection = T.simd.vcadd(T.simd.vmul(state_6_lo, klo), full)
                if DK > 64:
                    state_6_hi = T.simd.vmul(state_6_hi, dhi)
                    projection = T.simd.vadd(
                        projection, T.simd.vcadd(T.simd.vmul(state_6_hi, khi), full)
                    )
                value = T.simd.vcvt(T.simd.vld(vu[token, 6], dist="BRC_B16"), "float32", part=0)
                correction = T.simd.vsub(
                    T.simd.vmul(value, step), T.simd.vmul(T.simd.vdupv(projection), step)
                )
                T.simd.vmula(state_6_lo, klo, correction, full)
                answer = T.simd.vcadd(T.simd.vmul(state_6_lo, qlo), full)
                if DK > 64:
                    T.simd.vmula(state_6_hi, khi, correction, full)
                    answer = T.simd.vadd(answer, T.simd.vcadd(T.simd.vmul(state_6_hi, qhi), full))
                T.simd.vsts(yf[token, 6], answer, first_lane, dist="ONEPT_B32")
                state_7_lo = T.simd.vmul(state_7_lo, dlo)
                projection = T.simd.vcadd(T.simd.vmul(state_7_lo, klo), full)
                if DK > 64:
                    state_7_hi = T.simd.vmul(state_7_hi, dhi)
                    projection = T.simd.vadd(
                        projection, T.simd.vcadd(T.simd.vmul(state_7_hi, khi), full)
                    )
                value = T.simd.vcvt(T.simd.vld(vu[token, 7], dist="BRC_B16"), "float32", part=0)
                correction = T.simd.vsub(
                    T.simd.vmul(value, step), T.simd.vmul(T.simd.vdupv(projection), step)
                )
                T.simd.vmula(state_7_lo, klo, correction, full)
                answer = T.simd.vcadd(T.simd.vmul(state_7_lo, qlo), full)
                if DK > 64:
                    T.simd.vmula(state_7_hi, khi, correction, full)
                    answer = T.simd.vadd(answer, T.simd.vcadd(T.simd.vmul(state_7_hi, qhi), full))
                T.simd.vsts(yf[token, 7], answer, first_lane, dist="ONEPT_B32")
            T.simd.vsts(su[0, 0], state_0_lo)
            if DK > 64:
                T.simd.vsts(su[0, 64], state_0_hi)
            T.simd.vsts(su[1, 0], state_1_lo)
            if DK > 64:
                T.simd.vsts(su[1, 64], state_1_hi)
            T.simd.vsts(su[2, 0], state_2_lo)
            if DK > 64:
                T.simd.vsts(su[2, 64], state_2_hi)
            T.simd.vsts(su[3, 0], state_3_lo)
            if DK > 64:
                T.simd.vsts(su[3, 64], state_3_hi)
            T.simd.vsts(su[4, 0], state_4_lo)
            if DK > 64:
                T.simd.vsts(su[4, 64], state_4_hi)
            T.simd.vsts(su[5, 0], state_5_lo)
            if DK > 64:
                T.simd.vsts(su[5, 64], state_5_hi)
            T.simd.vsts(su[6, 0], state_6_lo)
            if DK > 64:
                T.simd.vsts(su[6, 64], state_6_hi)
            T.simd.vsts(su[7, 0], state_7_lo)
            if DK > 64:
                T.simd.vsts(su[7, 64], state_7_hi)
        with T.SimtVF(threads=256):
            for t, c in T.Parallel(CT, BV):
                if t < valid_t:
                    yu[t, c] = T.cast(yf[t, c], dtype)

    @T.prim_func
    def kernel(
        q: T.Tensor((B, S, H, DK), dtype),
        k: T.Tensor((B, S, H, DK), dtype),
        v: T.Tensor((B, S, HV, DV), dtype),
        g: T.Tensor((B, S, HV, DK), dtype),
        beta: T.Tensor((B, S, HV), dtype),
        initial: T.Tensor(state_shape, "float32"),
        offsets: T.Tensor((N + 1,), "int64"),
        a_log: T.Tensor((HV,), "float32"),
        bias: T.Tensor((HV * DK,), "float32"),
        output: T.Tensor((B, S, HV, DV), dtype),
        final: T.Tensor(state_shape, "float32"),
    ):
        with T.Kernel(NB) as block:
            qu = T.alloc_shared((CT, BK), dtype)
            ku = T.alloc_shared((CT, BK), dtype)
            gu = T.alloc_shared((CT, BK), dtype)
            vu = T.alloc_shared((CT, PV), dtype)
            yu = T.alloc_shared((CT, PV), dtype)
            bu = T.alloc_shared((CT,), dtype)
            su = T.alloc_shared(tile_shape, "float32")
            if VECTOR:
                yf = T.alloc_shared((CT, PV), "float32")
                warp_flags = T.alloc_shared((8,), "float32")
                tile_flag = T.alloc_shared((8,), "float32")
                T.annotate_buffer_versions({yf: 1, warp_flags: 1, tile_flag: 1})
            nq = T.alloc_shared((CT, BK), "float32")
            nk = T.alloc_shared((CT, BK), "float32")
            ng = T.alloc_shared((CT, BK), "float32")
            strength = T.alloc_shared((CT,), "float32")
            qf = T.alloc_shared((CT,), "float32")
            kf = T.alloc_shared((CT,), "float32")
            safe_token = T.alloc_shared((CT,), "int32")
            T.annotate_buffer_versions(
                {
                    qu: 1,
                    ku: 1,
                    gu: 1,
                    vu: 1,
                    yu: 1,
                    bu: 1,
                    su: 1,
                    nq: 1,
                    nk: 1,
                    ng: 1,
                    strength: 1,
                    qf: 1,
                    kf: 1,
                    safe_token: 1,
                }
            )
            for task_round in T.serial(T.ceildiv(TASKS, NB)):
                task = block + task_round * NB
                if task < TASKS:
                    seq = task // (HV * VT)
                    hv = task // VT % HV
                    h = hv // (HV // H)
                    voff = task % VT * BV
                    valid_v = T.min(BV, DV - voff)
                    row = T.alloc_var("int32", init=seq)
                    first = T.alloc_var("int32", init=0)
                    length = T.alloc_var("int32", init=S)
                    if packed:
                        row = 0
                        first = T.cast(offsets[seq], "int32")
                        length = T.cast(offsets[seq + 1], "int32") - first
                    if not has_state:
                        T.fill(su, 0.0)
                    elif STATE_DMA:
                        T.copy(initial[seq, hv, 0:DK, voff : voff + valid_v], su[0:DK, 0:valid_v])
                    else:
                        with T.SimtVF(threads=256):
                            for j, c in T.Parallel(DK, BV):
                                if c < valid_v:
                                    if vfirst:
                                        su[j, c] = initial[seq, hv, voff + c, j]
                                    else:
                                        su[j, c] = initial[seq, hv, j, voff + c]
                    for tile in T.serial(T.ceildiv(length, CT)):
                        t0 = first + tile * CT
                        valid_t = T.min(CT, length - tile * CT)
                        T.copy(q[row, t0 : t0 + valid_t, h, 0:DK], qu[0:valid_t, 0:DK])
                        T.copy(k[row, t0 : t0 + valid_t, h, 0:DK], ku[0:valid_t, 0:DK])
                        T.copy(g[row, t0 : t0 + valid_t, hv, 0:DK], gu[0:valid_t, 0:DK])
                        T.copy(
                            v[row, t0 : t0 + valid_t, hv, voff : voff + valid_v],
                            vu[0:valid_t, 0:valid_v],
                        )
                        T.copy(beta[row, t0 : t0 + valid_t, hv], bu[0:valid_t])
                        prepare(
                            qu,
                            ku,
                            gu,
                            bu,
                            nq,
                            nk,
                            ng,
                            strength,
                            qf,
                            kf,
                            safe_token,
                            valid_t,
                            a_log,
                            bias,
                            hv,
                        )
                        if VECTOR:
                            check_vector_range(su, vu, safe_token, warp_flags, tile_flag, valid_t)
                            if tile_flag[0] == 0.0:
                                advance_vector(su, nq, nk, ng, strength, vu, yf, yu, valid_t)
                            else:
                                with T.SimtVF(threads=256):
                                    tx = T.get_thread_binding()
                                    lane = tx % 32
                                    column = tx // 32 * VPW
                                    state = T.alloc_local((VPW, RD), "float32")
                                    unsafe = T.alloc_var("float32", init=0)
                                    for u in T.unroll(VPW):
                                        for j in T.unroll(RD):
                                            state[u, j] = T.float32(0)
                                            if lane + j * 32 < DK and column + u < valid_v:
                                                state[u, j] = su[column + u, lane + j * 32]
                                                if CHECK_RANGE:  # noqa: SIM102
                                                    if not (T.abs(state[u, j]) <= 1099511627776.0):
                                                        unsafe = T.float32(1)
                                    if CHECK_RANGE:
                                        # Normalized K, beta in [0,1], and decay <= 1 give a
                                        # contractive homogeneous update. The state column
                                        # norm grows by at most |v| per token. This range
                                        # leaves ample FP32 headroom for both dot products.
                                        for j in T.unroll(T.ceildiv(CT, 32)):
                                            t = lane + j * 32
                                            if t < valid_t:
                                                if safe_token[t] == 0:
                                                    unsafe = T.float32(1)
                                                for u in T.unroll(VPW):
                                                    if column + u < valid_v:  # noqa: SIM102
                                                        if not (
                                                            T.abs(
                                                                T.cast(vu[t, column + u], "float32")
                                                            )
                                                            <= 1099511627776.0
                                                        ):
                                                            unsafe = T.float32(1)
                                        if T.warp_reduce_max(unsafe) == 0.0:
                                            advance(
                                                state,
                                                nq,
                                                nk,
                                                ng,
                                                strength,
                                                qf,
                                                kf,
                                                vu,
                                                yu,
                                                valid_t,
                                                valid_v,
                                                lane,
                                                column,
                                                True,
                                            )
                                        else:
                                            advance(
                                                state,
                                                nq,
                                                nk,
                                                ng,
                                                strength,
                                                qf,
                                                kf,
                                                vu,
                                                yu,
                                                valid_t,
                                                valid_v,
                                                lane,
                                                column,
                                                False,
                                            )
                                    else:
                                        advance(
                                            state,
                                            nq,
                                            nk,
                                            ng,
                                            strength,
                                            qf,
                                            kf,
                                            vu,
                                            yu,
                                            valid_t,
                                            valid_v,
                                            lane,
                                            column,
                                            False,
                                        )
                                    for u in T.unroll(VPW):
                                        for j in T.unroll(RD):
                                            if lane + j * 32 < DK and column + u < valid_v:
                                                su[column + u, lane + j * 32] = state[u, j]
                        else:
                            with T.SimtVF(threads=256):
                                tx = T.get_thread_binding()
                                lane = tx % 32
                                column = tx // 32 * VPW
                                state = T.alloc_local((VPW, RD), "float32")
                                unsafe = T.alloc_var("float32", init=0)
                                for u in T.unroll(VPW):
                                    for j in T.unroll(RD):
                                        state[u, j] = T.float32(0)
                                        if lane + j * 32 < DK and column + u < valid_v:
                                            state[u, j] = su[lane + j * 32, column + u]
                                            if CHECK_RANGE:  # noqa: SIM102
                                                if not (T.abs(state[u, j]) <= 1099511627776.0):
                                                    unsafe = T.float32(1)
                                if CHECK_RANGE:
                                    # Normalized K, beta in [0,1], and decay <= 1 give a
                                    # contractive homogeneous update. The state column
                                    # norm grows by at most |v| per token. This range
                                    # leaves ample FP32 headroom for both dot products.
                                    for j in T.unroll(T.ceildiv(CT, 32)):
                                        t = lane + j * 32
                                        if t < valid_t:
                                            if safe_token[t] == 0:
                                                unsafe = T.float32(1)
                                            for u in T.unroll(VPW):
                                                if column + u < valid_v:  # noqa: SIM102
                                                    if not (
                                                        T.abs(T.cast(vu[t, column + u], "float32"))
                                                        <= 1099511627776.0
                                                    ):
                                                        unsafe = T.float32(1)
                                    if T.warp_reduce_max(unsafe) == 0.0:
                                        advance(
                                            state,
                                            nq,
                                            nk,
                                            ng,
                                            strength,
                                            qf,
                                            kf,
                                            vu,
                                            yu,
                                            valid_t,
                                            valid_v,
                                            lane,
                                            column,
                                            True,
                                        )
                                    else:
                                        advance(
                                            state,
                                            nq,
                                            nk,
                                            ng,
                                            strength,
                                            qf,
                                            kf,
                                            vu,
                                            yu,
                                            valid_t,
                                            valid_v,
                                            lane,
                                            column,
                                            False,
                                        )
                                else:
                                    advance(
                                        state,
                                        nq,
                                        nk,
                                        ng,
                                        strength,
                                        qf,
                                        kf,
                                        vu,
                                        yu,
                                        valid_t,
                                        valid_v,
                                        lane,
                                        column,
                                        False,
                                    )
                                for u in T.unroll(VPW):
                                    for j in T.unroll(RD):
                                        if lane + j * 32 < DK and column + u < valid_v:
                                            su[lane + j * 32, column + u] = state[u, j]
                        T.copy(
                            yu[0:valid_t, 0:valid_v],
                            output[row, t0 : t0 + valid_t, hv, voff : voff + valid_v],
                        )
                    if VECTOR:
                        with T.SimtVF(threads=256):
                            for j, c in T.Parallel(DK, BV):
                                if vfirst:
                                    final[seq, hv, voff + c, j] = su[c, j]
                                else:
                                    final[seq, hv, j, voff + c] = su[c, j]
                    elif STATE_DMA:
                        T.copy(su[0:DK, 0:valid_v], final[seq, hv, 0:DK, voff : voff + valid_v])
                    else:
                        with T.SimtVF(threads=256):
                            for j, c in T.Parallel(DK, BV):
                                if c < valid_v:
                                    if vfirst:
                                        final[seq, hv, voff + c, j] = su[j, c]
                                    else:
                                        final[seq, hv, j, voff + c] = su[j, c]

    return kernel


class AscendKDAFwdKernel(Kernel, KDAFwdInterface):
    """Run the Ascend KDA recurrence for the public inference interface."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: KDACall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: KDACall):
        index = call.device.index if call.device is not None else None
        super().__init__(device_index=index)
        self.call = call

    def forward(
        self,
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        A_log=None,
        dt_bias=None,
    ):
        # KDAFwdOp validates optional CPU offsets. The device program uses NPU offsets.
        del cu_seqlens_cpu
        call = self.call
        n, hv, dk, dv = call.sequences, call.value_heads, call.dim_k, call.dim_v
        state_shape = (n, hv, dv, dk) if call.state_v_first else (n, hv, dk, dv)
        output = torch.empty_like(v)
        final = torch.empty(state_shape, device=q.device, dtype=torch.float32)
        if not n:
            if call.seq_len:
                raise ValueError("nonempty packed tokens require at least one sequence")
            return output, final

        initial = initial_state
        if initial is None:
            initial = torch.empty_like(final)
        offsets = cu_seqlens
        if offsets is None:
            offsets = torch.empty(n + 1, device=q.device, dtype=torch.int64)
        rate = A_log
        if rate is None:
            rate = torch.empty(hv, device=q.device, dtype=torch.float32)
        bias = dt_bias
        if bias is None:
            bias = torch.empty(hv * dk, device=q.device, dtype=torch.float32)

        _kda_kernel(
            call.batch,
            call.seq_len,
            call.heads,
            hv,
            dk,
            dv,
            n,
            str(q.dtype).removeprefix("torch."),
            call.varlen,
            call.has_initial_state,
            call.state_v_first,
            call.l2norm,
            call.gate_in_kernel,
            call.has_dt_bias,
            call.lower_bound,
            call.beta_sigmoid,
            call.allow_neg_eigval,
            call.scale,
        )(q, k, v, g, beta, initial, offsets, rate, bias, output, final)
        return output, final


register_kernel_type("KDAFwdOp", "ascend", AscendKDAFwdKernel)
