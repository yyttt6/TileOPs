"""Ascend floor division and remainder with residual-based rounding corrections."""

import functools

import torch

from tileops.kernels.ascend.elementwise.arithmetic import ArithmeticKernelBase
from tileops.kernels.elementwise.call_spec import BinaryElementwiseFwdInterface

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None


_NATIVE_FLOOR_PRELUDE = r"""#include <type_traits>
#include <c_api/asc_simd.h>
__simd_callee__ inline void w1_native_floor_inplace(vector_f32 &dst, vector_bool mask) {
    vtrc(dst, dst, std::integral_constant<::ROUND, ROUND_F>(), mask, MODE_ZEROING);
}
__simd_callee__ inline void w1_native_trunc_inplace(vector_f32 &dst, vector_bool mask) {
    vtrc(dst, dst, std::integral_constant<::ROUND, ROUND_Z>(), mask, MODE_ZEROING);
}
"""

if T is not None:

    @T.macro
    def _divide_const(
        av,
        bv,
        dtype: str,
        rounding_mode: str,
        mask,
        zero,
        one,
        negative_one,
        nan,
        scale_tiny: bool,
        scale_big,
    ):
        if (
            rounding_mode == "floor_raw"
            or rounding_mode == "floor_refined"
            or rounding_mode == "floor_slim"
            or rounding_mode == "floor_native"
        ):
            if rounding_mode == "floor_slim" or rounding_mode == "floor_native":
                if scale_tiny:
                    tiny_b = T.simd.vcmps(
                        T.simd.vabs(bv), T.float32(1.1754943508222875e-38), mask, op="lt"
                    )
                    scale = T.simd.vsel(scale_big, one, tiny_b)
                    # Only quotient operands are scaled. Original operands retain
                    # correct NaN/Inf classification even if finite a*scale overflows.
                    raw = T.simd.vdiv(
                        T.simd.vmul(av, scale, mask),
                        T.simd.vmul(bv, scale, mask),
                        mask,
                        precision="exact",
                    )
                else:
                    raw = T.simd.vdiv(av, bv, mask, precision="exact")
            else:
                raw = T.simd.vdiv(av, bv, mask, precision="ftz_true")
            if rounding_mode == "floor_refined":
                error = T.simd.alloc_local(1, "float32")
                error[0] = av
                T.simd.vmula(error[0], T.simd.vneg(raw), bv, mask)
                q = T.simd.vadd(raw, T.simd.vdiv(error[0], bv, mask, precision="ftz_true"), mask)
                exceptional = T.simd.por(
                    T.simd.vcmps(T.simd.vabs(raw), T.float32(float("inf")), mask, op="ge"),
                    T.simd.vcmps(raw, T.float32(0), mask, op="eq"),
                    mask,
                )
                q = T.simd.vsel(raw, q, exceptional)
            else:
                q = raw
        elif rounding_mode == "remainder_raw" or rounding_mode == "remainder_native":
            q = T.simd.vdiv(av, bv, mask, precision="ftz_true")
        elif dtype == "float32" or rounding_mode != "true":
            q = T.simd.vdiv(av, bv, mask, precision="exact")
        else:
            q = T.simd.vdiv(av, bv, mask, precision="ftz_true")
        if rounding_mode == "true":
            result = q
        else:
            if rounding_mode == "trunc":
                if dtype != "float32":
                    narrow = T.simd.vcvt(q, dtype)
                    rounded_q = T.simd.vcvt(narrow, "float32", part=0)
                else:
                    rounded_q = q
                whole = T.simd.vcvt(T.simd.vcvt(rounded_q, "int32", round="ROUND_Z"), "float32")
                small = T.simd.vcmps(T.simd.vabs(rounded_q), T.float32(8388608), mask, op="lt")
                result = T.simd.vsel(whole, rounded_q, small)
            elif (
                rounding_mode == "floor"
                or rounding_mode == "floor_raw"
                or rounding_mode == "floor_refined"
                or rounding_mode == "floor_slim"
                or rounding_mode == "floor_native"
            ):
                if rounding_mode == "floor_native":
                    native_whole = T.simd.alloc_local(1, "float32")
                    native_whole[0] = q
                    T.evaluate(
                        T.call_extern("void", "w1_native_floor_inplace", native_whole[0], mask)
                    )
                    whole = native_whole[0]
                else:
                    small = T.simd.vcmps(T.simd.vabs(q), T.float32(8388608), mask, op="lt")
                    whole = T.simd.vcvt(T.simd.vcvt(q, "int32", round="ROUND_F"), "float32")
                    whole = T.simd.vsel(whole, q, small)
                residual = T.simd.alloc_local(1, "float32")
                residual[0] = av
                T.simd.vmula(residual[0], T.simd.vneg(whole), bv, mask)
                opposite = T.simd.pxor(
                    T.simd.vcmp(residual[0], zero, mask, op="lt"),
                    T.simd.vcmp(bv, zero, mask, op="lt"),
                    mask,
                )
                nonzero = T.simd.vcmp(residual[0], zero, mask, op="ne")
                if rounding_mode == "floor_slim" or rounding_mode == "floor_native":
                    correct = T.simd.pand(opposite, nonzero, mask)
                else:
                    finite_b = T.simd.vcmps(T.simd.vabs(bv), T.float32(float("inf")), mask, op="lt")
                    correct = T.simd.pand(T.simd.pand(opposite, nonzero, mask), finite_b, mask)
                result = T.simd.vsub(whole, T.simd.vsel(one, zero, correct), mask)
                if rounding_mode == "floor_raw" or rounding_mode == "floor_refined":
                    # Exclude floorq=-1: a tiny negative quotient has a
                    # residual that can round to b. Other integer candidates
                    # can safely repair a rounded-down quotient from this
                    # residual, including a raw quotient just below -1.
                    eligible = T.simd.vcmp(whole, negative_one, mask, op="ne")
                    above = T.simd.vcmp(T.simd.vabs(residual[0]), T.simd.vabs(bv), mask, op="ge")
                    if rounding_mode == "floor_slim" or rounding_mode == "floor_native":
                        upward = T.simd.pand(
                            T.simd.pand(T.simd.pnot(opposite, mask), above, mask),
                            T.simd.pand(eligible, small, mask),
                            mask,
                        )
                    else:
                        upward = T.simd.pand(
                            T.simd.pand(T.simd.pnot(opposite, mask), above, mask),
                            T.simd.pand(T.simd.pand(eligible, small, mask), finite_b, mask),
                            mask,
                        )
                    result = T.simd.vadd(result, T.simd.vsel(one, zero, upward), mask)
                infinite_b = T.simd.vcmps(T.simd.vabs(bv), T.float32(float("inf")), mask, op="eq")
                if rounding_mode == "floor_slim" or rounding_mode == "floor_native":
                    # For an infinite divisor, a*b<0 means finite nonzero a
                    # with the opposite sign. Zero and NaN products compare false.
                    negative_product = T.simd.vcmps(
                        T.simd.vmul(av, bv, mask), T.float32(0), mask, op="lt"
                    )
                    infinite_result = T.simd.vsel(negative_one, q, negative_product)
                    result = T.simd.vsel(infinite_result, result, infinite_b)
                    # a-a is zero for finite a, and NaN for infinite/NaN a.
                    # The zero-divisor override preserves IEEE division there.
                    result = T.simd.vadd(result, T.simd.vsub(av, av, mask), mask)
                    result = T.simd.vsel(q, result, T.simd.vcmp(bv, zero, mask, op="eq"))
                else:
                    opposite_ab = T.simd.pxor(
                        T.simd.vcmp(av, zero, mask, op="lt"),
                        T.simd.vcmp(bv, zero, mask, op="lt"),
                        mask,
                    )
                    nonzero_a = T.simd.vcmp(av, zero, mask, op="ne")
                    infinite_result = T.simd.vsel(
                        negative_one, q, T.simd.pand(opposite_ab, nonzero_a, mask)
                    )
                    result = T.simd.vsel(infinite_result, result, infinite_b)
                    infinite_a = T.simd.vcmps(
                        T.simd.vabs(av), T.float32(float("inf")), mask, op="eq"
                    )
                    result = T.simd.vsel(nan, result, infinite_a)
                    result = T.simd.vsel(q, result, T.simd.vcmp(bv, zero, mask, op="eq"))
            else:
                if rounding_mode == "remainder_native":
                    native_trunc = T.simd.alloc_local(1, "float32")
                    native_trunc[0] = q
                    T.evaluate(
                        T.call_extern("void", "w1_native_trunc_inplace", native_trunc[0], mask)
                    )
                    whole = native_trunc[0]
                else:
                    whole = T.simd.vcvt(T.simd.vcvt(q, "int32", round="ROUND_Z"), "float32")
                    small = T.simd.vcmps(T.simd.vabs(q), T.float32(8388608), mask, op="lt")
                    whole = T.simd.vsel(whole, q, small)
                residual = T.simd.alloc_local(1, "float32")
                residual[0] = av
                T.simd.vmula(residual[0], T.simd.vneg(whole), bv, mask)
                # A rounded quotient can cross an integer (e.g. 1 / 0.1).
                opposite_a = T.simd.pxor(
                    T.simd.vcmp(residual[0], zero, mask, op="lt"),
                    T.simd.vcmp(av, zero, mask, op="lt"),
                    mask,
                )
                nonzero = T.simd.vcmp(residual[0], zero, mask, op="ne")
                signed_b = T.simd.vsel(
                    T.simd.vneg(T.simd.vabs(bv)),
                    T.simd.vabs(bv),
                    T.simd.vcmp(av, zero, mask, op="lt"),
                )
                rem = T.simd.vadd(
                    residual[0],
                    T.simd.vsel(signed_b, zero, T.simd.pand(opposite_a, nonzero, mask)),
                    mask,
                )
                oversized = T.simd.vcmp(T.simd.vabs(rem), T.simd.vabs(bv), mask, op="ge")
                rem = T.simd.vsub(rem, T.simd.vsel(signed_b, zero, oversized), mask)
                opposite_b = T.simd.pxor(
                    T.simd.vcmp(rem, zero, mask, op="lt"),
                    T.simd.vcmp(bv, zero, mask, op="lt"),
                    mask,
                )
                nonzero = T.simd.vcmp(rem, zero, mask, op="ne")
                result = T.simd.vadd(
                    rem, T.simd.vsel(bv, zero, T.simd.pand(opposite_b, nonzero, mask)), mask
                )
                infinite_b = T.simd.vcmps(T.simd.vabs(bv), T.float32(float("inf")), mask, op="eq")
                opposite_ab = T.simd.pxor(
                    T.simd.vcmp(av, zero, mask, op="lt"), T.simd.vcmp(bv, zero, mask, op="lt"), mask
                )
                nonzero_a = T.simd.vcmp(av, zero, mask, op="ne")
                inf_result = T.simd.vsel(bv, av, T.simd.pand(opposite_ab, nonzero_a, mask))
                result = T.simd.vsel(inf_result, result, infinite_b)
                infinite_a = T.simd.vcmps(T.simd.vabs(av), T.float32(float("inf")), mask, op="eq")
                invalid = T.simd.por(infinite_a, T.simd.vcmp(bv, zero, mask, op="eq"), mask)
                result = T.simd.vsel(nan, result, invalid)
            if rounding_mode != "floor_slim" and rounding_mode != "floor_native":
                operand_nan = T.simd.por(
                    T.simd.vcmp(av, av, mask, op="ne"), T.simd.vcmp(bv, bv, mask, op="ne"), mask
                )
                result = T.simd.vsel(q, result, operand_nan)
            # Conversion to integers loses signed zero. Recover only its sign.
            if (
                rounding_mode == "remainder"
                or rounding_mode == "remainder_raw"
                or rounding_mode == "remainder_native"
            ):
                signed_zero = T.simd.vmul(av, zero, mask)
            else:
                signed_zero = T.simd.vmul(q, zero, mask)
            result = T.simd.vsel(signed_zero, result, T.simd.vcmp(result, zero, mask, op="eq"))
        return result

    @T.macro
    def _divide(av, bv, dtype: str, rounding_mode: str, mask):
        zero = T.simd.vdup(T.float32(0), "float32", mask)
        one = T.simd.vdup(T.float32(1), "float32", mask)
        negative_one = T.simd.vdup(T.float32(-1), "float32", mask)
        if rounding_mode == "floor_native" or rounding_mode == "floor_slim":
            nan = zero  # This branch propagates NaNs arithmetically.
        else:
            nan = T.simd.vdup(T.float32(float("nan")), "float32", mask)
        return _divide_const(
            av, bv, dtype, rounding_mode, mask, zero, one, negative_one, nan, False, one
        )

    @T.macro
    def _repair_remainder(a, b, c, width: int, begin, period: int, repeat: int):
        # Large quotients and subnormal divisors use native fmodf in SIMT.
        # The exponent-bit test is immune to SIMD flush-to-zero behavior.
        # This repairs only exceptional lanes and does not stage data on the host.
        with T.SimtVF(threads=256):
            for j in T.Parallel(width, coalesced_width=T.int32(1)):
                repair_av = T.cast(a[j], "float32")
                if period:
                    repair_bv = T.cast(b[((begin + j) // repeat) % period], "float32")
                else:
                    repair_bv = T.cast(b[j], "float32")
                if T.Or(
                    T.abs(repair_av) >= T.abs(repair_bv) * T.float32(4194304),
                    (T.reinterpret(repair_bv, "uint32") & T.uint32(0x7FFFFFFF))
                    < T.uint32(0x00800000),
                ):
                    repair_rem = T.call_extern("float32", "fmodf", repair_av, repair_bv)
                    c[j] = T.if_then_else(
                        T.And(repair_rem != 0, (repair_rem < 0) != (repair_bv < 0)),
                        repair_rem + repair_bv,
                        repair_rem,
                    )

    @T.macro
    def _repair_channel(a, b, c, channels: int, repeat: int, b_begin, threads: int):
        # Constant divisor per channel; avoid per-element integer division/modulo.
        with T.SimtVF(threads=threads):
            for ch in range(channels):
                channel_bv = T.cast(b[b_begin + ch], "float32")
                for j in T.Parallel(repeat, coalesced_width=T.int32(1)):
                    channel_av = T.cast(a[ch * repeat + j], "float32")
                    if T.Or(
                        T.abs(channel_av) >= T.abs(channel_bv) * T.float32(4194304),
                        (T.reinterpret(channel_bv, "uint32") & T.uint32(0x7FFFFFFF))
                        < T.uint32(0x00800000),
                    ):
                        channel_rem = T.call_extern("float32", "fmodf", channel_av, channel_bv)
                        c[ch * repeat + j] = T.if_then_else(
                            T.And(channel_rem != 0, (channel_rem < 0) != (channel_bv < 0)),
                            channel_rem + channel_bv,
                            channel_rem,
                        )

    @T.macro
    def _tiny_floor_vectors(
        a, b, c, width: int, dtype: str, period: int, repeat: int, begin, channel_offset, kind: int
    ):
        # Run only when a block contains an FP32 subnormal (or zero) divisor.
        # Scaling is exact, and moves every nonzero subnormal into normal range.
        mask = T.simd.pset(32)
        scale = T.simd.vdup(T.float32(16777216), "float32", mask)
        for i in range(width // 64):
            av = (
                T.simd.vld(a[i * 64])
                if dtype == "float32"
                else T.simd.vcvt(T.simd.vld(a[i * 64], dist="UNPK_B16"), "float32", part=0)
            )
            original = (
                T.simd.vld(c[i * 64])
                if dtype == "float32"
                else T.simd.vcvt(T.simd.vld(c[i * 64], dist="UNPK_B16"), "float32", part=0)
            )
            if kind == 2:
                index = channel_offset + (i * 64) // repeat
                bv = (
                    T.simd.vld(b[index], dist="BRC_B32")
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(b[index], dist="BRC_B16"), "float32", part=0)
                )
            elif period and repeat > 1:
                index = ((begin + i * 64) // repeat) % period
                bv = (
                    T.simd.vld(b[index], dist="BRC_B32")
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(b[index], dist="BRC_B16"), "float32", part=0)
                )
            else:
                index = (begin + i * 64) % period if period else i * 64
                bv = (
                    T.simd.vld(b[index])
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(b[index], dist="UNPK_B16"), "float32", part=0)
                )
            scaled = _divide(
                T.simd.vmul(av, scale, mask),
                T.simd.vmul(bv, scale, mask),
                "float32",
                "floor_native",
                mask,
            )
            tiny = T.simd.vcmps(T.simd.vabs(bv), T.float32(1.1754943508222875e-38), mask, op="lt")
            safe = T.simd.vcmps(T.simd.vabs(av), T.float32(1.2676506002282294e30), mask, op="lt")
            result = T.simd.vsel(scaled, original, T.simd.pand(tiny, safe, mask))
            if dtype == "float32":
                T.simd.vsts(c[i * 64], result, mask)
            else:
                T.simd.vsts(c[i * 64], T.simd.vcvt(result, dtype), mask, dist="PK_B32")

    @T.macro
    def _body_general(
        a,
        b,
        c,
        width: int,
        begin,
        period: int,
        repeat: int,
        dtype: str,
        alpha: float,
        rounding_mode: str,
    ):
        if rounding_mode == "floor_native" and dtype == "float32":
            tiny_flag = T.alloc_shared((64,), "uint32")
        with T.SimdVF():
            mask = T.simd.pset(32)
            if rounding_mode == "floor_native" and dtype == "float32":
                min_divisor = T.simd.alloc_local(1, "uint32")
                min_divisor[0] = T.simd.vdup(T.uint32(0x7F800000), "uint32", mask)
            if rounding_mode == "floor_native" and dtype == "bfloat16":
                constant_zero = T.simd.vdup(T.float32(0), "float32", mask)
                constant_one = T.simd.vdup(T.float32(1), "float32", mask)
                constant_neg_one = T.simd.vdup(T.float32(-1), "float32", mask)
                constant_scale = T.simd.vdup(T.float32(16777216), "float32", mask)
            for i in range(width // 64):
                av = (
                    T.simd.vld(a[i * 64])
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(a[i * 64], dist="UNPK_B16"), "float32", part=0)
                )
                if period and repeat > 1:
                    index = ((begin + i * 64) // repeat) % period
                    bv = (
                        T.simd.vld(b[index], dist="BRC_B32")
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(b[index], dist="BRC_B16"), "float32", part=0)
                    )
                elif period:
                    index = (begin + i * 64) % period
                    bv = (
                        T.simd.vld(b[index])
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(b[index], dist="UNPK_B16"), "float32", part=0)
                    )
                else:
                    bv = (
                        T.simd.vld(b[i * 64])
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(b[i * 64], dist="UNPK_B16"), "float32", part=0)
                    )
                if rounding_mode == "floor_native" and dtype == "float32":
                    min_divisor[0] = T.simd.vmin(
                        min_divisor[0], T.reinterpret(T.simd.vabs(bv), "uint32x64"), mask
                    )
                if rounding_mode == "floor_native" and dtype == "bfloat16":
                    cv = _divide_const(
                        av,
                        bv,
                        dtype,
                        rounding_mode,
                        mask,
                        constant_zero,
                        constant_one,
                        constant_neg_one,
                        constant_zero,
                        True,
                        constant_scale,
                    )
                else:
                    cv = _divide(av, bv, dtype, rounding_mode, mask)
                if dtype == "float32":
                    T.simd.vsts(c[i * 64], cv, mask)
                else:
                    cv16 = T.simd.vcvt(cv, dtype)
                    T.simd.vsts(c[i * 64], cv16, mask, dist="PK_B32")
            if rounding_mode == "floor_native" and dtype == "float32":
                T.simd.vsts(tiny_flag[0], T.simd.vcmin(min_divisor[0], mask), mask)
            if rounding_mode == "floor_native" and dtype == "float32":
                T.simd.mem_bar("VST_LD")
                if tiny_flag[0] < T.uint32(0x00800000):
                    _tiny_floor_vectors(a, b, c, width, dtype, period, repeat, begin, 0, 0)

    @T.macro
    def _bf16_floor_body(a, b, c, width: int, begin, period: int, repeat: int):
        # BF16 positive quotients below256 cannot round across an integer in
        # precise FP32 division. Every other value takes the complete fallback.
        guard = T.alloc_shared((128,), "uint32")
        with T.SimdVF():
            mask = T.simd.pset(32)
            min_q = T.simd.alloc_local(1, "uint32")
            max_q = T.simd.alloc_local(1, "uint32")
            min_q[0] = T.simd.vdup(T.uint32(0x7F800000), "uint32", mask)
            max_q[0] = T.simd.vdup(T.uint32(0), "uint32", mask)
            for i in range(width // 64):
                av = T.simd.vcvt(T.simd.vld(a[i * 64], dist="UNPK_B16"), "float32", part=0)
                if period and repeat > 1:
                    index = ((begin + i * 64) // repeat) % period
                    bv = T.simd.vcvt(T.simd.vld(b[index], dist="BRC_B16"), "float32", part=0)
                else:
                    index = (begin + i * 64) % period if period else i * 64
                    bv = T.simd.vcvt(T.simd.vld(b[index], dist="UNPK_B16"), "float32", part=0)
                q = T.simd.vdiv(av, bv, mask, precision="exact")
                q_bits = T.reinterpret(q, "uint32x64")
                min_q[0] = T.simd.vmin(min_q[0], q_bits, mask)
                max_q[0] = T.simd.vmax(max_q[0], q_bits, mask)
                whole = T.simd.alloc_local(1, "float32")
                whole[0] = q
                T.evaluate(T.call_extern("void", "w1_native_floor_inplace", whole[0], mask))
                T.simd.vsts(c[i * 64], T.simd.vcvt(whole[0], "bfloat16"), mask, dist="PK_B32")
            T.simd.vsts(guard[0], T.simd.vcmin(min_q[0], mask), mask)
            T.simd.vsts(guard[64], T.simd.vcmax(max_q[0], mask), mask)
            T.simd.mem_bar("VST_LD")
            if T.Or(guard[0] == T.uint32(0), guard[64] >= T.uint32(0x43800000)):
                zero = T.simd.vdup(T.float32(0), "float32", mask)
                one = T.simd.vdup(T.float32(1), "float32", mask)
                neg_one = T.simd.vdup(T.float32(-1), "float32", mask)
                large_scale = T.simd.vdup(T.float32(16777216), "float32", mask)
                for j in range(width // 64):
                    fallback_av = T.simd.vcvt(
                        T.simd.vld(a[j * 64], dist="UNPK_B16"), "float32", part=0
                    )
                    if period and repeat > 1:
                        fallback_index = ((begin + j * 64) // repeat) % period
                        fallback_bv = T.simd.vcvt(
                            T.simd.vld(b[fallback_index], dist="BRC_B16"), "float32", part=0
                        )
                    else:
                        fallback_index = (begin + j * 64) % period if period else j * 64
                        fallback_bv = T.simd.vcvt(
                            T.simd.vld(b[fallback_index], dist="UNPK_B16"), "float32", part=0
                        )
                    result = _divide_const(
                        fallback_av,
                        fallback_bv,
                        "bfloat16",
                        "floor_native",
                        mask,
                        zero,
                        one,
                        neg_one,
                        zero,
                        True,
                        large_scale,
                    )
                    T.simd.vsts(c[j * 64], T.simd.vcvt(result, "bfloat16"), mask, dist="PK_B32")

    @T.macro
    def _body(
        a,
        b,
        c,
        width: int,
        begin,
        period: int,
        repeat: int,
        dtype: str,
        alpha: float,
        rounding_mode: str,
    ):
        if rounding_mode == "floor_native" and dtype == "bfloat16":
            _bf16_floor_body(a, b, c, width, begin, period, repeat)
        else:
            _body_general(a, b, c, width, begin, period, repeat, dtype, alpha, rounding_mode)

    @functools.lru_cache(maxsize=512)
    def build(
        n, dtype, period=0, repeat=1, alpha=1, tile=8192, blocks=72, stages=2, rounding_mode="true"
    ):
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        b_width = period or tile
        b_tail = period or tail

        @T.prim_func
        def main(
            A: T.Tensor((n,), dtype), B: T.Tensor((period or n,), dtype), C: T.Tensor((n,), dtype)
        ):
            with T.Kernel(
                grid,
                prelude=_NATIVE_FLOOR_PRELUDE
                if rounding_mode in ("floor_native", "remainder_native")
                else None,
            ) as bx:
                if full:
                    a = T.alloc_shared((tile,), dtype)
                    b = T.alloc_shared((b_width,), dtype)
                    c = T.alloc_shared((tile,), dtype)
                    T.annotate_buffer_versions({a: stages, c: stages})
                    if period:
                        T.copy(B, b)
                    else:
                        T.annotate_buffer_versions({b: stages})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(A[begin : begin + tile], a)
                            if not period:
                                T.copy(B[begin : begin + tile], b)
                            _body(a, b, c, tile, begin, period, repeat, dtype, alpha, rounding_mode)
                            if (
                                rounding_mode == "remainder"
                                or rounding_mode == "remainder_raw"
                                or rounding_mode == "remainder_native"
                            ):
                                _repair_remainder(a, b, c, tile, begin, period, repeat)
                            T.copy(c, C[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        at = T.alloc_shared((tail,), dtype)
                        bt = T.alloc_shared((b_tail,), dtype)
                        ct = T.alloc_shared((tail,), dtype)
                        T.copy(A[full * tile : n], at)
                        if period:
                            T.copy(B, bt)
                        else:
                            T.copy(B[full * tile : n], bt)
                        _body(
                            at,
                            bt,
                            ct,
                            tail,
                            full * tile,
                            period,
                            repeat,
                            dtype,
                            alpha,
                            rounding_mode,
                        )
                        if (
                            rounding_mode == "remainder"
                            or rounding_mode == "remainder_raw"
                            or rounding_mode == "remainder_native"
                        ):
                            _repair_remainder(at, bt, ct, tail, full * tile, period, repeat)
                        T.copy(ct, C[full * tile : n], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @T.macro
    def _channel_body(
        a, b, c, channels: int, repeat: int, dtype: str, alpha: float, b_begin, rounding_mode: str
    ):
        if rounding_mode == "floor_native" and dtype != "float16":
            tiny_flag = T.alloc_shared((64,), "uint32")
        with T.SimdVF():
            mask = T.simd.pset(32)
            if rounding_mode == "floor_native" and dtype != "float16":
                min_divisor = T.simd.alloc_local(1, "uint32")
                min_divisor[0] = T.simd.vdup(T.uint32(0x7F800000), "uint32", mask)
            for ch in range(channels):
                bv = (
                    T.simd.vld(b[b_begin + ch], dist="BRC_B32")
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(b[b_begin + ch], dist="BRC_B16"), "float32", part=0)
                )
                if rounding_mode == "floor_native" and dtype != "float16":
                    min_divisor[0] = T.simd.vmin(
                        min_divisor[0], T.reinterpret(T.simd.vabs(bv), "uint32x64"), mask
                    )
                for i in range(repeat // 64):
                    index = ch * repeat + i * 64
                    av = (
                        T.simd.vld(a[index])
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(a[index], dist="UNPK_B16"), "float32", part=0)
                    )
                    cv = _divide(av, bv, dtype, rounding_mode, mask)
                    if dtype == "float32":
                        T.simd.vsts(c[index], cv, mask)
                    else:
                        cv16 = T.simd.vcvt(cv, dtype)
                        T.simd.vsts(c[index], cv16, mask, dist="PK_B32")
            if rounding_mode == "floor_native" and dtype != "float16":
                T.simd.vsts(tiny_flag[0], T.simd.vcmin(min_divisor[0], mask), mask)
            if rounding_mode == "floor_native" and dtype != "float16":
                T.simd.mem_bar("VST_LD")
                if tiny_flag[0] < T.uint32(0x00800000):
                    _tiny_floor_vectors(a, b, c, channels * repeat, dtype, 0, repeat, 0, b_begin, 2)

    @functools.lru_cache(maxsize=512)
    def build_channel(
        n,
        dtype,
        period,
        repeat,
        alpha=1,
        tile=4,
        blocks=72,
        stages=2,
        rounding_mode="true",
        repair_threads=256,
    ):
        width = tile * repeat
        count = n // width
        grid = min(blocks, count)
        assert repeat % 64 == 0 and period % tile == 0 and n % width == 0

        @T.prim_func
        def main(A: T.Tensor((n,), dtype), B: T.Tensor((period,), dtype), C: T.Tensor((n,), dtype)):
            with T.Kernel(
                grid,
                prelude=_NATIVE_FLOOR_PRELUDE
                if rounding_mode in ("floor_native", "remainder_native")
                else None,
            ) as bx:
                a = T.alloc_shared((width,), dtype)
                b = T.alloc_shared((period,), dtype)
                c = T.alloc_shared((width,), dtype)
                T.annotate_buffer_versions({a: stages, c: stages})
                T.copy(B, b)
                for step in T.Pipelined((count + grid - 1) // grid, num_stages=stages):
                    block = step * grid + bx
                    if block < count:
                        begin = block * width
                        T.copy(A[begin : begin + width], a)
                        _channel_body(
                            a,
                            b,
                            c,
                            tile,
                            repeat,
                            dtype,
                            alpha,
                            (block * tile) % period,
                            rounding_mode,
                        )
                        if (
                            rounding_mode == "remainder"
                            or rounding_mode == "remainder_raw"
                            or rounding_mode == "remainder_native"
                        ):
                            _repair_channel(
                                a, b, c, tile, repeat, (block * tile) % period, repair_threads
                            )
                        T.copy(c, C[begin : begin + width], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @T.macro
    def _bias_body(a, b, c, rows: int, period: int, dtype: str, alpha: float, rounding_mode: str):
        if rounding_mode == "floor_native" and dtype != "float16":
            tiny_flag = T.alloc_shared((64,), "uint32")
        with T.SimdVF():
            mask = T.simd.pset(32)
            if rounding_mode == "floor_native" and dtype != "float16":
                min_divisor = T.simd.alloc_local(1, "uint32")
                min_divisor[0] = T.simd.vdup(T.uint32(0x7F800000), "uint32", mask)
            for row in range(rows):
                for i in range(period // 64):
                    index = row * period + i * 64
                    av = (
                        T.simd.vld(a[index])
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(a[index], dist="UNPK_B16"), "float32", part=0)
                    )
                    bv = (
                        T.simd.vld(b[i * 64])
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(b[i * 64], dist="UNPK_B16"), "float32", part=0)
                    )
                    if rounding_mode == "floor_native" and dtype != "float16":
                        min_divisor[0] = T.simd.vmin(
                            min_divisor[0], T.reinterpret(T.simd.vabs(bv), "uint32x64"), mask
                        )
                    cv = _divide(av, bv, dtype, rounding_mode, mask)
                    if dtype == "float32":
                        T.simd.vsts(c[index], cv, mask)
                    else:
                        cv16 = T.simd.vcvt(cv, dtype)
                        T.simd.vsts(c[index], cv16, mask, dist="PK_B32")
            if rounding_mode == "floor_native" and dtype != "float16":
                T.simd.vsts(tiny_flag[0], T.simd.vcmin(min_divisor[0], mask), mask)
            if rounding_mode == "floor_native" and dtype != "float16":
                T.simd.mem_bar("VST_LD")
                if tiny_flag[0] < T.uint32(0x00800000):
                    _tiny_floor_vectors(a, b, c, rows * period, dtype, period, 1, 0, 0, 1)

    @functools.lru_cache(maxsize=256)
    def build_bias(n, dtype, period, alpha=1, rows=2, blocks=64, stages=2, rounding_mode="true"):
        width = rows * period
        count = n // width
        grid = min(blocks, count)
        assert n % width == 0 and period % 64 == 0

        @T.prim_func
        def main(A: T.Tensor((n,), dtype), B: T.Tensor((period,), dtype), C: T.Tensor((n,), dtype)):
            with T.Kernel(
                grid,
                prelude=_NATIVE_FLOOR_PRELUDE
                if rounding_mode in ("floor_native", "remainder_native")
                else None,
            ) as bx:
                a = T.alloc_shared((width,), dtype)
                b = T.alloc_shared((period,), dtype)
                c = T.alloc_shared((width,), dtype)
                T.annotate_buffer_versions({a: stages, c: stages})
                T.copy(B, b)
                for step in T.Pipelined((count + grid - 1) // grid, num_stages=stages):
                    block = step * grid + bx
                    if block < count:
                        begin = block * width
                        T.copy(A[begin : begin + width], a)
                        _bias_body(a, b, c, rows, period, dtype, alpha, rounding_mode)
                        if (
                            rounding_mode == "remainder"
                            or rounding_mode == "remainder_raw"
                            or rounding_mode == "remainder_native"
                        ):
                            _repair_remainder(a, b, c, width, begin, period, 1)
                        T.copy(c, C[begin : begin + width], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @T.macro
    def _widen_inputs(a, b, af, bf, width: int, dtype: str):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for i in range(width // 64):
                av = T.simd.vcvt(T.simd.vld(a[i * 64], dist="UNPK_B16"), "float32", part=0)
                bv = T.simd.vcvt(T.simd.vld(b[i * 64], dist="UNPK_B16"), "float32", part=0)
                T.simd.vsts(af[i * 64], av, mask)
                T.simd.vsts(bf[i * 64], bv, mask)

    @T.macro
    def _narrow_output(cf, c, width: int, dtype: str):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for i in range(width // 64):
                cv = T.simd.vcvt(T.simd.vld(cf[i * 64]), dtype)
                T.simd.vsts(c[i * 64], cv, mask, dist="PK_B32")

    @functools.lru_cache(maxsize=256)
    def build_widened(n, dtype, tile=4096, blocks=64, stages=2, rounding_mode="floor_refined"):
        """Separate exact conversion VFs from the homogeneous FP32 arithmetic VF."""
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        assert dtype in ("float16", "bfloat16") and n % 64 == 0
        assert (full > 0) * (6 * tile * stages + 12 * tile) + 18 * tail <= 248 * 1024

        @T.prim_func
        def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
            with T.Kernel(
                grid,
                prelude=_NATIVE_FLOOR_PRELUDE
                if rounding_mode in ("floor_native", "remainder_native")
                else None,
            ) as bx:
                if full:
                    a = T.alloc_shared((tile,), dtype)
                    b = T.alloc_shared((tile,), dtype)
                    c = T.alloc_shared((tile,), dtype)
                    af = T.alloc_shared((tile,), "float32")
                    bf = T.alloc_shared((tile,), "float32")
                    cf = T.alloc_shared((tile,), "float32")
                    T.annotate_buffer_versions(
                        {a: stages, b: stages, c: stages, af: 1, bf: 1, cf: 1}
                    )
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(A[begin : begin + tile], a)
                            T.copy(B[begin : begin + tile], b)
                            _widen_inputs(a, b, af, bf, tile, dtype)
                            _body(af, bf, cf, tile, 0, 0, 1, "float32", 1, rounding_mode)
                            if (
                                rounding_mode == "remainder"
                                or rounding_mode == "remainder_raw"
                                or rounding_mode == "remainder_native"
                            ):
                                _repair_remainder(af, bf, cf, tile, 0, 0, 1)
                            _narrow_output(cf, c, tile, dtype)
                            T.copy(c, C[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        at = T.alloc_shared((tail,), dtype)
                        bt = T.alloc_shared((tail,), dtype)
                        ct = T.alloc_shared((tail,), dtype)
                        aft = T.alloc_shared((tail,), "float32")
                        bft = T.alloc_shared((tail,), "float32")
                        cft = T.alloc_shared((tail,), "float32")
                        T.copy(A[full * tile : n], at)
                        T.copy(B[full * tile : n], bt)
                        _widen_inputs(at, bt, aft, bft, tail, dtype)
                        _body(aft, bft, cft, tail, 0, 0, 1, "float32", 1, rounding_mode)
                        if (
                            rounding_mode == "remainder"
                            or rounding_mode == "remainder_raw"
                            or rounding_mode == "remainder_native"
                        ):
                            _repair_remainder(aft, bft, cft, tail, 0, 0, 1)
                        _narrow_output(cft, ct, tail, dtype)
                        T.copy(ct, C[full * tile : n], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @T.macro
    def _widen_input(a, af, width: int):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for i in range(width // 64):
                av = T.simd.vcvt(T.simd.vld(a[i * 64], dist="UNPK_B16"), "float32", part=0)
                T.simd.vsts(af[i * 64], av, mask)

    @functools.lru_cache(maxsize=256)
    def build_widened_bias(
        n, dtype, period, tile=4096, blocks=64, stages=2, rounding_mode="floor_refined"
    ):
        """Retain the widened bias on chip, and convert each input/output tile once."""
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        assert dtype in ("float16", "bfloat16") and n % 64 == 0 and period % 64 == 0
        assert (full > 0) * (4 * tile * stages + 8 * tile) + 6 * period + 12 * tail <= 248 * 1024

        @T.prim_func
        def main(A: T.Tensor((n,), dtype), B: T.Tensor((period,), dtype), C: T.Tensor((n,), dtype)):
            with T.Kernel(
                grid,
                prelude=_NATIVE_FLOOR_PRELUDE
                if rounding_mode in ("floor_native", "remainder_native")
                else None,
            ) as bx:
                b = T.alloc_shared((period,), dtype)
                bf = T.alloc_shared((period,), "float32")
                T.copy(B, b)
                _widen_input(b, bf, period)
                if full:
                    a = T.alloc_shared((tile,), dtype)
                    c = T.alloc_shared((tile,), dtype)
                    af = T.alloc_shared((tile,), "float32")
                    cf = T.alloc_shared((tile,), "float32")
                    T.annotate_buffer_versions({a: stages, c: stages, af: 1, cf: 1})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(A[begin : begin + tile], a)
                            _widen_input(a, af, tile)
                            _body(af, bf, cf, tile, begin, period, 1, "float32", 1, rounding_mode)
                            if (
                                rounding_mode == "remainder"
                                or rounding_mode == "remainder_raw"
                                or rounding_mode == "remainder_native"
                            ):
                                _repair_remainder(af, bf, cf, tile, begin, period, 1)
                            _narrow_output(cf, c, tile, dtype)
                            T.copy(c, C[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        at = T.alloc_shared((tail,), dtype)
                        ct = T.alloc_shared((tail,), dtype)
                        aft = T.alloc_shared((tail,), "float32")
                        cft = T.alloc_shared((tail,), "float32")
                        T.copy(A[full * tile : n], at)
                        _widen_input(at, aft, tail)
                        _body(
                            aft, bf, cft, tail, full * tile, period, 1, "float32", 1, rounding_mode
                        )
                        if (
                            rounding_mode == "remainder"
                            or rounding_mode == "remainder_raw"
                            or rounding_mode == "remainder_native"
                        ):
                            _repair_remainder(aft, bf, cft, tail, full * tile, period, 1)
                        _narrow_output(cft, ct, tail, dtype)
                        T.copy(ct, C[full * tile : n], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=256)
    def build_widened_channel(
        n,
        dtype,
        period,
        repeat,
        tile=4,
        blocks=64,
        stages=2,
        rounding_mode="floor_refined",
        repair_threads=256,
    ):
        width = tile * repeat
        count = n // width
        grid = min(blocks, count)
        assert dtype in ("float16", "bfloat16") and period % 64 == 0
        assert repeat % 64 == 0 and period % tile == 0 and n % width == 0
        assert 4 * width * stages + 8 * width + 6 * period <= 248 * 1024

        @T.prim_func
        def main(A: T.Tensor((n,), dtype), B: T.Tensor((period,), dtype), C: T.Tensor((n,), dtype)):
            with T.Kernel(
                grid,
                prelude=_NATIVE_FLOOR_PRELUDE
                if rounding_mode in ("floor_native", "remainder_native")
                else None,
            ) as bx:
                a = T.alloc_shared((width,), dtype)
                b = T.alloc_shared((period,), dtype)
                c = T.alloc_shared((width,), dtype)
                af = T.alloc_shared((width,), "float32")
                bf = T.alloc_shared((period,), "float32")
                cf = T.alloc_shared((width,), "float32")
                T.annotate_buffer_versions({a: stages, c: stages, af: 1, bf: 1, cf: 1})
                T.copy(B, b)
                _widen_input(b, bf, period)
                for step in T.Pipelined((count + grid - 1) // grid, num_stages=stages):
                    block = step * grid + bx
                    if block < count:
                        begin = block * width
                        T.copy(A[begin : begin + width], a)
                        _widen_input(a, af, width)
                        _channel_body(
                            af,
                            bf,
                            cf,
                            tile,
                            repeat,
                            "float32",
                            1,
                            (block * tile) % period,
                            rounding_mode,
                        )
                        if (
                            rounding_mode == "remainder"
                            or rounding_mode == "remainder_raw"
                            or rounding_mode == "remainder_native"
                        ):
                            _repair_channel(
                                af, bf, cf, tile, repeat, (block * tile) % period, repair_threads
                            )
                        _narrow_output(cf, c, width, dtype)
                        T.copy(c, C[begin : begin + width], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=256)
    def build_widened_rows(n, dtype, period, blocks=64, stages=2, rounding_mode="floor_slim"):
        width = period
        count = n // width
        grid = min(blocks, count)
        assert dtype in ("float16", "bfloat16") and period % 64 == 0 and n % period == 0
        assert (4 * stages + 14) * period <= 248 * 1024

        @T.prim_func
        def main(A: T.Tensor((n,), dtype), B: T.Tensor((period,), dtype), C: T.Tensor((n,), dtype)):
            with T.Kernel(
                grid,
                prelude=_NATIVE_FLOOR_PRELUDE
                if rounding_mode in ("floor_native", "remainder_native")
                else None,
            ) as bx:
                a = T.alloc_shared((width,), dtype)
                b = T.alloc_shared((period,), dtype)
                c = T.alloc_shared((width,), dtype)
                af = T.alloc_shared((width,), "float32")
                bf = T.alloc_shared((period,), "float32")
                cf = T.alloc_shared((width,), "float32")
                T.annotate_buffer_versions({a: stages, c: stages, af: 1, bf: 1, cf: 1})
                T.copy(B, b)
                _widen_input(b, bf, period)
                for step in T.Pipelined((count + grid - 1) // grid, num_stages=stages):
                    block = step * grid + bx
                    if block < count:
                        begin = block * width
                        T.copy(A[begin : begin + width], a)
                        _widen_input(a, af, width)
                        _bias_body(af, bf, cf, 1, period, "float32", 1, rounding_mode)
                        if (
                            rounding_mode == "remainder"
                            or rounding_mode == "remainder_raw"
                            or rounding_mode == "remainder_native"
                        ):
                            _repair_remainder(af, bf, cf, width, begin, period, 1)
                        _narrow_output(cf, c, width, dtype)
                        T.copy(c, C[begin : begin + width], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @T.macro
    def _simt_body(
        a,
        b,
        c,
        width: int,
        begin,
        period: int,
        repeat: int,
        dtype: str,
        rounding_mode: str,
        threads: int,
    ):
        with T.SimtVF(threads=threads):
            for j in T.Parallel(width):
                av = T.cast(a[j], "float32")
                if period:
                    bv = T.cast(b[((begin + j) // repeat) % period], "float32")
                else:
                    bv = T.cast(b[j], "float32")
                if (
                    rounding_mode == "remainder"
                    or rounding_mode == "remainder_raw"
                    or rounding_mode == "remainder_native"
                ):
                    native = T.call_extern("float32", "fmodf", av, bv)
                    rem = T.if_then_else(T.abs(bv) == T.float32(float("inf")), av, native)
                    signed = T.if_then_else(T.And(rem != 0, (rem < 0) != (bv < 0)), rem + bv, rem)
                    result = T.if_then_else(
                        T.Or(T.abs(av) == T.float32(float("inf")), bv == 0),
                        T.float32(float("nan")),
                        signed,
                    )
                else:
                    q = av / bv
                    whole = T.call_extern("float32", "floorf", q)
                    residual = T.call_extern("float32", "fmaf", -whole, bv, av)
                    down = T.And(residual != 0, (residual < 0) != (bv < 0))
                    lower = T.if_then_else(down, whole - 1, whole)
                    up = T.And(
                        T.And(whole != -1, T.abs(q) < T.float32(8388608)),
                        T.And(T.abs(residual) >= T.abs(bv), (residual < 0) == (bv < 0)),
                    )
                    rounded = T.if_then_else(up, lower + 1, lower)
                    infinite_divisor = T.if_then_else(av * bv < 0, T.float32(-1), q)
                    with_divisor = T.if_then_else(
                        T.abs(bv) == T.float32(float("inf")), infinite_divisor, rounded
                    )
                    result = T.if_then_else(
                        T.And(T.abs(av) == T.float32(float("inf")), bv != 0),
                        T.float32(float("nan")),
                        with_divisor,
                    )
                c[j] = T.cast(result, dtype)

    @functools.lru_cache(maxsize=512)
    def build_simt(
        n,
        dtype,
        period=0,
        repeat=1,
        tile=8192,
        blocks=64,
        stages=3,
        rounding_mode="floor_slim",
        threads=512,
    ):
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        b_width, b_tail = period or tile, period or tail
        size = 4 if dtype == "float32" else 2
        assert (
            (full > 0) * ((2 * tile + (0 if period else tile)) * stages + (period if period else 0))
            + (2 * tail + (period or tail) if tail else 0)
        ) * size <= 248 * 1024

        @T.prim_func
        def main(
            A: T.Tensor((n,), dtype), B: T.Tensor((period or n,), dtype), C: T.Tensor((n,), dtype)
        ):
            with T.Kernel(grid) as bx:
                if full:
                    a = T.alloc_shared((tile,), dtype)
                    b = T.alloc_shared((b_width,), dtype)
                    c = T.alloc_shared((tile,), dtype)
                    T.annotate_buffer_versions({a: stages, c: stages})
                    if period:
                        T.copy(B, b)
                    else:
                        T.annotate_buffer_versions({b: stages})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(A[begin : begin + tile], a)
                            if not period:
                                T.copy(B[begin : begin + tile], b)
                            _simt_body(
                                a, b, c, tile, begin, period, repeat, dtype, rounding_mode, threads
                            )
                            T.copy(c, C[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        at = T.alloc_shared((tail,), dtype)
                        bt = T.alloc_shared((b_tail,), dtype)
                        ct = T.alloc_shared((tail,), dtype)
                        T.copy(A[full * tile : n], at)
                        if period:
                            T.copy(B, bt)
                        else:
                            T.copy(B[full * tile : n], bt)
                        _simt_body(
                            at,
                            bt,
                            ct,
                            tail,
                            full * tile,
                            period,
                            repeat,
                            dtype,
                            rounding_mode,
                            threads,
                        )
                        T.copy(ct, C[full * tile : n], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)


class _ModuloKernel(ArithmeticKernelBase, BinaryElementwiseFwdInterface):
    repair_threads = 256
    channel_tile = 4
    channel_tile_fp32 = 4

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.a_shape, call.b_shape, call.dtype)

    def _compile(self, dtype_name, tile, stages):
        # Native remainder repair also needs static kernel shared memory.
        # Leave headroom beyond the dynamically allocated tile buffers.
        ub_budget = (224 if self.rounding_mode.startswith("remainder") else 248) * 1024
        # Dedicated conversions avoid the measured mixed-dtype VF cost.
        if self.dtype.itemsize == 2:
            if not self.period:
                if self.rounding_mode.startswith("floor"):
                    return build(
                        self.padded,
                        dtype_name,
                        0,
                        1,
                        1,
                        8192,
                        64,
                        2 if dtype_name == "bfloat16" else 3,
                        self.rounding_mode,
                    )
                return build_widened(self.padded, dtype_name, 4096, 64, 2, self.rounding_mode)
            if self.repeat == 1:
                if not (self.period & (self.period - 1)):
                    return build_widened_bias(
                        self.padded, dtype_name, self.period, 4096, 64, 2, self.rounding_mode
                    )
                row_stages = 2 if 22 * self.period <= ub_budget else 1
                if (4 * row_stages + 14) * self.period <= ub_budget:
                    return build_widened_rows(
                        self.n, dtype_name, self.period, 64, row_stages, self.rounding_mode
                    )
            elif self.period % 64 == 0:
                channels = self.channel_tile if dtype_name != "float32" else self.channel_tile_fp32
                while channels > 1 and 16 * channels * self.repeat + 6 * self.period > ub_budget:
                    channels //= 2
                if 16 * channels * self.repeat + 6 * self.period <= ub_budget:
                    return build_widened_channel(
                        self.padded,
                        dtype_name,
                        self.period,
                        self.repeat,
                        channels,
                        64,
                        2,
                        self.rounding_mode,
                        self.repair_threads,
                    )
        if dtype_name == "float32" and not self.period:
            tile, stages = (4096, 2) if self.rounding_mode.startswith("floor") else (2048, 3)
        if self.period and self.repeat == 1 and self.period & (self.period - 1):
            rows = 2 if self.dtype.itemsize == 2 and self.n % (2 * self.period) == 0 else 1
            pipeline = 2
            while (
                rows * self.period * 2 * pipeline + self.period
            ) * self.dtype.itemsize > ub_budget:
                if rows > 1:
                    rows //= 2
                else:
                    pipeline = 1
            return build_bias(
                self.n, dtype_name, self.period, 1, rows, 64, pipeline, self.rounding_mode or "true"
            )
        channels = self.channel_tile if dtype_name != "float32" else self.channel_tile_fp32
        if (
            self.repeat > 1
            and self.period % channels == 0
            and (channels * self.repeat * 4 + self.period) * self.dtype.itemsize <= ub_budget
        ):
            return build_channel(
                self.padded,
                dtype_name,
                self.period,
                self.repeat,
                1,
                channels,
                64,
                2,
                self.rounding_mode or "true",
                self.repair_threads,
            )
        return build(
            self.padded,
            dtype_name,
            self.period,
            self.repeat,
            1,
            tile,
            64,
            stages,
            self.rounding_mode or "true",
        )

    def forward(self, input, other):
        # Integer promotion remains outside the floating performance manifest.
        if not self.floating:
            return (
                torch.floor_divide if self.rounding_mode.startswith("floor") else torch.remainder
            )(input, other)
        return super().forward(input, other)


class AscendFloorDivideKernel(_ModuloKernel):
    rounding_mode = "floor"

    def _compile(self, dtype_name, tile, stages):
        # FloorDivide and Div(rounding_mode="floor") share the same CUDA
        # arithmetic contract, including the large-quotient and tiny-input tiers.
        from tileops.kernels.ascend.elementwise._floor_divide import FloorDivideImplementation

        return FloorDivideImplementation._compile(self, dtype_name, tile, stages)


class AscendRemainderKernel(_ModuloKernel):
    rounding_mode = "remainder_native"
    repair_threads = 512
    channel_tile_fp32 = 2


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("FloorDivideFwdOp", "ascend", AscendFloorDivideKernel)
    register_kernel_type("RemainderFwdOp", "ascend", AscendRemainderKernel)
