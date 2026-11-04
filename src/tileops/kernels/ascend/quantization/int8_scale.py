"""INT8 scale and division helpers preserving float32 subnormal bit patterns."""

import tilelang.ascend.language as T


@T.macro
def int8_scale(peak, factor, tiny, normalized_scale):
    """Round amax / 127 once, including scales below the normal float32 range."""
    with T.SimtVF(threads=32):
        for j in T.Parallel(1):
            peak_bits = T.reinterpret(peak[j], "uint32")
            tiny[j] = 0
            if peak_bits > 0 and peak_bits < T.uint32(0x03FE0000):
                tiny[j] = 1
                exponent = peak_bits >> 23
                units = T.if_then_else(
                    exponent == 0,
                    peak_bits,
                    ((peak_bits & T.uint32(0x7FFFFF)) | T.uint32(0x800000))
                    << T.cast(T.max(T.cast(exponent, "int32") - 1, 0), "uint32"),
                )
                quotient = units // T.uint32(127)
                remainder = units % T.uint32(127)
                scale_bits = quotient + T.if_then_else(
                    remainder > T.uint32(63), T.uint32(1), T.uint32(0)
                )
                factor[j] = T.reinterpret(scale_bits, "float32")
                normalized_scale[j] = T.if_then_else(
                    scale_bits == T.uint32(0x800000),
                    T.float32(1),
                    T.reinterpret(scale_bits | T.uint32(0x3F800000), "float32") - T.float32(1),
                )
            else:
                factor[j] = T.if_then_else(peak[j] > 0, peak[j], T.float32(127))
    if tiny[0] == 0:
        with T.SimdVF():
            full = T.simd.pset(32)
            one = T.simd.pset(32, "PAT_VL1")
            scale = T.simd.vdiv(
                T.simd.vdup(factor[0], "float32"),
                T.simd.vdup(T.float32(127), "float32"),
                full,
                precision="exact",
            )
            T.simd.vsts(factor[0], scale, one, "ONEPT_B32")


@T.macro
def int8_divide(values, factor, tiny, normalized_scale, count: int):
    """Normalize both operands before division when the scale is subnormal."""
    if tiny[0] != 0:
        with T.SimtVF(threads=256):
            for j in T.Parallel(count):
                bits = T.reinterpret(values[j], "uint32")
                magnitude = bits & T.uint32(0x7FFFFFFF)
                normalized = T.if_then_else(
                    magnitude < T.uint32(0x800000),
                    T.reinterpret(magnitude | T.uint32(0x3F800000), "float32") - T.float32(1),
                    T.reinterpret(magnitude + T.uint32(0x3F000000), "float32"),
                )
                values[j] = T.reinterpret(
                    T.reinterpret(normalized, "uint32") | (bits & T.uint32(0x80000000)),
                    "float32",
                )
    with T.SimdVF():
        full = T.simd.pset(32)
        divisor = T.simd.alloc_local((1,), "float32")
        if tiny[0] != 0:
            divisor[0] = T.simd.vdup(normalized_scale[0], "float32")
        else:
            divisor[0] = T.simd.vdup(factor[0], "float32")
        for v in T.serial(count // 64):
            quotient = T.simd.vdiv(T.simd.vld(values[v * 64]), divisor[0], full, precision="exact")
            T.simd.vsts(values[v * 64], quotient, full)
