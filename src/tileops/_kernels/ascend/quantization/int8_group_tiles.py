"""Batch INT8 token groups with exact scales, including subnormal values."""

import functools

import tilelang
import tilelang.ascend.language as T


@functools.lru_cache(maxsize=64)
def build_group_tiles(rows: int, k: int, dtype: str):
    groups_per_row = (k + 127) // 128
    groups = rows * groups_per_row
    chunks = (groups_per_row + 63) // 64
    tiles = (groups + 63) // 64 if k % 128 == 0 else rows * chunks
    blocks = min(64, tiles)

    @T.prim_func
    def main(
        X: T.Tensor((rows * k,), dtype),
        Q: T.Tensor((rows * k,), "int8"),
        Scale: T.Tensor((groups,), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            raw = T.alloc_shared((8192,), dtype)
            values = T.alloc_shared((8192,), "float32")
            codes = T.alloc_shared((8192,), "int8")
            peaks = T.alloc_shared((64,), "float32")
            factors = T.alloc_shared((64,), "float32")
            normalized = T.alloc_shared((64,), "float32")
            tiny = T.alloc_shared((64,), "int32")
            for step in T.serial(T.ceildiv(tiles, blocks)):
                task = step * blocks + bx
                if task < tiles:
                    begin = (
                        task * 64
                        if k % 128 == 0
                        else task // chunks * groups_per_row + task % chunks * 64
                    )
                    count = (
                        T.min(64, groups - begin)
                        if k % 128 == 0
                        else T.min(64, groups_per_row - task % chunks * 64)
                    )
                    input_begin = (
                        begin * 128 if k % 128 == 0 else task // chunks * k + task % chunks * 8192
                    )
                    element_count = (
                        count * 128 if k % 128 == 0 else T.min(8192, k - task % chunks * 8192)
                    )
                    T.copy(X[input_begin : input_begin + element_count], raw[:element_count])
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        for v in T.serial(count * 2):
                            if dtype == "float32":
                                value = T.simd.vld(raw[v * 64])
                            else:
                                value = T.simd.vcvt(
                                    T.simd.vld(raw[v * 64], dist="UNPK_B16"), "float32", part=0
                                )
                            valid = T.simd.vcmps(
                                T.simd.vci(v * 64, "int32"), element_count, full, op="lt"
                            )
                            T.simd.vsts(
                                values[v * 64],
                                T.simd.vsel(value, T.simd.vdup(T.float32(0), "float32"), valid),
                                full,
                            )
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        for group in T.serial(count):
                            left = T.reinterpret(
                                T.simd.vabs(T.simd.vld(values[group * 128]), full), "int32x64"
                            )
                            right = T.reinterpret(
                                T.simd.vabs(T.simd.vld(values[group * 128 + 64]), full), "int32x64"
                            )
                            peak = T.simd.vmax(
                                T.simd.vcmax(left, full), T.simd.vcmax(right, full), full
                            )
                            T.simd.vsts(
                                peaks[group], T.reinterpret(peak, "float32x64"), one, "ONEPT_B32"
                            )
                    with T.SimtVF(threads=64):
                        for group in T.Parallel(64):
                            tiny[group] = 0
                            normalized[group] = T.float32(1)
                            factors[group] = T.float32(127)
                            if group < count:
                                bits = T.reinterpret(peaks[group], "uint32")
                                if bits > 0 and bits < T.uint32(0x03FE0000):
                                    tiny[group] = 1
                                    exponent = bits >> 23
                                    units = T.if_then_else(
                                        exponent == 0,
                                        bits,
                                        ((bits & T.uint32(0x7FFFFF)) | T.uint32(0x800000))
                                        << T.cast(
                                            T.max(T.cast(exponent, "int32") - 1, 0), "uint32"
                                        ),
                                    )
                                    rounded = units // T.uint32(127) + T.if_then_else(
                                        units % T.uint32(127) > T.uint32(63),
                                        T.uint32(1),
                                        T.uint32(0),
                                    )
                                    factors[group] = T.reinterpret(rounded, "float32")
                                    normalized[group] = T.if_then_else(
                                        rounded == T.uint32(0x800000),
                                        T.float32(1),
                                        T.reinterpret(rounded | T.uint32(0x3F800000), "float32")
                                        - T.float32(1),
                                    )
                                else:
                                    factors[group] = T.if_then_else(
                                        peaks[group] > 0, peaks[group], T.float32(127)
                                    )
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        original = T.simd.vld(factors[0])
                        divided = T.simd.vdiv(
                            original,
                            T.simd.vdup(T.float32(127), "float32"),
                            full,
                            precision="exact",
                        )
                        is_tiny = T.simd.vcmps(T.simd.vld(tiny[0]), T.int32(0), full, op="ne")
                        T.simd.vsts(factors[0], T.simd.vsel(original, divided, is_tiny), full)
                    with T.SimtVF(threads=256):
                        for j in T.Parallel(8192):
                            if j // 128 < count and tiny[j // 128] != 0:
                                bits = T.reinterpret(values[j], "uint32")
                                magnitude = bits & T.uint32(0x7FFFFFFF)
                                value = T.if_then_else(
                                    magnitude < T.uint32(0x800000),
                                    T.reinterpret(magnitude | T.uint32(0x3F800000), "float32")
                                    - T.float32(1),
                                    T.reinterpret(magnitude + T.uint32(0x3F000000), "float32"),
                                )
                                values[j] = T.reinterpret(
                                    T.reinterpret(value, "uint32") | (bits & T.uint32(0x80000000)),
                                    "float32",
                                )
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        divisor = T.simd.alloc_local((1,), "float32")
                        for group in T.serial(count):
                            if tiny[group] != 0:
                                divisor[0] = T.simd.vdup(normalized[group], "float32")
                            else:
                                divisor[0] = T.simd.vdup(factors[group], "float32")
                            for part in T.unroll(2):
                                index = group * 128 + part * 64
                                quotient = T.simd.vdiv(
                                    T.simd.vld(values[index]), divisor[0], full, precision="exact"
                                )
                                clipped = T.simd.vmin(
                                    T.simd.vmax(
                                        quotient, T.simd.vdup(T.float32(-127), "float32"), full
                                    ),
                                    T.simd.vdup(T.float32(127), "float32"),
                                    full,
                                )
                                integers = T.simd.vcvt(clipped, "int32", full, round="ROUND_R")
                                packed = T.simd.vcvt(
                                    T.simd.vand(integers, T.simd.vdup(T.int32(255), "int32"), full),
                                    "uint8",
                                    full,
                                    sat=False,
                                    part=0,
                                )
                                T.simd.vsts(
                                    codes[index], T.reinterpret(packed, "int8x256"), full, "PK4_B32"
                                )
                    T.copy(factors[:count], Scale[begin : begin + count])
                    T.copy(codes[:element_count], Q[input_begin : input_begin + element_count])

    return tilelang.compile(main, target="ascend", out_idx=[1, 2])
