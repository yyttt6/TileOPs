"""Batch aligned FP16 token groups into one DMA tile and one vector function."""

import functools

import tilelang
import tilelang.ascend.language as T


@functools.lru_cache(maxsize=64)
def build_fp16_groups(n: int):
    tile = 8192
    groups = n // 128
    blocks = min(64, (n + tile - 1) // tile)

    @T.prim_func
    def main(
        X: T.Tensor((n,), "float16"),
        Q: T.Tensor((n,), "int8"),
        Scale: T.Tensor((groups,), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            raw = T.alloc_shared((tile,), "float16")
            codes = T.alloc_shared((tile,), "int8")
            scales = T.alloc_shared((64,), "float32")
            for step in T.serial(T.ceildiv(n, blocks * tile)):
                begin = (step * blocks + bx) * tile
                if begin < n:
                    count = T.min(tile, n - begin)
                    T.copy(X[begin : begin + count], raw[:count])
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        denominator = T.simd.vdup(T.float32(127), "float32")
                        negative = T.simd.vdup(T.float32(-127), "float32")
                        for group in T.serial(64):
                            left = T.simd.vcvt(
                                T.simd.vld(raw[group * 128], dist="UNPK_B16"), "float32", part=0
                            )
                            right = T.simd.vcvt(
                                T.simd.vld(raw[group * 128 + 64], dist="UNPK_B16"),
                                "float32",
                                part=0,
                            )
                            peak = T.simd.vmax(
                                T.simd.vcmax(
                                    T.reinterpret(T.simd.vabs(left, full), "int32x64"), full
                                ),
                                T.simd.vcmax(
                                    T.reinterpret(T.simd.vabs(right, full), "int32x64"), full
                                ),
                                full,
                            )
                            maximum = T.reinterpret(T.simd.vdupv(peak, full), "float32x64")
                            numerator = T.simd.vsel(
                                maximum,
                                denominator,
                                T.simd.vcmps(maximum, T.float32(0), full, op="gt"),
                            )
                            scale = T.simd.vdiv(numerator, denominator, full, precision="exact")
                            T.simd.vsts(scales[group], scale, one, "ONEPT_B32")
                            for part in T.unroll(2):
                                value = T.simd.vcvt(
                                    T.simd.vld(raw[group * 128 + part * 64], dist="UNPK_B16"),
                                    "float32",
                                    part=0,
                                )
                                quotient = T.simd.vdiv(value, scale, full, precision="exact")
                                clipped = T.simd.vmin(
                                    T.simd.vmax(quotient, negative, full), denominator, full
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
                                    codes[group * 128 + part * 64],
                                    T.reinterpret(packed, "int8x256"),
                                    full,
                                    "PK4_B32",
                                )
                    T.copy(codes[:count], Q[begin : begin + count])
                    T.copy(scales[: count // 128], Scale[begin // 128 : (begin + count) // 128])

    return tilelang.compile(main, target="ascend", out_idx=[1, 2])
