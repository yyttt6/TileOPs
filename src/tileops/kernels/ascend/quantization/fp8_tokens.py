"""Batch short token rows and encode FP8 with the native vector conversion."""

import functools

import tilelang
import tilelang.ascend.language as T


@functools.lru_cache(maxsize=64)
def build_fp8_tokens(rows: int, width: int, dtype: str):
    tile = 8192
    n = rows * width
    blocks = min(64, (n + tile - 1) // tile)

    @T.prim_func
    def main(
        X: T.Tensor((n,), dtype),
        Q: T.Tensor((n,), "uint8"),
        Scale: T.Tensor((rows,), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            raw = T.alloc_shared((tile + 64,), dtype)
            codes = T.alloc_shared((tile + 64,), "uint8")
            scales = T.alloc_shared((tile // width,), "float32")
            for step in T.serial(T.ceildiv(n, blocks * tile)):
                begin = (step * blocks + bx) * tile
                if begin < n:
                    count = T.min(tile, n - begin)
                    T.copy(X[begin : begin + count], raw[:count])
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        row_mask = T.simd.pset(32, "PAT_VL32") if width == 32 else full
                        one = T.simd.pset(32, "PAT_VL1")
                        limit = T.simd.vdup(T.float32(448), "float32")
                        negative = T.simd.vdup(T.float32(-448), "float32")
                        for group in T.serial(tile // width):
                            peak = T.simd.alloc_local((1,), "float32")
                            peak[0] = T.simd.vdup(T.float32(1e-4), "float32")
                            for part in T.serial(T.ceildiv(width, 64)):
                                if dtype == "float32":
                                    value = T.simd.vld(raw[group * width + part * 64])
                                else:
                                    value = T.simd.vcvt(
                                        T.simd.vld(raw[group * width + part * 64], dist="UNPK_B16"),
                                        "float32",
                                        part=0,
                                    )
                                peak[0] = T.simd.vmax(
                                    peak[0],
                                    T.simd.vcmax(T.simd.vabs(value, row_mask), row_mask),
                                    full,
                                )
                            maximum = T.simd.vdupv(peak[0], full)
                            scale = T.simd.vdiv(maximum, limit, full, precision="exact")
                            T.simd.vsts(scales[group], scale, one, "ONEPT_B32")
                            for part in T.serial(T.ceildiv(width, 64)):
                                if dtype == "float32":
                                    value = T.simd.vld(raw[group * width + part * 64])
                                else:
                                    value = T.simd.vcvt(
                                        T.simd.vld(raw[group * width + part * 64], dist="UNPK_B16"),
                                        "float32",
                                        part=0,
                                    )
                                quotient = T.simd.vdiv(value, scale, row_mask, precision="exact")
                                clipped = T.simd.vmin(
                                    T.simd.vmax(quotient, negative, row_mask), limit, row_mask
                                )
                                encoded = T.simd.vcvt(
                                    clipped,
                                    "float8_e4m3fn",
                                    row_mask,
                                    round="ROUND_R",
                                    sat=True,
                                    part=0,
                                )
                                T.simd.vsts(
                                    codes[group * width + part * 64],
                                    T.reinterpret(encoded, "uint8x256"),
                                    row_mask,
                                    "PK4_B32",
                                )
                    T.copy(codes[:count], Q[begin : begin + count])
                    T.copy(
                        scales[: count // width], Scale[begin // width : (begin + count) // width]
                    )

    return tilelang.compile(main, target="ascend", out_idx=[1, 2])
