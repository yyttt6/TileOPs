"""Batch small INT4 groups into contiguous input and packed-output DMA tiles."""

import functools

import tilelang
import tilelang.ascend.language as T


@functools.lru_cache(maxsize=64)
def build_int4_groups(n: int, group_size: int):
    tile = 8192
    groups = n // group_size
    blocks = min(64, (n + tile - 1) // tile)

    @T.prim_func
    def main(
        X: T.Tensor((n,), "float16"),
        Q: T.Tensor((n // 2,), "int8"),
        Params: T.Tensor((groups * 2,), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            raw = T.alloc_shared((tile,), "float16")
            codes = T.alloc_shared((tile // 2,), "int8")
            parameters = T.alloc_shared((tile // group_size * 2,), "float32")
            for step in T.serial(T.ceildiv(n, blocks * tile)):
                begin = (step * blocks + bx) * tile
                if begin < n:
                    count = T.min(tile, n - begin)
                    T.copy(X[begin : begin + count], raw[:count])
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        pairs = T.simd.pset(32, "PAT_VL32")
                        sixteen = T.simd.vdup(T.float32(16), "float32")
                        mask_bits = T.simd.vdup(T.int32(15), "int32")
                        indices = T.simd.vand(
                            T.simd.vci(0, "int32"), T.simd.vdup(T.int32(31), "int32"), full
                        )
                        even = T.simd.vshls(indices, T.int32(1), full)
                        odd = T.simd.vadds(even, T.int32(1), full)
                        for group in T.serial(tile // group_size):
                            first = T.simd.vcvt(
                                T.simd.vld(raw[group * group_size], dist="UNPK_B16"),
                                "float32",
                                part=0,
                            )
                            low_acc = T.simd.alloc_local((1,), "float32")
                            high_acc = T.simd.alloc_local((1,), "float32")
                            low_acc[0] = T.simd.vcmin(first, full)
                            high_acc[0] = T.simd.vcmax(first, full)
                            for part in T.serial(1, group_size // 64):
                                values = T.simd.vcvt(
                                    T.simd.vld(
                                        raw[group * group_size + part * 64], dist="UNPK_B16"
                                    ),
                                    "float32",
                                    part=0,
                                )
                                low_acc[0] = T.simd.vmin(
                                    low_acc[0], T.simd.vcmin(values, full), full
                                )
                                high_acc[0] = T.simd.vmax(
                                    high_acc[0], T.simd.vcmax(values, full), full
                                )
                            low = T.simd.vdupv(low_acc[0], full)
                            high = T.simd.vdupv(high_acc[0], full)
                            offset = T.simd.vmuls(
                                T.simd.vadd(low, high, full), T.float32(0.5), full
                            )
                            denominator = T.simd.vsel(
                                sixteen,
                                T.simd.vsub(high, low, full),
                                T.simd.vcmp(high, low, full, op="eq"),
                            )
                            multiplier = T.simd.vdiv(sixteen, denominator, full, precision="exact")
                            scale = T.simd.vdiv(
                                T.simd.vdup(T.float32(1), "float32"),
                                multiplier,
                                full,
                                precision="exact",
                            )
                            T.simd.vsts(parameters[group * 2], scale, one, "ONEPT_B32")
                            T.simd.vsts(parameters[group * 2 + 1], offset, one, "ONEPT_B32")
                            for part in T.serial(group_size // 64):
                                values = T.simd.vcvt(
                                    T.simd.vld(
                                        raw[group * group_size + part * 64], dist="UNPK_B16"
                                    ),
                                    "float32",
                                    part=0,
                                )
                                scaled = T.simd.vmul(
                                    T.simd.vsub(values, offset, full), multiplier, full
                                )
                                clipped = T.simd.vmin(
                                    T.simd.vmax(
                                        scaled, T.simd.vdup(T.float32(-8), "float32"), full
                                    ),
                                    T.simd.vdup(T.float32(7), "float32"),
                                    full,
                                )
                                rounded = T.simd.vcvt(clipped, "int32", full, round="ROUND_R")
                                nibbles = T.simd.vand(rounded, mask_bits, full)
                                packed_bits = T.simd.vor(  # codespell:ignore vor
                                    T.simd.vshls(T.simd.vselr(nibbles, even), T.int32(4), full),
                                    T.simd.vselr(nibbles, odd),
                                    full,
                                )
                                packed = T.simd.vcvt(packed_bits, "uint8", full, sat=False, part=0)
                                T.simd.vsts(
                                    codes[group * group_size // 2 + part * 32],
                                    T.reinterpret(packed, "int8x256"),
                                    pairs,
                                    "PK4_B32",
                                )
                    T.copy(codes[: count // 2], Q[begin // 2 : (begin + count) // 2])
                    T.copy(
                        parameters[: count // group_size * 2],
                        Params[begin // group_size * 2 : (begin + count) // group_size * 2],
                    )

    return tilelang.compile(main, target="ascend", out_idx=[1, 2])
