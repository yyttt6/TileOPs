"""Two-pass INT4 quantization for groups larger than one shared-memory tile."""

import functools

import tilelang
import tilelang.ascend.language as T


@functools.lru_cache(maxsize=64)
def build_int4_stream(n: int, group_size: int):
    groups = n // group_size
    tile = 8192
    blocks = min(64, groups)

    @T.prim_func
    def main(
        X: T.Tensor((n,), "float16"),
        Q: T.Tensor((n // 2,), "int8"),
        Params: T.Tensor((groups, 2), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            raw = T.alloc_shared((tile,), "float16")
            codes = T.alloc_shared((tile // 2,), "int8")
            parameters = T.alloc_shared((3,), "float32")
            for step in T.serial(T.ceildiv(groups, blocks)):
                group = step * blocks + bx
                if group < groups:
                    with T.SimtVF(threads=32):
                        for j in T.Parallel(1):
                            parameters[j] = T.cast(X[group * group_size], "float32")
                            parameters[j + 1] = parameters[j]
                    for chunk in T.serial(T.ceildiv(group_size, tile)):
                        begin = group * group_size + chunk * tile
                        count = T.min(tile, group_size - chunk * tile)
                        T.copy(X[begin : begin + count], raw[:count])
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            one = T.simd.pset(32, "PAT_VL1")
                            low = T.simd.alloc_local((1,), "float32")
                            high = T.simd.alloc_local((1,), "float32")
                            low[0] = T.simd.vdup(parameters[0], "float32")
                            high[0] = T.simd.vdup(parameters[1], "float32")
                            for v in T.serial(T.ceildiv(count, 64)):
                                values = T.simd.vcvt(
                                    T.simd.vld(raw[v * 64], dist="UNPK_B16"), "float32", part=0
                                )
                                valid = T.simd.vcmps(
                                    T.simd.vci(v * 64, "int32"), count, full, op="lt"
                                )
                                # The last vector may contain stale tile data. A value
                                # already included in the extrema is a neutral fill.
                                values = T.simd.vsel(
                                    values, T.simd.vdup(parameters[0], "float32"), valid
                                )
                                low[0] = T.simd.vmin(low[0], T.simd.vcmin(values, full), full)
                                high[0] = T.simd.vmax(high[0], T.simd.vcmax(values, full), full)
                            T.simd.vsts(parameters[0], low[0], one, "ONEPT_B32")
                            T.simd.vsts(parameters[1], high[0], one, "ONEPT_B32")
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        low = T.simd.vdup(parameters[0], "float32")
                        high = T.simd.vdup(parameters[1], "float32")
                        sixteen = T.simd.vdup(T.float32(16), "float32")
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
                        offset = T.simd.vmuls(T.simd.vadd(low, high, full), T.float32(0.5), full)
                        T.simd.vsts(parameters[0], scale, one, "ONEPT_B32")
                        T.simd.vsts(parameters[1], offset, one, "ONEPT_B32")
                        T.simd.vsts(parameters[2], multiplier, one, "ONEPT_B32")
                    for chunk in T.serial(T.ceildiv(group_size, tile)):
                        begin = group * group_size + chunk * tile
                        count = T.min(tile, group_size - chunk * tile)
                        T.copy(X[begin : begin + count], raw[:count])
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            pairs = T.simd.pset(32, "PAT_VL32")
                            offset = T.simd.vdup(parameters[1], "float32")
                            multiplier = T.simd.vdup(parameters[2], "float32")
                            indices = T.simd.vand(
                                T.simd.vci(0, "int32"), T.simd.vdup(T.int32(31), "int32"), full
                            )
                            even = T.simd.vshls(indices, T.int32(1), full)
                            odd = T.simd.vadds(even, T.int32(1), full)
                            for v in T.serial(T.ceildiv(count, 64)):
                                values = T.simd.vcvt(
                                    T.simd.vld(raw[v * 64], dist="UNPK_B16"), "float32", part=0
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
                                nibbles = T.simd.vand(
                                    rounded, T.simd.vdup(T.int32(15), "int32"), full
                                )
                                bits = T.simd.vor(  # codespell:ignore vor
                                    T.simd.vshls(T.simd.vselr(nibbles, even), T.int32(4), full),
                                    T.simd.vselr(nibbles, odd),
                                    full,
                                )
                                packed = T.simd.vcvt(bits, "uint8", full, sat=False, part=0)
                                T.simd.vsts(
                                    codes[v * 32],
                                    T.reinterpret(packed, "int8x256"),
                                    pairs,
                                    "PK4_B32",
                                )
                        T.copy(codes[: count // 2], Q[begin // 2 : (begin + count) // 2])
                    T.copy(parameters[:2], Params[group, :])

    return tilelang.compile(main, target="ascend", out_idx=[1, 2])
