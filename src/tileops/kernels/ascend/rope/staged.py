"""Stage contiguous heads and reuse each frequency pair across heads."""

import functools

import tilelang
import tilelang.ascend.language as T

__all__ = ["build_staged_rotation"]


@functools.lru_cache(maxsize=64)
def build_staged_rotation(tokens, heads, dim, table_rows, dtype, interleaved, indexed):
    # A complete token tile allows all heads to reuse one pair of frequency rows.
    tile = 8192
    token_tile = tile // (heads * dim)
    half = dim // 2
    total = tokens * heads * dim
    tasks = total // tile
    blocks = min(64, tasks)
    positions_size = tokens if indexed else 1

    @T.prim_func
    def main(
        X: T.Tensor((total,), dtype),
        Cos: T.Tensor((table_rows * half,), dtype),
        Sin: T.Tensor((table_rows * half,), dtype),
        Positions: T.Tensor((positions_size,), "int32"),
        Y: T.Tensor((total,), dtype),
        Invalid: T.Tensor((positions_size,), "int32"),
    ):
        with T.Kernel(blocks) as bx:
            raw = T.alloc_shared((tile,), dtype)
            wide = T.alloc_shared((tile,), "float32")
            result = T.alloc_shared((tile,), dtype)
            cosine = T.alloc_shared((token_tile * half,), dtype)
            sine = T.alloc_shared((token_tile * half,), dtype)
            for step in T.serial(T.ceildiv(tasks, blocks)):
                task = step * blocks + bx
                if task < tasks:
                    begin = task * tile
                    token_begin = task * token_tile
                    T.copy(X[begin : begin + tile], raw)
                    if indexed:
                        with T.SimtVF(threads=256):
                            for j in T.Parallel(token_tile * half, coalesced_width=T.int32(1)):
                                token = token_begin + j // half
                                position = Positions[token]
                                safe = T.min(T.max(position, 0), table_rows - 1)
                                cosine[j] = Cos[safe * half + j % half]
                                sine[j] = Sin[safe * half + j % half]
                                if j % half == 0:
                                    Invalid[token] = T.cast(
                                        position < 0 or position >= table_rows, "int32"
                                    )
                    else:
                        table_begin = token_begin % table_rows * half
                        T.copy(Cos[table_begin : table_begin + token_tile * half], cosine)
                        T.copy(Sin[table_begin : table_begin + token_tile * half], sine)
                    if dim == 128:
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            if interleaved:
                                for v in T.serial(tile // 64):
                                    if dtype == "float32":
                                        converted = T.simd.vld(raw[v * 64])
                                    else:
                                        converted = T.simd.vcvt(
                                            T.simd.vld(raw[v * 64], dist="UNPK_B16"),
                                            "float32",
                                            part=0,
                                        )
                                    T.simd.vsts(wide[v * 64], converted, full)
                                T.simd.mem_bar("VST_VLD")
                            for token in T.serial(token_tile):
                                if dtype == "float32":
                                    c = T.simd.vld(cosine[token * 64])
                                    s = T.simd.vld(sine[token * 64])
                                else:
                                    c = T.simd.vcvt(
                                        T.simd.vld(cosine[token * 64], dist="UNPK_B16"),
                                        "float32",
                                        part=0,
                                    )
                                    s = T.simd.vcvt(
                                        T.simd.vld(sine[token * 64], dist="UNPK_B16"),
                                        "float32",
                                        part=0,
                                    )
                                for head in T.serial(heads):
                                    origin = (token * heads + head) * 128
                                    if interleaved:
                                        a, b = T.simd.vld2(wide[origin], dist="DINTLV_B32")
                                    elif dtype == "float32":
                                        a = T.simd.vld(raw[origin])
                                        b = T.simd.vld(raw[origin + 64])
                                    else:
                                        a = T.simd.vcvt(
                                            T.simd.vld(raw[origin], dist="UNPK_B16"),
                                            "float32",
                                            part=0,
                                        )
                                        b = T.simd.vcvt(
                                            T.simd.vld(raw[origin + 64], dist="UNPK_B16"),
                                            "float32",
                                            part=0,
                                        )
                                    lower = T.simd.vsub(
                                        T.simd.vmul(a, c, full), T.simd.vmul(b, s, full), full
                                    )
                                    upper = T.simd.vadd(
                                        T.simd.vmul(b, c, full), T.simd.vmul(a, s, full), full
                                    )
                                    if interleaved:
                                        first, second = T.simd.vintlv(lower, upper)
                                    else:
                                        first, second = lower, upper
                                    if dtype == "float32":
                                        T.simd.vsts(result[origin], first, full)
                                        T.simd.vsts(result[origin + 64], second, full)
                                    else:
                                        T.simd.vsts(
                                            result[origin],
                                            T.simd.vcvt(first, dtype, full, sat=False),
                                            full,
                                            "PK_B32",
                                        )
                                        T.simd.vsts(
                                            result[origin + 64],
                                            T.simd.vcvt(second, dtype, full, sat=False),
                                            full,
                                            "PK_B32",
                                        )
                    else:
                        with T.SimtVF(threads=256):
                            for j in T.Parallel(tile, coalesced_width=T.int32(1)):
                                col = j % dim
                                table_base = j // (heads * dim) * half
                                lane = (
                                    col // 2
                                    if interleaved
                                    else T.if_then_else(col < half, col, col - half)
                                )
                                other = (
                                    (col ^ 1)
                                    if interleaved
                                    else T.if_then_else(col < half, col + half, col - half)
                                )
                                subtract = col % 2 == 0 if interleaved else col < half
                                a = T.cast(raw[j], "float32")
                                b = T.cast(raw[j - col + other], "float32")
                                c = T.cast(cosine[table_base + lane], "float32")
                                s = T.cast(sine[table_base + lane], "float32")
                                result[j] = T.cast(
                                    a * c + T.if_then_else(subtract, -b, b) * s, dtype
                                )
                    T.copy(result, Y[begin : begin + tile])

    return tilelang.compile(main, target="ascend", out_idx=[4, 5])
