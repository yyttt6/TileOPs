"""Correctness-first T301C kernels, kept in the authorized GEMM ops directory.

Derived from the staging ideas in tileops/kernels/gemm.py and gemm_splitk.py,
plus tilelang-ascend/examples/deepseek_v4/int8_gemm.py (MIT, Copyright (c)
Tile-AI Corporation). Sources were read as text, never imported. Fixed tiles,
serial K loops and separate vector launches are deliberate; no tuning.
"""

from functools import lru_cache
from math import gcd

import tilelang as tl
import tilelang.language as T

CONFIG = {
    tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tl.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@lru_cache(maxsize=128)
def dense_accum(m, n, k, dtype, trans_b=False):
    """Cube contraction; preserve fp32 (int32 for INT8) until the epilogue."""
    bm, bn, bk = 32, 64, 64
    nt = (n + bn - 1) // bn
    tiles = ((m + bm - 1) // bm) * nt
    cores = min(tiles, 24)
    acc = "int32" if dtype == "int8" else "float32"
    bs = (n, k) if trans_b else (k, n)
    ls = (bn, bk) if trans_b else (bk, bn)

    @tl.jit(out_idx=[-1], pass_configs=CONFIG, compile_flags=["-O3", "-DENABLE_BF16"])
    def factory():
        @T.prim_func
        def main(
            a: T.Tensor((m, k), dtype), b: T.Tensor(bs, dtype), c: T.Tensor((m, n), acc)
        ):
            with T.Kernel(cores, is_npu=True) as (cid, _):
                for repeat in T.serial((tiles + cores - 1) // cores):
                    tile = repeat * cores + cid
                    if tile < tiles:
                        row = tile // nt * bm
                        col = tile % nt * bn
                        aa = T.alloc_L1((bm, bk), dtype)
                        bb = T.alloc_L1(ls, dtype)
                        cc = T.alloc_L0C((bm, bn), acc)
                        for group in T.serial((k + bk - 1) // bk):
                            T.copy(a[row : row + bm, group * bk : group * bk + bk], aa)
                            if trans_b:
                                T.copy(
                                    b[col : col + bn, group * bk : group * bk + bk], bb
                                )
                            else:
                                T.copy(
                                    b[group * bk : group * bk + bk, col : col + bn], bb
                                )
                            T.gemm_v0(
                                aa,
                                bb,
                                cc,
                                transpose_B=trans_b,
                                init=(group == 0),
                                kL0Size=64,
                            )
                        T.copy(cc, c[row : row + bm, col : col + bn])

        return main

    return factory()


@lru_cache(maxsize=128)
def cast_flat(count, source, dest):
    if source == "int8" and dest == "bfloat16":
        return lambda x: cast_flat(count, "float32", "bfloat16")(
            cast_flat(count, "float16", "float32")(
                cast_flat(count, "int8", "float16")(x)
            )
        )
    tile = 1024
    blocks = min((count + tile * 2 - 1) // (tile * 2), 24)

    @tl.jit(out_idx=[-1], pass_configs=CONFIG, compile_flags=["-O3", "-DENABLE_BF16"])
    def factory():
        @T.prim_func
        def main(x: T.Tensor((count,), source), y: T.Tensor((count,), dest)):
            with T.Kernel(blocks, is_npu=True) as (cid, vid):
                with T.Scope("V"):
                    a = T.alloc_ub((tile,), source)
                    b = T.alloc_ub((tile,), dest)
                    for r in T.serial(
                        (count + blocks * 2 * tile - 1) // (blocks * 2 * tile)
                    ):
                        start = (r * blocks * 2 + cid * 2 + vid) * tile
                        if start < count:
                            T.copy(x[start : start + tile], a)
                            T.tile.cast(
                                b,
                                a,
                                "CAST_NONE"
                                if source == "int8" or dest == "float32"
                                else "CAST_RINT",
                                tile,
                            )
                            T.copy(b, y[start : start + tile])

        return main

    return factory()


@lru_cache(maxsize=128)
def scaled_output(m, n, source, dest, row_scale):
    """Apply row/column fp32 scales after accumulation, then narrow once."""
    width = 256
    nt = (n + width - 1) // width
    blocks = min(m * nt, 24)

    @tl.jit(out_idx=[-1], pass_configs=CONFIG, compile_flags=["-O3", "-DENABLE_BF16"])
    def factory():
        @T.prim_func
        def main(
            x: T.Tensor((m, n), source),
            sa: T.Tensor((m,), "float32"),
            sb: T.Tensor((n,), "float32"),
            y: T.Tensor((m, n), dest),
        ):
            with T.Kernel(blocks, is_npu=True) as (cid, vid):
                with T.Scope("V"):
                    xx = T.alloc_ub((width,), source)
                    ff = T.alloc_ub((width,), "float32")
                    ss = T.alloc_ub((width,), "float32")
                    yy = T.alloc_ub((width,), dest)
                    for r in T.serial((m * nt + blocks * 2 - 1) // (blocks * 2)):
                        idx = r * blocks * 2 + cid * 2 + vid
                        if idx < m * nt:
                            row = idx // nt
                            col = idx % nt * width
                            T.copy(x[row, col : col + width], xx)
                            T.copy(sb[col : col + width], ss)
                            if source == "float32":
                                T.copy(xx, ff)
                            else:
                                T.tile.cast(ff, xx, "CAST_RINT", width)
                            T.tile.mul(ff, ff, ss)
                            if row_scale:
                                T.tile.mul(ff, ff, sa[row])
                            T.tile.cast(yy, ff, "CAST_RINT", width)
                            T.copy(yy, y[row, col : col + width])

        return main

    return factory()


@lru_cache(maxsize=128)
def swiglu(m, n, dtype):
    """silu(first N columns) * last N columns, all arithmetic in fp32."""
    width = 256
    nt = (n + width - 1) // width
    blocks = min(m * nt, 24)

    @tl.jit(out_idx=[-1], pass_configs=CONFIG, compile_flags=["-O3", "-DENABLE_BF16"])
    def factory():
        @T.prim_func
        def main(x: T.Tensor((m, 2 * n), "float32"), y: T.Tensor((m, n), dtype)):
            with T.Kernel(blocks, is_npu=True) as (cid, vid):
                with T.Scope("V"):
                    gate = T.alloc_ub((width,), "float32")
                    up = T.alloc_ub((width,), "float32")
                    tmp = T.alloc_ub((width,), "float32")
                    out = T.alloc_ub((width,), dtype)
                    for r in T.serial((m * nt + 2 * blocks - 1) // (2 * blocks)):
                        idx = r * 2 * blocks + 2 * cid + vid
                        if idx < m * nt:
                            row = idx // nt
                            col = idx % nt * width
                            T.copy(x[row, col : col + width], gate)
                            T.copy(x[row, n + col : n + col + width], up)
                            T.tile.mul(tmp, gate, -1.0)
                            T.tile.exp(tmp, tmp)
                            T.tile.add(tmp, tmp, 1.0)
                            T.tile.div(gate, gate, tmp)
                            T.tile.mul(gate, gate, up)
                            T.tile.cast(out, gate, "CAST_RINT", width)
                            T.copy(out, y[row, col : col + width])

        return main

    return factory()


@lru_cache(maxsize=64)
def sparse_expand(m, k, dtype):
    """Each metadata nibble stores ordered 2-bit positions of two values/4.

    Four nibbles per uint16 cover 16 logical K elements. This explicit encoding
    is the conventional ordered-index 2:4 layout. Positions must be ordered
    and distinct, as specified by the compressed input contract.
    """
    tile = gcd(k, 256)
    blocks = min((m * k + tile * 2 - 1) // (tile * 2), 24)

    @tl.jit(out_idx=[-1], pass_configs=CONFIG, compile_flags=["-O3", "-DENABLE_BF16"])
    def factory():
        @T.prim_func
        def main(
            a: T.Tensor((m, k // 2), dtype),
            meta: T.Tensor((m, k // 16), "uint16"),
            out: T.Tensor((m, k), dtype),
        ):
            with T.Kernel(blocks, is_npu=True) as (cid, vid):
                with T.Scope("V"):
                    vals = T.alloc_ub((tile // 2,), dtype)
                    vf = T.alloc_ub((tile // 2,), "float32")
                    expanded = T.alloc_ub((tile,), "float32")
                    result = T.alloc_ub((tile,), dtype)
                    for r in T.serial(
                        (m * k + 2 * blocks * tile - 1) // (2 * blocks * tile)
                    ):
                        start = (r * 2 * blocks + 2 * cid + vid) * tile
                        if start < m * k:
                            row = start // k
                            col = start % k
                            T.copy(a[row, col // 2 : col // 2 + tile // 2], vals)
                            T.tile.cast(vf, vals, "CAST_NONE", tile // 2)
                            for i in T.Parallel(tile):
                                code = T.Cast("int32", meta[row, (col + i) // 16])
                                shift = ((col + i) % 16) // 4 * 4
                                first = (code >> shift) & 3
                                second = (code >> (shift + 2)) & 3
                                lane = (col + i) % 4
                                expanded[i] = T.if_then_else(
                                    lane == first,
                                    vf[(i // 4) * 2],
                                    T.if_then_else(
                                        lane == second, vf[(i // 4) * 2 + 1], 0.0
                                    ),
                                )
                            T.tile.cast(result, expanded, "CAST_RINT", tile)
                            T.copy(result, out[row, col : col + tile])

        return main

    return factory()


@lru_cache(maxsize=64)
def unpack_fp4(count):
    """Low nibble first E2M1 -> exact unscaled fp16 values (including sign)."""
    tile = 512
    blocks = min((count + 2 * tile - 1) // (2 * tile), 24)

    @tl.jit(out_idx=[-1], pass_configs=CONFIG)
    def factory():
        @T.prim_func
        def main(x: T.Tensor((count // 2,), "uint8"), y: T.Tensor((count,), "float16")):
            with T.Kernel(blocks, is_npu=True) as (cid, vid):
                with T.Scope("V"):
                    packed = T.alloc_ub((tile // 2,), "uint8")
                    fp = T.alloc_ub((tile,), "float32")
                    out = T.alloc_ub((tile,), "float16")
                    for r in T.serial(
                        (count + 2 * blocks * tile - 1) // (2 * blocks * tile)
                    ):
                        start = (r * 2 * blocks + 2 * cid + vid) * tile
                        if start < count:
                            T.copy(x[start // 2 : start // 2 + tile // 2], packed)
                            for i in T.Parallel(tile):
                                code = (
                                    T.Cast("int32", packed[i // 2]) >> (4 * (i % 2))
                                ) & 15
                                mag = code & 7
                                val = T.if_then_else(
                                    mag == 0,
                                    0.0,
                                    T.if_then_else(
                                        mag == 1,
                                        0.5,
                                        T.if_then_else(
                                            mag == 2,
                                            1.0,
                                            T.if_then_else(
                                                mag == 3,
                                                1.5,
                                                T.if_then_else(
                                                    mag == 4,
                                                    2.0,
                                                    T.if_then_else(
                                                        mag == 5,
                                                        3.0,
                                                        T.if_then_else(
                                                            mag == 6, 4.0, 6.0
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                )
                                fp[i] = T.if_then_else(code >= 8, -val, val)
                            T.tile.cast(out, fp, "CAST_RINT", tile)
                            T.copy(out, y[start : start + tile])

        return main

    return factory()


@lru_cache(maxsize=64)
def accumulate_fp4_block(m, n, groups):
    """Scale a K=32 partial and add to fp32 state; E8M0 exponent 255 is NaN.

    Scale after the unscaled Cube product, so no large/small E8M0 value is
    prematurely rounded to fp16. Only O(MN) scratch, independent of K.
    """
    width = 256
    nt = (n + width - 1) // width
    blocks = min(m * nt, 24)

    @tl.jit(out_idx=[], pass_configs=CONFIG)
    def factory():
        @T.prim_func
        def main(
            part: T.Tensor((m, n), "float32"),
            sa: T.Tensor((m, groups), "uint8"),
            sb: T.Tensor((groups, n), "uint8"),
            acc: T.Tensor((m, n), "float32"),
            group: T.int32,
        ):
            with T.Kernel(blocks, is_npu=True) as (cid, vid):
                with T.Scope("V"):
                    pp = T.alloc_ub((width,), "float32")
                    old = T.alloc_ub((width,), "float32")
                    bs = T.alloc_ub((width,), "uint8")
                    for r in T.serial((m * nt + 2 * blocks - 1) // (2 * blocks)):
                        idx = r * 2 * blocks + 2 * cid + vid
                        if idx < m * nt:
                            row = idx // nt
                            col = idx % nt * width
                            T.copy(part[row, col : col + width], pp)
                            T.copy(sb[group, col : col + width], bs)
                            for i in T.Parallel(width):
                                ea = T.Cast("int32", sa[row, group])
                                eb = T.Cast("int32", bs[i])
                                exponent = ea + eb - 254
                                half = exponent // 2
                                rest = exponent - half
                                # Exact powers via IEEE bits; split the exponent
                                # to retain representable subnormal results.
                                bits0 = T.if_then_else(
                                    half == -127, 4194304, (half + 127) << 23
                                )
                                bits1 = T.if_then_else(
                                    rest == -127, 4194304, (rest + 127) << 23
                                )
                                f0 = T.reinterpret("float32", T.Cast("uint32", bits0))
                                f1 = T.reinterpret("float32", T.Cast("uint32", bits1))
                                nan = T.reinterpret("float32", T.uint32(2143289344))
                                pp[i] = T.if_then_else(
                                    (ea == 255) | (eb == 255), nan, (pp[i] * f0) * f1
                                )
                            if group != 0:
                                T.copy(acc[row, col : col + width], old)
                                T.tile.add(pp, pp, old)
                            T.copy(pp, acc[row, col : col + width])

        return main

    return factory()


@lru_cache(maxsize=64)
def compensated_add(count):
    """Kahan-add fp32 partials; used to protect SwiGLU's near-zero gate.

    A full-K fp32 Cube sum failed the unchanged tolerance in 8/16777216
    outputs even against fp64 (R301C). This is an accuracy change, applied
    uniformly to every SwiGLU input; no workload-specific branch or tuning.
    """
    width = 1024
    blocks = min((count + 2 * width - 1) // (2 * width), 24)

    @tl.jit(out_idx=[], pass_configs=CONFIG)
    def factory():
        @T.prim_func
        def main(
            part: T.Tensor((count,), "float32"),
            total: T.Tensor((count,), "float32"),
            correction: T.Tensor((count,), "float32"),
            first: T.int32,
        ):
            with T.Kernel(blocks, is_npu=True) as (cid, vid):
                with T.Scope("V"):
                    pp = T.alloc_ub((width,), "float32")
                    ss = T.alloc_ub((width,), "float32")
                    cc = T.alloc_ub((width,), "float32")
                    tt = T.alloc_ub((width,), "float32")
                    for r in T.serial(
                        (count + 2 * blocks * width - 1) // (2 * blocks * width)
                    ):
                        start = (r * 2 * blocks + 2 * cid + vid) * width
                        if start < count:
                            T.copy(part[start : start + width], pp)
                            if first != 0:
                                T.tile.fill(cc, 0.0)
                                T.copy(pp, total[start : start + width])
                            else:
                                T.copy(total[start : start + width], ss)
                                T.copy(correction[start : start + width], cc)
                                T.tile.sub(pp, pp, cc)
                                T.tile.add(tt, ss, pp)
                                T.tile.sub(cc, tt, ss)
                                T.tile.sub(cc, cc, pp)
                                T.copy(tt, total[start : start + width])
                            T.copy(cc, correction[start : start + width])

        return main

    return factory()
