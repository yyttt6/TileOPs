"""Keep a bounded row prefix resident; reread only its small remainder."""

import functools
import math

import tilelang
import tilelang.ascend.language as T

from tileops.kernels.ascend.reduction.softmax import _store_probability, _tile_output


@T.macro
def bounds(x, state, count: int, initial: bool):
    with T.SimdVF():
        p = T.simd.pset(32)
        one = T.simd.pset(32, "PAT_VL1")
        hi = T.simd.alloc_local(2, "float32")
        lo = T.simd.alloc_local(2, "float32")
        hi[0] = T.simd.vdup(-T.infinity("float32"), "float32")
        hi[1] = hi[0]
        lo[0] = T.simd.vdup(T.infinity("float32"), "float32")
        lo[1] = lo[0]
        for i in T.serial(count // 128):
            a = T.simd.vld(x[0, i * 128])
            b = T.simd.vld(x[0, i * 128 + 64])
            hi[0] = T.simd.vmax(hi[0], a, p)
            hi[1] = T.simd.vmax(hi[1], b, p)
            lo[0] = T.simd.vmin(lo[0], a, p)
            lo[1] = T.simd.vmin(lo[1], b, p)
        mx = T.simd.vdupv(T.simd.vcmax(T.simd.vmax(hi[0], hi[1], p), p), p)
        mn = T.simd.vdupv(T.simd.vcmin(T.simd.vmin(lo[0], lo[1], p), p), p)
        if not initial:
            mx = T.simd.vmax(mx, T.simd.vld(state[0, 0], "BRC_B32"), p)
            mn = T.simd.vmin(mn, T.simd.vld(state[0, 16], "BRC_B32"), p)
        T.simd.vsts(state[0, 0], mx, one, extent=1)
        T.simd.vsts(state[0, 16], mn, one, extent=1)
        T.simd.mem_bar("VST_VLD")


@T.macro
def exponentials(x, state, count: int, initial: bool, keep: bool):
    with T.SimdVF():
        p = T.simd.pset(32)
        one = T.simd.pset(32, "PAT_VL1")
        mx = T.simd.vld(state[0, 0], "BRC_B32")
        acc = T.simd.alloc_local(2, "float32")
        acc[0] = T.simd.vdup(T.float32(0), "float32")
        acc[1] = acc[0]
        for i in T.serial(count // 128):
            a = T.simd.vexp(
                T.simd.vadds(
                    T.simd.vsub(T.simd.vld(x[0, i * 128]), mx, p), T.float32(64 * math.log(2)), p
                ),
                p,
            )
            b = T.simd.vexp(
                T.simd.vadds(
                    T.simd.vsub(T.simd.vld(x[0, i * 128 + 64]), mx, p),
                    T.float32(64 * math.log(2)),
                    p,
                ),
                p,
            )
            acc[0] = T.simd.vadd(acc[0], a, p)
            acc[1] = T.simd.vadd(acc[1], b, p)
            if keep:
                T.simd.vsts(x[0, i * 128], a, p, extent=64)
                T.simd.vsts(x[0, i * 128 + 64], b, p, extent=64)
        sm = T.simd.vdupv(T.simd.vcadd(T.simd.vadd(acc[0], acc[1], p), p), p)
        if not initial:
            sm = T.simd.vadd(sm, T.simd.vld(state[0, 8], "BRC_B32"), p)
        T.simd.vsts(state[0, 8], sm, one, extent=1)
        T.simd.mem_bar("VST_VLD")


@T.macro
def cached_write(x, state, count: int, normal: bool):
    p = T.simd.pset(32)
    total = T.simd.vld(state[0, 8], "BRC_B32")
    inverse = T.simd.vdiv(
        T.simd.vdup(T.float32(2.0 ** (-64) if normal else 1), "float32"),
        total,
        p,
        precision="ftz_true",
    )
    for i in T.serial(count // 64):
        value = T.simd.vld(x[0, i * 64])
        prob = T.simd.vmul(value, inverse, p)
        if normal:
            T.simd.vsts(x[0, i * 64], prob, p, extent=64)
        else:
            _store_probability(x, 0, i * 64, value, prob, total, "float32")


@T.macro
def normalize_prefix(x, state, flag, count: int, n: int):
    with T.SimdVF():
        p = T.simd.pset(32)
        one = T.simd.pset(32, "PAT_VL1")
        mx = T.simd.vld(state[0, 0], "BRC_B32")
        mn = T.simd.vld(state[0, 16], "BRC_B32")
        sm = T.simd.vmuls(T.simd.vld(state[0, 8], "BRC_B32"), T.float32(2.0 ** (-64)), p)
        T.simd.vsts(state[0, 8], sm, one, extent=1)
        large = T.simd.vcmps(
            T.simd.vsub(mx, mn, p), T.float32(126 * math.log(2) - math.log(n) - 1), p, op="gt"
        )
        bad = T.simd.vsel(T.simd.vdup(T.int32(1), "int32"), T.simd.vdup(T.int32(0), "int32"), large)
        T.simd.vsts(flag[0], bad, one, extent=1)
        T.simd.mem_bar("VST_VLD")
        T.simd.mem_bar("VST_LD")
        if flag[0] == 0:
            cached_write(x, state, count, True)
        else:
            cached_write(x, state, count, False)


@functools.lru_cache(maxsize=64)
def build(m, n, dtype, out_dtype):
    assert n == 65536 and dtype == out_dtype == "float32"
    prefix = 53248
    tile = 4096
    parts = (n - prefix) // tile
    blocks = min(64, m)

    @T.prim_func
    def main(X: T.Tensor((m, n), "float32"), Y: T.Tensor((m, n), "float32")):
        with T.Kernel(blocks) as bx:
            T.annotate_unlimit_memory("shared")
            x = T.alloc_shared((1, prefix), "float32")
            tail = T.alloc_shared((1, tile), "float32")
            state = T.alloc_shared((1, 24), "float32")
            flag = T.alloc_shared((8,), "int32")
            for row_group in T.serial((m + blocks - 1) // blocks):
                row = row_group * blocks + bx
                if row < m:
                    T.copy(X[row : row + 1, :prefix], x)
                    bounds(x, state, prefix, True)
                    for part in T.serial(parts):
                        begin = prefix + part * tile
                        T.copy(X[row : row + 1, begin : begin + tile], tail)
                        bounds(tail, state, tile, False)
                    exponentials(x, state, prefix, True, True)
                    for part in T.serial(parts):
                        begin = prefix + part * tile
                        T.copy(X[row : row + 1, begin : begin + tile], tail)
                        exponentials(tail, state, tile, False, False)
                    normalize_prefix(x, state, flag, prefix, n)
                    T.copy(x, Y[row : row + 1, :prefix], l2_cache_ctrl=0)
                    for part in T.serial(parts):
                        begin = prefix + part * tile
                        T.copy(X[row : row + 1, begin : begin + tile], tail)
                        _tile_output(tail, tail, state, flag, tile, "float32", "float32", n)
                        T.copy(tail, Y[row : row + 1, begin : begin + tile], l2_cache_ctrl=0)

    return tilelang.compile(main, target="ascend", out_idx=[1])
