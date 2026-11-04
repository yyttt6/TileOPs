"""Interleave independent FP32 rows while retaining stable scaled exponentials."""

import functools
import math

import tilelang
import tilelang.ascend.language as T

from tileops.kernels.ascend.reduction.softmax import _store_probability


@T.macro
def _write(e, y, state, rows: int, count, out_dtype: str, normal: bool, mask):
    power = 0 if out_dtype == "float16" else 64
    for r in T.serial(rows):
        summed = T.simd.vld(state[r, 0], dist="BRC_B32")
        inverse = T.simd.vdiv(
            T.simd.vdup(T.float32(1 if normal else 2.0**power), "float32"),
            summed,
            mask,
            precision="ftz_true",
        )
        for i in T.serial(count // 64):
            value = T.simd.vld(e[r, i * 64])
            prob = T.simd.vmul(value, inverse, mask)
            if normal:
                if out_dtype == "float32":
                    T.simd.vsts(y[r, i * 64], prob, mask, extent=64)
                else:
                    T.simd.vsts(
                        y[r, i * 64],
                        T.simd.vcvt(prob, out_dtype, sat=False),
                        mask,
                        dist="PK_B32",
                        extent=64,
                    )
            else:
                _store_probability(y, r, i * 64, value, prob, summed, out_dtype)


@T.macro
def _output(e, y, state, flag, rows: int, count, out_dtype: str, guarded: bool):
    with T.SimdVF():
        mask = T.simd.pset(32)
        if guarded:
            T.simd.mem_bar("VST_LD")
            if flag[0] == 0:
                _write(e, y, state, rows, count, out_dtype, True, mask)
            else:
                _write(e, y, state, rows, count, out_dtype, False, mask)
        else:
            _write(e, y, state, rows, count, out_dtype, False, mask)


@T.macro
def _stats(x, e, state, flag, rows: int, count: int):
    with T.SimdVF():
        mask = T.simd.pset(32)
        one = T.simd.pset(32, "PAT_VL1")
        bad = T.simd.alloc_local(1, "int32")
        bad[0] = T.simd.vdup(T.int32(0), "int32")
        for group in T.serial(rows // 4):
            maximum = T.simd.alloc_local(8, "float32")
            minimum = T.simd.alloc_local(8, "float32")
            mx = T.simd.alloc_local(4, "float32")
            acc = T.simd.alloc_local(8, "float32")
            for r in T.unroll(4, explicit=True):
                maximum[r * 2] = T.simd.vdup(-T.infinity("float32"), "float32")
                maximum[r * 2 + 1] = maximum[r * 2]
                minimum[r * 2] = T.simd.vdup(T.infinity("float32"), "float32")
                minimum[r * 2 + 1] = minimum[r * 2]
            for i in T.serial(count // 128):
                for r in T.unroll(4, explicit=True):
                    a = T.simd.vld(x[group * 4 + r, i * 128])
                    b = T.simd.vld(x[group * 4 + r, i * 128 + 64])
                    maximum[r * 2] = T.simd.vmax(maximum[r * 2], a, mask)
                    maximum[r * 2 + 1] = T.simd.vmax(maximum[r * 2 + 1], b, mask)
                    minimum[r * 2] = T.simd.vmin(minimum[r * 2], a, mask)
                    minimum[r * 2 + 1] = T.simd.vmin(minimum[r * 2 + 1], b, mask)
            for r in T.unroll(4, explicit=True):
                mx[r] = T.simd.vdupv(
                    T.simd.vcmax(T.simd.vmax(maximum[r * 2], maximum[r * 2 + 1], mask), mask), mask
                )
                mn = T.simd.vdupv(
                    T.simd.vcmin(T.simd.vmin(minimum[r * 2], minimum[r * 2 + 1], mask), mask), mask
                )
                large = T.simd.vcmps(
                    T.simd.vsub(mx[r], mn, mask),
                    T.float32(126 * math.log(2) - math.log(count) - 1),
                    mask,
                    op="gt",
                )
                bad[0] = T.simd.vsel(T.simd.vdup(T.int32(1), "int32"), bad[0], large)
                acc[r * 2] = T.simd.vdup(T.float32(0), "float32")
                acc[r * 2 + 1] = acc[r * 2]
            for i in T.serial(count // 128):
                for r in T.unroll(4, explicit=True):
                    a = T.simd.vexp(
                        T.simd.vadds(
                            T.simd.vsub(T.simd.vld(x[group * 4 + r, i * 128]), mx[r], mask),
                            T.float32(64 * math.log(2)),
                            mask,
                        ),
                        mask,
                    )
                    b = T.simd.vexp(
                        T.simd.vadds(
                            T.simd.vsub(T.simd.vld(x[group * 4 + r, i * 128 + 64]), mx[r], mask),
                            T.float32(64 * math.log(2)),
                            mask,
                        ),
                        mask,
                    )
                    acc[r * 2] = T.simd.vadd(acc[r * 2], a, mask)
                    acc[r * 2 + 1] = T.simd.vadd(acc[r * 2 + 1], b, mask)
                    T.simd.vsts(e[group * 4 + r, i * 128], a, mask, extent=64)
                    T.simd.vsts(e[group * 4 + r, i * 128 + 64], b, mask, extent=64)
            for r in T.unroll(4, explicit=True):
                total = T.simd.vcadd(T.simd.vadd(acc[r * 2], acc[r * 2 + 1], mask), mask)
                T.simd.vsts(state[group * 4 + r, 0], total, one, extent=1)
        T.simd.vsts(flag[0], bad[0], one, extent=1)
        T.simd.mem_bar("VST_VLD")


@functools.lru_cache(maxsize=64)
def build(m, n, dtype, out_dtype):
    rows = 56
    assert dtype == out_dtype == "float32" and n == 512 and m % 4 == 0 and rows % 4 == 0
    groups = (m + rows - 1) // rows
    blocks = min(64, groups)
    stages = 2

    @T.prim_func
    def main(X: T.Tensor((m, n), "float32"), Y: T.Tensor((m, n), "float32")):
        with T.Kernel(blocks) as bx:
            T.annotate_unlimit_memory("shared")
            e = T.alloc_shared((rows, n), "float32")
            state = T.alloc_shared((rows, 8), "float32")
            flag = T.alloc_shared((8,), "int32")
            if stages == 2:
                T.annotate_buffer_versions({e: 2, state: 2, flag: 2})
            for group in T.Pipelined((groups + blocks - 1) // blocks, num_stages=stages):
                row = (group * blocks + bx) * rows
                if row < m:
                    valid_rows = T.min(rows, m - row)
                    T.copy(X[row : row + valid_rows, :], e[:valid_rows, :])
                    _stats(e, e, state, flag, valid_rows, n)
                    _output(e, e, state, flag, valid_rows, n, "float32", True)
                    T.copy(e[:valid_rows, :], Y[row : row + valid_rows, :], l2_cache_ctrl=0)

    return tilelang.compile(main, target="ascend", out_idx=[1])
