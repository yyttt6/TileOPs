"""Bounded row batches with reusable FP32 exponentials and guarded normal output.

Half inputs expand backwards in UB, so unread input elements are preserved.
The second VF consumes completed stores; padded row statistics are 32-byte aligned."""

import functools
import math

import tilelang
import tilelang.ascend.language as T

from tileops.kernels.ascend.reduction.softmax import _store_probability


@T.macro
def _stats(x, e, state, flag, rows: int, count, dtype: str, out_dtype: str, guarded: bool):
    with T.SimdVF():
        mask = T.simd.pset(32)
        one = T.simd.pset(32, "PAT_VL1")
        power = 0 if out_dtype == "float16" else 64
        bad = T.simd.alloc_local(1, "int32")
        bad[0] = T.simd.vdup(T.int32(0), "int32")
        for r in T.serial(rows):
            maximum = T.simd.alloc_local(2, "float32")
            minimum = T.simd.alloc_local(2, "float32")
            minimum[0] = T.simd.vdup(T.infinity("float32"), "float32")
            minimum[1] = minimum[0]
            maximum[0] = T.simd.vdup(-T.infinity("float32"), "float32")
            maximum[1] = maximum[0]
            for i in T.serial(count // 128):
                if dtype == "float32":
                    av = T.simd.vld(x[r, i * 128])
                    bv = T.simd.vld(x[r, i * 128 + 64])
                else:
                    packed = T.simd.vld(x[r, i * 128])
                    av = T.simd.vcvt(packed, "float32", part=0)
                    bv = T.simd.vcvt(packed, "float32", part=1)
                maximum[0] = T.simd.vmax(maximum[0], av, mask)
                maximum[1] = T.simd.vmax(maximum[1], bv, mask)
                if guarded and out_dtype != "float16":
                    minimum[0] = T.simd.vmin(minimum[0], av, mask)
                    minimum[1] = T.simd.vmin(minimum[1], bv, mask)
            reduced = T.simd.vcmax(T.simd.vmax(maximum[0], maximum[1], mask), mask)
            mx = T.simd.vdupv(reduced, mask)
            if guarded and out_dtype != "float16":
                mn = T.simd.vdupv(
                    T.simd.vcmin(T.simd.vmin(minimum[0], minimum[1], mask), mask), mask
                )
                # exp(-span)/count bounds every probability from below. One natural-log
                # unit of margin conservatively keeps all fast-path outputs normal.
                large = T.simd.vcmps(
                    T.simd.vsub(mx, mn, mask),
                    T.float32(126 * math.log(2) - math.log(count) - 1),
                    mask,
                    op="gt",
                )
                bad[0] = T.simd.vsel(T.simd.vdup(T.int32(1), "int32"), bad[0], large)
            acc = T.simd.alloc_local(2, "float32")
            acc[0] = T.simd.vdup(T.float32(0), "float32")
            acc[1] = acc[0]
            for step in T.serial(count // 128):
                i = count // 128 - 1 - step
                if dtype == "float32":
                    av = T.simd.vld(x[r, i * 128])
                    bv = T.simd.vld(x[r, i * 128 + 64])
                else:
                    packed = T.simd.vld(x[r, i * 128])
                    av = T.simd.vcvt(packed, "float32", part=0)
                    bv = T.simd.vcvt(packed, "float32", part=1)
                a = T.simd.vexp(
                    T.simd.vadds(T.simd.vsub(av, mx, mask), T.float32(power * math.log(2)), mask),
                    mask,
                )
                b = T.simd.vexp(
                    T.simd.vadds(T.simd.vsub(bv, mx, mask), T.float32(power * math.log(2)), mask),
                    mask,
                )
                acc[0] = T.simd.vadd(acc[0], a, mask)
                acc[1] = T.simd.vadd(acc[1], b, mask)
                if dtype == "float32":
                    lo = a
                    hi = b
                else:
                    lo, hi = T.simd.vintlv(a, b)
                T.simd.vsts(e[r, i * 128], lo, mask, extent=64)
                T.simd.vsts(e[r, i * 128 + 64], hi, mask, extent=64)
            total = T.simd.vcadd(T.simd.vadd(acc[0], acc[1], mask), mask)
            T.simd.vsts(state[r, 0], total, one, extent=1)
        if guarded:
            T.simd.vsts(flag[0], bad[0], one, extent=1)
        T.simd.mem_bar("VST_VLD")


@T.macro
def _write(e, y, state, rows: int, count, out_dtype: str, normal: bool):
    mask = T.simd.pset(32)
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
        if guarded:
            T.simd.mem_bar("VST_LD")
            if flag[0] == 0:
                _write(e, y, state, rows, count, out_dtype, True)
            else:
                _write(e, y, state, rows, count, out_dtype, False)
        else:
            _write(e, y, state, rows, count, out_dtype, False)


@functools.lru_cache(maxsize=256)
def build(m, n, dtype, out_dtype, rows, guarded):
    assert n % 128 == 0 and m % rows == 0
    groups = m // rows
    blocks = min(64, groups)
    stages = 2 if rows * n * 8 <= 184 * 1024 else 1
    in_pitch = n * (1 if dtype == "float32" else 2)
    out_pitch = n * (1 if out_dtype == "float32" else 2)

    @T.prim_func
    def main(X: T.Tensor((m, n), dtype), Y: T.Tensor((m, n), out_dtype)):
        with T.Kernel(blocks) as bx:
            flag = T.alloc_shared((8,), "int32")
            state = T.alloc_shared((rows, 8), "float32")
            e = T.alloc_shared((rows, n), "float32")
            x = T.view(e, shape=(rows, in_pitch), dtype=dtype)
            y = T.view(e, shape=(rows, out_pitch), dtype=out_dtype)
            if stages == 2:
                T.annotate_buffer_versions({e: 2, state: 2, flag: 2})
            for group in T.Pipelined((groups + blocks - 1) // blocks, num_stages=stages):
                row = (group * blocks + bx) * rows
                if row < m:
                    T.copy(X[row : row + rows, :], x[:, :n])
                    _stats(x, e, state, flag, rows, n, dtype, out_dtype, guarded)
                    _output(e, y, state, flag, rows, n, out_dtype, guarded)
                    T.copy(y[:, :n], Y[row : row + rows, :], l2_cache_ctrl=0)

    return tilelang.compile(main, target="ascend", out_idx=[1])
