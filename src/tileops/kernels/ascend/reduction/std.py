"""Ascend standard deviation using centered block statistics and Chan merging."""

import functools
from typing import ClassVar

import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import restore_reduced, rows_for_axes
from tileops.kernels.reduction.call_spec import VarianceFwdInterface

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None

if T is not None:

    @T.macro
    def _native_std(x, y, rows: int, n: int, dtype: str, correction: float):
        with T.SimdVF():
            for r in T.serial(rows):
                mask = T.simd.pset(32)
                first = T.simd.vld(x[r, 0])
                if dtype == "float32":
                    anchor = T.simd.vdupv(first, mask)
                else:
                    anchor = T.simd.vdupv(T.simd.vcvt(first, "float32", part=0), mask)
                factor = T.simd.vdup(T.float32(1), "float32")
                unscale = T.simd.vdup(T.float32(1), "float32")
                if dtype != "float16":
                    if dtype == "bfloat16":
                        # BF16 positive magnitude bit order is monotonic; widen only the maximum.
                        narrow_mask = T.simd.pset(16)
                        signless = T.simd.vdup(T.int16(32767), "int16")
                        largest_narrow = T.simd.alloc_local(1, "int16")
                        largest_narrow[0] = T.simd.vdup(T.int16(0), "int16")
                        for i in range(n // 256):
                            low = T.simd.vand(
                                T.reinterpret(T.simd.vld(x[r, i * 256]), "int16x128"),
                                signless,
                                narrow_mask,
                            )
                            high = T.simd.vand(
                                T.reinterpret(T.simd.vld(x[r, i * 256 + 128]), "int16x128"),
                                signless,
                                narrow_mask,
                            )
                            largest_narrow[0] = T.simd.vmax(
                                largest_narrow[0], T.simd.vmax(low, high, narrow_mask), narrow_mask
                            )
                        peak_bits = T.simd.vcmax(largest_narrow[0], narrow_mask)
                        peak = T.simd.vdupv(
                            T.simd.vcvt(
                                T.reinterpret(peak_bits, "bfloat16x128"), "float32", part=0
                            ),
                            mask,
                        )
                    else:
                        largest = T.simd.alloc_local(1, "float32")
                        largest[0] = T.simd.vdup(T.float32(0), "float32")
                        for i in range(n // 256):
                            if dtype == "float32":
                                av = T.simd.vld(x[r, i * 256])
                                bv = T.simd.vld(x[r, i * 256 + 64])
                                cv = T.simd.vld(x[r, i * 256 + 128])
                                dv = T.simd.vld(x[r, i * 256 + 192])
                            else:
                                low = T.simd.vld(x[r, i * 256])
                                high = T.simd.vld(x[r, i * 256 + 128])
                                av = T.simd.vcvt(low, "float32", part=0)
                                bv = T.simd.vcvt(low, "float32", part=1)
                                cv = T.simd.vcvt(high, "float32", part=0)
                                dv = T.simd.vcvt(high, "float32", part=1)
                            ab = T.simd.vmax(T.simd.vabs(av, mask), T.simd.vabs(bv, mask), mask)
                            cd = T.simd.vmax(T.simd.vabs(cv, mask), T.simd.vabs(dv, mask), mask)
                            largest[0] = T.simd.vmax(largest[0], T.simd.vmax(ab, cd, mask), mask)
                        peak = T.simd.vdupv(T.simd.vcmax(largest[0], mask), mask)
                    big = T.simd.pand(
                        T.simd.vcmps(peak, T.float32(1e15), mask, op="gt"),
                        T.simd.vcmps(peak, T.float32(float("inf")), mask, op="lt"),
                        mask,
                    )
                    small = T.simd.pand(
                        T.simd.vcmps(peak, T.float32(1e-15), mask, op="lt"),
                        T.simd.vcmps(peak, T.float32(0), mask, op="gt"),
                        mask,
                    )
                    factor = T.simd.vsel(T.simd.vdup(T.float32(2.0**-96), "float32"), factor, big)
                    factor = T.simd.vsel(T.simd.vdup(T.float32(2.0**96), "float32"), factor, small)
                    unscale = T.simd.vsel(T.simd.vdup(T.float32(2.0**96), "float32"), unscale, big)
                    unscale = T.simd.vsel(
                        T.simd.vdup(T.float32(2.0**-96), "float32"), unscale, small
                    )
                    anchor = T.simd.vmul(anchor, factor, mask)
                a = T.simd.alloc_local(1, "float32")
                b = T.simd.alloc_local(1, "float32")
                c = T.simd.alloc_local(1, "float32")
                d = T.simd.alloc_local(1, "float32")
                a[0] = T.simd.vdup(T.float32(0), "float32")
                b[0] = T.simd.vdup(T.float32(0), "float32")
                c[0] = T.simd.vdup(T.float32(0), "float32")
                d[0] = T.simd.vdup(T.float32(0), "float32")
                for i in range(n // 256):
                    if dtype == "float32":
                        av = T.simd.vld(x[r, i * 256])
                        bv = T.simd.vld(x[r, i * 256 + 64])
                        cv = T.simd.vld(x[r, i * 256 + 128])
                        dv = T.simd.vld(x[r, i * 256 + 192])
                    else:
                        low = T.simd.vld(x[r, i * 256])
                        high = T.simd.vld(x[r, i * 256 + 128])
                        av = T.simd.vcvt(low, "float32", part=0)
                        bv = T.simd.vcvt(low, "float32", part=1)
                        cv = T.simd.vcvt(high, "float32", part=0)
                        dv = T.simd.vcvt(high, "float32", part=1)
                    if dtype != "float16":
                        av = T.simd.vmul(av, factor, mask)
                        bv = T.simd.vmul(bv, factor, mask)
                        cv = T.simd.vmul(cv, factor, mask)
                        dv = T.simd.vmul(dv, factor, mask)
                    a[0] = T.simd.vadd(a[0], T.simd.vsub(av, anchor, mask), mask)
                    b[0] = T.simd.vadd(b[0], T.simd.vsub(bv, anchor, mask), mask)
                    c[0] = T.simd.vadd(c[0], T.simd.vsub(cv, anchor, mask), mask)
                    d[0] = T.simd.vadd(d[0], T.simd.vsub(dv, anchor, mask), mask)
                total = T.simd.vcadd(
                    T.simd.vadd(T.simd.vadd(a[0], b[0], mask), T.simd.vadd(c[0], d[0], mask), mask),
                    mask,
                )
                shift_mean = T.simd.vdupv(T.simd.vmuls(total, T.float32(1.0 / n), mask), mask)
                a[0] = T.simd.vdup(T.float32(0), "float32")
                b[0] = T.simd.vdup(T.float32(0), "float32")
                c[0] = T.simd.vdup(T.float32(0), "float32")
                d[0] = T.simd.vdup(T.float32(0), "float32")
                for i in range(n // 256):
                    if dtype == "float32":
                        av = T.simd.vld(x[r, i * 256])
                        bv = T.simd.vld(x[r, i * 256 + 64])
                        cv = T.simd.vld(x[r, i * 256 + 128])
                        dv = T.simd.vld(x[r, i * 256 + 192])
                    else:
                        low = T.simd.vld(x[r, i * 256])
                        high = T.simd.vld(x[r, i * 256 + 128])
                        av = T.simd.vcvt(low, "float32", part=0)
                        bv = T.simd.vcvt(low, "float32", part=1)
                        cv = T.simd.vcvt(high, "float32", part=0)
                        dv = T.simd.vcvt(high, "float32", part=1)
                    if dtype != "float16":
                        av = T.simd.vmul(av, factor, mask)
                        bv = T.simd.vmul(bv, factor, mask)
                        cv = T.simd.vmul(cv, factor, mask)
                        dv = T.simd.vmul(dv, factor, mask)
                    adelta = T.simd.vsub(T.simd.vsub(av, anchor, mask), shift_mean, mask)
                    asquare = T.simd.vmul(adelta, adelta, mask)
                    a[0] = T.simd.vadd(a[0], asquare, mask)
                    bdelta = T.simd.vsub(T.simd.vsub(bv, anchor, mask), shift_mean, mask)
                    bsquare = T.simd.vmul(bdelta, bdelta, mask)
                    b[0] = T.simd.vadd(b[0], bsquare, mask)
                    cdelta = T.simd.vsub(T.simd.vsub(cv, anchor, mask), shift_mean, mask)
                    csquare = T.simd.vmul(cdelta, cdelta, mask)
                    c[0] = T.simd.vadd(c[0], csquare, mask)
                    ddelta = T.simd.vsub(T.simd.vsub(dv, anchor, mask), shift_mean, mask)
                    dsquare = T.simd.vmul(ddelta, ddelta, mask)
                    d[0] = T.simd.vadd(d[0], dsquare, mask)
                total = T.simd.vcadd(
                    T.simd.vadd(T.simd.vadd(a[0], b[0], mask), T.simd.vadd(c[0], d[0], mask), mask),
                    mask,
                )
                variance = T.simd.vsqrt(
                    T.simd.vmuls(total, T.float32(1.0 / (n - correction)), mask), mask
                )
                variance = T.simd.vmul(variance, unscale, mask)
                if dtype == "float32":
                    one = T.simd.pset(32, "PAT_VL1")
                    T.simd.vsts(y[r, 0], variance, one, extent=1)
                else:
                    narrowed = T.simd.vcvt(variance, dtype)
                    one = T.simd.pset(16, "PAT_VL1")
                    T.simd.vsts(y[r, 0], narrowed, one, extent=1)

    @functools.lru_cache(maxsize=256)
    def build_native(m, n, dtype, correction, layout_outer=0):
        rows = min(4, max(1, m // 64))
        while rows > 1 and 2 * rows * n * (4 if dtype == "float32" else 2) > 192 * 1024:
            rows //= 2
        while m % rows:
            rows //= 2
        groups = m // rows
        blocks = min(64, groups)
        input_shape = (layout_outer, m, n // layout_outer) if layout_outer else (m, n)

        @T.prim_func
        def main(X: T.Tensor(input_shape, dtype), Y: T.Tensor((m, 1), dtype)):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((rows, n), dtype)
                y = T.alloc_shared((rows, 16), dtype)
                T.annotate_buffer_versions({x: 2, y: 2})
                for step in T.Pipelined((groups + blocks - 1) // blocks, num_stages=2):
                    group = step * blocks + bx
                    if group < groups:
                        row = group * rows
                        if layout_outer:
                            for outer in T.serial(layout_outer):
                                T.copy(
                                    X[outer, row : row + rows, :],
                                    x[
                                        :,
                                        outer * (n // layout_outer) : (outer + 1)
                                        * (n // layout_outer),
                                    ],
                                )
                        else:
                            T.copy(X[row : row + rows, :], x)
                        _native_std(x, y, rows, n, dtype, correction)
                        T.copy(y[:, :1], Y[row : row + rows, :], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])

    @functools.lru_cache(maxsize=256)
    def build(m, n, dtype, correction):
        width = max(32, min(16384 if dtype == "float32" else 32768, tilelang.next_power_of_2(n)))
        pitch = min(width, (n + 15) // 16 * 16)
        rows = min(4, m)
        while rows > 1 and 2 * rows * pitch * (4 if dtype == "float32" else 2) > 192 * 1024:
            rows //= 2
        groups = (m + rows - 1) // rows
        blocks = min(64, groups)

        @T.prim_func
        def main(X: T.Tensor((m, n), dtype), Y: T.Tensor((m, 1), dtype)):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((rows, pitch), dtype)
                factor = T.alloc_shared((32,), "float32")
                mean = T.alloc_shared((32,), "float32")
                m2 = T.alloc_shared((32,), "float32")
                y = T.alloc_shared((rows, 16), dtype)
                T.annotate_buffer_versions({x: 2})
                for group in T.serial((groups + blocks - 1) // blocks):
                    row = (group * blocks + bx) * rows
                    if row < m:
                        valid_rows = T.min(rows, m - row)
                        T.clear(factor)
                        if dtype != "float16":
                            for tile in T.serial((n + width - 1) // width):
                                begin = tile * width
                                count = T.min(width, n - begin)
                                T.copy(
                                    X[row : row + valid_rows, begin : begin + count],
                                    x[:valid_rows, :count],
                                )
                                with T.SimtVF(threads=min(256, width)):
                                    peak = T.alloc_reducer((rows,), "float32", op="max")
                                    T.reducer_init(peak)
                                    for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                        if r < valid_rows and i < count:
                                            T.reducer_update(
                                                peak[r], T.abs(T.cast(x[r, i], "float32"))
                                            )
                                    maxima = T.alloc_fragment((rows,), "float32")
                                    T.finalize_reducer(peak, maxima)
                                    for r in T.Parallel(rows):
                                        factor[r] = T.max(factor[r], maxima[r])
                        with T.SimtVF(threads=32):
                            for r in T.Parallel(rows):
                                peak = factor[r]
                                factor[r] = T.if_then_else(
                                    peak > 1e15 and not T.isinf(peak),
                                    T.float32(2.0**-96),
                                    T.if_then_else(
                                        peak > 0 and peak < 1e-15, T.float32(2.0**96), T.float32(1)
                                    ),
                                )
                        T.clear(mean)
                        T.clear(m2)
                        for step in T.Pipelined((n + width - 1) // width, num_stages=2):
                            begin = step * width
                            count = T.min(width, n - begin)
                            T.copy(
                                X[row : row + valid_rows, begin : begin + count],
                                x[:valid_rows, :count],
                            )
                            with T.SimtVF(threads=min(256, width)):
                                shifted_sum = T.alloc_reducer((rows,), "float32", op="sum")
                                T.reducer_init(shifted_sum)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < count:
                                        T.reducer_update(
                                            shifted_sum[r],
                                            (T.cast(x[r, i], "float32") * factor[r])
                                            - (T.cast(x[r, 0], "float32") * factor[r]),
                                        )
                                summed = T.alloc_fragment((rows,), "float32")
                                T.finalize_reducer(shifted_sum, summed)
                                squared = T.alloc_reducer((rows,), "float32", op="sum")
                                T.reducer_init(squared)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < count:
                                        delta = (
                                            (T.cast(x[r, i], "float32") * factor[r])
                                            - (T.cast(x[r, 0], "float32") * factor[r])
                                        ) - summed[r] / count
                                        value = T.cast(x[r, i], "float32") * factor[r]
                                        anchor = T.cast(x[r, 0], "float32") * factor[r]
                                        finite_overflow = (
                                            T.isnan(delta)
                                            and not T.isnan(value)
                                            and not T.isinf(value)
                                            and not T.isnan(anchor)
                                            and not T.isinf(anchor)
                                        )
                                        T.reducer_update(
                                            squared[r],
                                            T.if_then_else(
                                                finite_overflow,
                                                T.float32(float("inf")),
                                                delta * delta,
                                            ),
                                        )
                                centered = T.alloc_fragment((rows,), "float32")
                                T.finalize_reducer(squared, centered)
                                for r in T.Parallel(rows):
                                    chunk_mean = (T.cast(x[r, 0], "float32") * factor[r]) + summed[
                                        r
                                    ] / count
                                    delta = chunk_mean - mean[r]
                                    if step == 0:
                                        mean[r] = chunk_mean
                                        m2[r] = centered[r]
                                    else:
                                        mean[r] = mean[r] + delta * (
                                            count / T.cast(begin + count, "float32")
                                        )
                                        combined = (
                                            m2[r]
                                            + centered[r]
                                            + delta
                                            * delta
                                            * (T.cast(begin, "float32") * count / (begin + count))
                                        )
                                        m2[r] = T.if_then_else(
                                            T.isnan(m2[r]) or T.isnan(centered[r]),
                                            T.float32(float("nan")),
                                            T.if_then_else(
                                                T.isinf(m2[r]) or T.isinf(centered[r]),
                                                T.float32(float("inf")),
                                                combined,
                                            ),
                                        )
                        with T.SimtVF(threads=32):
                            for r in T.Parallel(rows):
                                y[r, 0] = T.sqrt(m2[r] / (n - correction)) / factor[r]
                        T.copy(y[:valid_rows, :1], Y[row : row + valid_rows, :], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])


class AscendStdKernel(Kernel, VarianceFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        if call.op_kind != "std":
            raise ValueError("Expected std")
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("Unsupported variance dtype")
        cap = 16384 if call.dtype == torch.float32 else 32768
        factory = build_native if call.n <= cap and call.n % 256 == 0 else build
        # DMA segments require block-aligned inner strides; other layouts use the generic transform.
        self.direct_layout = (
            factory is build_native
            and len(call.shape) == 3
            and tuple(call.axes) == (0, 2)
            and call.shape[2] % 32 == 0
        )
        args = (call.m, call.n, str(call.dtype).split(".")[-1], call.correction)
        self.compiled = (
            factory(*args, layout_outer=call.shape[0]) if self.direct_layout else factory(*args)
        )

    def forward(self, x):
        if tuple(x.shape) != self.call.shape or x.dtype != self.call.dtype:
            raise ValueError("Input signature changed")
        result = self.compiled(x if self.direct_layout else rows_for_axes(x, self.call.axes))
        return restore_reduced(result, self.call.shape, self.call.axes, self.call.keepdim)


if T is not None:
    register_kernel_type("StdFwdOp", "ascend", AscendStdKernel)
