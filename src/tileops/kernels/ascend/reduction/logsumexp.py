"""Stable Ascend logsumexp with bounded tiles and FP32 scaled partial sums."""

import functools
import math
from typing import ClassVar

import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import restore_reduced, rows_for_axes
from tileops.kernels.reduction.call_spec import LogSumExpFwdInterface

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None

if T is not None:

    @T.macro
    def _native_stats(x, y, rows: int, count, dtype: str):
        with T.SimdVF():
            mask = T.simd.pset(32)
            one = T.simd.pset(32, "PAT_VL1")  # noqa: F841 - preserve validated TileLang DSL structure
            zeros = T.simd.vdup(T.float32(0), "float32")
            for r in T.serial(rows):
                maximum = T.simd.alloc_local(2, "float32")
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
                reduced = T.simd.vcmax(T.simd.vmax(maximum[0], maximum[1], mask), mask)
                mx = T.simd.vdupv(reduced, mask)
                shift = T.simd.vsel(
                    zeros,
                    mx,
                    T.simd.vcmps(T.simd.vabs(mx, mask), T.infinity("float32"), mask, op="eq"),
                )
                acc = T.simd.alloc_local(2, "float32")
                acc[0] = zeros
                acc[1] = zeros
                for i in T.serial(count // 128):
                    if dtype == "float32":
                        av = T.simd.vld(x[r, i * 128])
                        bv = T.simd.vld(x[r, i * 128 + 64])
                    else:
                        packed = T.simd.vld(x[r, i * 128])
                        av = T.simd.vcvt(packed, "float32", part=0)
                        bv = T.simd.vcvt(packed, "float32", part=1)
                    acc[0] = T.simd.vadd(
                        acc[0], T.simd.vexp(T.simd.vsub(av, shift, mask), mask), mask
                    )
                    acc[1] = T.simd.vadd(
                        acc[1], T.simd.vexp(T.simd.vsub(bv, shift, mask), mask), mask
                    )
                total = T.simd.vcadd(T.simd.vadd(acc[0], acc[1], mask), mask)
                summed = T.simd.vdupv(total, mask)
                packed, unused = T.simd.vintlv(mx, summed)
                two = T.simd.pset(32, "PAT_VL2")
                T.simd.vsts(y[r, 0], packed, two, extent=2)

    @T.macro
    def _native_result(x, y, rows: int, count, dtype: str):
        with T.SimdVF():
            mask = T.simd.pset(32)
            one = T.simd.pset(32, "PAT_VL1")
            zeros = T.simd.vdup(T.float32(0), "float32")
            for r in T.serial(rows):
                maximum = T.simd.alloc_local(2, "float32")
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
                reduced = T.simd.vcmax(T.simd.vmax(maximum[0], maximum[1], mask), mask)
                mx = T.simd.vdupv(reduced, mask)
                shift = T.simd.vsel(
                    zeros,
                    mx,
                    T.simd.vcmps(T.simd.vabs(mx, mask), T.infinity("float32"), mask, op="eq"),
                )
                acc = T.simd.alloc_local(2, "float32")
                acc[0] = zeros
                acc[1] = zeros
                for i in T.serial(count // 128):
                    if dtype == "float32":
                        av = T.simd.vld(x[r, i * 128])
                        bv = T.simd.vld(x[r, i * 128 + 64])
                    else:
                        packed = T.simd.vld(x[r, i * 128])
                        av = T.simd.vcvt(packed, "float32", part=0)
                        bv = T.simd.vcvt(packed, "float32", part=1)
                    acc[0] = T.simd.vadd(
                        acc[0], T.simd.vexp(T.simd.vsub(av, shift, mask), mask), mask
                    )
                    acc[1] = T.simd.vadd(
                        acc[1], T.simd.vexp(T.simd.vsub(bv, shift, mask), mask), mask
                    )
                total = T.simd.vcadd(T.simd.vadd(acc[0], acc[1], mask), mask)
                summed = T.simd.vdupv(total, mask)
                value = T.simd.vadd(shift, T.simd.vln(summed, mask, precision="ftz_true"), mask)
                if dtype == "float32":
                    T.simd.vsts(y[r, 0], value, one, extent=1)
                else:
                    T.simd.vsts(
                        y[r, 0], T.simd.vcvt(value, dtype, sat=False), one, dist="PK_B32", extent=1
                    )

    @functools.lru_cache(maxsize=256)
    def build_fused(m, n, dtype):
        size = 4 if dtype == "float32" else 2
        rows = min(4, m)
        while rows > 1 and (m % rows or 2 * rows * n * size > 184 * 1024):
            rows //= 2
        groups = m // rows
        blocks = min(64, groups)

        @T.prim_func
        def main(X: T.Tensor((m, n), dtype), Y: T.Tensor((m, 1), dtype)):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((rows, n), dtype)
                y = T.alloc_shared((rows, 16), dtype)
                T.annotate_buffer_versions({x: 2, y: 2})
                for step in T.Pipelined((groups + blocks - 1) // blocks, num_stages=2):
                    group = step * blocks + bx
                    if group < groups:
                        row = group * rows
                        T.copy(X[row : row + rows, :], x)
                        _native_result(x, y, rows, n, dtype)
                        T.copy(y[:, :1], Y[row : row + rows, :], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])

    @functools.lru_cache(maxsize=256)
    def build(m, n, dtype, outer, inner):
        extent = inner if outer else n
        width = max(32, min(4096 if m <= 64 else 8192, tilelang.next_power_of_2(extent)))
        parts = (extent + width - 1) // width
        chunks = parts * (outer if outer else 1)
        rows = min(2, m) if m > 64 else 1
        pitch = min(width, (extent + 15) // 16 * 16)
        native = extent % 128 == 0 and m % rows == 0
        tasks = ((m + rows - 1) // rows) * chunks
        blocks = min(64, tasks)
        shape = (outer, m, inner) if outer else (m, n)

        @T.prim_func
        def main(X: T.Tensor(shape, dtype), Y: T.Tensor((m, chunks, 8), "float32")):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((rows, pitch), dtype)
                y = T.alloc_shared((rows, 8), "float32")
                T.annotate_buffer_versions({x: 2, y: 2})
                for step in T.Pipelined((tasks + blocks - 1) // blocks, num_stages=2):
                    task = step * blocks + bx
                    if task < tasks:
                        row = (task // chunks) * rows
                        chunk = task % chunks
                        begin = (chunk % parts) * width
                        valid_rows = T.min(rows, m - row)
                        count = T.min(width, extent - begin)
                        if outer:
                            T.copy(
                                X[chunk // parts, row : row + valid_rows, begin : begin + count],
                                x[:valid_rows, :count],
                            )
                        else:
                            T.copy(
                                X[row : row + valid_rows, begin : begin + count],
                                x[:valid_rows, :count],
                            )
                        if native:
                            _native_stats(x, y, rows, count, dtype)
                        else:
                            with T.SimtVF(threads=min(256, width)):
                                maximum = T.alloc_reducer((rows,), "float32", op="max")
                                T.reducer_init(maximum)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < count:
                                        T.reducer_update(maximum[r], T.cast(x[r, i], "float32"))
                                mx = T.alloc_fragment((rows,), "float32")
                                T.finalize_reducer(maximum, mx)
                                total = T.alloc_reducer((rows,), "float32", op="sum")
                                T.reducer_init(total)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < count:
                                        value = T.cast(x[r, i], "float32")
                                        shift = T.if_then_else(
                                            T.abs(mx[r]) == T.infinity("float32"),
                                            T.float32(0),
                                            mx[r],
                                        )
                                        T.reducer_update(total[r], T.exp(value - shift))
                                sm = T.alloc_fragment((rows,), "float32")
                                T.finalize_reducer(total, sm)
                                for r in T.Parallel(rows):
                                    y[r, 0] = mx[r]
                                    y[r, 1] = sm[r]
                        T.copy(
                            y[:valid_rows, :2],
                            Y[row : row + valid_rows, chunk, :2],
                            l2_cache_ctrl=0,
                        )

        return tilelang.compile(main, target="ascend", out_idx=[1]), chunks

    @functools.lru_cache(maxsize=256)
    def build_finish(m, chunks, dtype):
        width = max(32, tilelang.next_power_of_2(chunks))
        rows = min(4, m)
        groups = (m + rows - 1) // rows
        blocks = min(64, groups)

        @T.prim_func
        def main(X: T.Tensor((m, chunks, 8), "float32"), Y: T.Tensor((m, 1), dtype)):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((rows, chunks, 8), "float32")
                y = T.alloc_shared((rows, 16), dtype)
                for group in T.serial((groups + blocks - 1) // blocks):
                    row = (group * blocks + bx) * rows
                    if row < m:
                        valid_rows = T.min(rows, m - row)
                        T.copy(X[row : row + valid_rows, :, :], x[:valid_rows, :, :])
                        with T.SimtVF(threads=min(256, width)):
                            maximum = T.alloc_reducer((rows,), "float32", op="max")
                            T.reducer_init(maximum)
                            for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                if r < valid_rows and i < chunks:
                                    T.reducer_update(maximum[r], x[r, i, 0])
                            mx = T.alloc_fragment((rows,), "float32")
                            T.finalize_reducer(maximum, mx)
                            total = T.alloc_reducer((rows,), "float32", op="sum")
                            T.reducer_init(total)
                            for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                if r < valid_rows and i < chunks:
                                    # Infinite local maxima used shift zero in the first stage.
                                    local = T.if_then_else(
                                        T.abs(x[r, i, 0]) == T.infinity("float32"),
                                        T.float32(0),
                                        x[r, i, 0],
                                    )
                                    global_shift = T.if_then_else(
                                        T.abs(mx[r]) == T.infinity("float32"), T.float32(0), mx[r]
                                    )
                                    T.reducer_update(
                                        total[r],
                                        T.if_then_else(
                                            x[r, i, 1] == 0,
                                            T.float32(0),
                                            x[r, i, 1] * T.exp(local - global_shift),
                                        ),
                                    )
                            sm = T.alloc_fragment((rows,), "float32")
                            T.finalize_reducer(total, sm)
                            for r in T.Parallel(rows):
                                shift = T.if_then_else(
                                    T.abs(mx[r]) == T.infinity("float32"), T.float32(0), mx[r]
                                )
                                y[r, 0] = T.cast(shift + T.log(sm[r]), dtype)
                        T.copy(y[:valid_rows, :1], Y[row : row + valid_rows, :], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])


class AscendLogSumExpKernel(Kernel, LogSumExpFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("Unsupported logsumexp dtype")
        self.m, self.n = call.m, call.n
        dtype = str(call.dtype).split(".")[-1]
        kept = [i for i in range(len(call.shape)) if i not in call.axes]
        self.edge = (
            bool(kept)
            and kept == list(range(kept[0], kept[-1] + 1))
            and kept[0] > 0
            and kept[-1] < len(call.shape) - 1
        )
        self.outer = math.prod(call.shape[: kept[0]]) if self.edge else 0
        self.inner = math.prod(call.shape[kept[-1] + 1 :]) if self.edge else 0
        self.fused = not self.edge and self.m >= 64 and self.n % 128 == 0 and self.n <= 8192
        if self.fused:
            self.compiled = build_fused(self.m, self.n, dtype)
            self.chunks = 1
            self.finish = None
        else:
            self.compiled, self.chunks = build(self.m, self.n, dtype, self.outer, self.inner)
            self.finish = build_finish(self.m, self.chunks, dtype)

    def forward(self, x):
        if tuple(x.shape) != self.call.shape or x.dtype != self.call.dtype:
            raise ValueError("Input signature changed")
        arg = (
            x.reshape(self.outer, self.m, self.inner)
            if self.edge
            else rows_for_axes(x, self.call.axes)
        )
        result = self.compiled(arg)
        if self.finish is not None:
            result = self.finish(result)
        return restore_reduced(result, self.call.shape, self.call.axes, self.call.keepdim)


if T is not None:
    register_kernel_type("LogSumExpFwdOp", "ascend", AscendLogSumExpKernel)
