"""Ascend FP32 mean with bounded row tiles and split-axis partials."""

import functools
import math
from typing import ClassVar

import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import restore_reduced, rows_for_axes
from tileops.kernels.reduction.call_spec import ReduceFwdInterface

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None

if T is not None:

    @T.macro
    def _long_sum(x, y, width: int, out_dtype: str, denominator: int):
        with T.SimdVF():
            mask = T.simd.pset(32)
            a = T.simd.alloc_local(1, "float32")
            b = T.simd.alloc_local(1, "float32")
            c = T.simd.alloc_local(1, "float32")
            d = T.simd.alloc_local(1, "float32")
            a[0] = T.simd.vdup(T.float32(0), "float32")
            b[0] = T.simd.vdup(T.float32(0), "float32")
            c[0] = T.simd.vdup(T.float32(0), "float32")
            d[0] = T.simd.vdup(T.float32(0), "float32")
            for i in range(width // 256):
                low = T.simd.vld(x[0, i * 256])
                high = T.simd.vld(x[0, i * 256 + 128])
                av = T.simd.vcvt(low, "float32", part=0)
                bv = T.simd.vcvt(low, "float32", part=1)
                cv = T.simd.vcvt(high, "float32", part=0)
                dv = T.simd.vcvt(high, "float32", part=1)
                a[0] = T.simd.vadd(a[0], av, mask)
                b[0] = T.simd.vadd(b[0], bv, mask)
                c[0] = T.simd.vadd(c[0], cv, mask)
                d[0] = T.simd.vadd(d[0], dv, mask)
            total = T.simd.vcadd(
                T.simd.vadd(T.simd.vadd(a[0], b[0], mask), T.simd.vadd(c[0], d[0], mask), mask),
                mask,
            )
            total = T.simd.vmuls(total, T.float32(1.0 / denominator), mask)
            if out_dtype == "float32":
                one = T.simd.pset(32, "PAT_VL1")
                T.simd.vsts(y[0, 0], total, one, extent=1)
            else:
                one = T.simd.pset(16, "PAT_VL1")
                narrowed = T.simd.vcvt(total, out_dtype)
                T.simd.vsts(y[0, 0], narrowed, one, extent=1)

    @T.macro
    def _row_sum(x, y, width: int, rows: int, dtype: str, out_dtype: str, denominator: int):
        with T.SimdVF():
            for r in range(rows):
                mask = T.simd.pset(32)
                a = T.simd.alloc_local(1, "float32")
                b = T.simd.alloc_local(1, "float32")
                c = T.simd.alloc_local(1, "float32")
                d = T.simd.alloc_local(1, "float32")
                a[0] = T.simd.vdup(T.float32(0), "float32")
                b[0] = T.simd.vdup(T.float32(0), "float32")
                c[0] = T.simd.vdup(T.float32(0), "float32")
                d[0] = T.simd.vdup(T.float32(0), "float32")
                for i in range(width // 256):
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
                    a[0] = T.simd.vadd(a[0], av, mask)
                    b[0] = T.simd.vadd(b[0], bv, mask)
                    c[0] = T.simd.vadd(c[0], cv, mask)
                    d[0] = T.simd.vadd(d[0], dv, mask)
                total = T.simd.vcadd(
                    T.simd.vadd(T.simd.vadd(a[0], b[0], mask), T.simd.vadd(c[0], d[0], mask), mask),
                    mask,
                )
                total = T.simd.vmuls(total, T.float32(1.0 / denominator), mask)
                if out_dtype == "float32":
                    one = T.simd.pset(32, "PAT_VL1")
                    T.simd.vsts(y[r, 0], total, one, extent=1)
                else:
                    one = T.simd.pset(16, "PAT_VL1")
                    narrowed = T.simd.vcvt(total, out_dtype)
                    T.simd.vsts(y[r, 0], narrowed, one, extent=1)

    @functools.lru_cache(maxsize=256)
    def build(m, n, dtype, out_dtype, denominator=1):
        width = max(32, min(16384 if dtype == "float32" else 32768, tilelang.next_power_of_2(n)))
        chunks = (n + width - 1) // width
        pitch = min(width, (n + 15) // 16 * 16)
        rows = min(4, m)
        while rows > 1 and 2 * rows * pitch * (4 if dtype == "float32" else 2) > 192 * 1024:
            rows //= 2
        native = rows == 1 and n == width and width >= 16384 and dtype != "float32"
        native_rows = n % 256 == 0 and n <= width and m % rows == 0
        groups = (m + rows - 1) // rows
        blocks = min(64, groups * chunks)

        @T.prim_func
        def main(X: T.Tensor((m, n), dtype), Y: T.Tensor((m, chunks), out_dtype)):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((rows, pitch), dtype)
                y = T.alloc_shared((rows, 16), out_dtype)
                T.annotate_buffer_versions({x: 2, y: 2})
                for step in T.Pipelined((groups * chunks + blocks - 1) // blocks, num_stages=2):
                    work = step * blocks + bx
                    if work < groups * chunks:
                        row = (work // chunks) * rows
                        chunk = work % chunks
                        valid_rows = T.min(rows, m - row)
                        valid_cols = T.min(width, n - chunk * width)
                        T.copy(
                            X[row : row + valid_rows, chunk * width : chunk * width + valid_cols],
                            x[:valid_rows, :valid_cols],
                        )
                        if native_rows:
                            _row_sum(x, y, n, rows, dtype, out_dtype, denominator)
                        elif native:
                            _long_sum(x, y, width, out_dtype, denominator)
                        else:
                            with T.SimtVF(threads=min(256, width)):
                                acc = T.alloc_reducer((rows,), "float32", op="sum")
                                T.reducer_init(acc)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < valid_cols:
                                        T.reducer_update(acc[r], T.cast(x[r, i], "float32"))
                                result = T.alloc_fragment((rows,), "float32")
                                T.finalize_reducer(acc, result)
                                for r in T.Parallel(rows):
                                    y[r, 0] = result[r] / denominator
                        T.copy(
                            y[:valid_rows, :1],
                            Y[row : row + valid_rows, chunk : chunk + 1],
                            l2_cache_ctrl=0,
                        )

        return tilelang.compile(main, target="ascend", out_idx=[1]), chunks

    @functools.lru_cache(maxsize=256)
    def build_leading(m, n, dtype, out_dtype):
        cols, tile_r = 64, 128
        blocks = min(64, (m + cols - 1) // cols)
        groups = (m + cols - 1) // cols

        @T.prim_func
        def main(X: T.Tensor((n, m), dtype), Y: T.Tensor((m,), out_dtype)):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((tile_r, cols), dtype)
                acc = T.alloc_shared((cols,), "float32")
                y = T.alloc_shared((cols,), out_dtype)
                T.annotate_buffer_versions({x: 2})
                for group in T.serial((groups + blocks - 1) // blocks):
                    col = (group * blocks + bx) * cols
                    if col < m:
                        valid_cols = T.min(cols, m - col)
                        T.clear(acc)
                        for step in T.Pipelined((n + tile_r - 1) // tile_r, num_stages=2):
                            begin = step * tile_r
                            valid_rows = T.min(tile_r, n - begin)
                            T.copy(
                                X[begin : begin + valid_rows, col : col + valid_cols],
                                x[:valid_rows, :valid_cols],
                            )
                            with T.SimtVF(threads=256):
                                total = T.alloc_reducer((cols,), "float32", op="sum")
                                T.reducer_init(total)
                                for r, c in T.Parallel(tile_r, cols, coalesced_width=T.int32(1)):
                                    if r < valid_rows and c < valid_cols:
                                        T.reducer_update(total[c], T.cast(x[r, c], "float32"))
                                reduced = T.alloc_fragment((cols,), "float32")
                                T.finalize_reducer(total, reduced)
                                for c in T.Parallel(cols):
                                    acc[c] = acc[c] + reduced[c]
                        with T.SimtVF(threads=64):
                            for c in T.Parallel(cols):
                                y[c] = acc[c] / n
                        T.copy(y[:valid_cols], Y[col : col + valid_cols], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])

    @functools.lru_cache(maxsize=256)
    def build_edge(outer, m, inner, dtype, out_dtype):
        width = max(
            32, min(16384 if dtype == "float32" else 32768, tilelang.next_power_of_2(inner))
        )
        chunks = (inner + width - 1) // width
        rows = min(2, m) if width <= 8192 else 1
        groups = (m + rows - 1) // rows
        blocks = min(64, groups)

        @T.prim_func
        def main(X: T.Tensor((outer, m, inner), dtype), Y: T.Tensor((m,), out_dtype)):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((rows, width), dtype)
                acc = T.alloc_shared((32,), "float32")
                y = T.alloc_shared((32,), out_dtype)
                T.annotate_buffer_versions({x: 2})
                for group in T.serial((groups + blocks - 1) // blocks):
                    row = (group * blocks + bx) * rows
                    if row < m:
                        valid_rows = T.min(rows, m - row)
                        T.clear(acc)
                        for step in T.Pipelined(outer * chunks, num_stages=2):
                            a = step // chunks
                            begin = (step % chunks) * width
                            valid_cols = T.min(width, inner - begin)
                            T.copy(
                                X[a, row : row + valid_rows, begin : begin + valid_cols],
                                x[:valid_rows, :valid_cols],
                            )
                            with T.SimtVF(threads=min(256, width)):
                                total = T.alloc_reducer((rows,), "float32", op="sum")
                                T.reducer_init(total)
                                for r, c in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and c < valid_cols:
                                        T.reducer_update(total[r], T.cast(x[r, c], "float32"))
                                reduced = T.alloc_fragment((rows,), "float32")
                                T.finalize_reducer(total, reduced)
                                for r in T.Parallel(rows):
                                    acc[r] = acc[r] + reduced[r]
                        with T.SimtVF(threads=32):
                            for r in T.Parallel(rows):
                                y[r] = acc[r] / (outer * inner)
                        T.copy(y[:valid_rows], Y[row : row + valid_rows], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])


class AscendMeanKernel(Kernel, ReduceFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        if call.op_kind != "mean":
            raise ValueError("Expected mean")
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("Unsupported mean dtype")
        dtype = str(call.dtype).split(".")[-1]
        out = str(call.out_dtype or call.dtype).split(".")[-1]
        kept = [i for i in range(len(call.shape)) if i not in call.axes]
        self.edge = (
            bool(kept)
            and kept[0] > 0
            and kept[-1] < len(call.shape) - 1
            and kept == list(range(kept[0], kept[-1] + 1))
        )
        if self.edge:
            self.outer = math.prod(call.shape[: kept[0]])
            self.inner = math.prod(call.shape[kept[-1] + 1 :])
            self.compiled = build_edge(self.outer, call.m, self.inner, dtype, out)
            self.finish = None
            self.chunks = 1
            self.leading = False
            return
        self.leading = call.axes == tuple(range(len(call.axes))) and len(call.axes) < len(
            call.shape
        )
        if self.leading:
            self.compiled = build_leading(call.m, call.n, dtype, out)
            self.finish = None
            self.chunks = 1
            return
        cap = 16384 if call.dtype == torch.float32 else 32768
        chunks = (call.n + cap - 1) // cap
        self.compiled, self.chunks = build(
            call.m, call.n, dtype, "float32" if chunks > 1 else out, 1 if chunks > 1 else call.n
        )
        self.finish = (
            build(call.m, self.chunks, "float32", out, call.n)[0] if self.chunks > 1 else None
        )

    def forward(self, x):
        if tuple(x.shape) != self.call.shape or x.dtype != self.call.dtype:
            raise ValueError("Input signature changed")
        rows = (
            x.reshape(self.outer, self.call.m, self.inner)
            if self.edge
            else (
                x.reshape(self.call.n, self.call.m)
                if self.leading
                else rows_for_axes(x, self.call.axes)
            )
        )
        result = self.compiled(rows)
        if self.finish is not None:
            result = self.finish(result)
        return restore_reduced(result, self.call.shape, self.call.axes, self.call.keepdim)


if T is not None:
    register_kernel_type("MeanFwdOp", "ascend", AscendMeanKernel)
