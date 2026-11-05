"""Ascend FP32 tree prod with bounded row tiles and split-axis partials."""

import functools
from typing import ClassVar

import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import restore_reduced, rows_for_axes
from tileops.kernels.reduction.call_spec import ProdFwdInterface

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None

if T is not None:

    @T.macro
    def _row_product(x, y, rows: int, width: int, dtype: str, out_dtype: str):
        with T.SimdVF():
            for r in T.serial(rows):
                mask = T.simd.pset(32)
                a = T.simd.alloc_local(1, "float32")
                b = T.simd.alloc_local(1, "float32")
                c = T.simd.alloc_local(1, "float32")
                d = T.simd.alloc_local(1, "float32")
                a[0] = T.simd.vdup(T.float32(1), "float32")
                b[0] = T.simd.vdup(T.float32(1), "float32")
                c[0] = T.simd.vdup(T.float32(1), "float32")
                d[0] = T.simd.vdup(T.float32(1), "float32")
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
                    a[0] = T.simd.vmul(a[0], av, mask)
                    b[0] = T.simd.vmul(b[0], bv, mask)
                    c[0] = T.simd.vmul(c[0], cv, mask)
                    d[0] = T.simd.vmul(d[0], dv, mask)
                total = T.simd.alloc_local(1, "float32")
                total[0] = T.simd.vmul(
                    T.simd.vmul(a[0], b[0], mask), T.simd.vmul(c[0], d[0], mask), mask
                )
                index = T.simd.vci(T.int32(0), "int32")
                for shift in T.unroll(6):
                    other = T.simd.vxor(index, T.simd.vdup(T.int32(1 << shift), "int32"), mask)
                    total[0] = T.simd.vmul(total[0], T.simd.vselr(total[0], other), mask)
                if out_dtype == "float32":
                    one = T.simd.pset(32, "PAT_VL1")
                    T.simd.vsts(y[r, 0], total[0], one, extent=1)
                else:
                    one = T.simd.pset(16, "PAT_VL1")
                    narrowed = T.simd.vcvt(total[0], out_dtype)
                    T.simd.vsts(y[r, 0], narrowed, one, extent=1)

    @functools.lru_cache(maxsize=256)
    def build(m, n, dtype, out_dtype):
        width = max(32, min(16384 if dtype == "float32" else 32768, tilelang.next_power_of_2(n)))
        chunks = (n + width - 1) // width
        pitch = min(width, (n + 15) // 16 * 16)
        rows = min(4, m)
        if rows == 3:
            rows = 2
        while rows > 1 and 2 * rows * pitch * (4 if dtype == "float32" else 2) > 192 * 1024:
            rows //= 2
        native = chunks == 1 and n % 256 == 0 and m % rows == 0
        lanes = 256 // rows
        warps = lanes // 32
        groups = (m + rows - 1) // rows
        blocks = min(64, groups * chunks)

        @T.prim_func
        def main(X: T.Tensor((m, n), dtype), Y: T.Tensor((m, chunks), out_dtype)):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((rows, pitch), dtype)
                y = T.alloc_shared((rows, 16), out_dtype)
                partial = T.alloc_shared((rows, 32), "float32")
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
                        if native:
                            _row_product(x, y, rows, n, dtype, out_dtype)
                        else:
                            with T.SimtVF(threads=256):
                                tx = T.get_thread_binding()
                                r = tx // lanes
                                lane = tx % lanes
                                acc = T.alloc_local((1,), "float32")
                                acc[0] = T.float32(1)
                                for i in T.serial((width + lanes - 1) // lanes):
                                    col = i * lanes + lane
                                    if r < valid_rows and col < valid_cols:
                                        acc[0] = acc[0] * T.cast(x[r, col], "float32")
                                for shift in T.unroll(5):
                                    acc[0] = (
                                        acc[0]
                                        * T.call_extern(  # lint: extern-target allow: CANN shuffle has no wrapper in the pinned Ascend API.
                                            "float32", "asc_shfl_xor", acc[0], T.int32(1) << shift
                                        )
                                    )
                                if lane % 32 == 0:
                                    partial[r, lane // 32] = acc[0]
                            with T.SimtVF(threads=32):
                                acc = T.alloc_local((1,), "float32")
                                for r in T.Parallel(rows):
                                    acc[0] = T.float32(1)
                                    for w in T.serial(warps):
                                        acc[0] = acc[0] * partial[r, w]
                                    y[r, 0] = acc[0]
                        T.copy(
                            y[:valid_rows, :1],
                            Y[row : row + valid_rows, chunk : chunk + 1],
                            l2_cache_ctrl=0,
                        )

        return tilelang.compile(main, target="ascend", out_idx=[1]), chunks

    @functools.lru_cache(maxsize=256)
    def build_short_leading(m, n, dtype, out_dtype):
        cols = min(8192, max(32, tilelang.next_power_of_2(m)))
        itemsize = 4 if dtype == "float32" else 2
        outsize = 4 if out_dtype == "float32" else 2
        while 2 * cols * (n * itemsize + outsize) > 192 * 1024:
            cols //= 2
        groups = (m + cols - 1) // cols
        blocks = min(64, groups)

        @T.prim_func
        def main(X: T.Tensor((n, m), dtype), Y: T.Tensor((m,), out_dtype)):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((n, cols), dtype)
                y = T.alloc_shared((cols,), out_dtype)
                T.annotate_buffer_versions({x: 2, y: 2})
                for group in T.Pipelined((groups + blocks - 1) // blocks, num_stages=2):
                    col = (group * blocks + bx) * cols
                    if col < m:
                        valid_cols = T.min(cols, m - col)
                        T.copy(X[:, col : col + valid_cols], x[:, :valid_cols])
                        with T.SimtVF(threads=256):
                            value = T.alloc_local((1,), "float32")
                            for c in T.Parallel(cols):
                                if c < valid_cols:
                                    value[0] = T.float32(1)
                                    for r in T.unroll(n):
                                        value[0] = value[0] * T.cast(x[r, c], "float32")
                                    y[c] = value[0]
                        T.copy(y[:valid_cols], Y[col : col + valid_cols], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])

    @functools.lru_cache(maxsize=256)
    def build_leading(m, n, dtype, out_dtype):
        cols, tile_r = (1024 if n <= 16 else 64), min(128, n)
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
                        T.fill(acc, T.float32(1))
                        for step in T.Pipelined((n + tile_r - 1) // tile_r, num_stages=2):
                            begin = step * tile_r
                            valid_rows = T.min(tile_r, n - begin)
                            T.copy(
                                X[begin : begin + valid_rows, col : col + valid_cols],
                                x[:valid_rows, :valid_cols],
                            )
                            with T.SimtVF(threads=min(256, cols)):
                                value = T.alloc_local((1,), "float32")
                                for c in T.Parallel(cols):
                                    value[0] = acc[c]
                                    for r in T.serial(tile_r):
                                        if r < valid_rows and c < valid_cols:
                                            value[0] = value[0] * T.cast(x[r, c], "float32")
                                    acc[c] = value[0]
                        with T.SimtVF(threads=min(256, cols)):
                            for c in T.Parallel(cols):
                                y[c] = acc[c]
                        T.copy(y[:valid_cols], Y[col : col + valid_cols], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])


class AscendProdKernel(Kernel, ProdFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        if call.op_kind != "prod":
            raise ValueError("Expected prod")
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("Unsupported prod dtype")
        dtype = str(call.dtype).split(".")[-1]
        out = str(call.out_dtype or call.dtype).split(".")[-1]
        self.leading = call.axes == tuple(range(len(call.axes))) and len(call.axes) < len(
            call.shape
        )
        if self.leading:
            self.compiled = (build_short_leading if call.n <= 16 else build_leading)(
                call.m, call.n, dtype, out
            )
            self.finish = None
            self.chunks = 1
            return
        cap = 16384 if call.dtype == torch.float32 else 32768
        chunks = (call.n + cap - 1) // cap
        self.compiled, self.chunks = build(call.m, call.n, dtype, "float32" if chunks > 1 else out)
        self.finish = build(call.m, self.chunks, "float32", out)[0] if self.chunks > 1 else None

    def forward(self, x):
        if tuple(x.shape) != self.call.shape or x.dtype != self.call.dtype:
            raise ValueError("Input signature changed")
        rows = (
            x.reshape(self.call.n, self.call.m)
            if self.leading
            else rows_for_axes(x, self.call.axes)
        )
        result = self.compiled(rows)
        if self.finish is not None:
            result = self.finish(result)
        return restore_reduced(result, self.call.shape, self.call.axes, self.call.keepdim)


if T is not None:
    register_kernel_type("ProdFwdOp", "ascend", AscendProdKernel)
