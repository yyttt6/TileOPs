"""Ascend vector norms with FP32 folds and bitwise NaN-preserving infinity norm."""

import functools
import math
from typing import ClassVar

import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import restore_reduced, rows_for_axes
from tileops.kernels.reduction.call_spec import VectorNormFwdInterface

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None

if T is not None:

    @T.macro
    def _native_norm(x, y, rows: int, n: int, dtype: str, out_dtype: str, kind: str):
        with T.SimdVF():
            for r in T.serial(rows):
                mask = T.simd.pset(32)
                if kind == "inf":
                    a = T.simd.alloc_local(1, "int32")
                    b = T.simd.alloc_local(1, "int32")
                    c = T.simd.alloc_local(1, "int32")
                    d = T.simd.alloc_local(1, "int32")
                    a[0] = T.simd.vdup(T.int32(0), "int32")
                    b[0] = T.simd.vdup(T.int32(0), "int32")
                    c[0] = T.simd.vdup(T.int32(0), "int32")
                    d[0] = T.simd.vdup(T.int32(0), "int32")
                else:
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
                    if kind == "l2":
                        a[0] = T.simd.vadd(a[0], T.simd.vmul(av, av, mask), mask)
                        b[0] = T.simd.vadd(b[0], T.simd.vmul(bv, bv, mask), mask)
                        c[0] = T.simd.vadd(c[0], T.simd.vmul(cv, cv, mask), mask)
                        d[0] = T.simd.vadd(d[0], T.simd.vmul(dv, dv, mask), mask)
                    elif kind == "l1":
                        a[0] = T.simd.vadd(a[0], T.simd.vabs(av, mask), mask)
                        b[0] = T.simd.vadd(b[0], T.simd.vabs(bv, mask), mask)
                        c[0] = T.simd.vadd(c[0], T.simd.vabs(cv, mask), mask)
                        d[0] = T.simd.vadd(d[0], T.simd.vabs(dv, mask), mask)
                    else:
                        a[0] = T.simd.vmax(
                            a[0], T.reinterpret(T.simd.vabs(av, mask), "int32x64"), mask
                        )
                        b[0] = T.simd.vmax(
                            b[0], T.reinterpret(T.simd.vabs(bv, mask), "int32x64"), mask
                        )
                        c[0] = T.simd.vmax(
                            c[0], T.reinterpret(T.simd.vabs(cv, mask), "int32x64"), mask
                        )
                        d[0] = T.simd.vmax(
                            d[0], T.reinterpret(T.simd.vabs(dv, mask), "int32x64"), mask
                        )
                if kind == "inf":
                    bits = T.simd.vcmax(
                        T.simd.vmax(
                            T.simd.vmax(a[0], b[0], mask), T.simd.vmax(c[0], d[0], mask), mask
                        ),
                        mask,
                    )
                    result = T.reinterpret(bits, "float32x64")
                else:
                    total = T.simd.vcadd(
                        T.simd.vadd(
                            T.simd.vadd(a[0], b[0], mask), T.simd.vadd(c[0], d[0], mask), mask
                        ),
                        mask,
                    )
                    if kind == "l2":  # noqa: SIM108 - preserve validated TileLang DSL structure
                        result = T.simd.vsqrt(total, mask)
                    else:
                        result = total
                if out_dtype == "float32":
                    one = T.simd.pset(32, "PAT_VL1")
                    T.simd.vsts(y[r, 0], result, one, extent=1)
                else:
                    narrowed = T.simd.vcvt(result, out_dtype)
                    one = T.simd.pset(16, "PAT_VL1")
                    T.simd.vsts(y[r, 0], narrowed, one, extent=1)

    @functools.lru_cache(maxsize=256)
    def build_native(m, n, dtype, out_dtype, kind):
        rows = min(4, m)
        while rows > 1 and 2 * rows * n * (4 if dtype == "float32" else 2) > 192 * 1024:
            rows //= 2
        while m % rows:
            rows //= 2
        groups = m // rows
        blocks = min(64, groups)

        @T.prim_func
        def main(X: T.Tensor((m, n), dtype), Y: T.Tensor((m, 1), out_dtype)):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((rows, n), dtype)
                y = T.alloc_shared((rows, 16), out_dtype)
                T.annotate_buffer_versions({x: 2, y: 2})
                for step in T.Pipelined((groups + blocks - 1) // blocks, num_stages=2):
                    group = step * blocks + bx
                    if group < groups:
                        row = group * rows
                        T.copy(X[row : row + rows, :], x)
                        _native_norm(x, y, rows, n, dtype, out_dtype, kind)
                        T.copy(y[:, :1], Y[row : row + rows, :], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])

    @functools.lru_cache(maxsize=256)
    def build_edge(outer, m, inner, dtype, out_dtype, kind):
        n = outer * inner
        rows = min(4, m)
        while rows > 1 and 2 * rows * n * (4 if dtype == "float32" else 2) > 192 * 1024:
            rows //= 2
        while m % rows:
            rows //= 2
        groups = m // rows
        blocks = min(64, groups)

        @T.prim_func
        def main(X: T.Tensor((outer, m, inner), dtype), Y: T.Tensor((m, 1), out_dtype)):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((rows, n), dtype)
                y = T.alloc_shared((rows, 16), out_dtype)
                T.annotate_buffer_versions({x: 2, y: 2})
                for step in T.Pipelined((groups + blocks - 1) // blocks, num_stages=2):
                    group = step * blocks + bx
                    if group < groups:
                        row = group * rows
                        for a in T.serial(outer):
                            T.copy(X[a, row : row + rows, :], x[:, a * inner : (a + 1) * inner])
                        _native_norm(x, y, rows, n, dtype, out_dtype, kind)
                        T.copy(y[:, :1], Y[row : row + rows, :], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])

    @functools.lru_cache(maxsize=256)
    def build(m, n, dtype, out_dtype, kind):
        width = max(32, min(16384 if dtype == "float32" else 32768, tilelang.next_power_of_2(n)))
        pitch = min(width, (n + 15) // 16 * 16)
        rows = min(4, m)
        while rows > 1 and 2 * rows * pitch * (4 if dtype == "float32" else 2) > 192 * 1024:
            rows //= 2
        groups = (m + rows - 1) // rows
        blocks = min(64, groups)
        acc_dtype = "int32" if kind == "inf" else "float32"
        reduce_kind = "max" if kind == "inf" else "sum"

        @T.prim_func
        def main(X: T.Tensor((m, n), dtype), Y: T.Tensor((m, 1), out_dtype)):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((rows, pitch), dtype)
                acc = T.alloc_shared((32,), acc_dtype)
                y = T.alloc_shared((rows, 16), out_dtype)
                T.annotate_buffer_versions({x: 2})
                for group in T.serial((groups + blocks - 1) // blocks):
                    row = (group * blocks + bx) * rows
                    if row < m:
                        valid_rows = T.min(rows, m - row)
                        T.clear(acc)
                        for step in T.Pipelined((n + width - 1) // width, num_stages=2):
                            begin = step * width
                            count = T.min(width, n - begin)
                            T.copy(
                                X[row : row + valid_rows, begin : begin + count],
                                x[:valid_rows, :count],
                            )
                            with T.SimtVF(threads=min(256, width)):
                                fold = T.alloc_reducer((rows,), acc_dtype, op=reduce_kind)
                                T.reducer_init(fold)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < count:
                                        value = T.cast(x[r, i], "float32")
                                        if kind == "inf":
                                            T.reducer_update(
                                                fold[r], T.reinterpret(T.abs(value), "int32")
                                            )
                                        elif kind == "l2":
                                            T.reducer_update(fold[r], value * value)
                                        else:
                                            T.reducer_update(fold[r], T.abs(value))
                                partial = T.alloc_fragment((rows,), acc_dtype)
                                T.finalize_reducer(fold, partial)
                                for r in T.Parallel(rows):
                                    if kind == "inf":
                                        acc[r] = T.max(acc[r], partial[r])
                                    else:
                                        acc[r] = acc[r] + partial[r]
                        with T.SimtVF(threads=32):
                            for r in T.Parallel(rows):
                                if kind == "inf":
                                    y[r, 0] = T.reinterpret(acc[r], "float32")
                                elif kind == "l2":
                                    y[r, 0] = T.sqrt(acc[r])
                                else:
                                    y[r, 0] = acc[r]
                        T.copy(y[:valid_rows, :1], Y[row : row + valid_rows, :], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])


class AscendVectorNormKernel(Kernel, VectorNormFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        if call.op_kind not in ("l1", "l2", "inf"):
            raise ValueError("Unsupported norm order")
        output = call.out_dtype or call.dtype
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32) or output not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise TypeError("Unsupported norm dtype")
        cap = 16384 if call.dtype == torch.float32 else 32768
        kept = [i for i in range(len(call.shape)) if i not in call.axes]
        edge_axes = (
            bool(kept)
            and kept[0] > 0
            and kept[-1] < len(call.shape) - 1
            and kept == list(range(kept[0], kept[-1] + 1))
        )
        self.outer = math.prod(call.shape[: kept[0]]) if edge_axes else 1
        self.inner = math.prod(call.shape[kept[-1] + 1 :]) if edge_axes else call.n
        self.edge = edge_axes and call.n <= cap and call.n % 256 == 0 and self.inner % 16 == 0
        dtype = str(call.dtype).split(".")[-1]
        out_dtype = str(output).split(".")[-1]
        if self.edge:
            self.compiled = build_edge(
                self.outer, call.m, self.inner, dtype, out_dtype, call.op_kind
            )
        else:
            factory = build_native if call.n <= cap and call.n % 256 == 0 else build
            self.compiled = factory(call.m, call.n, dtype, out_dtype, call.op_kind)

    def forward(self, x):
        if tuple(x.shape) != self.call.shape or x.dtype != self.call.dtype:
            raise ValueError("Input signature changed")
        arg = (
            x.reshape(self.outer, self.call.m, self.inner)
            if self.edge
            else rows_for_axes(x, self.call.axes)
        )
        result = self.compiled(arg)
        return restore_reduced(result, self.call.shape, self.call.axes, self.call.keepdim)


if T is not None:
    register_kernel_type("VectorNormFwdOp", "ascend", AscendVectorNormKernel)
