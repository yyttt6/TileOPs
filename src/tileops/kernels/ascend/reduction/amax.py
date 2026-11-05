"""Ascend FP32 tree amax with bounded row tiles and split-axis partials."""

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

    def ordered_key(value):
        value = T.cast(value, "float32")
        bits = T.reinterpret(value, "int32")
        # Signed integer ordering matches numerical float ordering. Canonical
        # NaN sorts above positive infinity, so one max also propagates NaN.
        return T.if_then_else(
            T.isnan(value),
            T.int32(2147483647),
            T.if_then_else(bits < 0, bits ^ T.int32(2147483647), bits),
        )

    def ordered_value(key):
        bits = T.if_then_else(key < 0, key ^ T.int32(2147483647), key)
        return T.reinterpret(bits, "float32")

    @T.macro
    def _long_max(x, y, width: int, out_dtype: str):
        with T.SimdVF():
            mask = T.simd.pset(32)
            a = T.simd.alloc_local(1, "int32")
            b = T.simd.alloc_local(1, "int32")
            c = T.simd.alloc_local(1, "int32")
            d = T.simd.alloc_local(1, "int32")
            a[0] = T.simd.vdup(T.int32(-2147483648), "int32")
            b[0] = T.simd.vdup(T.int32(-2147483648), "int32")
            c[0] = T.simd.vdup(T.int32(-2147483648), "int32")
            d[0] = T.simd.vdup(T.int32(-2147483648), "int32")
            for i in range(width // 256):
                low = T.simd.vld(x[0, i * 256])
                high = T.simd.vld(x[0, i * 256 + 128])
                av = T.simd.vcvt(low, "float32", part=0)
                av_bits = T.reinterpret(av, "int32x64")
                av_neg = T.simd.vxor(av_bits, T.simd.vdup(T.int32(2147483647), "int32"), mask)
                av_key = T.simd.vsel(
                    av_neg, av_bits, T.simd.vcmps(av_bits, T.int32(0), mask, op="lt")
                )
                av_abs = T.simd.vand(av_bits, T.simd.vdup(T.int32(2147483647), "int32"), mask)
                av_keyn = T.simd.vsel(
                    T.simd.vdup(T.int32(2147483647), "int32"),
                    av_key,
                    T.simd.vcmps(av_abs, T.int32(2139095040), mask, op="gt"),
                )
                bv = T.simd.vcvt(low, "float32", part=1)
                bv_bits = T.reinterpret(bv, "int32x64")
                bv_neg = T.simd.vxor(bv_bits, T.simd.vdup(T.int32(2147483647), "int32"), mask)
                bv_key = T.simd.vsel(
                    bv_neg, bv_bits, T.simd.vcmps(bv_bits, T.int32(0), mask, op="lt")
                )
                bv_abs = T.simd.vand(bv_bits, T.simd.vdup(T.int32(2147483647), "int32"), mask)
                bv_keyn = T.simd.vsel(
                    T.simd.vdup(T.int32(2147483647), "int32"),
                    bv_key,
                    T.simd.vcmps(bv_abs, T.int32(2139095040), mask, op="gt"),
                )
                cv = T.simd.vcvt(high, "float32", part=0)
                cv_bits = T.reinterpret(cv, "int32x64")
                cv_neg = T.simd.vxor(cv_bits, T.simd.vdup(T.int32(2147483647), "int32"), mask)
                cv_key = T.simd.vsel(
                    cv_neg, cv_bits, T.simd.vcmps(cv_bits, T.int32(0), mask, op="lt")
                )
                cv_abs = T.simd.vand(cv_bits, T.simd.vdup(T.int32(2147483647), "int32"), mask)
                cv_keyn = T.simd.vsel(
                    T.simd.vdup(T.int32(2147483647), "int32"),
                    cv_key,
                    T.simd.vcmps(cv_abs, T.int32(2139095040), mask, op="gt"),
                )
                dv = T.simd.vcvt(high, "float32", part=1)
                dv_bits = T.reinterpret(dv, "int32x64")
                dv_neg = T.simd.vxor(dv_bits, T.simd.vdup(T.int32(2147483647), "int32"), mask)
                dv_key = T.simd.vsel(
                    dv_neg, dv_bits, T.simd.vcmps(dv_bits, T.int32(0), mask, op="lt")
                )
                dv_abs = T.simd.vand(dv_bits, T.simd.vdup(T.int32(2147483647), "int32"), mask)
                dv_keyn = T.simd.vsel(
                    T.simd.vdup(T.int32(2147483647), "int32"),
                    dv_key,
                    T.simd.vcmps(dv_abs, T.int32(2139095040), mask, op="gt"),
                )
                a[0] = T.simd.vmax(a[0], av_keyn, mask)
                b[0] = T.simd.vmax(b[0], bv_keyn, mask)
                c[0] = T.simd.vmax(c[0], cv_keyn, mask)
                d[0] = T.simd.vmax(d[0], dv_keyn, mask)
            total = T.simd.vcmax(
                T.simd.vmax(T.simd.vmax(a[0], b[0], mask), T.simd.vmax(c[0], d[0], mask), mask),
                mask,
            )
            flipped = T.simd.vxor(total, T.simd.vdup(T.int32(2147483647), "int32"), mask)
            decoded = T.simd.vsel(flipped, total, T.simd.vcmps(total, T.int32(0), mask, op="lt"))
            value = T.reinterpret(decoded, "float32x64")
            if out_dtype == "float32":
                one = T.simd.pset(32, "PAT_VL1")
                T.simd.vsts(y[0, 0], value, one, extent=1)
            else:
                one = T.simd.pset(16, "PAT_VL1")
                narrowed = T.simd.vcvt(value, out_dtype)
                T.simd.vsts(y[0, 0], narrowed, one, extent=1)

    @T.macro
    def _row_max(x, y, width: int, rows: int, dtype: str, out_dtype: str):
        with T.SimdVF():
            for r in range(rows):
                mask = T.simd.pset(32)
                a = T.simd.alloc_local(1, "int32")
                b = T.simd.alloc_local(1, "int32")
                c = T.simd.alloc_local(1, "int32")
                d = T.simd.alloc_local(1, "int32")
                a[0] = T.simd.vdup(T.int32(-2147483648), "int32")
                b[0] = T.simd.vdup(T.int32(-2147483648), "int32")
                c[0] = T.simd.vdup(T.int32(-2147483648), "int32")
                d[0] = T.simd.vdup(T.int32(-2147483648), "int32")
                for i in range(width // 256):
                    if dtype != "float32":
                        low = T.simd.vld(x[r, i * 256])
                        high = T.simd.vld(x[r, i * 256 + 128])
                    if dtype == "float32":
                        av = T.simd.vld(x[r, i * 256 + 0])
                    else:
                        av = T.simd.vcvt(low, "float32", part=0)
                    av_bits = T.reinterpret(av, "int32x64")
                    av_neg = T.simd.vxor(av_bits, T.simd.vdup(T.int32(2147483647), "int32"), mask)
                    av_key = T.simd.vsel(
                        av_neg, av_bits, T.simd.vcmps(av_bits, T.int32(0), mask, op="lt")
                    )
                    av_abs = T.simd.vand(av_bits, T.simd.vdup(T.int32(2147483647), "int32"), mask)
                    av_keyn = T.simd.vsel(
                        T.simd.vdup(T.int32(2147483647), "int32"),
                        av_key,
                        T.simd.vcmps(av_abs, T.int32(2139095040), mask, op="gt"),
                    )
                    if dtype == "float32":
                        bv = T.simd.vld(x[r, i * 256 + 64])
                    else:
                        bv = T.simd.vcvt(low, "float32", part=1)
                    bv_bits = T.reinterpret(bv, "int32x64")
                    bv_neg = T.simd.vxor(bv_bits, T.simd.vdup(T.int32(2147483647), "int32"), mask)
                    bv_key = T.simd.vsel(
                        bv_neg, bv_bits, T.simd.vcmps(bv_bits, T.int32(0), mask, op="lt")
                    )
                    bv_abs = T.simd.vand(bv_bits, T.simd.vdup(T.int32(2147483647), "int32"), mask)
                    bv_keyn = T.simd.vsel(
                        T.simd.vdup(T.int32(2147483647), "int32"),
                        bv_key,
                        T.simd.vcmps(bv_abs, T.int32(2139095040), mask, op="gt"),
                    )
                    if dtype == "float32":
                        cv = T.simd.vld(x[r, i * 256 + 128])
                    else:
                        cv = T.simd.vcvt(high, "float32", part=0)
                    cv_bits = T.reinterpret(cv, "int32x64")
                    cv_neg = T.simd.vxor(cv_bits, T.simd.vdup(T.int32(2147483647), "int32"), mask)
                    cv_key = T.simd.vsel(
                        cv_neg, cv_bits, T.simd.vcmps(cv_bits, T.int32(0), mask, op="lt")
                    )
                    cv_abs = T.simd.vand(cv_bits, T.simd.vdup(T.int32(2147483647), "int32"), mask)
                    cv_keyn = T.simd.vsel(
                        T.simd.vdup(T.int32(2147483647), "int32"),
                        cv_key,
                        T.simd.vcmps(cv_abs, T.int32(2139095040), mask, op="gt"),
                    )
                    if dtype == "float32":
                        dv = T.simd.vld(x[r, i * 256 + 192])
                    else:
                        dv = T.simd.vcvt(high, "float32", part=1)
                    dv_bits = T.reinterpret(dv, "int32x64")
                    dv_neg = T.simd.vxor(dv_bits, T.simd.vdup(T.int32(2147483647), "int32"), mask)
                    dv_key = T.simd.vsel(
                        dv_neg, dv_bits, T.simd.vcmps(dv_bits, T.int32(0), mask, op="lt")
                    )
                    dv_abs = T.simd.vand(dv_bits, T.simd.vdup(T.int32(2147483647), "int32"), mask)
                    dv_keyn = T.simd.vsel(
                        T.simd.vdup(T.int32(2147483647), "int32"),
                        dv_key,
                        T.simd.vcmps(dv_abs, T.int32(2139095040), mask, op="gt"),
                    )
                    a[0] = T.simd.vmax(a[0], av_keyn, mask)
                    b[0] = T.simd.vmax(b[0], bv_keyn, mask)
                    c[0] = T.simd.vmax(c[0], cv_keyn, mask)
                    d[0] = T.simd.vmax(d[0], dv_keyn, mask)
                total = T.simd.vcmax(
                    T.simd.vmax(T.simd.vmax(a[0], b[0], mask), T.simd.vmax(c[0], d[0], mask), mask),
                    mask,
                )
                flipped = T.simd.vxor(total, T.simd.vdup(T.int32(2147483647), "int32"), mask)
                decoded = T.simd.vsel(
                    flipped, total, T.simd.vcmps(total, T.int32(0), mask, op="lt")
                )
                value = T.reinterpret(decoded, "float32x64")
                if out_dtype == "float32":
                    one = T.simd.pset(32, "PAT_VL1")
                    T.simd.vsts(y[r, 0], value, one, extent=1)
                else:
                    one = T.simd.pset(16, "PAT_VL1")
                    narrowed = T.simd.vcvt(value, out_dtype)
                    T.simd.vsts(y[r, 0], narrowed, one, extent=1)

    @T.macro
    def _row_max_narrow(x, y, width: int, rows: int, dtype: str):
        with T.SimdVF():
            for r in range(rows):
                mask = T.simd.pset(16)
                signless = T.simd.vdup(T.int16(32767), "int16")
                infinity = T.int16(31744 if dtype == "float16" else 32640)
                a = T.simd.alloc_local(1, "int16")
                b = T.simd.alloc_local(1, "int16")
                a[0] = T.simd.vdup(T.int16(-32768), "int16")
                b[0] = T.simd.vdup(T.int16(-32768), "int16")
                for i in range(width // 256):
                    av_bits = T.reinterpret(T.simd.vld(x[r, i * 256]), "int16x128")
                    av_neg = T.simd.vxor(av_bits, signless, mask)
                    av_key = T.simd.vsel(
                        av_neg, av_bits, T.simd.vcmps(av_bits, T.int16(0), mask, op="lt")
                    )
                    av_abs = T.simd.vand(av_bits, signless, mask)
                    av_keyn = T.simd.vsel(
                        signless, av_key, T.simd.vcmps(av_abs, infinity, mask, op="gt")
                    )
                    bv_bits = T.reinterpret(T.simd.vld(x[r, i * 256 + 128]), "int16x128")
                    bv_neg = T.simd.vxor(bv_bits, signless, mask)
                    bv_key = T.simd.vsel(
                        bv_neg, bv_bits, T.simd.vcmps(bv_bits, T.int16(0), mask, op="lt")
                    )
                    bv_abs = T.simd.vand(bv_bits, signless, mask)
                    bv_keyn = T.simd.vsel(
                        signless, bv_key, T.simd.vcmps(bv_abs, infinity, mask, op="gt")
                    )
                    a[0] = T.simd.vmax(a[0], av_keyn, mask)
                    b[0] = T.simd.vmax(b[0], bv_keyn, mask)
                total = T.simd.vcmax(T.simd.vmax(a[0], b[0], mask), mask)
                flipped = T.simd.vxor(total, signless, mask)
                decoded = T.simd.vsel(
                    flipped, total, T.simd.vcmps(total, T.int16(0), mask, op="lt")
                )
                value = T.reinterpret(decoded, dtype + "x128")
                one = T.simd.pset(16, "PAT_VL1")
                T.simd.vsts(y[r, 0], value, one, extent=1)

    @functools.lru_cache(maxsize=256)
    def build(m, n, dtype, out_dtype):
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
                            if dtype != "float32" and dtype == out_dtype:
                                _row_max_narrow(x, y, n, rows, dtype)
                            else:
                                _row_max(x, y, n, rows, dtype, out_dtype)
                        elif native:
                            _long_max(x, y, width, out_dtype)
                        else:
                            with T.SimtVF(threads=min(256, width)):
                                acc = T.alloc_reducer((rows,), "int32", op="max")
                                T.reducer_init(acc)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < valid_cols:
                                        T.reducer_update(acc[r], ordered_key(x[r, i]))
                                result = T.alloc_fragment((rows,), "int32")
                                T.finalize_reducer(acc, result)
                                for r in T.Parallel(rows):
                                    y[r, 0] = ordered_value(result[r])
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
                        T.fill(acc, T.float32(float("-inf")))
                        for step in T.Pipelined((n + tile_r - 1) // tile_r, num_stages=2):
                            begin = step * tile_r
                            valid_rows = T.min(tile_r, n - begin)
                            T.copy(
                                X[begin : begin + valid_rows, col : col + valid_cols],
                                x[:valid_rows, :valid_cols],
                            )
                            with T.SimtVF(threads=256):
                                total = T.alloc_reducer((cols,), "int32", op="max")
                                T.reducer_init(total)
                                for r, c in T.Parallel(tile_r, cols, coalesced_width=T.int32(1)):
                                    if r < valid_rows and c < valid_cols:
                                        T.reducer_update(total[c], ordered_key(x[r, c]))
                                reduced = T.alloc_fragment((cols,), "int32")
                                T.finalize_reducer(total, reduced)
                                for c in T.Parallel(cols):
                                    acc[c] = T.if_then_else(
                                        T.isnan(acc[c]) or T.isnan(ordered_value(reduced[c])),
                                        T.float32(float("nan")),
                                        T.max(acc[c], ordered_value(reduced[c])),
                                    )
                        with T.SimtVF(threads=64):
                            for c in T.Parallel(cols):
                                y[c] = acc[c]
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
                        T.fill(acc, T.float32(float("-inf")))
                        for step in T.Pipelined(outer * chunks, num_stages=2):
                            a = step // chunks
                            begin = (step % chunks) * width
                            valid_cols = T.min(width, inner - begin)
                            T.copy(
                                X[a, row : row + valid_rows, begin : begin + valid_cols],
                                x[:valid_rows, :valid_cols],
                            )
                            with T.SimtVF(threads=min(256, width)):
                                total = T.alloc_reducer((rows,), "int32", op="max")
                                T.reducer_init(total)
                                for r, c in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and c < valid_cols:
                                        T.reducer_update(total[r], ordered_key(x[r, c]))
                                reduced = T.alloc_fragment((rows,), "int32")
                                T.finalize_reducer(total, reduced)
                                for r in T.Parallel(rows):
                                    acc[r] = T.if_then_else(
                                        T.isnan(acc[r]) or T.isnan(ordered_value(reduced[r])),
                                        T.float32(float("nan")),
                                        T.max(acc[r], ordered_value(reduced[r])),
                                    )
                        with T.SimtVF(threads=32):
                            for r in T.Parallel(rows):
                                y[r] = acc[r]
                        T.copy(y[:valid_rows], Y[row : row + valid_rows], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])


class AscendAmaxKernel(Kernel, ReduceFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        if call.op_kind != "amax":
            raise ValueError("Expected amax")
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("Unsupported amax dtype")
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
        self.compiled, self.chunks = build(call.m, call.n, dtype, "float32" if chunks > 1 else out)
        self.finish = build(call.m, self.chunks, "float32", out)[0] if self.chunks > 1 else None

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
    register_kernel_type("AmaxFwdOp", "ascend", AscendAmaxKernel)
