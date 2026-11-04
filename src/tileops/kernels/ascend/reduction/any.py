"""Ascend any reduction with exact input-width truth and direct edge-axis gathers."""

import functools
import math
from typing import ClassVar

import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import restore_reduced, rows_for_axes
from tileops.kernels.reduction.call_spec import LogicalReduceFwdInterface

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None

# Read storage bits directly: no narrowing can erase tiny complex128 components.
_STORAGE = {
    torch.bool: (torch.int8, "int8", 1, "integer"),
    torch.uint8: (torch.int8, "int8", 1, "integer"),
    torch.int8: (torch.int8, "int8", 1, "integer"),
    torch.int16: (torch.int16, "int16", 1, "integer"),
    torch.int32: (torch.int32, "int32", 1, "integer"),
    torch.int64: (torch.int32, "int32", 2, "int64"),
    torch.float16: (torch.int16, "int16", 1, "float16"),
    torch.bfloat16: (torch.int16, "int16", 1, "float16"),
    torch.float32: (torch.int32, "int32", 1, "float32"),
    torch.float64: (torch.int32, "int32", 2, "float64"),
    torch.complex64: (torch.int32, "int32", 2, "complex64"),
    torch.complex128: (torch.int32, "int32", 4, "complex128"),
}
if T is not None:

    def truth(x, r, i, kind):
        if kind == "float16":
            return (x[r, i] & T.int16(32767)) != 0
        if kind == "float32":
            return (x[r, i] & T.int32(2147483647)) != 0
        if kind == "int64":
            return (x[r, i * 2] | x[r, i * 2 + 1]) != 0
        if kind == "float64":
            return (x[r, i * 2] | (x[r, i * 2 + 1] & T.int32(2147483647))) != 0
        if kind == "complex64":
            return ((x[r, i * 2] | x[r, i * 2 + 1]) & T.int32(2147483647)) != 0
        if kind == "complex128":
            return (
                x[r, i * 4]
                | (x[r, i * 4 + 1] & T.int32(2147483647))
                | x[r, i * 4 + 2]
                | (x[r, i * 4 + 3] & T.int32(2147483647))
            ) != 0
        return x[r, i] != 0

    @T.macro
    def _native_bytes(x, y, rows: int, count):
        with T.SimdVF():
            mask8 = T.simd.pset(8)
            mask32 = T.simd.pset(32)
            ones = T.simd.vdup(T.int8(1), "int8")
            zeros = T.simd.vdup(T.int8(0), "int8")
            for r in T.serial(rows):
                acc = T.simd.alloc_local(1, "int32")
                acc[0] = T.simd.vdup(T.int32(0), "int32")
                for i in T.serial(count // 256):
                    value = T.simd.vld(x[r, i * 256])
                    normalized = T.simd.vsel(
                        ones, zeros, T.simd.vcmps(value, T.int8(0), mask8, op="ne")
                    )
                    words = T.reinterpret(normalized, "int32x64")
                    acc[0] = T.simd.vmax(acc[0], words, mask32)
                result = T.simd.vcmax(acc[0], mask32)
                flag = T.simd.vsel(
                    T.simd.vdup(T.int32(1), "int32"),
                    T.simd.vdup(T.int32(0), "int32"),
                    T.simd.vcmps(result, T.int32(0), mask32, op="ne"),
                )
                one = T.simd.pset(32, "PAT_VL1")
                T.simd.vsts(y[r, 0], flag, one, extent=1)

    @T.macro
    def _native_words(x, y, rows: int, count, words: int, kind: str, update: bool):
        with T.SimdVF():
            mask = T.simd.pset(32)
            ones = T.simd.vdup(T.int32(1), "int32")
            zeros = T.simd.vdup(T.int32(0), "int32")
            signless = T.simd.vdup(T.int32(2147483647), "int32")
            for r in T.serial(rows):
                acc = T.simd.alloc_local(1, "int32")
                acc[0] = zeros
                for i in T.serial(count // 64):
                    av = T.simd.vld(x[r, i * 64 * words])
                    if words == 1:
                        if kind == "float32":  # noqa: SIM108 - preserve validated TileLang DSL structure
                            value = T.simd.vand(av, signless, mask)
                        else:
                            value = av
                    elif words == 2:
                        bv = T.simd.vld(x[r, i * 128 + 64])
                        low, high = T.simd.vdintlv(av, bv)
                        if kind == "float64":
                            value = T.simd.vor(low, T.simd.vand(high, signless, mask), mask)
                        elif kind == "complex64":
                            value = T.simd.vand(T.simd.vor(low, high, mask), signless, mask)
                        else:
                            value = T.simd.vor(low, high, mask)
                    else:
                        bv = T.simd.vld(x[r, i * 256 + 64])
                        cv = T.simd.vld(x[r, i * 256 + 128])
                        dv = T.simd.vld(x[r, i * 256 + 192])
                        low0, high0 = T.simd.vdintlv(av, bv)
                        low1, high1 = T.simd.vdintlv(cv, dv)
                        part0 = T.simd.vor(low0, T.simd.vand(high0, signless, mask), mask)
                        part1 = T.simd.vor(low1, T.simd.vand(high1, signless, mask), mask)
                        real, imag = T.simd.vdintlv(part0, part1)
                        value = T.simd.vor(real, imag, mask)
                    flag = T.simd.vsel(ones, zeros, T.simd.vcmps(value, T.int32(0), mask, op="ne"))
                    acc[0] = T.simd.vmax(acc[0], flag, mask)
                result = T.simd.vcmax(acc[0], mask)
                if update:
                    result = T.simd.vmax(result, T.simd.vld(y[r, 0], "BRC_B32"), mask)
                one = T.simd.pset(32, "PAT_VL1")
                T.simd.vsts(y[r, 0], result, one, extent=1)

    @T.macro
    def _native_complex(x, y, rows: int, count, words: int, kind: str, update: bool):
        with T.SimdVF():
            mask = T.simd.pset(32)
            ones = T.simd.vdup(T.int32(1), "int32")
            zeros = T.simd.vdup(T.int32(0), "int32")
            signless = T.simd.vdup(T.int32(2147483647), "int32")
            for r in T.serial(rows):
                acc = T.simd.alloc_local(4, "bool")
                for j in T.unroll(4):
                    acc[j] = T.simd.vcmps(zeros, T.int32(0), mask, op="ne")
                for group in T.serial((count + 255) // 256):
                    for j in T.unroll(4):
                        i = group * 4 + j
                        if i < count // 64:
                            if words == 1:
                                av = T.simd.vld(x[r, i * 64 * words])
                                if kind == "float32":  # noqa: SIM108 - preserve validated TileLang DSL structure
                                    value = T.simd.vand(av, signless, mask)
                                else:
                                    value = av
                            elif words == 2:
                                low, high = T.simd.vld2(x[r, i * 128], "DINTLV_B32")
                                if kind == "float64":
                                    value = T.simd.vor(low, T.simd.vand(high, signless, mask), mask)
                                elif kind == "complex64":
                                    value = T.simd.vand(T.simd.vor(low, high, mask), signless, mask)
                                else:
                                    value = T.simd.vor(low, high, mask)
                            else:
                                low0, high0 = T.simd.vld2(x[r, i * 256], "DINTLV_B32")
                                low1, high1 = T.simd.vld2(x[r, i * 256 + 128], "DINTLV_B32")
                                part0 = T.simd.vor(low0, T.simd.vand(high0, signless, mask), mask)
                                part1 = T.simd.vor(low1, T.simd.vand(high1, signless, mask), mask)
                                real, imag = T.simd.vdintlv(part0, part1)
                                value = T.simd.vor(real, imag, mask)
                            flag = T.simd.vcmps(value, T.int32(0), mask, op="ne")
                            acc[j] = T.simd.por(acc[j], flag, mask)
                combined = T.simd.por(
                    T.simd.por(acc[0], acc[1], mask), T.simd.por(acc[2], acc[3], mask), mask
                )
                result = T.simd.vcmax(T.simd.vsel(ones, zeros, combined), mask)
                if update:
                    result = T.simd.vmax(result, T.simd.vld(y[r, 0], "BRC_B32"), mask)
                one = T.simd.pset(32, "PAT_VL1")
                T.simd.vsts(y[r, 0], result, one, extent=1)

    @functools.lru_cache(maxsize=256)
    def build(m, n, raw_dtype, words, kind, outer, inner):
        byte_width = {"int8": 1, "int16": 2, "int32": 4}[raw_dtype] * words
        extent = inner if outer else n
        width = max(32, min(32768 // byte_width, tilelang.next_power_of_2(extent)))
        parts = (extent + width - 1) // width
        chunks = parts * (outer if outer else 1)
        pitch = min(width, (extent + 31) // 32 * 32)
        rows = min(4, m)
        while rows > 1 and 2 * rows * (pitch * byte_width + 128) > 192 * 1024:
            rows //= 2
        groups = (m + rows - 1) // rows
        tasks = groups * chunks
        blocks = min(64, tasks)
        native = (
            (raw_dtype == "int8" and extent % 256 == 0)
            or (raw_dtype == "int32" and extent % 64 == 0)
        ) and m % rows == 0
        out_dtype = "bool" if chunks == 1 else "int32"
        shape = (outer, m, inner * words) if outer else (m, n * words)

        @T.prim_func
        def main(X: T.Tensor(shape, raw_dtype), Y: T.Tensor((m, chunks), out_dtype)):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((rows, pitch * words), raw_dtype)
                y = T.alloc_shared((rows, 32), out_dtype)
                scratch = T.alloc_shared((rows, 8), "int32")
                T.annotate_buffer_versions({x: 2, y: 2, scratch: 2})
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
                                X[
                                    chunk // parts,
                                    row : row + valid_rows,
                                    begin * words : (begin + count) * words,
                                ],
                                x[:valid_rows, : count * words],
                            )
                        else:
                            T.copy(
                                X[row : row + valid_rows, begin * words : (begin + count) * words],
                                x[:valid_rows, : count * words],
                            )
                        if native:
                            if out_dtype == "int32":
                                if raw_dtype == "int8":
                                    _native_bytes(x, y, rows, count)
                                else:
                                    _native_words(x, y, rows, count, words, kind, False)
                            else:
                                if raw_dtype == "int8":
                                    _native_bytes(x, scratch, rows, count)
                                else:
                                    _native_words(x, scratch, rows, count, words, kind, False)
                                with T.SimtVF(threads=32):
                                    for r in T.Parallel(rows):
                                        y[r, 0] = scratch[r, 0] != 0
                        else:
                            with T.SimtVF(threads=min(256, width)):
                                acc = T.alloc_reducer((rows,), "int32", op="max")
                                T.reducer_init(acc)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < count:
                                        T.reducer_update(
                                            acc[r],
                                            T.if_then_else(
                                                truth(x, r, i, kind), T.int32(1), T.int32(0)
                                            ),
                                        )
                                values = T.alloc_fragment((rows,), "int32")
                                T.finalize_reducer(acc, values)
                                for r in T.Parallel(rows):
                                    y[r, 0] = T.cast(values[r] != 0, out_dtype)
                        T.copy(
                            y[:valid_rows, :1],
                            Y[row : row + valid_rows, chunk : chunk + 1],
                            l2_cache_ctrl=0,
                        )

        return tilelang.compile(main, target="ascend", out_idx=[1]), chunks

    @functools.lru_cache(maxsize=256)
    def build_edge_bytes(outer, m, inner):
        n = outer * inner
        rows = min(4, m)
        while rows > 1 and 2 * rows * (n + 64) > 192 * 1024:
            rows //= 2
        groups = (m + rows - 1) // rows
        blocks = min(64, groups)

        @T.prim_func
        def main(X: T.Tensor((outer, m, inner), "int8"), Y: T.Tensor((m, 1), "bool")):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((rows, n), "int8")
                scratch = T.alloc_shared((rows, 8), "int32")
                y = T.alloc_shared((rows, 32), "bool")
                T.annotate_buffer_versions({x: 2, scratch: 2, y: 2})
                for group in T.Pipelined((groups + blocks - 1) // blocks, num_stages=2):
                    row = (group * blocks + bx) * rows
                    if row < m:
                        valid_rows = T.min(rows, m - row)
                        for a in T.serial(outer):
                            T.copy(
                                X[a, row : row + valid_rows, :],
                                x[:valid_rows, a * inner : (a + 1) * inner],
                            )
                        if m % rows == 0:
                            _native_bytes(x, scratch, rows, n)
                        else:
                            with T.SimtVF(threads=256):
                                acc = T.alloc_reducer((rows,), "int32", op="max")
                                T.reducer_init(acc)
                                for r, i in T.Parallel(rows, n, coalesced_width=T.int32(1)):
                                    if r < valid_rows:
                                        T.reducer_update(
                                            acc[r],
                                            T.if_then_else(x[r, i] != 0, T.int32(1), T.int32(0)),
                                        )
                                values = T.alloc_fragment((rows,), "int32")
                                T.finalize_reducer(acc, values)
                                for r in T.Parallel(rows):
                                    scratch[r, 0] = values[r]
                        with T.SimtVF(threads=32):
                            for r in T.Parallel(rows):
                                y[r, 0] = scratch[r, 0] != 0
                        T.copy(y[:valid_rows, :1], Y[row : row + valid_rows, :], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])

    @functools.lru_cache(maxsize=256)
    def build_edge_stream(outer, m, inner, words, kind):
        width = max(64, min(32768 // (4 * words), tilelang.next_power_of_2(inner)))
        pitch = min(width, (inner + 31) // 32 * 32)
        parts = (inner + width - 1) // width
        rows = min(2, m)
        while rows > 1 and 2 * rows * pitch * words * 4 + rows * 64 > 192 * 1024:
            rows //= 2
        groups = (m + rows - 1) // rows
        blocks = min(64, groups)
        chunks = outer * parts

        @T.prim_func
        def main(X: T.Tensor((outer, m, inner * words), "int32"), Y: T.Tensor((m, 32), "bool")):
            with T.Kernel(blocks) as bx:
                x0 = T.alloc_shared((rows, pitch * words), "int32")
                x1 = T.alloc_shared((rows, pitch * words), "int32")
                state = T.alloc_shared((rows, 8), "int32")
                y = T.alloc_shared((rows, 32), "bool")
                for group in T.serial((groups + blocks - 1) // blocks):
                    row = (group * blocks + bx) * rows
                    if row < m:
                        valid_rows = T.min(rows, m - row)
                        T.fill(state, T.int32(0))
                        initial = T.min(width, inner)
                        T.copy(
                            X[0, row : row + valid_rows, : initial * words],
                            x0[:valid_rows, : initial * words],
                        )
                        for pair in T.serial((chunks + 1) // 2):
                            even = pair * 2
                            odd = even + 1
                            begin0 = (even % parts) * width
                            count0 = T.min(width, inner - begin0)
                            if odd < chunks:
                                begin1 = (odd % parts) * width
                                count1 = T.min(width, inner - begin1)
                                T.copy(
                                    X[
                                        odd // parts,
                                        row : row + valid_rows,
                                        begin1 * words : (begin1 + count1) * words,
                                    ],
                                    x1[:valid_rows, : count1 * words],
                                )
                            _native_complex(x0, state, rows, count0, words, kind, True)
                            if even + 2 < chunks:
                                next_begin = ((even + 2) % parts) * width
                                next_count = T.min(width, inner - next_begin)
                                T.copy(
                                    X[
                                        (even + 2) // parts,
                                        row : row + valid_rows,
                                        next_begin * words : (next_begin + next_count) * words,
                                    ],
                                    x0[:valid_rows, : next_count * words],
                                )
                            if odd < chunks:
                                count1 = T.min(width, inner - (odd % parts) * width)
                                _native_complex(x1, state, rows, count1, words, kind, True)
                        with T.SimtVF(threads=32):
                            for r in T.Parallel(rows):
                                y[r, 0] = state[r, 0] != 0
                        T.copy(y[:valid_rows, :1], Y[row : row + valid_rows, :1], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])

    @functools.lru_cache(maxsize=256)
    def build_finish(m, chunks):
        width = max(32, tilelang.next_power_of_2(chunks))
        pitch = (chunks + 7) // 8 * 8
        rows = min(4, m)
        groups = (m + rows - 1) // rows
        blocks = min(64, groups)

        @T.prim_func
        def main(X: T.Tensor((m, chunks), "int32"), Y: T.Tensor((m, 1), "bool")):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((rows, pitch), "int32")
                y = T.alloc_shared((rows, 32), "bool")
                for group in T.serial((groups + blocks - 1) // blocks):
                    row = (group * blocks + bx) * rows
                    if row < m:
                        valid_rows = T.min(rows, m - row)
                        T.copy(X[row : row + valid_rows, :], x[:valid_rows, :chunks])
                        with T.SimtVF(threads=min(256, width)):
                            acc = T.alloc_reducer((rows,), "int32", op="max")
                            T.reducer_init(acc)
                            for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                if r < valid_rows and i < chunks:
                                    T.reducer_update(acc[r], x[r, i])
                            values = T.alloc_fragment((rows,), "int32")
                            T.finalize_reducer(acc, values)
                            for r in T.Parallel(rows):
                                y[r, 0] = values[r] != 0
                        T.copy(y[:valid_rows, :1], Y[row : row + valid_rows, :], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])


class AscendAnyKernel(Kernel, LogicalReduceFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        if call.op_kind != "any":
            raise ValueError("Expected any")
        if call.dtype not in _STORAGE:
            raise TypeError("Unsupported logical dtype")
        self.view_dtype, raw_dtype, self.words, kind = _STORAGE[call.dtype]
        self.n = math.prod(call.shape[a] for a in call.axes)
        self.m = math.prod(call.shape) // self.n
        kept = [i for i in range(len(call.shape)) if i not in call.axes]
        self.edge = (
            bool(kept)
            and kept == list(range(kept[0], kept[-1] + 1))
            and kept[0] > 0
            and kept[-1] < len(call.shape) - 1
        )
        self.outer = math.prod(call.shape[: kept[0]]) if self.edge else 0
        self.inner = math.prod(call.shape[kept[-1] + 1 :]) if self.edge else 0
        self.stream = (
            self.edge
            and kind in ("complex64", "complex128")
            and self.inner % 64 == 0
            and self.m % 2 == 0
        )
        self.fused = self.stream or (
            self.edge
            and raw_dtype == "int8"
            and self.n <= 65536
            and self.n % 256 == 0
            and self.inner % 32 == 0
        )
        if self.stream:
            self.compiled = build_edge_stream(self.outer, self.m, self.inner, self.words, kind)
            self.chunks = 1
        elif self.fused:
            self.compiled = build_edge_bytes(self.outer, self.m, self.inner)
            self.chunks = 1
        else:
            self.compiled, self.chunks = build(
                self.m, self.n, raw_dtype, self.words, kind, self.outer, self.inner
            )
        self.output_pitch = 32 if self.stream else 1
        self.finish = build_finish(self.m, self.chunks) if self.chunks > 1 else None

    def forward(self, x):
        if tuple(x.shape) != self.call.shape or x.dtype != self.call.dtype:
            raise ValueError("Input signature changed")
        # Conjugation only changes a sign; logical truth can read the physical values.
        if x.is_conj():
            x = x.conj()
        if x.is_neg():
            x = x.resolve_neg()
        if self.edge:
            arg = (
                x.reshape(self.outer, self.m, self.inner)
                .view(self.view_dtype)
                .reshape(self.outer, self.m, self.inner * self.words)
            )
        else:
            arg = (
                rows_for_axes(x, self.call.axes)
                .view(self.view_dtype)
                .reshape(self.m, self.n * self.words)
            )
        y = self.compiled(arg)
        if self.output_pitch > 1:
            y = y[:, :1]
        if self.finish is not None:
            y = self.finish(y)
        return restore_reduced(y, self.call.shape, self.call.axes, self.call.keepdim)


if T is not None:
    register_kernel_type("AnyFwdOp", "ascend", AscendAnyKernel)
