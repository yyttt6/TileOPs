"""Ascend exact integer nonzero count with exact input-width truth and direct edge-axis gathers."""

import functools
import math
from typing import ClassVar

import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import restore_reduced, rows_for_axes
from tileops.kernels.reduction.call_spec import CountNonzeroFwdInterface

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
    def _native_small(x, y, rows: int, count, bits: int, kind: str):
        with T.SimdVF():
            mask32 = T.simd.pset(32)
            mask16 = T.simd.pset(16)
            mask8 = T.simd.pset(8)
            zeros32 = T.simd.vdup(T.int32(0), "int32")
            for r in T.serial(rows):
                acc = T.simd.alloc_local(1, "int32")
                acc[0] = zeros32
                for i in T.serial(count // (256 if bits == 8 else 128)):
                    if bits == 8:
                        value = T.simd.vld(x[r, i * 256])
                        flags = T.simd.vsel(
                            T.simd.vdup(T.int8(1), "int8"),
                            T.simd.vdup(T.int8(0), "int8"),
                            T.simd.vcmps(value, T.int8(0), mask8, op="ne"),
                        )
                        halves = T.simd.vadd(
                            T.simd.vunpack(flags, 0), T.simd.vunpack(flags, 1), mask16
                        )
                    else:
                        value = T.simd.vld(x[r, i * 128])
                        if kind == "float16":
                            value = T.simd.vand(value, T.simd.vdup(T.int16(32767), "int16"), mask16)
                        halves = T.simd.vsel(
                            T.simd.vdup(T.int16(1), "int16"),
                            T.simd.vdup(T.int16(0), "int16"),
                            T.simd.vcmps(value, T.int16(0), mask16, op="ne"),
                        )
                    expanded = T.simd.vadd(
                        T.simd.vunpack(halves, 0), T.simd.vunpack(halves, 1), mask32
                    )
                    acc[0] = T.simd.vadd(acc[0], expanded, mask32)
                result = T.simd.vcadd(acc[0], mask32)
                one = T.simd.pset(32, "PAT_VL1")
                T.simd.vsts(y[r, 0], result, one, extent=1)

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
                    acc[0] = T.simd.vadd(acc[0], flag, mask)
                result = T.simd.vcadd(acc[0], mask)
                if update:
                    result = T.simd.vadd(result, T.simd.vld(y[r, 0], "BRC_B32"), mask)
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
            or (raw_dtype == "int16" and extent % 128 == 0)
            or (raw_dtype == "int32" and extent % 64 == 0)
        ) and m % rows == 0
        out_dtype = "int64"
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
                            if raw_dtype == "int8":
                                _native_small(x, scratch, rows, count, 8, kind)
                            elif raw_dtype == "int16":
                                _native_small(x, scratch, rows, count, 16, kind)
                            else:
                                _native_words(x, scratch, rows, count, words, kind, False)
                            with T.SimtVF(threads=32):
                                for r in T.Parallel(rows):
                                    y[r, 0] = T.cast(scratch[r, 0], "int64")
                        else:
                            with T.SimtVF(threads=min(256, width)):
                                acc = T.alloc_reducer((rows,), "int32", op="sum")
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
                                    y[r, 0] = T.cast(values[r], "int64")
                        T.copy(
                            y[:valid_rows, :1],
                            Y[row : row + valid_rows, chunk : chunk + 1],
                            l2_cache_ctrl=0,
                        )

        return tilelang.compile(main, target="ascend", out_idx=[1]), chunks

    @functools.lru_cache(maxsize=256)
    def build_finish(m, chunks):
        width = max(32, tilelang.next_power_of_2(chunks))
        pitch = (chunks + 7) // 8 * 8
        rows = min(4, m)
        groups = (m + rows - 1) // rows
        blocks = min(64, groups)

        @T.prim_func
        def main(X: T.Tensor((m, chunks), "int64"), Y: T.Tensor((m, 1), "int64")):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((rows, pitch), "int64")
                y = T.alloc_shared((rows, 32), "int64")
                for group in T.serial((groups + blocks - 1) // blocks):
                    row = (group * blocks + bx) * rows
                    if row < m:
                        valid_rows = T.min(rows, m - row)
                        T.copy(X[row : row + valid_rows, :], x[:valid_rows, :chunks])
                        with T.SimtVF(threads=32):
                            acc = T.alloc_reducer((rows,), "int64", op="sum")
                            T.reducer_init(acc)
                            for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                if r < valid_rows and i < chunks:
                                    T.reducer_update(acc[r], x[r, i])
                            values = T.alloc_fragment((rows,), "int64")
                            T.finalize_reducer(acc, values)
                            for r in T.Parallel(rows):
                                y[r, 0] = values[r]
                        T.copy(y[:valid_rows, :1], Y[row : row + valid_rows, :], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])


class AscendCountNonzeroKernel(Kernel, CountNonzeroFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        if call.op_kind != "count_nonzero":
            raise ValueError("Expected count_nonzero")
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
        self.stream = False
        self.fused = False
        self.compiled, self.chunks = build(
            self.m, self.n, raw_dtype, self.words, kind, self.outer, self.inner
        )
        self.output_pitch = 1
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
    register_kernel_type("CountNonzeroFwdOp", "ascend", AscendCountNonzeroKernel)
