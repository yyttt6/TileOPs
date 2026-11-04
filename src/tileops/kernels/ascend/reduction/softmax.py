"""Ascend softmax: fused bounded rows and stable split statistics for wide rows."""

import functools
import math
from typing import ClassVar

import torch

from tileops.backend import register_kernel_type
from tileops.kernels.ascend.reduction.logsumexp import build as build_parts
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.call_spec import SoftmaxFwdInterface

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None

if T is not None:

    def scalar_probability(delta, total, dtype):
        logp = delta - T.log(total)
        prob = T.exp(delta) / total
        threshold = (-14 if dtype == "float16" else -126) * math.log(2)
        scale = (24 if dtype == "float16" else 133 if dtype == "bfloat16" else 149) * math.log(2)
        bits = T.cast(T.nearbyint(T.exp(logp + scale)), "int32")
        tiny = (
            T.reinterpret(bits, "float32")
            if dtype == "float32"
            else T.reinterpret(T.cast(bits, "int16"), dtype)
        )
        return T.if_then_else(logp < threshold, tiny, T.cast(prob, dtype))

    @T.macro
    def _store_probability(y, r, offset, delta, prob, logsum, out_dtype: str):
        if out_dtype == "float16":
            mask = T.simd.pset(32)
            converted = T.simd.vcvt(prob, "float16", sat=False)
            T.simd.vsts(y[r, offset], converted, mask, dist="PK_B32", extent=64)
        else:
            mask = T.simd.pset(32)
            power = 0 if out_dtype == "float16" else 64
            threshold = 2.0 ** (power + (-14 if out_dtype == "float16" else -126))
            scale = 2.0 ** (
                (24 if out_dtype == "float16" else 133 if out_dtype == "bfloat16" else 149) - power
            )
            small = T.simd.vcmps(prob, T.float32(threshold), mask, op="lt")
            scaled = T.simd.vmuls(prob, T.float32(scale), small)
            bits = T.simd.vcvt(scaled, "int32", mask=small, round="ROUND_R")
            normal_probability = T.simd.vmuls(prob, T.float32(2.0 ** (-power)), mask)
            if out_dtype == "float32":
                normal = T.reinterpret(normal_probability, "int32x64")
                picked = T.simd.vsel(bits, normal, small)
                T.simd.vsts(y[r, offset], T.reinterpret(picked, "float32x64"), mask, extent=64)
            else:
                converted = T.simd.vcvt(normal_probability, out_dtype, sat=False)
                normal = T.reinterpret(converted, "int32x64")
                picked = T.simd.vsel(bits, normal, small)
                if out_dtype == "float16":
                    packed = T.reinterpret(picked, "float16x128")
                else:
                    packed = T.reinterpret(picked, "bfloat16x128")
                T.simd.vsts(y[r, offset], packed, mask, dist="PK_B32", extent=64)

    @T.macro
    def _native_fused(x, y, rows: int, count, dtype: str, out_dtype: str):
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
                shift = mx
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
                inverse = T.simd.vdiv(
                    T.simd.vdup(T.float32(1), "float32"), summed, mask, precision="ftz_true"
                )
                logsum = summed
                for i in T.serial(count // 64):
                    if dtype == "float32":
                        value = T.simd.vld(x[r, i * 64])
                    else:
                        value = T.simd.vcvt(
                            T.simd.vld(x[r, i * 64], dist="UNPK_B16"), "float32", part=0
                        )
                    delta = T.simd.vsub(value, mx, mask)
                    prob = T.simd.vmul(
                        T.simd.vexp(
                            T.simd.vadds(
                                delta,
                                T.float32((0 if out_dtype == "float16" else 64) * math.log(2)),
                                mask,
                            ),
                            mask,
                        ),
                        inverse,
                        mask,
                    )
                    _store_probability(y, r, i * 64, delta, prob, logsum, out_dtype)

    @functools.lru_cache(maxsize=256)
    def build_fused(m, n, dtype, out_dtype):
        width = max(32, tilelang.next_power_of_2(n))
        pitch = (n + 63) // 64 * 64
        size = 4 if dtype == "float32" else 2
        out_size = 4 if out_dtype == "float32" else 2
        alias = dtype == out_dtype
        rows = min(1 if n >= 16384 else 2, m)
        while rows > 1 and rows * pitch * (size + (0 if alias else out_size)) > 128 * 1024:
            rows //= 2
        stages = 2 if 2 * rows * pitch * (size + (0 if alias else out_size)) <= 184 * 1024 else 1
        native = n % 128 == 0 and m % rows == 0
        groups = (m + rows - 1) // rows
        blocks = min(64, groups)

        @T.prim_func
        def main(X: T.Tensor((m, n), dtype), Y: T.Tensor((m, n), out_dtype)):
            with T.Kernel(blocks) as bx:
                x = T.alloc_shared((rows, pitch), dtype)
                if alias:  # noqa: SIM108 - preserve validated TileLang DSL structure
                    y = x
                else:
                    y = T.alloc_shared((rows, pitch), out_dtype)
                if stages == 2:
                    if alias:
                        T.annotate_buffer_versions({x: 2})
                    else:
                        T.annotate_buffer_versions({x: 2, y: 2})
                for group in T.Pipelined((groups + blocks - 1) // blocks, num_stages=stages):
                    row = (group * blocks + bx) * rows
                    if row < m:
                        valid_rows = T.min(rows, m - row)
                        T.copy(X[row : row + valid_rows, :], x[:valid_rows, :n])
                        if native:
                            _native_fused(x, y, rows, n, dtype, out_dtype)
                        else:
                            with T.SimtVF(threads=min(256, width)):
                                maximum = T.alloc_reducer((rows,), "float32", op="max")
                                T.reducer_init(maximum)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < n:
                                        T.reducer_update(maximum[r], T.cast(x[r, i], "float32"))
                                mx = T.alloc_fragment((rows,), "float32")
                                T.finalize_reducer(maximum, mx)
                                total = T.alloc_reducer((rows,), "float32", op="sum")
                                T.reducer_init(total)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < n:
                                        T.reducer_update(
                                            total[r], T.exp(T.cast(x[r, i], "float32") - mx[r])
                                        )
                                sm = T.alloc_fragment((rows,), "float32")
                                T.finalize_reducer(total, sm)
                                for r, i in T.Parallel(rows, width, coalesced_width=T.int32(1)):
                                    if r < valid_rows and i < n:
                                        y[r, i] = scalar_probability(
                                            T.cast(x[r, i], "float32") - mx[r], sm[r], out_dtype
                                        )
                        T.copy(y[:valid_rows, :n], Y[row : row + valid_rows, :], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])

    @functools.lru_cache(maxsize=256)
    def build_statistics(m, chunks):
        dtype = "float32"
        width = max(32, tilelang.next_power_of_2(chunks))
        rows = min(4, m)
        groups = (m + rows - 1) // rows
        blocks = min(64, groups)

        @T.prim_func
        def main(X: T.Tensor((m, chunks, 8), "float32"), Y: T.Tensor((m, 16), dtype)):
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
                            for r, i in T.Parallel(rows, 16):
                                y[r, i] = T.float32(0)
                            for r in T.Parallel(rows):
                                y[r, 0] = mx[r]
                                y[r, 8] = T.if_then_else(
                                    T.abs(mx[r]) == T.infinity("float32"),
                                    T.float32(float("nan")),
                                    sm[r],
                                )
                        T.copy(y[:valid_rows, :16], Y[row : row + valid_rows, :], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])

    @T.macro
    def _tile_write(x, y, count, mx, inverse, dtype: str, out_dtype: str, normal: bool):
        mask = T.simd.pset(32)
        for i in T.serial(count // 64):
            if dtype == "float32":
                value = T.simd.vld(x[0, i * 64])
            else:
                value = T.simd.vcvt(T.simd.vld(x[0, i * 64], dist="UNPK_B16"), "float32", part=0)
            delta = T.simd.vsub(value, mx, mask)
            if normal:
                prob = T.simd.vmul(T.simd.vexp(delta, mask), inverse, mask)
                if out_dtype == "float32":
                    T.simd.vsts(y[0, i * 64], prob, mask, extent=64)
                else:
                    T.simd.vsts(
                        y[0, i * 64],
                        T.simd.vcvt(prob, out_dtype, sat=False),
                        mask,
                        dist="PK_B32",
                        extent=64,
                    )
            else:
                prob = T.simd.vmul(
                    T.simd.vexp(T.simd.vadds(delta, T.float32(64 * math.log(2)), mask), mask),
                    inverse,
                    mask,
                )
                _store_probability(y, 0, i * 64, delta, prob, inverse, out_dtype)

    @T.macro
    def _tile_output(x, y, state, flag, count, dtype: str, out_dtype: str, n: int):
        with T.SimdVF():
            mask = T.simd.pset(32)
            one = T.simd.pset(32, "PAT_VL1")
            mx = T.simd.vld(state[0, 0], "BRC_B32")
            summed = T.simd.vld(state[0, 8], "BRC_B32")
            inverse = T.simd.vdiv(
                T.simd.vdup(T.float32(1), "float32"), summed, mask, precision="ftz_true"
            )
            if out_dtype == "float16":
                _tile_write(x, y, count, mx, inverse, dtype, out_dtype, True)
            else:
                minimum = T.simd.alloc_local(2, "float32")
                minimum[0] = T.simd.vdup(T.infinity("float32"), "float32")
                minimum[1] = minimum[0]
                for i in T.serial(count // 128):
                    if dtype == "float32":
                        av = T.simd.vld(x[0, i * 128])
                        bv = T.simd.vld(x[0, i * 128 + 64])
                    else:
                        packed = T.simd.vld(x[0, i * 128])
                        av = T.simd.vcvt(packed, "float32", part=0)
                        bv = T.simd.vcvt(packed, "float32", part=1)
                    minimum[0] = T.simd.vmin(minimum[0], av, mask)
                    minimum[1] = T.simd.vmin(minimum[1], bv, mask)
                mn = T.simd.vdupv(
                    T.simd.vcmin(T.simd.vmin(minimum[0], minimum[1], mask), mask), mask
                )
                large = T.simd.vcmps(
                    T.simd.vsub(mx, mn, mask),
                    T.float32(126 * math.log(2) - math.log(n) - 1),
                    mask,
                    op="gt",
                )
                bad = T.simd.vsel(
                    T.simd.vdup(T.int32(1), "int32"), T.simd.vdup(T.int32(0), "int32"), large
                )
                T.simd.vsts(flag[0], bad, one, extent=1)
                T.simd.mem_bar("VST_LD")
                if flag[0] == 0:
                    _tile_write(x, y, count, mx, inverse, dtype, out_dtype, True)
                else:
                    _tile_write(x, y, count, mx, inverse, dtype, out_dtype, False)

    @functools.lru_cache(maxsize=256)
    def build_output(m, n, dtype, out_dtype):
        streaming = m >= 4096 and n >= 65536
        policy = 4 if streaming else 0
        alias = dtype == out_dtype
        tile = 8192 if m < 64 or not alias else 16384 if dtype == "float32" else 32768
        width = min(tile, max(32, tilelang.next_power_of_2(n)))
        parts = (n + width - 1) // width
        blocks = min(64, m * parts)
        native = n % 128 == 0

        @T.prim_func
        def main(
            X: T.Tensor((m, n), dtype),
            S: T.Tensor((m, 16), "float32"),
            Y: T.Tensor((m, n), out_dtype),
        ):
            with T.Kernel(blocks) as bx:
                flag = T.alloc_shared((8,), "int32")
                x = T.alloc_shared((1, width), dtype)
                state = T.alloc_shared((1, 16), "float32")
                if alias:  # noqa: SIM108 - preserve validated TileLang DSL structure
                    y = x
                else:
                    y = T.alloc_shared((1, width), out_dtype)
                if alias:
                    T.annotate_buffer_versions({x: 2, state: 2})
                else:
                    T.annotate_buffer_versions({x: 2, y: 2, state: 2})
                for step in T.Pipelined((m * parts + blocks - 1) // blocks, num_stages=2):
                    task = step * blocks + bx
                    if task < m * parts:
                        row = task // parts
                        begin = (task % parts) * width
                        count = T.min(width, n - begin)
                        T.copy(
                            X[row : row + 1, begin : begin + count],
                            x[:, :count],
                            l2_cache_ctrl=policy,
                        )
                        T.copy(S[row : row + 1, :], state)
                        if native:
                            _tile_output(x, y, state, flag, count, dtype, out_dtype, n)
                        else:
                            with T.SimtVF(threads=min(256, width)):
                                for i in T.Parallel(width):
                                    if i < count:
                                        y[0, i] = scalar_probability(
                                            T.cast(x[0, i], "float32") - state[0, 0],
                                            state[0, 8],
                                            out_dtype,
                                        )
                        T.copy(
                            y[:, :count],
                            Y[row : row + 1, begin : begin + count],
                            l2_cache_ctrl=policy,
                        )

        return tilelang.compile(main, target="ascend", out_idx=[2])


class AscendSoftmaxKernel(Kernel, SoftmaxFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        if call.op_kind != "softmax":
            raise ValueError("Expected softmax")
        if call.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ) or call.out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("Unsupported softmax dtype")
        self.n = call.shape[call.axis]
        self.m = math.prod(call.shape[: call.axis] + call.shape[call.axis + 1 :])
        self.empty = self.m == 0 or self.n == 0
        if self.empty:
            return
        dtype = str(call.dtype).split(".")[-1]
        out_dtype = str(call.out_dtype).split(".")[-1]
        bytes_in = 4 if dtype == "float32" else 2
        bytes_out = 4 if out_dtype == "float32" else 2
        self.fused = self.n * (
            bytes_in + (0 if dtype == out_dtype else bytes_out)
        ) <= 128 * 1024 and (self.m >= 64 or self.n <= 8192)
        if dtype == out_dtype == "bfloat16" and self.n >= 65536:
            self.fused = False
        cached = self.m >= 4096 and self.n == 65536 and dtype == out_dtype == "float32"
        if cached:
            self.fused = True
        self.statistics = None
        self.output = None
        if self.fused:
            if cached:
                from tileops.kernels.ascend.reduction.softmax_cached import build as build_cached

                self.compiled = build_cached(self.m, self.n, dtype, out_dtype)
            elif (
                self.m >= 4096
                and self.n == 512
                and self.m % 4 == 0
                and dtype == out_dtype == "float32"
            ):
                from tileops.kernels.ascend.reduction.softmax_short import build as build_short

                self.compiled = build_short(self.m, self.n, dtype, out_dtype)
            else:
                rows = min(
                    self.m,
                    64
                    if self.n <= 256
                    else 32
                    if self.n <= 512
                    else max(1, min(16, 16384 // self.n)),
                )
                # Use cached exponentials at full vector-core occupancy when two UB tiles fit.
                buffered = self.m >= 4096 or (
                    self.m >= 64 * rows and rows * self.n * 8 <= 184 * 1024
                )
                if buffered and self.n <= 32768 and self.n % 128 == 0 and self.m % rows == 0:
                    from tileops.kernels.ascend.reduction.softmax_fused import (
                        build as build_buffered,
                    )

                    self.compiled = build_buffered(self.m, self.n, dtype, out_dtype, rows, True)
                else:
                    self.compiled = build_fused(self.m, self.n, dtype, out_dtype)
            self.chunks = 1
        else:
            parts_builder = build_parts
            if self.m >= 4096 and self.n >= 65536:
                from tileops.kernels.ascend.reduction.softmax_parts import build as parts_builder
            self.compiled, self.chunks = parts_builder(self.m, self.n, dtype, 0, 0)
            self.statistics = build_statistics(self.m, self.chunks)
            self.output = build_output(self.m, self.n, dtype, out_dtype)

    def forward(self, x):
        if tuple(x.shape) != self.call.shape or x.dtype != self.call.dtype:
            raise ValueError("Input signature changed")
        if self.empty:
            return torch.empty_like(x, dtype=self.call.out_dtype)
        if x.is_neg():
            x = x.resolve_neg()
        arg = x.movedim(self.call.axis, -1).contiguous().reshape(self.m, self.n)
        if self.fused:
            y = self.compiled(arg)
        else:
            y = self.output(arg, self.statistics(self.compiled(arg)))
        kept = self.call.shape[: self.call.axis] + self.call.shape[self.call.axis + 1 :]
        return y.reshape(*kept, self.n).movedim(-1, self.call.axis)


if T is not None:
    register_kernel_type("SoftmaxFwdOp", "ascend", AscendSoftmaxKernel)
