"""Ascend arithmetic with persistent tiles and on-chip broadcast reuse."""

import functools
import math
from typing import ClassVar

import torch

from tileops.kernels.elementwise.call_spec import AlphaScaledBinaryFwdInterface
from tileops.kernels.kernel_base import Kernel

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None


if T is not None:

    @T.macro
    def _body(a, b, c, width: int, begin, period: int, repeat: int, dtype: str, alpha: float):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for i in range(width // 64):
                av = (
                    T.simd.vld(a[i * 64])
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(a[i * 64], dist="UNPK_B16"), "float32", part=0)
                )
                if period and repeat > 1:
                    index = ((begin + i * 64) // repeat) % period
                    bv = (
                        T.simd.vld(b[index], dist="BRC_B32")
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(b[index], dist="BRC_B16"), "float32", part=0)
                    )
                elif period:
                    # Both the tile start and broadcast period are 64-aligned.
                    # Reduce in vector units to avoid wide division lowering
                    # for a non-power-of-two period in the tail VF.
                    index = ((begin // 64 + i) % (period // 64)) * 64
                    bv = (
                        T.simd.vld(b[index])
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(b[index], dist="UNPK_B16"), "float32", part=0)
                    )
                else:
                    bv = (
                        T.simd.vld(b[i * 64])
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(b[i * 64], dist="UNPK_B16"), "float32", part=0)
                    )
                if alpha == 1:
                    cv = T.simd.vadd(av, bv, mask)
                elif alpha == -1:
                    cv = T.simd.vsub(av, bv, mask)
                else:
                    # Fuse alpha scaling to avoid an overflowing intermediate product.
                    accumulator = T.simd.alloc_local(1, "float32")
                    accumulator[0] = av
                    T.simd.vaxpy(accumulator[0], bv, T.float32(alpha), mask)
                    cv = accumulator[0]
                if dtype == "float32":
                    T.simd.vsts(c[i * 64], cv, mask)
                else:
                    cv16 = T.simd.vcvt(cv, dtype)
                    T.simd.vsts(c[i * 64], cv16, mask, dist="PK_B32")

    @functools.lru_cache(maxsize=512)
    def build(n, dtype, period=0, repeat=1, alpha=1, tile=8192, blocks=72, stages=2):
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        b_width = period or tile
        b_tail = period or tail

        @T.prim_func
        def main(
            A: T.Tensor((n,), dtype), B: T.Tensor((period or n,), dtype), C: T.Tensor((n,), dtype)
        ):
            with T.Kernel(grid) as bx:
                if full:
                    a = T.alloc_shared((tile,), dtype)
                    b = T.alloc_shared((b_width,), dtype)
                    c = T.alloc_shared((tile,), dtype)
                    T.annotate_buffer_versions({a: stages, c: stages})
                    if period:
                        T.copy(B, b)
                    else:
                        T.annotate_buffer_versions({b: stages})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(A[begin : begin + tile], a)
                            if not period:
                                T.copy(B[begin : begin + tile], b)
                            _body(a, b, c, tile, begin, period, repeat, dtype, alpha)
                            T.copy(c, C[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        at = T.alloc_shared((tail,), dtype)
                        bt = T.alloc_shared((b_tail,), dtype)
                        ct = T.alloc_shared((tail,), dtype)
                        T.copy(A[full * tile : n], at)
                        if period:
                            T.copy(B, bt)
                        else:
                            T.copy(B[full * tile : n], bt)
                        _body(at, bt, ct, tail, full * tile, period, repeat, dtype, alpha)
                        T.copy(ct, C[full * tile : n], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @T.macro
    def _channel_body(a, b, c, channels: int, repeat: int, dtype: str, alpha: float, b_begin):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for ch in range(channels):
                bv = (
                    T.simd.vld(b[b_begin + ch], dist="BRC_B32")
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(b[b_begin + ch], dist="BRC_B16"), "float32", part=0)
                )
                for i in range(repeat // 64):
                    index = ch * repeat + i * 64
                    av = (
                        T.simd.vld(a[index])
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(a[index], dist="UNPK_B16"), "float32", part=0)
                    )
                    if alpha == 1:
                        cv = T.simd.vadd(av, bv, mask)
                    elif alpha == -1:
                        cv = T.simd.vsub(av, bv, mask)
                    else:
                        # Fuse alpha scaling to avoid an overflowing intermediate product.
                        accumulator = T.simd.alloc_local(1, "float32")
                        accumulator[0] = av
                        T.simd.vaxpy(accumulator[0], bv, T.float32(alpha), mask)
                        cv = accumulator[0]
                    if dtype == "float32":
                        T.simd.vsts(c[index], cv, mask)
                    else:
                        cv16 = T.simd.vcvt(cv, dtype)
                        T.simd.vsts(c[index], cv16, mask, dist="PK_B32")

    @functools.lru_cache(maxsize=512)
    def build_channel(n, dtype, period, repeat, alpha=1, tile=4, blocks=72, stages=2):
        width = tile * repeat
        count = n // width
        grid = min(blocks, count)
        assert repeat % 64 == 0 and period % tile == 0 and n % width == 0

        @T.prim_func
        def main(A: T.Tensor((n,), dtype), B: T.Tensor((period,), dtype), C: T.Tensor((n,), dtype)):
            with T.Kernel(grid) as bx:
                a = T.alloc_shared((width,), dtype)
                b = T.alloc_shared((period,), dtype)
                c = T.alloc_shared((width,), dtype)
                T.annotate_buffer_versions({a: stages, c: stages})
                T.copy(B, b)
                for step in T.Pipelined((count + grid - 1) // grid, num_stages=stages):
                    block = step * grid + bx
                    if block < count:
                        begin = block * width
                        T.copy(A[begin : begin + width], a)
                        _channel_body(a, b, c, tile, repeat, dtype, alpha, (block * tile) % period)
                        T.copy(c, C[begin : begin + width], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)


def layout(a_shape, b_shape):
    shape = tuple(torch.broadcast_shapes(a_shape, b_shape))
    n = math.prod(shape)
    padded = (1,) * (len(shape) - len(b_shape)) + tuple(b_shape)
    if math.prod(a_shape) == math.prod(b_shape) == n:
        return shape, 0, 1, False
    axes = [i for i, size in enumerate(padded) if size > 1]
    if math.prod(a_shape) == n and axes:
        lo, hi = axes[0], axes[-1]
        period, repeat = math.prod(padded[lo : hi + 1]), math.prod(shape[hi + 1 :])
        if (
            all(padded[i] == shape[i] for i in range(lo, hi + 1))
            and period <= 16384
            and ((repeat == 1 and period % 64 == 0) or repeat % 64 == 0)
        ):
            return shape, period, repeat, False
    return shape, 0, 1, True


class ArithmeticKernelBase(Kernel):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def refusal(cls, call):
        return None

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.a_shape, call.b_shape, call.dtype, call.alpha)

    def __init__(self, a_shape, b_shape, dtype, alpha=1):
        super().__init__()
        self.shape, self.period, self.repeat, self.expand = layout(a_shape, b_shape)
        self.dtype, self.alpha = dtype, alpha
        self.n = math.prod(self.shape)
        self.padded = (self.n + 63) // 64 * 64
        if self.n != self.padded:
            self.period, self.repeat, self.expand = 0, 1, True
        self.floating = dtype in (torch.float16, torch.bfloat16, torch.float32)
        if self.floating and self.n:
            # The 4K/three-stage FP32 path won same-host msprof searches.
            tile = 4096 if dtype == torch.float32 else 8192
            stages = 3
            # Tail allocations coexist with pipeline buffers: bound the total
            # UB footprint, including any broadcast vector, below 248 KiB.
            size = dtype.itemsize
            while (
                (2 * tile + (0 if self.period else tile)) * stages
                + self.period
                + (2 * (self.padded % tile) + (self.period or self.padded % tile))
            ) * size > 248 * 1024:
                tile //= 2
            dtype_name = str(dtype).removeprefix("torch.")
            self.compiled = self._compile(dtype_name, tile, stages)

    def _compile(self, dtype_name, tile, stages):
        channels = 4
        if (
            self.repeat > 1
            and self.period % channels == 0
            and (channels * self.repeat * 4 + self.period) * self.dtype.itemsize <= 248 * 1024
        ):
            return build_channel(
                self.padded, dtype_name, self.period, self.repeat, self.alpha, channels, 64, 2
            )
        return build(
            self.padded, dtype_name, self.period, self.repeat, self.alpha, tile, 64, stages
        )

    def forward(self, input, other):
        if not self.n:
            return torch.empty(self.shape, dtype=self.dtype, device=input.device)
        if not self.floating:
            # The manifest's integer/bool path follows the existing fallback
            # contract, outside the floating performance workload set.
            if self.dtype in (torch.uint8, torch.int8, torch.int16):
                bits = self.dtype.itemsize * 8
                alpha = int(self.alpha) % (1 << bits)
                return torch.add(input.to(torch.int32), other.to(torch.int32), alpha=alpha).to(
                    self.dtype
                )
            return torch.add(input, other, alpha=self.alpha)
        a = (
            input.expand(self.shape).contiguous().reshape(-1)
            if self.expand
            else input.contiguous().reshape(-1)
        )
        b = (
            other.expand(self.shape).contiguous().reshape(-1)
            if self.expand
            else other.contiguous().reshape(-1)
        )
        if self.padded != self.n:
            a = torch.nn.functional.pad(a, (0, self.padded - self.n))
            b = torch.nn.functional.pad(b, (0, self.padded - self.n))
        return self.compiled(a, b)[: self.n].reshape(self.shape)


class AscendAddKernel(ArithmeticKernelBase, AlphaScaledBinaryFwdInterface):
    def __init__(self, a_shape, b_shape, dtype, alpha=1):
        super().__init__(a_shape, b_shape, dtype, alpha)
        self.bool_impl = None
        if dtype == torch.bool:
            from tileops.kernels.ascend.elementwise._bitwise_fast import FastBitwiseBinaryBase

            class BoolAddKernel(FastBitwiseBinaryBase):
                kind = "or"

            # Boolean alpha is a truth value, not a byte multiplier. Native
            # torch_npu add can wrap noncanonical true bytes back to zero.
            shapes = (a_shape, b_shape) if alpha else (self.shape, self.shape)
            self.bool_impl = BoolAddKernel(*shapes, dtype)
            self.compiled = self.bool_impl.compiled

    def forward(self, input, other):
        if self.bool_impl is not None:
            if not self.alpha:
                expanded = input.expand(self.shape)
                return self.bool_impl(expanded, expanded)
            return self.bool_impl(input, other)
        return super().forward(input, other)


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("AddFwdOp", "ascend", AscendAddKernel)
