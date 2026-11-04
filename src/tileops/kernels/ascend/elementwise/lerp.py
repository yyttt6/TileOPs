"""Ascend scalar/tensor lerp with FP32 interpolation and broadcast reuse."""

import functools
from ctypes import c_float
from typing import ClassVar

import torch

from tileops.kernels.ascend.elementwise.arithmetic import ArithmeticKernelBase
from tileops.kernels.elementwise.call_spec import LerpFwdInterface, LerpTensorFwdInterface
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
                    # Tile starts and bias periods are 64-element aligned.
                    # Reduce vector indices to avoid wide division in tail VFs.
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
                delta = T.simd.vsub(bv, av, mask)
                accumulator = T.simd.alloc_local(1, "float32")
                if abs(alpha) < 0.5:
                    accumulator[0] = av
                    T.simd.vaxpy(accumulator[0], delta, T.float32(alpha), mask)
                else:
                    accumulator[0] = bv
                    T.simd.vaxpy(accumulator[0], delta, T.float32(alpha - 1.0), mask)
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
                if tail:  # noqa: SIM102 -- compile-time guard excludes zero-sized buffers
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
                    delta = T.simd.vsub(bv, av, mask)
                    accumulator = T.simd.alloc_local(1, "float32")
                    if abs(alpha) < 0.5:
                        accumulator[0] = av
                        T.simd.vaxpy(accumulator[0], delta, T.float32(alpha), mask)
                    else:
                        accumulator[0] = bv
                        T.simd.vaxpy(accumulator[0], delta, T.float32(alpha - 1.0), mask)
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

    @T.macro
    def _tensor_body(a, b, w, c, width: int, dtype: str):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for i in range(width // 64):
                av = (
                    T.simd.vld(a[i * 64])
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(a[i * 64], dist="UNPK_B16"), "float32", part=0)
                )
                bv = (
                    T.simd.vld(b[i * 64])
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(b[i * 64], dist="UNPK_B16"), "float32", part=0)
                )
                wv = (
                    T.simd.vld(w[i * 64])
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(w[i * 64], dist="UNPK_B16"), "float32", part=0)
                )
                delta = T.simd.vsub(bv, av, mask)
                near_start = T.simd.vcmps(T.simd.vabs(wv), T.float32(0.5), mask, op="lt")
                factor = T.simd.vsel(wv, T.simd.vadds(wv, T.float32(-1), mask), near_start)
                result = T.simd.alloc_local(1, "float32")
                result[0] = T.simd.vsel(av, bv, near_start)
                T.simd.vmula(result[0], delta, factor, mask)
                if dtype == "float32":
                    T.simd.vsts(c[i * 64], result[0], mask)
                else:
                    cv = T.simd.vcvt(result[0], dtype)
                    T.simd.vsts(c[i * 64], cv, mask, dist="PK_B32")

    @functools.lru_cache(maxsize=256)
    def build_tensor(n, dtype, tile=8192, blocks=64, stages=3):
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1

        @T.prim_func
        def main(
            A: T.Tensor((n,), dtype),
            B: T.Tensor((n,), dtype),
            W: T.Tensor((n,), dtype),
            C: T.Tensor((n,), dtype),
        ):
            with T.Kernel(grid) as bx:
                if full:
                    a = T.alloc_shared((tile,), dtype)
                    b = T.alloc_shared((tile,), dtype)
                    w = T.alloc_shared((tile,), dtype)
                    c = T.alloc_shared((tile,), dtype)
                    T.annotate_buffer_versions({a: stages, b: stages, w: stages, c: stages})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(A[begin : begin + tile], a)
                            T.copy(B[begin : begin + tile], b)
                            T.copy(W[begin : begin + tile], w)
                            _tensor_body(a, b, w, c, tile, dtype)
                            T.copy(c, C[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 -- compile-time guard excludes zero-sized buffers
                    if bx == 0:
                        at = T.alloc_shared((tail,), dtype)
                        bt = T.alloc_shared((tail,), dtype)
                        wt = T.alloc_shared((tail,), dtype)
                        ct = T.alloc_shared((tail,), dtype)
                        T.copy(A[full * tile : n], at)
                        T.copy(B[full * tile : n], bt)
                        T.copy(W[full * tile : n], wt)
                        _tensor_body(at, bt, wt, ct, tail, dtype)
                        T.copy(ct, C[full * tile : n], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)


_FLOATS = (torch.float16, torch.bfloat16, torch.float32)


class AscendLerpScalarKernel(ArithmeticKernelBase, LerpFwdInterface):
    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.a_shape, call.b_shape, call.dtype, call.weight)

    def __init__(self, a_shape, b_shape, dtype, weight=0.5):
        if dtype not in _FLOATS:
            raise TypeError("lerp supports float16, bfloat16 and float32")
        # Match opmath precision before selecting the interpolation branch.
        # In particular, values just below 0.5 may round to exactly 0.5.
        super().__init__(a_shape, b_shape, dtype, c_float(weight).value)

    def _compile(self, dtype_name, tile, stages):
        channels = 4
        # Keep odd channel indexing outside the vector loop. Reduce the group
        # until it divides the channel count and fits the pipelined UB buffers.
        if self.repeat > 1:
            while channels > 1 and (
                self.period % channels != 0
                or (channels * self.repeat * 4 + self.period) * self.dtype.itemsize > 248 * 1024
            ):
                channels //= 2
            if (channels * self.repeat * 4 + self.period) * self.dtype.itemsize <= 248 * 1024:
                return build_channel(
                    self.padded, dtype_name, self.period, self.repeat, self.alpha, channels, 64, 2
                )
        return build(
            self.padded, dtype_name, self.period, self.repeat, self.alpha, tile, 64, stages
        )


class AscendLerpTensorKernel(Kernel, LerpTensorFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def refusal(cls, call):
        return None

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n, dtype):
        super().__init__()
        if dtype not in _FLOATS:
            raise TypeError("lerp supports float16, bfloat16 and float32")
        self.n, self.dtype = n, dtype
        self.padded = (n + 63) // 64 * 64
        if n:
            tile, stages = (4096 if dtype == torch.float32 else 8192), 3
            while (4 * tile * stages + 4 * (self.padded % tile)) * dtype.itemsize > 248 * 1024:
                tile //= 2
            self.compiled = build_tensor(self.padded, str(dtype).split(".")[-1], tile, 64, stages)

    def forward(self, input, end, weight):
        if end.dtype != self.dtype:
            raise ValueError("end.dtype must match input.dtype")
        if weight.dtype != self.dtype:
            raise ValueError("weight.dtype must match input.dtype")
        shape = tuple(torch.broadcast_shapes(input.shape, end.shape, weight.shape))
        if not self.n:
            return torch.empty(shape, dtype=self.dtype, device=input.device)
        xs = [x.expand(shape).contiguous().reshape(-1) for x in (input, end, weight)]
        if self.padded != self.n:
            xs = [torch.nn.functional.pad(x, (0, self.padded - self.n)) for x in xs]
        return self.compiled(*xs)[: self.n].reshape(shape)


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("LerpScalarFwdOp", "ascend", AscendLerpScalarKernel)
    register_kernel_type("LerpTensorFwdOp", "ascend", AscendLerpTensorKernel)
