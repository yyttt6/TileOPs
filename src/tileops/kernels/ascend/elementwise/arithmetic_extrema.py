"""NaN-propagating Ascend extrema with persistent tiles and on-chip broadcasts."""

import functools

import torch

from tileops.kernels.ascend.elementwise.arithmetic import ArithmeticKernelBase
from tileops.kernels.elementwise.call_spec import BinaryElementwiseFwdInterface

# Real AutoTuner selections keyed by operation, dtype and flattened length.
# Other shapes retain the control configuration until independently measured.
_TUNED_FLAT = {
    (True, "float32", 11272192): (8192, 2),
    (False, "bfloat16", 8388608): (16384, 2),
}

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None


if T is not None:

    @T.macro
    def _extreme(av, bv, mask, maximum: bool):
        # Ascend 950 native extrema propagate NaN from either input and use
        # canonical IEEE zero ties. Both operand orders are device-tested,
        # including every FP16/BF16 NaN encoding, instead of assuming CUDA
        # fmax/fmin or CPU scalar-comparison behavior applies to this ISA.
        return T.simd.vmax(av, bv, mask) if maximum else T.simd.vmin(av, bv, mask)

    @T.macro
    def _body(a, b, c, width: int, begin, period: int, repeat: int, dtype: str, maximum: bool):
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
                    index = (begin + i * 64) % period
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
                cv = _extreme(av, bv, mask, maximum)
                if dtype == "float32":
                    T.simd.vsts(c[i * 64], cv, mask)
                else:
                    cv16 = T.simd.vcvt(cv, dtype)
                    T.simd.vsts(c[i * 64], cv16, mask, dist="PK_B32")

    @functools.lru_cache(maxsize=512)
    def build(n, dtype, period=0, repeat=1, maximum=True, tile=8192, blocks=72, stages=2):
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        b_width = period or tile
        b_tail = period or tail
        load_policy = 4 if maximum and dtype == "float32" else 0

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
                            T.copy(A[begin : begin + tile], a, l2_cache_ctrl=load_policy)
                            if not period:
                                T.copy(B[begin : begin + tile], b, l2_cache_ctrl=load_policy)
                            _body(a, b, c, tile, begin, period, repeat, dtype, maximum)
                            T.copy(c, C[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        at = T.alloc_shared((tail,), dtype)
                        bt = T.alloc_shared((b_tail,), dtype)
                        ct = T.alloc_shared((tail,), dtype)
                        T.copy(A[full * tile : n], at, l2_cache_ctrl=load_policy)
                        if period:
                            T.copy(B, bt)
                        else:
                            T.copy(B[full * tile : n], bt, l2_cache_ctrl=load_policy)
                        _body(at, bt, ct, tail, full * tile, period, repeat, dtype, maximum)
                        T.copy(ct, C[full * tile : n], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @T.macro
    def _channel_body(a, b, c, channels: int, repeat: int, dtype: str, maximum: bool, b_begin):
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
                    cv = _extreme(av, bv, mask, maximum)
                    if dtype == "float32":
                        T.simd.vsts(c[index], cv, mask)
                    else:
                        cv16 = T.simd.vcvt(cv, dtype)
                        T.simd.vsts(c[index], cv16, mask, dist="PK_B32")

    @functools.lru_cache(maxsize=512)
    def build_channel(n, dtype, period, repeat, maximum=True, tile=4, blocks=72, stages=2):
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
                        _channel_body(
                            a, b, c, tile, repeat, dtype, maximum, (block * tile) % period
                        )
                        T.copy(c, C[begin : begin + width], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @T.macro
    def _bias_body(a, b, c, rows: int, period: int, dtype: str, maximum: bool):
        with T.SimdVF():
            mask = T.simd.pset(32)
            # Reuse each converted bias vector across the rows in this tile.
            for i in range(period // 64):
                bv = (
                    T.simd.vld(b[i * 64])
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(b[i * 64], dist="UNPK_B16"), "float32", part=0)
                )
                for row in range(rows):
                    index = row * period + i * 64
                    av = (
                        T.simd.vld(a[index])
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(a[index], dist="UNPK_B16"), "float32", part=0)
                    )
                    cv = _extreme(av, bv, mask, maximum)
                    if dtype == "float32":
                        T.simd.vsts(c[index], cv, mask)
                    else:
                        cv16 = T.simd.vcvt(cv, dtype)
                        T.simd.vsts(c[index], cv16, mask, dist="PK_B32")

    @functools.lru_cache(maxsize=256)
    def build_bias(n, dtype, period, maximum=True, rows=2, blocks=64, stages=2):
        width = rows * period
        count = n // width
        grid = min(blocks, count)
        assert n % width == 0 and period % 64 == 0

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
                        _bias_body(a, b, c, rows, period, dtype, maximum)
                        T.copy(c, C[begin : begin + width], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)


class AscendExtremaKernel(ArithmeticKernelBase, BinaryElementwiseFwdInterface):
    maximum = True

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.a_shape, call.b_shape, call.dtype)

    def _compile(self, dtype_name, tile, stages):
        if dtype_name == "float32" and not self.period:
            tile, stages = 4096, 3
        if not self.period and self.n == self.padded:
            tile, stages = _TUNED_FLAT.get((self.maximum, dtype_name, self.n), (tile, stages))
        if (
            self.period == 4096
            and self.repeat == 1
            and self.dtype.itemsize == 2
            and self.n % (4 * self.period) == 0
        ):
            return build_bias(self.n, dtype_name, self.period, self.maximum, 4, 64, 2)
        if self.period and self.repeat == 1 and self.period & (self.period - 1):
            rows = 2 if self.dtype.itemsize == 2 and self.n % (2 * self.period) == 0 else 1
            pipeline = 2
            while (
                rows * self.period * 2 * pipeline + self.period
            ) * self.dtype.itemsize > 248 * 1024:
                if rows > 1:
                    rows //= 2
                else:
                    pipeline = 1
            return build_bias(self.n, dtype_name, self.period, self.maximum, rows, 64, pipeline)
        channels = 4
        # Keep odd channel indexing outside the vector loop and fit wide
        # broadcasts by reducing the pipelined channel group when necessary.
        if self.repeat > 1:
            while channels > 1 and (
                self.period % channels != 0
                or (channels * self.repeat * 4 + self.period) * self.dtype.itemsize > 248 * 1024
            ):
                channels //= 2
            if (channels * self.repeat * 4 + self.period) * self.dtype.itemsize <= 248 * 1024:
                return build_channel(
                    self.padded, dtype_name, self.period, self.repeat, self.maximum, channels, 64, 2
                )
        return build(
            self.padded, dtype_name, self.period, self.repeat, self.maximum, tile, 64, stages
        )

    def forward(self, input, other):
        if not self.floating:
            function = torch.maximum if self.maximum else torch.minimum
            return function(input, other)
        return super().forward(input, other)


class AscendMaximumKernel(AscendExtremaKernel):
    maximum = True


class AscendMinimumKernel(AscendExtremaKernel):
    maximum = False

    def __init__(self, a_shape, b_shape, dtype):
        super().__init__(a_shape, b_shape, dtype)
        self.bool_impl = None
        if dtype == torch.bool:
            from tileops.kernels.ascend.elementwise._bitwise_fast import FastBitwiseBinaryBase

            class BoolMinimumKernel(FastBitwiseBinaryBase):
                kind = "and"

            # Native minimum can AND storage bytes instead of their truth values.
            # Normalize each nonzero byte before computing Boolean minimum.
            self.bool_impl = BoolMinimumKernel(a_shape, b_shape, dtype)
            self.compiled = self.bool_impl.compiled

    def forward(self, input, other):
        if self.bool_impl is not None:
            return self.bool_impl(input, other)
        return super().forward(input, other)


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("MaximumFwdOp", "ascend", AscendMaximumKernel)
    register_kernel_type("MinimumFwdOp", "ascend", AscendMinimumKernel)
