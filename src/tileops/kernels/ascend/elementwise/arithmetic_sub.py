"""Ascend subtraction through the shared alpha-scaled vector pipeline."""

import functools

import torch

from tileops.kernels.ascend.elementwise.arithmetic import AscendAddKernel, T

if T is not None:
    import tilelang

    @T.macro
    def _bias_body(a, b, c, rows: int, period: int, dtype: str, alpha: float):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for row in range(rows):
                for i in range(period // 64):
                    index = row * period + i * 64
                    av = (
                        T.simd.vld(a[index])
                        if dtype == "float32"
                        else T.simd.vcvt(T.simd.vld(a[index], dist="UNPK_B16"), "float32", part=0)
                    )
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
                        T.simd.vsts(c[index], cv, mask)
                    else:
                        cv16 = T.simd.vcvt(cv, dtype)
                        T.simd.vsts(c[index], cv16, mask, dist="PK_B32")

    @functools.lru_cache(maxsize=256)
    def build_bias(n, dtype, period, alpha=-1, rows=2, blocks=64, stages=2):
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
                        _bias_body(a, b, c, rows, period, dtype, alpha)
                        T.copy(c, C[begin : begin + width], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)


class AscendSubKernel(AscendAddKernel):
    """Compute ``input - alpha * other`` without materializing a negation."""

    def __init__(self, a_shape, b_shape, dtype, alpha=1):
        negated_alpha = -alpha
        if dtype == torch.int64 and alpha == -(1 << 63):
            # Negating INT64_MIN overflows a host scalar. In modulo-2**64
            # arithmetic its additive inverse is the same signed value.
            negated_alpha = alpha
        super().__init__(a_shape, b_shape, dtype, negated_alpha)

    def _compile(self, dtype_name, tile, stages):
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
            return build_bias(self.n, dtype_name, self.period, self.alpha, rows, 64, pipeline)
        return super()._compile(dtype_name, tile, stages)


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("SubFwdOp", "ascend", AscendSubKernel)
