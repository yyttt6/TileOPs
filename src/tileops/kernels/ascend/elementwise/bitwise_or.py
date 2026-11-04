"""Ascend bitwise OR with grouped channel broadcasts."""

import functools
import math

import torch

from tileops.kernels.ascend.elementwise.bitwise_binary import BitwiseBinaryBase, T

if T is not None:
    import tilelang

    from tileops.kernels.ascend.elementwise._bitwise_fast import (
        FastBitwiseBinaryBase as BitwiseBinaryBase,
    )

    @functools.lru_cache(maxsize=128)
    def build_grouped(n, period, repeat):
        channels = n // repeat
        group = min(4, 16384 // repeat)
        tile = group * repeat
        groups = (channels + group - 1) // group
        grid = min(64, groups)

        @T.prim_func
        def main(
            A: T.Tensor((n,), "int32"), B: T.Tensor((period,), "int32"), C: T.Tensor((n,), "int32")
        ):
            with T.Kernel(grid) as bx:
                a = T.alloc_shared((tile,), "int32")
                b = T.alloc_shared((period,), "int32")
                c = T.alloc_shared((tile,), "int32")
                T.copy(B, b)
                T.annotate_buffer_versions({a: 2, c: 2})
                for step in T.Pipelined((groups + grid - 1) // grid, num_stages=2):
                    block = step * grid + bx
                    if block < groups:
                        begin = block * tile
                        count = tile
                        T.copy(A[begin : begin + count], a[:count])
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            for j in T.unroll(group, explicit=True):
                                channel = block * group + j
                                bv = T.simd.vld(b[channel % period], dist="BRC_B32")
                                for i in range(repeat // 64):
                                    av = T.simd.vld(a[j * repeat + i * 64])
                                    cv = T.simd.vor(av, bv, mask)
                                    T.simd.vsts(c[j * repeat + i * 64], cv, mask)
                        T.copy(c[:count], C[begin : begin + count], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)


class AscendBitwiseOrKernel(BitwiseBinaryBase):
    kind = "or"

    def __init__(self, a_shape, b_shape, dtype):
        shape = tuple(torch.broadcast_shapes(a_shape, b_shape))
        padded = (1,) * (len(shape) - len(b_shape)) + tuple(b_shape)
        axes = [i for i, s in enumerate(padded) if s > 1]
        if (
            T is not None
            and dtype == torch.int32
            and axes
            and math.prod(a_shape) == math.prod(shape)
        ):
            lo, hi = axes[0], axes[-1]
            period = math.prod(padded[lo : hi + 1])
            repeat = math.prod(shape[hi + 1 :])
            if (
                all(padded[i] == shape[i] for i in range(lo, hi + 1))
                and 64 <= repeat <= 4096
                and repeat % 64 == 0
                and math.prod(shape) // repeat % 4 == 0
                and period * 4 <= 8192
                and repeat * 4 * 16 + period * 4 <= 248 * 1024
            ):
                super().__init__((0,), (0,), dtype)
                self.shape, self.n, self.dtype = shape, math.prod(shape), dtype
                self.compiled = build_grouped(self.n, period, repeat)
                self.fast = True
                return
        super().__init__(a_shape, b_shape, dtype)


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("BitwiseOrFwdOp", "ascend", AscendBitwiseOrKernel)
