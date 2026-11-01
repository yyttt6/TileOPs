"""Vector exponential kernels with FP32 evaluation and bounded tail storage."""

import functools
from typing import ClassVar

import torch

from tileops.kernels.elementwise.call_spec import UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Kernel

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None


if T is not None:

    @T.macro
    def _body(x, y, width: int, dtype: str, minus_one: bool):
        with T.SimdVF():
            mask = T.simd.pset(32)
            for i in range(width // 64):
                v = (
                    T.simd.vld(x[i * 64])
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(x[i * 64], dist="UNPK_B16"), "float32", part=0)
                )
                result = T.simd.vexp(v, mask)
                if minus_one:
                    large = T.simd.vadds(result, T.float32(-1), mask)
                    # On [-1/32, 1/32], degree three has absolute truncation
                    # error <= exp(1/32)*(1/32)**4/4! < 4.2e-8. Outside
                    # this interval exp(x)-1 avoids severe cancellation.
                    p = T.simd.vadds(
                        T.simd.vmuls(v, T.float32(1 / 6), mask), T.float32(1 / 2), mask
                    )
                    small = T.simd.vmul(
                        v, T.simd.vadds(T.simd.vmul(v, p, mask), T.float32(1), mask), mask
                    )
                    near_zero = T.simd.vcmps(T.simd.vabs(v), T.float32(1 / 32), mask, op="le")
                    result = T.simd.vsel(small, large, near_zero)
                if dtype == "float32":
                    T.simd.vsts(y[i * 64], result, mask)
                else:
                    result16 = T.simd.vcvt(result, dtype)
                    T.simd.vsts(y[i * 64], result16, mask, dist="PK_B32")

    @functools.lru_cache(maxsize=256)
    def build(n, dtype, minus_one=False, tile=16384, blocks=64, stages=3):
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1

        @T.prim_func
        def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
            with T.Kernel(grid) as bx:
                if full:
                    x = T.alloc_shared((tile,), dtype)
                    y = T.alloc_shared((tile,), dtype)
                    T.annotate_buffer_versions({x: stages, y: stages})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(X[begin : begin + tile], x)
                            _body(x, y, tile, dtype, minus_one)
                            T.copy(y, Y[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 -- compile-time guard excludes zero-sized buffers
                    if bx == 0:
                        xt = T.alloc_shared((tail,), dtype)
                        yt = T.alloc_shared((tail,), dtype)
                        T.copy(X[full * tile : n], xt)
                        _body(xt, yt, tail, dtype, minus_one)
                        T.copy(yt, Y[full * tile : n], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)


class ExponentialKernel(Kernel, UnaryElementwiseFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    minus_one: ClassVar[bool] = False
    fp32_tile: ClassVar[int] = 8192
    fp32_stages: ClassVar[int] = 3

    @classmethod
    def refusal(cls, call):
        return None

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total, dtype):
        super().__init__()
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("exponential supports float16, bfloat16 and float32")
        if T is None:
            raise ImportError("tilelang.ascend is required")
        self.n_total, self.dtype = n_total, dtype
        self.padded = (n_total + 63) // 64 * 64
        if n_total:
            tile, stages = (
                (self.fp32_tile, self.fp32_stages) if dtype == torch.float32 else (16384, 3)
            )
            while (2 * tile * stages + 2 * (self.padded % tile)) * dtype.itemsize > 248 * 1024:
                tile //= 2
            self.kernel = build(
                self.padded, str(dtype).split(".")[-1], self.minus_one, tile, 64, stages
            )

    def forward(self, input):
        if input.device.type != "npu" or input.dtype != self.dtype or input.numel() != self.n_total:
            raise ValueError("input device, dtype and size must match the compiled NPU kernel")
        if not self.n_total:
            return torch.empty_like(input)
        flat = input.contiguous().reshape(-1)
        if self.padded != self.n_total:
            flat = torch.nn.functional.pad(flat, (0, self.padded - self.n_total))
        return self.kernel(flat)[: self.n_total].reshape(input.shape)
