"""Ascend SIMD sigmoid with FP32 evaluation and bounded arbitrary tails."""

import functools
from typing import ClassVar

import torch

import tileops.backend
from tileops.kernels.elementwise.call_spec import UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Kernel

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None

if T is not None:

    @T.macro
    def _body(x, y, width: int, dtype: str):
        with T.SimdVF():
            mask = T.simd.pset(32)
            one = T.simd.vdup(1.0, "float32")
            for i in range(width // 64):
                value = (
                    T.simd.vld(x[i * 64])
                    if dtype == "float32"
                    else T.simd.vcvt(T.simd.vld(x[i * 64], dist="UNPK_B16"), "float32", part=0)
                )
                negative = T.simd.vmuls(value, -1.0, mask)
                denominator = T.simd.vadds(T.simd.vexp(negative, mask), 1.0, mask)
                result = T.simd.vdiv(one, denominator, mask, precision="ftz_true")
                if dtype == "float32":
                    T.simd.vsts(y[i * 64], result, mask)
                else:
                    T.simd.vsts(y[i * 64], T.simd.vcvt(result, dtype), mask, dist="PK_B32")

    @functools.lru_cache(maxsize=256)
    def build(n, dtype, tile=16384, blocks=64, stages=3):
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        padded_tail = (tail + 255) // 256 * 256

        @T.prim_func
        def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
            with T.Kernel(grid) as bx:
                if full:
                    x = T.alloc_shared((tile,), dtype)
                    y = x
                    T.annotate_buffer_versions({x: stages})
                    for step in T.Pipelined((full + grid - 1) // grid, num_stages=stages):
                        block = step * grid + bx
                        if block < full:
                            begin = block * tile
                            T.copy(X[begin : begin + tile], x, l2_cache_ctrl=4)
                            _body(x, y, tile, dtype)
                            T.copy(y, Y[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        xt = T.alloc_shared((padded_tail,), dtype)
                        yt = xt
                        T.copy(X[full * tile : n], xt[:tail], l2_cache_ctrl=4)
                        _body(xt, yt, padded_tail, dtype)
                        T.copy(yt[:tail], Y[full * tile : n], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)


class AscendSigmoidKernel(Kernel, UnaryElementwiseFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def refusal(cls, call):
        return None

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total, dtype):
        super().__init__()
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("unsupported sigmoid dtype")
        if T is None:
            raise ImportError("tilelang.ascend is required")
        self.n_total, self.dtype = n_total, dtype
        if n_total:
            tile = 32768 if dtype.itemsize <= 2 else 16384
            stages = 3
            while (
                tile * stages + ((n_total % tile + 255) // 256 * 256)
            ) * dtype.itemsize > 248 * 1024:
                tile //= 2
            self.compiled = build(n_total, str(dtype).split(".")[-1], tile, 64, stages)

    def forward(self, input):
        if input.device.type != "npu" or input.dtype != self.dtype or input.numel() != self.n_total:
            raise ValueError("input must match compiled NPU kernel")
        if not self.n_total:
            return torch.empty_like(input)
        flat = input.contiguous().reshape(-1)
        return self.compiled(flat).reshape(input.shape)


tileops.backend.register_kernel_type("SigmoidFwdOp", "ascend", AscendSigmoidKernel)
