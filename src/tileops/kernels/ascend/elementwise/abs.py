"""Ascend absolute value with exact floating sign-bit clearing and bounded tails."""

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
    def _body(x, y, width: int, dtype: str, floating: bool):
        if floating:
            with T.SimdVF():
                if dtype == "int16":
                    mask = T.simd.pset(16)
                    signless = T.simd.vdup(T.int16(32767), "int16")
                    for i in range(width // 128):
                        value = T.simd.vld(x[i * 128])
                        T.simd.vsts(y[i * 128], T.simd.vand(value, signless, mask), mask)
                else:
                    mask = T.simd.pset(32)
                    signless = T.simd.vdup(T.int32(2147483647), "int32")
                    for i in range(width // 64):
                        value = T.simd.vld(x[i * 64])
                        T.simd.vsts(y[i * 64], T.simd.vand(value, signless, mask), mask)
        elif dtype == "uint8":
            with T.SimdVF():
                mask = T.simd.pset(8)
                for i in range(width // 256):
                    T.simd.vsts(y[i * 256], T.simd.vld(x[i * 256]), mask)
        else:
            with T.SimtVF(threads=256):
                for i in T.Parallel(width):
                    y[i] = T.if_then_else(x[i] < 0, -x[i], x[i])

    @functools.lru_cache(maxsize=256)
    def build(n, dtype, floating, tile=16384, blocks=64, stages=3):
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
                            _body(x, y, tile, dtype, floating)
                            T.copy(y, Y[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        xt = T.alloc_shared((padded_tail,), dtype)
                        yt = xt
                        T.copy(X[full * tile : n], xt[:tail], l2_cache_ctrl=4)
                        _body(xt, yt, padded_tail, dtype, floating)
                        T.copy(yt[:tail], Y[full * tile : n], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=-1)


class AscendAbsKernel(Kernel, UnaryElementwiseFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def refusal(cls, call):
        return None

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total, dtype):
        super().__init__()
        if dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("unsupported abs dtype")
        if T is None:
            raise ImportError("tilelang.ascend is required")
        self.n_total, self.dtype = n_total, dtype
        self.storage_dtype = (
            (torch.int32 if dtype == torch.float32 else torch.int16)
            if dtype.is_floating_point
            else dtype
        )
        if n_total:
            tile = 16384 if dtype.itemsize <= 2 else 8192 if dtype.itemsize == 4 else 4096
            stages = 3
            if dtype.is_floating_point:
                tile *= 2
            buffers = 1 if dtype.is_floating_point else 2
            while (
                buffers * tile * stages + buffers * ((n_total % tile + 255) // 256 * 256)
            ) * dtype.itemsize > 248 * 1024:
                tile //= 2
            self.compiled = build(
                n_total,
                str(self.storage_dtype).split(".")[-1],
                dtype.is_floating_point,
                tile,
                64,
                stages,
            )

    def forward(self, input):
        if input.device.type != "npu" or input.dtype != self.dtype or input.numel() != self.n_total:
            raise ValueError("input must match compiled NPU kernel")
        if not self.n_total:
            return torch.empty_like(input)
        flat = input.contiguous().reshape(-1).view(self.storage_dtype)
        return self.compiled(flat).view(self.dtype).reshape(input.shape)


tileops.backend.register_kernel_type("AbsFwdOp", "ascend", AscendAbsKernel)
