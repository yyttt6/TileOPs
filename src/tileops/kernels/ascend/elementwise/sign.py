"""Ascend sign with integer SIMD classification of floating bit patterns and bounded tails."""
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
    def _body(x, y, width: int, dtype: str, floating: bool, one_bits: int, inf_bits: int):
        if floating:
            with T.SimdVF():
                bits = 16 if dtype == "int16" else 32
                lanes = 256 // (bits // 8)
                mask = T.simd.pset(bits)
                magnitude_mask = T.simd.vdup(T.cast((1 << (bits - 1)) - 1, dtype), dtype)
                zero = T.simd.vdup(T.cast(0, dtype), dtype)
                positive = T.simd.vdup(T.cast(one_bits, dtype), dtype)
                negative = T.simd.vdup(T.cast(one_bits - (1 << (bits - 1)), dtype), dtype)
                for i in range(width // lanes):
                    value = T.simd.vld(x[i * lanes])
                    magnitude = T.simd.vand(value, magnitude_mask, mask)
                    nonzero = T.simd.vcmps(magnitude, T.cast(0, dtype), mask, "gt")
                    not_nan = T.simd.vcmps(magnitude, T.cast(inf_bits, dtype), mask, "le")
                    valid = T.simd.pand(nonzero, not_nan, mask)
                    negative_sign = T.simd.vcmps(value, T.cast(0, dtype), mask, "lt")
                    signed_one = T.simd.vsel(negative, positive, negative_sign)
                    result = T.simd.vsel(signed_one, zero, valid)
                    T.simd.vsts(y[i * lanes], result, mask)
        else:
            with T.SimtVF(threads=256):
                for i in T.Parallel(width):
                    if dtype == "uint8":
                        y[i] = T.if_then_else(x[i] != 0, T.uint8(1), T.uint8(0))
                    else:
                        y[i] = T.cast(T.if_then_else(x[i] > 0, 1, T.if_then_else(x[i] < 0, -1, 0)), dtype)

    @functools.lru_cache(maxsize=256)
    def build(n, dtype, floating, one_bits, inf_bits, tile=16384, blocks=64, stages=3):
        full, tail = divmod(n, tile)
        grid = min(blocks, full) if full else 1
        padded_tail = (tail + 255) // 256 * 256
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
                            T.copy(X[begin:begin + tile], x)
                            _body(x, y, tile, dtype, floating, one_bits, inf_bits)
                            T.copy(y, Y[begin:begin + tile], l2_cache_ctrl=0)
                if tail:
                    if bx == 0:
                        xt = T.alloc_shared((padded_tail,), dtype)
                        yt = T.alloc_shared((padded_tail,), dtype)
                        T.copy(X[full * tile:n], xt[:tail])
                        _body(xt, yt, padded_tail, dtype, floating, one_bits, inf_bits)
                        T.copy(yt[:tail], Y[full * tile:n], l2_cache_ctrl=0)
        return tilelang.compile(main, target="ascend", out_idx=-1)

class AscendSignKernel(Kernel, UnaryElementwiseFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    @classmethod
    def refusal(cls, call):
        return None
    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.n_total, call.dtype)
    def __init__(self, n_total, dtype):
        super().__init__()
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
            raise TypeError("unsupported sign dtype")
        if T is None:
            raise ImportError("tilelang.ascend is required")
        self.n_total, self.dtype = n_total, dtype
        self.storage_dtype = (torch.int32 if dtype == torch.float32 else torch.int16) if dtype.is_floating_point else dtype
        if n_total:
            tile = 16384 if dtype.itemsize <= 2 else 8192 if dtype.itemsize == 4 else 4096
            stages = 3
            while (2 * tile * stages + 2 * ((n_total % tile + 255) // 256 * 256)) * dtype.itemsize > 248 * 1024:
                tile //= 2
            one_bits, inf_bits = {
                torch.float16: (0x3c00, 0x7c00),
                torch.bfloat16: (0x3f80, 0x7f80),
                torch.float32: (0x3f800000, 0x7f800000),
            }.get(dtype, (0, 0))
            self.compiled = build(n_total, str(self.storage_dtype).split('.')[-1], dtype.is_floating_point, one_bits, inf_bits, tile, 64, stages)
    def forward(self, input):
        if input.device.type != 'npu' or input.dtype != self.dtype or input.numel() != self.n_total:
            raise ValueError("input must match compiled NPU kernel")
        if not self.n_total:
            return torch.empty_like(input)
        flat = input.contiguous().reshape(-1).view(self.storage_dtype)
        return self.compiled(flat).view(self.dtype).reshape(input.shape)

import tileops.backend
tileops.backend.register_kernel_type("SignFwdOp", "ascend", AscendSignKernel)
