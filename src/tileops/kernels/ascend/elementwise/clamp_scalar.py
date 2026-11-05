"""Ascend scalar clamp with strict zero ties, optional bounds and exact tails."""

import functools
import math
import struct
from typing import ClassVar

import torch

from tileops.kernels.elementwise.call_spec import BoundedUnaryFwdInterface
from tileops.kernels.kernel_base import Kernel

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None


def _bound_key(value, dtype):
    if value is None:
        return None
    value = float(value)
    if math.isfinite(value) and abs(value) > torch.finfo(dtype).max:
        raise RuntimeError(f"value cannot be converted to {dtype} without overflow")
    if dtype == torch.float16:
        value = struct.unpack("<e", struct.pack("<e", value))[0]
    elif dtype == torch.bfloat16 and math.isfinite(value):
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        value = struct.unpack("<f", struct.pack("<I", bits))[0]
    elif dtype == torch.float32:
        value = struct.unpack("<f", struct.pack("<f", value))[0]
    return value.hex()


if T is not None:

    @T.macro
    def _body(
        x,
        y,
        width: int,
        dtype: str,
        lo: float,
        hi: float,
        has_min: bool,
        has_max: bool,
        nan_bound: bool,
    ):
        with T.SimdVF():
            lanes = 64 if dtype == "float32" else 128
            mask = T.simd.pset(32 if dtype == "float32" else 16)
            if nan_bound:
                nan = T.simd.vdup(T.cast(float("nan"), dtype), dtype)
            else:
                if has_min:
                    lower = T.simd.vdup(T.cast(lo, dtype), dtype)
                if has_max:
                    upper = T.simd.vdup(T.cast(hi, dtype), dtype)
            for i in range(width // lanes):
                if nan_bound:
                    result = nan
                else:
                    value = T.simd.vld(x[i * lanes])
                    if has_min:
                        low = T.simd.vsel(lower, value, T.simd.vcmp(value, lower, mask, "lt"))
                    else:
                        low = value
                    if has_max:
                        result = T.simd.vsel(upper, low, T.simd.vcmp(low, upper, mask, "gt"))
                    else:
                        result = low
                if dtype == "float32":
                    T.simd.vsts(y[i * lanes], result, mask)
                else:
                    T.simd.vsts(y[i * lanes], result, mask, dist="NORM_B16")

    @functools.lru_cache(maxsize=256)
    def build(n, dtype, lower_key, upper_key):
        has_min, has_max = lower_key is not None, upper_key is not None
        lo = float.fromhex(lower_key) if has_min else 0.0
        hi = float.fromhex(upper_key) if has_max else 0.0
        nan_bound = math.isnan(lo) or math.isnan(hi)
        tile, stages = (16384 if dtype == "float32" else 32768), 2
        full, tail = divmod(n, tile)
        padded = (tail + 127) // 128 * 128
        grid = min(64, full) if full else 1

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
                            _body(x, y, tile, dtype, lo, hi, has_min, has_max, nan_bound)
                            T.copy(y, Y[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        xt = T.alloc_shared((padded,), dtype)
                        yt = xt
                        T.copy(X[full * tile : n], xt[:tail], l2_cache_ctrl=4)
                        _body(xt, yt, padded, dtype, lo, hi, has_min, has_max, nan_bound)
                        T.copy(yt[:tail], Y[full * tile : n], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])


class AscendClampScalarKernel(Kernel, BoundedUnaryFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.n_total, call.dtype, call.min_val, call.max_val)

    def __init__(self, n_total, dtype, min_val=None, max_val=None):
        super().__init__()
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"Unsupported clamp dtype: {dtype}")
        if min_val is None and max_val is None:
            raise ValueError("At least one scalar bound is required")
        self.n_total, self.dtype = n_total, dtype
        self.lower_key, self.upper_key = _bound_key(min_val, dtype), _bound_key(max_val, dtype)
        self.compiled = (
            build(n_total, str(dtype).removeprefix("torch."), self.lower_key, self.upper_key)
            if n_total
            else None
        )

    def forward(self, input):
        if input.device.type != "npu" or input.dtype != self.dtype or input.numel() != self.n_total:
            raise ValueError("Clamp input differs from the compiled NPU signature")
        if not self.n_total:
            return torch.empty_like(input)
        return self.compiled(input.contiguous().reshape(-1)).reshape(input.shape)


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("ClampScalarFwdOp", "ascend", AscendClampScalarKernel)
