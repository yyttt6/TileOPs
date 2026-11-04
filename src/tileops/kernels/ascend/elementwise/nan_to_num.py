"""Ascend NaN/Inf replacement with native SIMD and bounded exact tails."""

import ctypes
import functools
import math
import struct
from typing import ClassVar

import torch

from tileops.kernels.elementwise.call_spec import NanToNumFwdInterface
from tileops.kernels.kernel_base import Kernel

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:
    T = None


def _replacement_key(value, dtype):
    value = float(value)
    if dtype == torch.float16:
        try:
            value = struct.unpack("<e", struct.pack("<e", value))[0]
        except OverflowError:
            value = math.copysign(float("inf"), value)
    else:
        value = ctypes.c_float(value).value
        if dtype == torch.bfloat16 and math.isfinite(value):
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
            value = struct.unpack("<f", struct.pack("<I", bits))[0]
    return value.hex()


if T is not None:

    @T.macro
    def _body(
        x,
        y,
        width: int,
        dtype: str,
        nan_value: float,
        pos_value: float,
        neg_value: float,
        saturate: bool,
    ):
        with T.SimdVF():
            lanes = 64 if dtype == "float32" else 128
            mask = T.simd.pset(32 if dtype == "float32" else 16)
            nan_replacement = T.simd.vdup(T.cast(nan_value, dtype), dtype)
            pos_replacement = T.simd.vdup(T.cast(pos_value, dtype), dtype)
            neg_replacement = T.simd.vdup(T.cast(neg_value, dtype), dtype)
            if not saturate:
                positive_inf = T.simd.vdup(T.cast(float("inf"), dtype), dtype)
                negative_inf = T.simd.vdup(T.cast(-float("inf"), dtype), dtype)
            for i in range(width // lanes):
                value = T.simd.vld(x[i * lanes])
                if saturate:
                    low = T.simd.vmax(value, neg_replacement, mask)
                    finite = T.simd.vmin(low, pos_replacement, mask)
                else:
                    # Every predicate observes the original input. Replacements
                    # may themselves be NaN/Inf and must never be reclassified.
                    positive = T.simd.vcmp(value, positive_inf, mask, "eq")
                    negative = T.simd.vcmp(value, negative_inf, mask, "eq")
                    intermediate = T.simd.vsel(pos_replacement, value, positive)
                    finite = T.simd.vsel(neg_replacement, intermediate, negative)
                is_nan = T.simd.vcmp(value, value, mask, "ne")
                result = T.simd.vsel(nan_replacement, finite, is_nan)
                if dtype == "float32":
                    T.simd.vsts(y[i * lanes], result, mask)
                else:
                    T.simd.vsts(y[i * lanes], result, mask, dist="NORM_B16")

    @functools.lru_cache(maxsize=256)
    def build(n, dtype, nan_key, pos_key, neg_key, saturate):
        nan_value, pos_value, neg_value = (float.fromhex(v) for v in (nan_key, pos_key, neg_key))
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
                            _body(x, y, tile, dtype, nan_value, pos_value, neg_value, saturate)
                            T.copy(y, Y[begin : begin + tile], l2_cache_ctrl=0)
                if tail:  # noqa: SIM102 - preserve validated TileLang DSL structure
                    if bx == 0:
                        xt = T.alloc_shared((padded,), dtype)
                        yt = xt
                        T.copy(X[full * tile : n], xt[:tail], l2_cache_ctrl=4)
                        _body(xt, yt, padded, dtype, nan_value, pos_value, neg_value, saturate)
                        T.copy(yt[:tail], Y[full * tile : n], l2_cache_ctrl=0)

        return tilelang.compile(main, target="ascend", out_idx=[1])


class AscendNanToNumKernel(Kernel, NanToNumFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call.n_total, call.dtype, call.nan, call.posinf, call.neginf)

    def __init__(self, n_total, dtype, nan=0.0, posinf=None, neginf=None):
        super().__init__()
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"Unsupported nan_to_num dtype: {dtype}")
        self.n_total, self.dtype = n_total, dtype
        largest = torch.finfo(dtype).max
        nan_key = _replacement_key(nan, dtype)
        pos_key = _replacement_key(largest if posinf is None else posinf, dtype)
        neg_key = _replacement_key(-largest if neginf is None else neginf, dtype)
        saturate = float.fromhex(pos_key) == largest and float.fromhex(neg_key) == -largest
        self.compiled = (
            build(n_total, str(dtype).removeprefix("torch."), nan_key, pos_key, neg_key, saturate)
            if n_total
            else None
        )

    def forward(self, input):
        if input.device.type != "npu" or input.dtype != self.dtype or input.numel() != self.n_total:
            raise ValueError("NanToNum input differs from the compiled NPU signature")
        if not self.n_total:
            return torch.empty_like(input)
        return self.compiled(input.contiguous().reshape(-1)).reshape(input.shape)


if T is not None:
    from tileops.backend import register_kernel_type

    register_kernel_type("NanToNumFwdOp", "ascend", AscendNanToNumKernel)
