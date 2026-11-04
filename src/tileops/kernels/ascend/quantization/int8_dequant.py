"""INT8 dequantization with staged code and scale tiles on Ascend."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.quantization.call_spec import (
    DequantizeCall,
    INT8DequantPerBlockFwdInterface,
    INT8DequantPerChannelFwdInterface,
    INT8DequantPerTensorFwdInterface,
)

__all__ = [
    "AscendINT8DequantPerBlockKernel",
    "AscendINT8DequantPerChannelKernel",
    "AscendINT8DequantPerTensorKernel",
]


@functools.lru_cache(maxsize=64)
def _build_channel(m: int, k: int, dtype: str):
    tile = min(16384, max(64, (k + 63) // 64 * 64))
    chunks = (k + tile - 1) // tile
    tasks = m * chunks
    blocks = min(64, tasks)

    @T.prim_func
    def main(
        Q: T.Tensor((m * k,), "int8"), S: T.Tensor((m,), "float32"), Y: T.Tensor((m * k,), dtype)
    ):
        with T.Kernel(blocks) as bx:
            codes = T.alloc_shared((tile,), "int8")
            factor = T.alloc_shared((1,), "float32")
            result = T.alloc_shared((tile,), dtype)
            for step in T.serial(T.ceildiv(tasks, blocks)):
                task = step * blocks + bx
                if task < tasks:
                    row = task // chunks
                    column = task % chunks * tile
                    count = T.min(tile, k - column)
                    begin = row * k + column
                    T.copy(Q[begin : begin + count], codes[:count])
                    T.copy(S[row : row + 1], factor)
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        multiplier = T.simd.vdup(factor[0], "float32")
                        for v in T.serial(tile // 64):
                            raw = T.simd.vld(codes[v * 64], dist="UNPK4_B8")
                            half_vector = T.simd.vcvt(raw, "float16", part=0)
                            wide = T.simd.vcvt(half_vector, "float32", part=0)
                            value = T.simd.vmul(wide, multiplier, full)
                            if dtype == "float32":
                                T.simd.vsts(result[v * 64], value, full)
                            else:
                                narrowed = T.simd.vcvt(value, dtype, full, sat=False)
                                T.simd.vsts(result[v * 64], narrowed, full, "PK_B32")
                    T.copy(result[:count], Y[begin : begin + count])

    return tilelang.compile(main, target="ascend", out_idx=-1)


@functools.lru_cache(maxsize=64)
def _build(m: int, k: int, dtype: str, mode: str):
    if mode == "channel":
        return _build_channel(m, k, dtype)
    n = m * k
    # Large flat tensors amortize DMA and VF dispatch over a larger UB tile.
    tile = 32768 if mode in {"tensor", "channel"} and n >= 1_000_000 else 8192
    blocks = min(64, (n + tile - 1) // tile)
    groups = (k + 127) // 128
    scales = 1 if mode == "tensor" else m if mode == "channel" else m * groups
    scale_tile = (
        1
        if mode == "tensor"
        else tile // k + 2
        if mode == "channel"
        else tile // 128 + tile // k + 3
    )

    @T.prim_func
    def main(
        Q: T.Tensor((n,), "int8"),
        S: T.Tensor((scales,), "float32"),
        Y: T.Tensor((n,), dtype),
    ):
        with T.Kernel(blocks) as bx:
            codes = T.alloc_shared((tile,), "int8")
            factors = T.alloc_shared((scale_tile,), "float32")
            result = T.alloc_shared((tile,), dtype)
            for step in T.serial(T.ceildiv(n, blocks * tile)):
                begin = (step * blocks + bx) * tile
                if begin < n:
                    start_scale = (
                        0
                        if mode == "tensor"
                        else begin // k
                        if mode == "channel"
                        else begin // k * groups + begin % k // 128
                    )
                    T.copy(Q[begin : begin + tile], codes)
                    T.copy(S[start_scale : start_scale + scale_tile], factors)
                    if mode == "tensor":
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            multiplier = T.simd.alloc_local((1,), "float32")
                            multiplier[0] = T.simd.vdup(factors[0], "float32")
                            for v in T.serial(tile // 64):
                                bytes_vector = T.simd.vld(codes[v * 64], dist="UNPK4_B8")
                                half_vector = T.simd.vcvt(bytes_vector, "float16", part=0)
                                float_vector = T.simd.vcvt(half_vector, "float32", part=0)
                                value = T.simd.vmul(float_vector, multiplier[0], full)
                                if dtype == "float32":
                                    T.simd.vsts(result[v * 64], value, full)
                                else:
                                    narrowed = T.simd.vcvt(value, dtype, full, sat=False)
                                    T.simd.vsts(result[v * 64], narrowed, full, "PK_B32")
                    else:
                        with T.SimtVF(threads=256):
                            for j in T.Parallel(tile, coalesced_width=T.int32(1)):
                                idx = begin + j
                                if idx < n:
                                    scale_idx = (
                                        0
                                        if mode == "tensor"
                                        else idx // k
                                        if mode == "channel"
                                        else idx // k * groups + idx % k // 128
                                    )
                                    result[j] = T.Cast(
                                        dtype,
                                        T.Cast("float32", codes[j])
                                        * factors[scale_idx - start_scale],
                                    )
                    T.copy(result, Y[begin : begin + tile])

    return tilelang.compile(main, target="ascend", out_idx=-1)


class _DequantKernel(Kernel):
    devices = frozenset({"npu"})
    supported_archs = None
    mode = "tensor"

    def __init__(self, call: DequantizeCall):
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self._run = _build(call.m, call.k, str(call.out_dtype).split(".")[-1], self.mode)

    def forward(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        self._require_devices(q=q, scale=scale)
        return self._run(q.view(-1), scale.view(-1)).view(self.call.m, self.call.k)


class AscendINT8DequantPerTensorKernel(_DequantKernel, INT8DequantPerTensorFwdInterface):
    """Convert INT8 codes and apply one FP32 scale to the whole tensor."""


class AscendINT8DequantPerChannelKernel(_DequantKernel, INT8DequantPerChannelFwdInterface):
    """Convert INT8 codes and apply one FP32 scale to each row."""

    mode = "channel"


class AscendINT8DequantPerBlockKernel(_DequantKernel, INT8DequantPerBlockFwdInterface):
    """Convert INT8 codes with one FP32 scale per 128-element row segment."""

    mode = "block"


register_kernel_type("INT8DequantPerTensorFwdOp", "ascend", AscendINT8DequantPerTensorKernel)
register_kernel_type("INT8DequantPerChannelFwdOp", "ascend", AscendINT8DequantPerChannelKernel)
register_kernel_type("INT8DequantPerBlockFwdOp", "ascend", AscendINT8DequantPerBlockKernel)
