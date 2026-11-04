"""Two-pass Ascend INT8 tensor quantization with a global maximum."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.ascend.quantization.int8_scale import int8_divide, int8_scale
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.quantization.call_spec import INT8QuantPerTensorFwdInterface, QuantizeCall

__all__ = ["AscendINT8QuantPerTensorKernel"]


@functools.lru_cache(maxsize=64)
def _build(n: int, dtype: str):
    tile = 16384 if n >= 1_000_000 else 8192
    blocks = min(64, (n + tile - 1) // tile)

    @T.prim_func
    def maximum(X: T.Tensor((n,), dtype), Peaks: T.Tensor((64,), "float32")):
        with T.Kernel(64) as bx:
            raw_input = T.alloc_shared((tile,), dtype)
            peak = T.alloc_shared((1,), "float32")
            with T.SimtVF(threads=32):
                for j in T.Parallel(1):
                    peak[j] = T.float32(0)
            for step in T.serial(T.ceildiv(n, 64 * tile)):
                begin = (step * 64 + bx) * tile
                if begin < n:
                    count = T.min(tile, n - begin)
                    T.copy(X[begin : begin + count], raw_input[:count])
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        acc = T.simd.alloc_local((1,), "int32")
                        acc[0] = T.simd.vdup(T.reinterpret(peak[0], "int32"), "int32")
                        for v in T.serial(tile // 64):
                            if dtype == "float32":
                                wide = T.simd.vld(raw_input[v * 64])
                            else:
                                wide = T.simd.vcvt(
                                    T.simd.vld(raw_input[v * 64], dist="UNPK_B16"),
                                    "float32",
                                    part=0,
                                )
                            valid = T.simd.vcmps(T.simd.vci(v * 64, "int32"), count, full, op="lt")
                            magnitude = T.simd.vsel(
                                T.simd.vabs(wide, full), T.simd.vdup(T.float32(0), "float32"), valid
                            )
                            acc[0] = T.simd.vmax(
                                acc[0],
                                T.simd.vcmax(T.reinterpret(magnitude, "int32x64"), full),
                                full,
                            )
                        T.simd.vsts(peak[0], T.reinterpret(acc[0], "float32x64"), one, "ONEPT_B32")
            T.copy(peak, Peaks[bx : bx + 1])

    @T.prim_func
    def quantize(
        X: T.Tensor((n,), dtype),
        Peaks: T.Tensor((64,), "float32"),
        Q: T.Tensor((n,), "int8"),
        Scale: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            peaks = T.alloc_shared((64,), "float32")
            factor = T.alloc_shared((1,), "float32")
            tiny = T.alloc_shared((1,), "int32")
            normalized_scale = T.alloc_shared((1,), "float32")
            values = T.alloc_shared((tile,), "float32")
            raw_input = T.alloc_shared((tile,), dtype)
            codes = T.alloc_shared((tile,), "int8")
            T.copy(Peaks, peaks)
            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.pset(32, "PAT_VL1")
                maximum = T.simd.vcmax(T.reinterpret(T.simd.vld(peaks[0]), "int32x64"), full)
                T.simd.vsts(factor[0], T.reinterpret(maximum, "float32x64"), one, "ONEPT_B32")
            int8_scale(factor, factor, tiny, normalized_scale)
            if bx == 0:
                T.copy(factor, Scale)
            for step in T.serial(T.ceildiv(n, blocks * tile)):
                begin = (step * blocks + bx) * tile
                if begin < n:
                    count = T.min(tile, n - begin)
                    T.copy(X[begin : begin + count], raw_input[:count])
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        for v in T.serial(tile // 64):
                            if dtype == "float32":
                                wide = T.simd.vld(raw_input[v * 64])
                            else:
                                wide = T.simd.vcvt(
                                    T.simd.vld(raw_input[v * 64], dist="UNPK_B16"),
                                    "float32",
                                    part=0,
                                )
                            valid = T.simd.vcmps(T.simd.vci(v * 64, "int32"), count, full, op="lt")
                            staged = T.simd.vsel(wide, T.simd.vdup(T.float32(0), "float32"), valid)
                            T.simd.vsts(values[v * 64], staged, full)
                    int8_divide(values, factor, tiny, normalized_scale, tile)
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        for v in T.serial(tile // 64):
                            clipped = T.simd.vmin(
                                T.simd.vmax(
                                    T.simd.vld(values[v * 64]),
                                    T.simd.vdup(T.float32(-127), "float32"),
                                    full,
                                ),
                                T.simd.vdup(T.float32(127), "float32"),
                                full,
                            )
                            integers = T.simd.vcvt(clipped, "int32", full, round="ROUND_R")
                            packed = T.simd.vcvt(
                                T.simd.vand(integers, T.simd.vdup(T.int32(255), "int32"), full),
                                "uint8",
                                full,
                                sat=False,
                                part=0,
                            )
                            T.simd.vsts(
                                codes[v * 64], T.reinterpret(packed, "int8x256"), full, "PK4_B32"
                            )
                    T.copy(codes[:count], Q[begin : begin + count])

    return (
        tilelang.compile(maximum, target="ascend", out_idx=1),
        tilelang.compile(quantize, target="ascend", out_idx=[2, 3]),
    )


class AscendINT8QuantPerTensorKernel(Kernel, INT8QuantPerTensorFwdInterface):
    """Reduce global magnitude before quantizing disjoint output tiles."""

    devices = frozenset({"npu"})
    supported_archs = None

    def __init__(self, call: QuantizeCall):
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self._maximum, self._quantize = _build(
            call.rows * call.cols, str(call.dtype).split(".")[-1]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_devices(x=x)
        flat = x.view(-1)
        peaks = self._maximum(flat)
        q, scale = self._quantize(flat, peaks)
        return q.view(self.call.rows, self.call.cols), scale


register_kernel_type("INT8QuantPerTensorFwdOp", "ascend", AscendINT8QuantPerTensorKernel)
