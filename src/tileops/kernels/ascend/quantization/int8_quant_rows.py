"""Ascend row and token-group symmetric INT8 quantization."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.ascend.quantization.int8_group_tiles import build_group_tiles
from tileops.kernels.ascend.quantization.int8_groups import build_fp16_groups
from tileops.kernels.ascend.quantization.int8_row_stream import build_int8_row_stream
from tileops.kernels.ascend.quantization.int8_scale import int8_divide, int8_scale
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.quantization.call_spec import (
    INT8QuantPerBlockFwdInterface,
    INT8QuantPerChannelFwdInterface,
    QuantizeCall,
    SmoothQuantFwdInterface,
)

__all__ = [
    "AscendINT8QuantPerBlockKernel",
    "AscendINT8QuantPerChannelKernel",
    "AscendSmoothQuantKernel",
]


@functools.lru_cache(maxsize=64)
def _build(rows: int, k: int, dtype: str, grouped: bool, smoothed: bool):
    width = 128 if grouped else k
    cols = max(64, (width + 63) // 64 * 64)
    groups = (k + width - 1) // width
    total = rows * groups
    blocks = min(64, total)

    @T.prim_func
    def main(
        X: T.Tensor((rows * k,), dtype),
        Smooth: T.Tensor((k,), "float32"),
        Q: T.Tensor((rows * k,), "int8"),
        Scale: T.Tensor((total,), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            magnitude = T.alloc_shared((cols,), "float32")
            raw_input = T.alloc_shared((cols,), dtype)
            smooth_values = T.alloc_shared((cols if smoothed else 1,), "float32")
            peak = T.alloc_shared((1,), "float32")
            factor = T.alloc_shared((1,), "float32")
            tiny = T.alloc_shared((1,), "int32")
            normalized_scale = T.alloc_shared((1,), "float32")
            codes = T.alloc_shared((cols,), "int8")
            for step in T.serial(T.ceildiv(total, blocks)):
                group = step * blocks + bx
                if group < total:
                    row = group // groups
                    column = group % groups * width
                    count = T.min(width, k - column)
                    T.copy(X[row * k + column : row * k + column + count], raw_input[:count])
                    if smoothed:
                        T.copy(Smooth[column : column + count], smooth_values[:count])
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        for v in T.serial(cols // 64):
                            valid = T.simd.vcmps(T.simd.vci(v * 64, "int32"), count, full, op="lt")
                            if dtype == "float32":
                                wide_input = T.simd.vld(raw_input[v * 64])
                            else:
                                wide_input = T.simd.vcvt(
                                    T.simd.vld(raw_input[v * 64], dist="UNPK_B16"),
                                    "float32",
                                    part=0,
                                )
                            staged_input = T.simd.vsel(
                                wide_input, T.simd.vdup(T.float32(0), "float32"), valid
                            )
                            T.simd.vsts(magnitude[v * 64], staged_input, full)
                            if smoothed:
                                staged_smooth = T.simd.vsel(
                                    T.simd.vld(smooth_values[v * 64]),
                                    T.simd.vdup(T.float32(1), "float32"),
                                    valid,
                                )
                                T.simd.vsts(smooth_values[v * 64], staged_smooth, full)
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        acc = T.simd.alloc_local((1,), "int32")
                        if smoothed:
                            for v in T.serial(cols // 64):
                                values = T.simd.vdiv(
                                    T.simd.vld(magnitude[v * 64]),
                                    T.simd.vld(smooth_values[v * 64]),
                                    full,
                                    precision="exact",
                                )
                                T.simd.vsts(magnitude[v * 64], values, full)
                            T.simd.mem_bar("VST_VLD")
                        acc[0] = T.simd.vcmax(
                            T.reinterpret(T.simd.vabs(T.simd.vld(magnitude[0]), full), "int32x64"),
                            full,
                        )
                        for v in T.serial(1, cols // 64):
                            acc[0] = T.simd.vmax(
                                acc[0],
                                T.simd.vcmax(
                                    T.reinterpret(
                                        T.simd.vabs(T.simd.vld(magnitude[v * 64]), full), "int32x64"
                                    ),
                                    full,
                                ),
                                full,
                            )
                        T.simd.vsts(peak[0], T.reinterpret(acc[0], "float32x64"), one, "ONEPT_B32")
                    int8_scale(peak, factor, tiny, normalized_scale)
                    int8_divide(magnitude, factor, tiny, normalized_scale, cols)
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        for v in T.serial(cols // 64):
                            quotient = T.simd.vld(magnitude[v * 64])
                            clipped = T.simd.vmin(
                                T.simd.vmax(
                                    quotient, T.simd.vdup(T.float32(-127), "float32"), full
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
                    T.copy(factor, Scale[group : group + 1])
                    T.copy(
                        codes[: T.min(width, k - column)],
                        Q[row * k + column : row * k + column + T.min(width, k - column)],
                    )

    return tilelang.compile(main, target="ascend", out_idx=[2, 3])


class _RowsKernel(Kernel):
    devices = frozenset({"npu"})
    supported_archs = None
    grouped = False
    smoothed = False

    def __init__(self, call: QuantizeCall):
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self._batched = (
            self.grouped and call.dtype == torch.float16 and call.cols % 128 == 0 and call.rows > 0
        )
        # Leave room for scalar buffers and compiler alignment in the 216 KiB UB.
        # The existing one-pass path stays in use whenever its buffers fit.
        padded = (call.cols + 63) // 64 * 64
        bytes_per_column = 4 + (4 if call.dtype == torch.float32 else 2) + 1
        if self.smoothed:
            bytes_per_column += 4
        streamed = not self.grouped and padded * bytes_per_column > 216 * 1024 - 1024
        self._run = (
            build_int8_row_stream(
                call.rows, call.cols, str(call.dtype).split(".")[-1], self.smoothed
            )
            if streamed and call.rows
            else build_fp16_groups(call.rows * call.cols)
            if self._batched
            else build_group_tiles(call.rows, call.cols, str(call.dtype).split(".")[-1])
            if self.grouped and call.rows
            else (
                _build(
                    call.rows,
                    call.cols,
                    str(call.dtype).split(".")[-1],
                    self.grouped,
                    self.smoothed,
                )
                if call.rows
                else None
            )
        )

    def forward(self, x: torch.Tensor, smooth: torch.Tensor | None = None):
        self._require_devices(x=x)
        if self.call.rows == 0:
            scale_shape = (0, (self.call.cols + 127) // 128) if self.grouped else (0,)
            return (
                torch.empty(x.shape, dtype=torch.int8, device=x.device),
                torch.empty(scale_shape, dtype=torch.float32, device=x.device),
            )
        if smooth is None:
            smooth = torch.empty((self.call.cols,), dtype=torch.float32, device=x.device)
        else:
            self._require_devices(smooth=smooth)
        q, scale = self._run(x.view(-1)) if self.grouped else self._run(x.view(-1), smooth)
        shape = (
            (self.call.rows, (self.call.cols + 127) // 128) if self.grouped else (self.call.rows,)
        )
        return q.view(self.call.rows, self.call.cols), scale.view(shape)


class AscendINT8QuantPerChannelKernel(_RowsKernel, INT8QuantPerChannelFwdInterface):
    """Quantize each matrix row using its own maximum magnitude."""


class AscendINT8QuantPerBlockKernel(_RowsKernel, INT8QuantPerBlockFwdInterface):
    """Quantize each row's groups of at most 128 elements."""

    grouped = True


class AscendSmoothQuantKernel(_RowsKernel, SmoothQuantFwdInterface):
    """Divide by smoothing factors before per-row INT8 quantization."""

    smoothed = True


register_kernel_type("INT8QuantPerChannelFwdOp", "ascend", AscendINT8QuantPerChannelKernel)
register_kernel_type("INT8QuantPerBlockFwdOp", "ascend", AscendINT8QuantPerBlockKernel)
register_kernel_type("SmoothQuantFwdOp", "ascend", AscendSmoothQuantKernel)
