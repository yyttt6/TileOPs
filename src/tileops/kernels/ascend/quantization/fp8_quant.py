"""Ascend FP8 scaling and explicit round-to-even E4M3FN encoding."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.ascend.quantization.fp8_tokens import build_fp8_tokens
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.quantization.call_spec import FP8QuantPerBlockFwdInterface, QuantizeCall
from tileops.kernels.quantization.fp8_quant import FP8QuantCall, FP8QuantFwdInterface

__all__ = ["AscendFP8QuantPerBlockKernel", "AscendFP8QuantPerTokenGroupKernel"]


@functools.lru_cache(maxsize=64)
def _build(m: int, k: int, dtype: str, block: bool):
    width = 128 if block else k
    height = 128 if block else 1
    cols = max(64, 1 << (height * width - 1).bit_length())
    groups_k = (k + width - 1) // width
    groups = ((m + height - 1) // height) * groups_k
    blocks = min(64, groups)

    @T.prim_func
    def main(
        X: T.Tensor((m * k,), dtype),
        Q: T.Tensor((m * k,), "uint8"),
        Scale: T.Tensor((groups,), "float32"),
    ):
        with T.Kernel(blocks) as bx:
            input_matrix = T.reshape(X, (m, k))
            output_matrix = T.reshape(Q, (m, k))
            raw = T.alloc_shared((cols,), dtype)
            raw_matrix = T.reshape(raw, (height, cols // height))
            values = T.alloc_shared((cols,), "float32")
            codes = T.alloc_shared((cols,), "uint8")
            code_matrix = T.reshape(codes, (height, cols // height))
            factor = T.alloc_shared((1,), "float32")
            tiny = T.alloc_shared((1,), "int32")
            normalized_scale = T.alloc_shared((1,), "float32")
            for step in T.serial(T.ceildiv(groups, blocks)):
                group = step * blocks + bx
                if group < groups:
                    row = group // groups_k * height
                    column = group % groups_k * width
                    real_height = T.min(height, m - row)
                    real_width = T.min(width, k - column)
                    if block:
                        T.copy(
                            input_matrix[row : row + real_height, column : column + real_width],
                            raw_matrix[:real_height, :real_width],
                        )
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            for v in T.serial(cols // 64):
                                if dtype == "float32":
                                    value = T.simd.vld(raw[v * 64])
                                else:
                                    value = T.simd.vcvt(
                                        T.simd.vld(raw[v * 64], dist="UNPK_B16"), "float32", part=0
                                    )
                                row_valid = T.simd.vcmps(
                                    T.simd.vdup(v // 2, "int32"), real_height, full, op="lt"
                                )
                                col_valid = T.simd.vcmps(
                                    T.simd.vci(v % 2 * 64, "int32"), real_width, full, op="lt"
                                )
                                valid = T.simd.pand(row_valid, col_valid, full)
                                T.simd.vsts(
                                    values[v * 64],
                                    T.simd.vsel(value, T.simd.vdup(T.float32(0), "float32"), valid),
                                    full,
                                )
                    else:
                        with T.SimtVF(threads=256):
                            for j in T.Parallel(cols, coalesced_width=T.int32(1)):
                                r = row + j // width
                                c = column + j % width
                                values[j] = T.float32(0)
                                if j < height * width and r < m and c < k:
                                    values[j] = T.cast(X[r * k + c], "float32")
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        # Positive IEEE bit patterns order finite magnitudes and preserve NaNs.
                        acc = T.simd.alloc_local((1,), "int32")
                        acc[0] = T.simd.vcmax(
                            T.reinterpret(T.simd.vabs(T.simd.vld(values[0]), full), "int32x64"),
                            full,
                        )
                        for v in T.serial(1, cols // 64):
                            bits = T.reinterpret(
                                T.simd.vabs(T.simd.vld(values[v * 64]), full), "int32x64"
                            )
                            acc[0] = T.simd.vmax(acc[0], T.simd.vcmax(bits, full), full)
                        T.simd.vsts(
                            factor[0], T.reinterpret(acc[0], "float32x64"), one, "ONEPT_B32"
                        )
                    with T.SimtVF(threads=32):
                        for j in T.Parallel(1):
                            peak_bits = T.reinterpret(factor[j], "uint32")
                            tiny[j] = 0
                            if block and peak_bits > 0 and peak_bits < T.uint32(0x04E00000):
                                tiny[j] = 1
                                exponent = peak_bits >> 23
                                units = T.if_then_else(
                                    exponent == 0,
                                    peak_bits,
                                    ((peak_bits & T.uint32(0x7FFFFF)) | T.uint32(0x800000))
                                    << T.cast(T.max(T.cast(exponent, "int32") - 1, 0), "uint32"),
                                )
                                quotient_bits = units // T.uint32(448)
                                remainder_bits = units % T.uint32(448)
                                bump = (remainder_bits > T.uint32(224)) | (
                                    (remainder_bits == T.uint32(224))
                                    & ((quotient_bits & T.uint32(1)) != 0)
                                )
                                scale_bits = quotient_bits + T.if_then_else(
                                    bump, T.uint32(1), T.uint32(0)
                                )
                                factor[j] = T.reinterpret(scale_bits, "float32")
                                normalized_scale[j] = T.if_then_else(
                                    scale_bits == T.uint32(0x800000),
                                    T.float32(1),
                                    T.reinterpret(scale_bits | T.uint32(0x3F800000), "float32")
                                    - T.float32(1),
                                )
                            else:
                                if block:
                                    factor[j] = T.if_then_else(
                                        factor[j] > 0, factor[j], T.float32(448)
                                    )
                                else:
                                    factor[j] = T.max(factor[j], T.float32(1e-4))
                    if tiny[0] != 0:
                        # Normalize through IEEE exponent bits so the SFU never sees denormals.
                        with T.SimtVF(threads=256):
                            for j in T.Parallel(cols):
                                bits = T.reinterpret(values[j], "uint32")
                                magnitude_bits = bits & T.uint32(0x7FFFFFFF)
                                normalized = T.if_then_else(
                                    magnitude_bits < T.uint32(0x800000),
                                    T.reinterpret(magnitude_bits | T.uint32(0x3F800000), "float32")
                                    - T.float32(1),
                                    T.reinterpret(magnitude_bits + T.uint32(0x3F000000), "float32"),
                                )
                                values[j] = T.reinterpret(
                                    T.reinterpret(normalized, "uint32")
                                    | (bits & T.uint32(0x80000000)),
                                    "float32",
                                )
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.pset(32, "PAT_VL1")
                        divisor = T.simd.alloc_local((1,), "float32")
                        if tiny[0] != 0:
                            divisor[0] = T.simd.vdup(normalized_scale[0], "float32")
                        else:
                            divisor[0] = T.simd.vdiv(
                                T.simd.vdup(factor[0], "float32"),
                                T.simd.vdup(T.float32(448), "float32"),
                                full,
                                precision="exact",
                            )
                            T.simd.vsts(factor[0], divisor[0], one, "ONEPT_B32")
                        for v in T.serial(cols // 64):
                            quotient = T.simd.vdiv(
                                T.simd.vld(values[v * 64]), divisor[0], full, precision="exact"
                            )
                            T.simd.vsts(values[v * 64], quotient, full)
                            clipped = T.simd.vmin(
                                T.simd.vmax(
                                    quotient, T.simd.vdup(T.float32(-448), "float32"), full
                                ),
                                T.simd.vdup(T.float32(448), "float32"),
                                full,
                            )
                            encoded = T.simd.vcvt(
                                clipped, "float8_e4m3fn", full, round="ROUND_R", sat=True, part=0
                            )
                            T.simd.vsts(
                                codes[v * 64], T.reinterpret(encoded, "uint8x256"), full, "PK4_B32"
                            )
                    with T.SimtVF(threads=256):
                        for j in T.Parallel(cols):
                            if T.isnan(values[j]):
                                sign = (T.reinterpret(values[j], "uint32") >> 24) & T.uint32(128)
                                codes[j] = T.cast(sign | T.uint32(127), "uint8")
                    T.copy(
                        code_matrix[:real_height, :real_width],
                        output_matrix[row : row + real_height, column : column + real_width],
                    )
                    T.copy(factor, Scale[group : group + 1])

    return tilelang.compile(main, target="ascend", out_idx=[1, 2])


class AscendFP8QuantPerBlockKernel(Kernel, FP8QuantPerBlockFwdInterface):
    """Quantize each 128 by 128 tile, including incomplete edge tiles."""

    devices = frozenset({"npu"})
    supported_archs = None

    def __init__(self, call: QuantizeCall):
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self._run = _build(call.rows, call.cols, str(call.dtype).split(".")[-1], True)

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_devices(w=w)
        codes, scale = self._run(w.view(-1))
        return codes.view(torch.float8_e4m3fn).view(self.call.rows, self.call.cols), scale.view(
            (self.call.rows + 127) // 128, (self.call.cols + 127) // 128
        )


class AscendFP8QuantPerTokenGroupKernel(Kernel, FP8QuantFwdInterface):
    """Quantize the final dimension of each token/group; keep the branch's return order."""

    devices = frozenset({"npu"})
    supported_archs = None

    def __init__(self, call: FP8QuantCall):
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        rows = call.batch * call.seq_len_kv * call.kv_group
        self._run = (
            build_fp8_tokens(rows, call.index_dim, str(call.dtype).split(".")[-1])
            if call.index_dim in (32, 64, 128)
            else _build(rows, call.index_dim, str(call.dtype).split(".")[-1], False)
        )

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_devices(input_tensor=input_tensor)
        codes, scale = self._run(input_tensor.view(-1))
        return scale.view(self.call.batch, self.call.seq_len_kv, self.call.kv_group), codes.view(
            torch.float8_e4m3fn
        ).view(input_tensor.shape)


register_kernel_type("FP8QuantPerBlockFwdOp", "ascend", AscendFP8QuantPerBlockKernel)
register_kernel_type("FP8QuantFwdOp", "ascend", AscendFP8QuantPerTokenGroupKernel)
