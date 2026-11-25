"""Per-group power-of-two scaling and packed nearest-even e2m1 quantization."""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.quantization.fp4_quant import FP4QuantFwdInterface
from tileops.backend._registry import register_kernel_type


@lru_cache(maxsize=64)
def _programs(elements, group, inplace):
    ng = elements // group
    tile = 128

    @T.prim_func
    def make_scales(x: T.Tensor((elements,), "bfloat16"), scales: T.Tensor((ng,), "uint8")):
        with T.Kernel(T.ceildiv(ng, tile)) as bx:  # noqa: SIM117 - separate TileLang block and vector scopes
            with T.SimtVF(threads=tile):
                for j in T.Parallel(tile):
                    g = bx * tile + j
                    if g < ng:
                        peak = T.alloc_local((1,), "float32")
                        peak[0] = T.float32(6 * 2.0**-126)
                        for i in T.serial(group):
                            peak[0] = T.max(peak[0], T.abs(T.cast(x[g * group + i], "float32")))
                        bits = T.reinterpret(peak[0] * T.float32(1 / 6), "uint32")
                        exponent = (bits >> T.uint32(23)) + T.cast(
                            (bits & T.uint32(0x7FFFFF)) != 0, "uint32"
                        )
                        scales[g] = T.cast(exponent, "uint8")

    @T.macro
    def quantize(x, scales, i, code):
        value = T.cast(x[i], "float32")
        sign = (T.reinterpret(value, "uint32") >> T.uint32(28)) & T.uint32(8)
        scale = T.reinterpret(T.cast(scales[i // group], "uint32") << T.uint32(23), "float32")
        a = T.abs(value) / scale
        mag = (
            T.cast(a > 0.25, "uint32")
            + T.cast(a >= 0.75, "uint32")
            + T.cast(a > 1.25, "uint32")
            + T.cast(a >= 1.75, "uint32")
            + T.cast(a > 2.5, "uint32")
            + T.cast(a >= 3.5, "uint32")
            + T.cast(a > 5, "uint32")
        )
        code[0] = mag | sign

    count = elements if inplace else elements // 2

    @T.prim_func
    def encode(
        x: T.Tensor((elements,), "bfloat16"),
        scales: T.Tensor((ng,), "uint8"),
        out: T.Tensor((count,), "bfloat16" if inplace else "uint8"),
    ):
        with T.Kernel(T.ceildiv(count, 256)) as bx:  # noqa: SIM117 - separate TileLang block and vector scopes
            with T.SimtVF(threads=256):
                for j in T.Parallel(256):
                    index = bx * 256 + j
                    if index < count:
                        code = T.alloc_local((1,), "uint32")
                        if inplace:
                            quantize(x, scales, index, code)
                            m = code[0] & T.uint32(7)
                            val = T.if_then_else(
                                m <= 4,
                                T.cast(m, "float32") * 0.5,
                                T.if_then_else(m == 5, 3.0, T.if_then_else(m == 6, 4.0, 6.0)),
                            )
                            scale = T.reinterpret(
                                T.cast(scales[index // group], "uint32") << T.uint32(23), "float32"
                            )
                            bits = T.reinterpret(val * scale, "uint32") | (
                                (code[0] & T.uint32(8)) << T.uint32(28)
                            )
                            out[index] = T.cast(T.reinterpret(bits, "float32"), "bfloat16")
                        else:
                            upper = T.alloc_local((1,), "uint32")
                            quantize(x, scales, index * 2, code)
                            quantize(x, scales, index * 2 + 1, upper)
                            out[index] = T.cast(code[0] | (upper[0] << T.uint32(4)), "uint8")

    return tilelang.compile(make_scales, target="ascend", out_idx=[1]), tilelang.compile(
        encode, target="ascend", out_idx=[2]
    )


class AscendFP4QuantKernel(Kernel, FP4QuantFwdInterface):
    devices = frozenset({"npu"})
    supported_archs = None

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call

    def forward(self, input_tensor):
        x = input_tensor.contiguous().view(-1)
        group = self.call.block_size
        if not x.numel():
            if self.call.inplace:
                return None, None
            return (
                x.new_empty((*input_tensor.shape[:-1], self.call.n // 2), dtype=torch.uint8),
                x.new_empty(
                    (*input_tensor.shape[:-1], self.call.n // group), dtype=torch.float8_e8m0fnu
                ),
            )
        scale_kernel, encode = _programs(x.numel(), group, self.call.inplace)
        scales = scale_kernel(x)
        result = encode(x, scales)
        if self.call.inplace:
            input_tensor.copy_(result.view_as(input_tensor))
            return None, None
        return (
            result.view(*input_tensor.shape[:-1], self.call.n // 2),
            scales.view(torch.float8_e8m0fnu).view(*input_tensor.shape[:-1], self.call.n // group),
        )


register_kernel_type("FP4QuantPerGroupFwdOp", "ascend", AscendFP4QuantKernel)
