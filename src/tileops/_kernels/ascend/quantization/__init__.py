"""Ascend quantization and dequantization implementations."""

from tileops._kernels.ascend.quantization.fp8_quant import (
    AscendFP8QuantPerBlockKernel,
    AscendFP8QuantPerTokenGroupKernel,
)
from tileops._kernels.ascend.quantization.int4_quant import AscendINT4QuantPerGroupKernel
from tileops._kernels.ascend.quantization.int8_dequant import (
    AscendINT8DequantPerBlockKernel,
    AscendINT8DequantPerChannelKernel,
    AscendINT8DequantPerTensorKernel,
)
from tileops._kernels.ascend.quantization.int8_quant_tensor import AscendINT8QuantPerTensorKernel

__all__ = [
    "AscendFP8QuantPerBlockKernel",
    "AscendFP8QuantPerTokenGroupKernel",
    "AscendINT4QuantPerGroupKernel",
    "AscendINT8DequantPerBlockKernel",
    "AscendINT8DequantPerChannelKernel",
    "AscendINT8DequantPerTensorKernel",
    "AscendINT8QuantPerBlockKernel",
    "AscendINT8QuantPerChannelKernel",
    "AscendINT8QuantPerTensorKernel",
    "AscendSmoothQuantKernel",
]

from tileops._kernels.ascend.quantization.int8_quant_rows import (
    AscendINT8QuantPerBlockKernel,
    AscendINT8QuantPerChannelKernel,
    AscendSmoothQuantKernel,
)
