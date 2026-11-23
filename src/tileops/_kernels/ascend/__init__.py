"""Ascend 950 kernels, registered through the backend entry point."""

from tileops._kernels.ascend import (
    convolution,
    elementwise,
    fft,
    gemm,
    linear_attention,
    moe,
    norm,
    pool,
    quantization,
    reduction,
    rope,
    sequence_modeling,
    transform,
)

__all__ = [
    "convolution",
    "elementwise",
    "fft",
    "gemm",
    "moe",
    "pool",
    "quantization",
    "reduction",
    "rope",
    "sequence_modeling",
    "transform",
]

from tileops._kernels.ascend import attention, mamba, sampling
