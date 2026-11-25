"""Ascend 950 kernels, registered through the backend entry point."""

from tileops._kernels.ascend import (
    convolution,
    elementwise,
    fft,
    gemm,
    indexing,
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
    "indexing",
    "moe",
    "pool",
    "quantization",
    "reduction",
    "rope",
    "sequence_modeling",
    "transform",
]

from tileops._kernels.ascend import attention, mamba, sampling
