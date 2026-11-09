"""Ascend 950 kernels, registered through the backend entry point."""

from tileops._kernels.ascend import (
    elementwise,
    gemm,
    linear_attention,
    moe,
    norm,
    pool,
    quantization,
    reduction,
    rope,
    sequence_modeling,
)

__all__ = [
    "elementwise",
    "gemm",
    "moe",
    "pool",
    "quantization",
    "reduction",
    "rope",
    "sequence_modeling",
]

from tileops._kernels.ascend import attention, mamba, sampling
