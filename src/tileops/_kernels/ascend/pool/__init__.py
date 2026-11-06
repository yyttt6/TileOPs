"""Ascend pooling implementations."""

from tileops._kernels.ascend.pool.adaptive import (
    AscendAdaptiveAvgPool2dKernel,
    AscendAdaptiveMaxPool2dKernel,
    AscendAdaptiveMaxPool2dWithIndicesKernel,
)
from tileops._kernels.ascend.pool.average import (
    AscendAvgPool1dKernel,
    AscendAvgPool2dKernel,
    AscendAvgPool3dKernel,
)
from tileops._kernels.ascend.pool.maximum import (
    AscendMaxPool1dIndicesKernel,
    AscendMaxPool1dKernel,
    AscendMaxPool2dIndicesKernel,
    AscendMaxPool2dKernel,
    AscendMaxPool3dIndicesKernel,
    AscendMaxPool3dKernel,
)
from tileops._kernels.ascend.pool.mean import AscendMeanPoolingFwdKernel

__all__ = [
    "AscendAdaptiveAvgPool2dKernel",
    "AscendAdaptiveMaxPool2dKernel",
    "AscendAdaptiveMaxPool2dWithIndicesKernel",
    "AscendAvgPool1dKernel",
    "AscendAvgPool2dKernel",
    "AscendAvgPool3dKernel",
    "AscendMaxPool1dIndicesKernel",
    "AscendMaxPool1dKernel",
    "AscendMaxPool2dIndicesKernel",
    "AscendMaxPool2dKernel",
    "AscendMaxPool3dIndicesKernel",
    "AscendMaxPool3dKernel",
    "AscendMeanPoolingFwdKernel",
]
