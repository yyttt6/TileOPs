"""Ascend implementations of Mamba State Space Model operators."""

from tileops.kernels.ascend.mamba.scans import (
    AscendSSDChunkCumsumKernel,
    AscendSSDRecurrentKernel,
    AscendSSDStatePassingKernel,
)

__all__ = [
    "AscendCausalConv1dKernel",
    "AscendCausalConv1dUpdateKernel",
    "AscendSSDChunkCouplingKernel",
    "AscendSSDChunkCumsumKernel",
    "AscendSSDChunkScanKernel",
    "AscendSSDChunkStateKernel",
    "AscendSSDRecurrentKernel",
    "AscendSSDStatePassingKernel",
    "AscendSelectiveScanKernel",
]

from tileops.kernels.ascend.mamba.causal_conv import (
    AscendCausalConv1dKernel,
    AscendCausalConv1dUpdateKernel,
)
from tileops.kernels.ascend.mamba.chunks import (
    AscendSSDChunkCouplingKernel,
    AscendSSDChunkScanKernel,
    AscendSSDChunkStateKernel,
)
from tileops.kernels.ascend.mamba.selective_scan import AscendSelectiveScanKernel
