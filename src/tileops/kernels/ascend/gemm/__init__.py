"""Ascend 950 gemm kernels."""

from tileops.kernels.ascend.gemm.bmm import AscendBmmKernel
from tileops.kernels.ascend.gemm.grouped import AscendGroupedGemmKernel

__all__ = ["AscendBmmKernel", "AscendGroupedGemmKernel"]
