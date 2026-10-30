"""Ascend 950 gemm kernels."""

from tileops.kernels.ascend.gemm.bmm import AscendBmmKernel
from tileops.kernels.ascend.gemm.gemm import AscendGemmKernel

__all__ = ["AscendBmmKernel", "AscendGemmKernel"]
