"""Ascend kernels for linear attention operators."""

from tileops.kernels.ascend.linear_attention.deltanet import (
    AscendDeltaNetChunkedKernel,
    AscendDeltaNetDecodeKernel,
)

__all__ = ["AscendDeltaNetChunkedKernel", "AscendDeltaNetDecodeKernel"]
