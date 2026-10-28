"""Ascend attention implementations."""

from tileops.kernels.ascend.attention.dense import AscendGQADenseKernel

__all__ = ["AscendGQADenseKernel"]
