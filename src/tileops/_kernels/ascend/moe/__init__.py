"""Ascend implementations of MoE routing boundaries."""

from tileops._kernels.ascend.moe.pre_permute import AscendMoEPrePermuteKernel

__all__ = ["AscendMoEPrePermuteKernel"]
