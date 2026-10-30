"""Ascend implementations of MoE routing boundaries."""

from tileops.kernels.ascend.moe.pre_permute import AscendMoEPrePermuteKernel

__all__ = ["AscendMoEPrePermuteKernel"]
