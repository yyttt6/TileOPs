"""Ascend implementations of MoE routing boundaries."""

from tileops.kernels.ascend.moe.permute_align import AscendMoEPermuteAlignKernel
from tileops.kernels.ascend.moe.pre_permute import AscendMoEPrePermuteKernel

__all__ = ["AscendMoEPermuteAlignKernel", "AscendMoEPrePermuteKernel"]
