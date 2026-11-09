"""Ascend implementations of MoE routing boundaries."""

from tileops.kernels.ascend.moe.pre_permute import AscendMoEPrePermuteKernel
from tileops.kernels.ascend.moe.shared_expert_mlp import AscendSharedExpertMLPKernel

__all__ = [
    "AscendMoEPrePermuteKernel",
    "AscendSharedExpertMLPKernel",
]
